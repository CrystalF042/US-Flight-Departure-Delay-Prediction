from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from meteostat import Daily, Point


OPENFLIGHTS_AIRPORTS_URL = (
    "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
)

AIRPORTS_DAT_COLUMNS = [
    "id",
    "name",
    "city",
    "country",
    "iata",
    "icao",
    "lat",
    "lon",
    "altitude",
    "timezone",
    "dst",
    "tz_name",
    "type",
    "source",
]

WEATHER_BASE_COLS = ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres"]
WEATHER_JOIN_KEYS = ["fl_date"]


def load_us_airports(url: str = OPENFLIGHTS_AIRPORTS_URL) -> pd.DataFrame:
    """Load US airports with valid IATA codes from OpenFlights."""
    airports = pd.read_csv(url, header=None, names=AIRPORTS_DAT_COLUMNS)

    us_airports = airports.loc[
        (airports["country"] == "United States")
        & airports["iata"].notna()
        & (airports["iata"] != "\\N"),
        ["iata", "lat", "lon", "altitude", "tz_name"],
    ].copy()

    us_airports["iata"] = us_airports["iata"].astype("string").str.strip()
    us_airports["lat"] = pd.to_numeric(us_airports["lat"], errors="coerce")
    us_airports["lon"] = pd.to_numeric(us_airports["lon"], errors="coerce")

    us_airports = (
        us_airports.dropna(subset=["iata", "lat", "lon"])
        .drop_duplicates(subset=["iata"])
        .reset_index(drop=True)
    )
    return us_airports


def fetch_daily_weather_for_airports(
    airport_coords: pd.DataFrame,
    start: str | datetime,
    end: str | datetime,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch daily Meteostat weather for each airport IATA code."""
    start_dt = pd.Timestamp(start).to_pydatetime()
    end_dt = pd.Timestamp(end).to_pydatetime()

    weather_dfs: list[pd.DataFrame] = []
    skip_reasons: dict[str, str] = {}

    required_cols = {"iata", "lat", "lon"}
    missing = required_cols - set(airport_coords.columns)
    if missing:
        raise ValueError(f"airport_coords is missing required columns: {sorted(missing)}")

    coords = (
        airport_coords.loc[:, ["iata", "lat", "lon"]]
        .dropna(subset=["iata", "lat", "lon"])
        .drop_duplicates(subset=["iata"])
        .copy()
    )

    for _, row in coords.iterrows():
        iata = str(row["iata"]).strip()
        try:
            point = Point(float(row["lat"]), float(row["lon"]))
            weather_data = Daily(point, start_dt, end_dt).fetch()

            if weather_data is None or weather_data.empty:
                skip_reasons[iata] = "no data"
                continue

            weather_data = weather_data.copy()
            weather_data["iata"] = iata
            weather_dfs.append(weather_data)
        except Exception as exc:
            skip_reasons[iata] = str(exc)[:80]

    if not weather_dfs:
        empty = pd.DataFrame(columns=["iata", "fl_date", *WEATHER_BASE_COLS])
        return empty, skip_reasons

    weather_df = pd.concat(weather_dfs, ignore_index=False).reset_index()
    weather_df = weather_df.rename(columns={"time": "fl_date"})
    weather_df = weather_df[["iata", "fl_date", *WEATHER_BASE_COLS]].copy()
    weather_df["fl_date"] = pd.to_datetime(weather_df["fl_date"], errors="coerce")

    return weather_df, skip_reasons


def prepare_weather_table(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the weather table before joining."""
    df = weather_df.copy()

    expected_cols = {"iata", "fl_date", *WEATHER_BASE_COLS}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"weather_df is missing required columns: {sorted(missing)}")

    df["iata"] = df["iata"].astype("string").str.strip()
    df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")

    for col in WEATHER_BASE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["iata", "fl_date"]).copy()
    return df


def _prefixed_weather_table(weather_df: pd.DataFrame, airport_key: str) -> pd.DataFrame:
    """Rename weather columns for origin or dest joins."""
    if airport_key not in {"origin", "dest"}:
        raise ValueError("airport_key must be 'origin' or 'dest'")

    rename_map = {"iata": airport_key}
    rename_map.update({col: f"{airport_key}_{col}" for col in WEATHER_BASE_COLS})
    return weather_df.rename(columns=rename_map)


def merge_weather_features(
    flights_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    drop_missing_temp: bool = True,
) -> pd.DataFrame:
    """Join origin and destination weather to the flights table."""
    df = flights_df.copy()
    df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")

    if "month" not in df.columns:
        df["month"] = df["fl_date"].dt.month.astype("Int64")

    weather_df = prepare_weather_table(weather_df)

    origin_weather = _prefixed_weather_table(weather_df, "origin")
    dest_weather = _prefixed_weather_table(weather_df, "dest")

    df = df.merge(origin_weather, on=["origin", *WEATHER_JOIN_KEYS], how="left")
    df = df.merge(dest_weather, on=["dest", *WEATHER_JOIN_KEYS], how="left")

    if drop_missing_temp:
        df = df.dropna(subset=["origin_tavg", "dest_tavg"]).copy()

    return df


def impute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Impute weather columns after join.

    Rules copied from the notebook:
    - snow / precipitation -> fill with 0
    - wind speed / pressure -> airport-month mean, then global median
    - min / max temperature -> airport-month mean, then global median
    """
    df = df.copy()

    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["fl_date"], errors="coerce").dt.month.astype("Int64")

    zero_fill_cols = ["origin_snow", "dest_snow", "origin_prcp", "dest_prcp"]
    mean_then_median_cols = ["origin_wspd", "dest_wspd", "origin_pres", "dest_pres"]
    temp_mean_then_median_cols = ["origin_tmin", "origin_tmax", "dest_tmin", "dest_tmax"]

    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in mean_then_median_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            airport_col = "origin" if col.startswith("origin") else "dest"
            df[col] = df.groupby([airport_col, "month"], dropna=False)[col].transform(
                lambda x: x.fillna(x.mean())
            )
            df[col] = df[col].fillna(df[col].median())

    for col in temp_mean_then_median_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            airport_col = "origin" if col.startswith("origin") else "dest"
            df[col] = df.groupby([airport_col, "month"], dropna=False)[col].transform(
                lambda x: x.fillna(x.mean())
            )
            df[col] = df[col].fillna(df[col].median())

    return df


def _wind_category(series: pd.Series) -> pd.Series:
    """0=calm, 1=breezy, 2=windy, 3=high_wind."""
    return pd.cut(
        series,
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=[0, 1, 2, 3],
    ).astype("Int64")


def add_weather_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create bad-weather flags and wind categories."""
    df = df.copy()

    df["origin_bad_weather"] = (
        (pd.to_numeric(df["origin_prcp"], errors="coerce") > 10)
        | (pd.to_numeric(df["origin_snow"], errors="coerce") > 0)
        | (pd.to_numeric(df["origin_wspd"], errors="coerce") > 30)
    ).astype("Int64")

    df["dest_bad_weather"] = (
        (pd.to_numeric(df["dest_prcp"], errors="coerce") > 10)
        | (pd.to_numeric(df["dest_snow"], errors="coerce") > 0)
        | (pd.to_numeric(df["dest_wspd"], errors="coerce") > 30)
    ).astype("Int64")

    df["origin_wind_cat"] = _wind_category(pd.to_numeric(df["origin_wspd"], errors="coerce"))
    df["dest_wind_cat"] = _wind_category(pd.to_numeric(df["dest_wspd"], errors="coerce"))

    return df


def build_weather_enriched_flights(
    flights_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    drop_missing_temp: bool = True,
) -> pd.DataFrame:
    """Full weather pipeline for an already cleaned / feature-engineered flights table."""
    df = merge_weather_features(
        flights_df=flights_df,
        weather_df=weather_df,
        drop_missing_temp=drop_missing_temp,
    )
    df = impute_weather_features(df)
    df = add_weather_derived_features(df)
    return df


def build_weather_enriched_flights_from_live_sources(
    flights_df: pd.DataFrame,
    start: str | datetime,
    end: str | datetime,
    airports_url: str = OPENFLIGHTS_AIRPORTS_URL,
    drop_missing_temp: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Convenience wrapper:
    1. load airport coordinates
    2. fetch daily weather
    3. join + impute + add derived weather features
    """
    airport_coords = load_us_airports(airports_url)
    weather_df, skip_reasons = fetch_daily_weather_for_airports(
        airport_coords=airport_coords,
        start=start,
        end=end,
    )
    flights_weather = build_weather_enriched_flights(
        flights_df=flights_df,
        weather_df=weather_df,
        drop_missing_temp=drop_missing_temp,
    )
    return flights_weather, weather_df, skip_reasons


__all__ = [
    "load_us_airports",
    "fetch_daily_weather_for_airports",
    "prepare_weather_table",
    "merge_weather_features",
    "impute_weather_features",
    "add_weather_derived_features",
    "build_weather_enriched_flights",
    "build_weather_enriched_flights_from_live_sources",
]
