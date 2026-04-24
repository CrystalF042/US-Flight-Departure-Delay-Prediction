from __future__ import annotations

import numpy as np
import pandas as pd

from Model.data_cleaning import clean_flight_data


DELAY_COMPONENT_COLS = [
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]


def _get_season(month):
    if pd.isna(month):
        return pd.NA
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "fall"


def _delay_bucket(delay):
    if pd.isna(delay):
        return "unknown"
    if delay <= 15:
        return "on_time"
    if delay <= 60:
        return "minor"
    if delay <= 180:
        return "major"
    return "extreme"


def _distance_bucket(distance):
    if pd.isna(distance):
        return pd.NA
    if distance < 500:
        return "short"
    if distance < 1500:
        return "medium"
    return "long"


def _to_datetime_series(series: pd.Series) -> pd.Series:
    """
    Robustly parse a column like crs_dep_time that may be:
    - '12:52:00'
    - '12:52'
    - python datetime.time objects rendered as strings
    """
    s = series.astype("string").str.strip()

    dt = pd.to_datetime(s, format="%H:%M:%S", errors="coerce")
    missing = dt.isna()

    if missing.any():
        dt2 = pd.to_datetime(s[missing], format="%H:%M", errors="coerce")
        dt.loc[missing] = dt2

    return dt


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature engineering columns to an already cleaned flight dataframe.

    Expected input:
    cleaned dataframe from Model.data_cleaning.clean_flight_data(...)

    Returns:
    dataframe with additional engineered features
    """
    df = df.copy()

    df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")

    numeric_cols = [
        "dep_delay",
        "arr_delay",
        "crs_elapsed_time",
        "actual_elapsed_time",
        "air_time",
        "taxi_out",
        "taxi_in",
        "distance",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------
    # Step 1: Time-Based Features
    # -------------------------
    df["day_name"] = df["fl_date"].dt.day_name()
    df["is_weekend"] = (df["fl_date"].dt.dayofweek >= 5).astype("Int64")
    df["season"] = df["fl_date"].dt.month.apply(_get_season)

    crs_dep_dt = _to_datetime_series(df["crs_dep_time"])
    df["scheduled_dep_hour"] = crs_dep_dt.dt.hour.astype("Int64")
    df["scheduled_dep_minute_of_day"] = (
        crs_dep_dt.dt.hour * 60 + crs_dep_dt.dt.minute
    ).astype("Int64")

    df["is_peak_hour"] = (
        df["scheduled_dep_hour"].between(7, 9)
        | df["scheduled_dep_hour"].between(16, 19)
    ).astype("Int64")

    # -------------------------
    # Step 2: Delay Outcome Features
    # -------------------------
    df["is_dep_delayed"] = (df["dep_delay"] > 15).astype("Int64")
    df["is_arr_delayed"] = (df["arr_delay"] > 15).astype("Int64")
    df["delay_bucket"] = df["arr_delay"].apply(_delay_bucket)

    # -------------------------
    # Step 3: Operational Efficiency Features
    # -------------------------
    df["schedule_padding"] = df["crs_elapsed_time"] - df["air_time"]
    df["excess_elapsed_time"] = df["actual_elapsed_time"] - df["crs_elapsed_time"]
    df["taxi_total"] = df["taxi_out"] + df["taxi_in"]

    df["taxi_share_of_trip"] = np.where(
        df["actual_elapsed_time"] > 0,
        df["taxi_total"] / df["actual_elapsed_time"],
        np.nan,
    )

    df["air_speed_mph"] = np.where(
        df["air_time"] > 0,
        df["distance"] / (df["air_time"] / 60),
        np.nan,
    )

    
    for col in DELAY_COMPONENT_COLS:
        df[col] = df[col].fillna(0)

    df["total_reported_delay"] = df[DELAY_COMPONENT_COLS].sum(axis=1)
    df["has_attributed_delay"] = (df["total_reported_delay"] > 0).astype("Int64")

    for col in DELAY_COMPONENT_COLS:
        share_col = f"{col}_share"
        df[share_col] = np.where(
            df["total_reported_delay"] > 0,
            df[col] / df["total_reported_delay"],
            0.0,
        )

    df["primary_delay_cause"] = np.where(
        df["total_reported_delay"] > 0,
        df[DELAY_COMPONENT_COLS].idxmax(axis=1),
        pd.NA,
    )

    winsor_cols = [
        "taxi_out",
        "taxi_in",
        "taxi_total",
        "excess_elapsed_time",
    ]

    for col in winsor_cols:
        series = pd.to_numeric(df[col], errors="coerce")

        p99_origin = df.groupby("origin", dropna=False)[col].quantile(0.99)
        p99_global = series.quantile(0.99)

        cap = df["origin"].map(p99_origin).fillna(p99_global)

        df[f"{col}_cap"] = cap
        df[f"{col}_outlier_flag"] = (
            series.notna() & (series > cap)
        ).astype("Int64")
        df[f"{col}_winsor"] = np.minimum(series, cap)

    return df


def build_feature_engineered_flights_from_raw(
    path_2024: str,
) -> pd.DataFrame:
    """
    Raw file -> cleaned dataframe -> feature engineered dataframe.

    clean_flight_data returns:
    (cleaned_dataframe, report)
    """
    df_clean, _ = clean_flight_data(path_2024)
    return add_feature_engineering(df_clean)


def build_and_save_feature_engineered_flights_from_raw(
    path_2024: str,
    output_path: str,
    file_format: str = "parquet",
) -> pd.DataFrame:
    """
    Convenience wrapper to go directly from raw CSV to saved feature-engineered file.
    """
    df_fe = build_feature_engineered_flights_from_raw(path_2024)
    save_feature_engineered_flights(df_fe, output_path, file_format=file_format)
    return df_fe


def save_feature_engineered_flights(
    df: pd.DataFrame,
    output_path: str,
    file_format: str = "parquet",
) -> None:
    """
    Save an already feature-engineered dataframe.
    """
    file_format = file_format.lower()

    if file_format == "parquet":
        df.to_parquet(output_path, index=False)
    elif file_format == "csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("file_format must be either 'parquet' or 'csv'")
