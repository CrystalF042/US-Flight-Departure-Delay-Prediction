from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


US_STATE_ABBR = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "TT",
]

US_STATE_NAMES = [
    "Texas", "California", "Florida", "Illinois", "Georgia", "New York", "Colorado",
    "North Carolina", "Virginia", "Arizona", "Nevada", "Washington", "Michigan",
    "Pennsylvania", "Tennessee", "Massachusetts", "New Jersey", "Minnesota", "Missouri",
    "Hawaii", "Utah", "Maryland", "Ohio", "Oregon", "Kentucky", "Louisiana",
    "South Carolina", "Indiana", "Wisconsin", "Oklahoma", "Puerto Rico", "Alaska",
    "Alabama", "Idaho", "Arkansas", "New Mexico", "Nebraska", "Montana", "Iowa",
    "Connecticut", "North Dakota", "Maine", "Rhode Island", "Kansas", "South Dakota",
    "Mississippi", "Wyoming", "Vermont", "U.S. Virgin Islands", "New Hampshire",
    "West Virginia", "U.S. Pacific Trust Territories and Possessions",
]

CANCELLED_RELATED_COLS = [
    "dep_time",
    "dep_delay",
    "taxi_out",
    "wheels_off",
    "wheels_on",
    "taxi_in",
    "arr_time",
    "arr_delay",
    "actual_elapsed_time",
    "air_time",
]

DELAY_COMPONENT_COLS = [
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]


def _clean_string(series: pd.Series) -> pd.Series:
    """Trim whitespace and convert empty strings to missing values."""
    s = series.astype("string").str.strip()
    return s.mask(s.eq(""), pd.NA)


def _hhmm_to_time(series: pd.Series) -> pd.Series:
    """Convert HHMM values like 930 or '0930' into python time objects."""
    numeric = pd.to_numeric(series, errors="coerce")
    padded = numeric.dropna().astype(int).astype(str).str.zfill(4)

    time_strings = pd.Series(pd.NA, index=series.index, dtype="string")
    time_strings.loc[padded.index] = pd.to_datetime(
        padded,
        format="%H%M",
        errors="coerce",
    ).dt.strftime("%H:%M")

    return pd.to_datetime(time_strings, format="%H:%M", errors="coerce").dt.time


def _hhmm_to_minutes(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Convert HHMM values into minutes after midnight and return validity mask."""
    numeric = pd.to_numeric(series, errors="coerce")
    hours = numeric // 100
    minutes = numeric % 100
    valid = hours.between(0, 23) & minutes.between(0, 59)
    return (hours * 60 + minutes).where(valid), valid


def _delay_mismatch_count(
    scheduled_series: pd.Series,
    actual_series: pd.Series,
    delay_series: pd.Series,
) -> int:
    """Count rows where reported delay is inconsistent with HHMM timestamps.

    The logic matches the notebook's intention: a difference is acceptable when it
    differs by whole days (overnight wrap-around), so delta % 1440 must be 0.
    """
    scheduled_min, scheduled_valid = _hhmm_to_minutes(scheduled_series)
    actual_min, actual_valid = _hhmm_to_minutes(actual_series)
    delay = pd.to_numeric(delay_series, errors="coerce")

    valid = scheduled_valid & actual_valid
    raw_diff = actual_min - scheduled_min
    mask = delay.notna() & valid

    delta = delay - raw_diff
    ok = (delta % 1440 == 0)
    return int((mask & ~ok).sum())


def _split_city_and_state(city_state_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split values like 'Detroit, MI' into ('Detroit', 'MI')."""
    raw = city_state_series.astype("string").str.strip()

    state_abbr = raw.str.extract(r",\s*([A-Za-z]{2})\s*$", expand=False).str.upper()
    city_name = raw.str.replace(r",\s*[A-Za-z]{2}\s*$", "", regex=True).str.strip()

    return city_name, state_abbr


def load_flight_data(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw flights CSV file."""
    return pd.read_csv(csv_path)


def part1_clean_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and validate year/month/day/fl_date columns."""
    df = df.copy()

    report = {
        "missing_values_per_column": df.isna().sum().to_dict(),
        "duplicated_rows": int(df.duplicated().sum()),
        "date_columns_missing": df[["year", "month", "day_of_month", "day_of_week", "fl_date"]]
        .isna()
        .sum()
        .to_dict(),
        "abnormal_month_rows": int(((df["month"] < 1) | (df["month"] > 12)).sum()),
        "abnormal_day_of_month_rows": int(((df["day_of_month"] < 1) | (df["day_of_month"] > 31)).sum()),
        "abnormal_day_of_week_rows": int(((df["day_of_week"] < 1) | (df["day_of_week"] > 7)).sum()),
    }

    df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce")
    report["fl_date_parse_failed"] = int(df["fl_date"].isna().sum())

    rebuilt = pd.to_datetime(
        {
            "year": pd.to_numeric(df["year"], errors="coerce"),
            "month": pd.to_numeric(df["month"], errors="coerce"),
            "day": pd.to_numeric(df["day_of_month"], errors="coerce"),
        },
        errors="coerce",
    )

    mask = df["fl_date"].notna() & rebuilt.notna()
    report["fl_date_mismatch_rows"] = int((mask & (df["fl_date"] != rebuilt)).sum())

    return df, report


def part2_clean_carrier_and_route(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean carrier, flight number, origin, and destination fields."""
    df = df.copy()
    report: dict[str, Any] = {}

    carrier = _clean_string(df["op_unique_carrier"])
    carrier_bad = carrier.isna() | (carrier.str.len() != 2)
    report["op_unique_carrier_bad_rows"] = int(carrier_bad.sum())

    flight_num_raw = _clean_string(df["op_carrier_fl_num"])
    flight_num_numeric = pd.to_numeric(flight_num_raw, errors="coerce")
    flight_num_missing = flight_num_raw.isna()
    flight_num_not_int = (~flight_num_missing) & (
        flight_num_numeric.isna() | (flight_num_numeric % 1 != 0)
    )
    flight_num_int = flight_num_numeric.round(0).astype("Int64")
    flight_num_gt_4_digits = (~flight_num_missing) & (~flight_num_not_int) & (
        (flight_num_int < 0) | (flight_num_int > 9999)
    )
    flight_num_bad = flight_num_missing | flight_num_not_int | flight_num_gt_4_digits

    report["op_carrier_fl_num"] = {
        "missing": int(flight_num_missing.sum()),
        "not_int": int(flight_num_not_int.sum()),
        "gt_4_digits": int(flight_num_gt_4_digits.sum()),
        "bad_total": int(flight_num_bad.sum()),
    }

    # Notebook imputation: the missing flight number was manually set to 2483.
    fill_value = 2483
    cleaned_flight_num = flight_num_int.copy()
    cleaned_flight_num.loc[flight_num_bad] = fill_value
    df["op_carrier_fl_num"] = cleaned_flight_num.astype("int64")

    for col in ["origin", "dest"]:
        code = _clean_string(df[col])
        code_missing = code.isna()
        code_not_3 = (~code_missing) & (code.str.len() != 3)
        report[f"{col}_validation"] = {
            "missing": int(code_missing.sum()),
            "not_3_chars": int(code_not_3.sum()),
            "bad_total": int((code_missing | code_not_3).sum()),
        }

    df["origin_city_name"], df["origin_state_abbr"] = _split_city_and_state(df["origin_city_name"])
    origin_abbr = df["origin_state_abbr"].astype("string").str.strip().str.upper()
    report["origin_state_abbr_validation"] = {
        "missing": int(origin_abbr.isna().sum()),
        "not_in_us_state_list": int((~origin_abbr.isna() & ~origin_abbr.isin(US_STATE_ABBR)).sum()),
        "bad_total": int((origin_abbr.isna() | ~origin_abbr.isin(US_STATE_ABBR)).sum()),
    }

    origin_state_name = _clean_string(df["origin_state_nm"])
    report["origin_state_nm_validation"] = {
        "missing": int(origin_state_name.isna().sum()),
        "not_in_us_state_list": int((~origin_state_name.isna() & ~origin_state_name.isin(US_STATE_NAMES)).sum()),
        "bad_total": int((origin_state_name.isna() | ~origin_state_name.isin(US_STATE_NAMES)).sum()),
    }

    df["dest_city_name"], df["dest_state_abbr"] = _split_city_and_state(df["dest_city_name"])
    dest_abbr = df["dest_state_abbr"].astype("string").str.strip().str.upper()
    report["dest_state_abbr_validation"] = {
        "missing": int(dest_abbr.isna().sum()),
        "not_in_us_state_list": int((~dest_abbr.isna() & ~dest_abbr.isin(US_STATE_ABBR)).sum()),
        "bad_total": int((dest_abbr.isna() | ~dest_abbr.isin(US_STATE_ABBR)).sum()),
    }

    dest_state_name = _clean_string(df["dest_state_nm"])
    report["dest_state_nm_validation"] = {
        "missing": int(dest_state_name.isna().sum()),
        "not_in_us_state_list": int((~dest_state_name.isna() & ~dest_state_name.isin(US_STATE_NAMES)).sum()),
        "bad_total": int((dest_state_name.isna() | ~dest_state_name.isin(US_STATE_NAMES)).sum()),
    }

    return df, report


def part3_clean_time_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean departure/arrival HHMM fields and validate time-based delays."""
    df = df.copy()

    report = {
        "missing_crs_dep_time": int(df["crs_dep_time"].isna().sum()),
    }

    cancelled_mask = pd.to_numeric(df["cancelled"], errors="coerce").eq(1)
    df.loc[cancelled_mask, ["dep_delay", "taxi_out", "taxi_in"]] = np.nan

    dep_delay_mismatch = _delay_mismatch_count(
        df["crs_dep_time"],
        df["dep_time"],
        df["dep_delay"],
    )
    arr_delay_mismatch = _delay_mismatch_count(
        df["crs_arr_time"],
        df["arr_time"],
        df["arr_delay"],
    )

    df["crs_dep_time"] = _hhmm_to_time(df["crs_dep_time"])
    df["dep_time"] = _hhmm_to_time(df["dep_time"])
    df["wheels_off"] = _hhmm_to_time(df["wheels_off"])
    df["wheels_on"] = _hhmm_to_time(df["wheels_on"])
    df["crs_arr_time"] = _hhmm_to_time(df["crs_arr_time"])
    df["arr_time"] = _hhmm_to_time(df["arr_time"])

    report.update(
        {
            "dep_delay_mismatch_rows": dep_delay_mismatch,
            "arr_delay_mismatch_rows": arr_delay_mismatch,
            "negative_taxi_out_rows": int((pd.to_numeric(df["taxi_out"], errors="coerce") < 0).sum()),
            "negative_taxi_in_rows": int((pd.to_numeric(df["taxi_in"], errors="coerce") < 0).sum()),
        }
    )

    return df, report


def part4_clean_cancellation_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate cancellation flags and related cancellation code logic."""
    df = df.copy()

    cancelled_raw = _clean_string(df["cancelled"])
    cancelled_num = pd.to_numeric(cancelled_raw, errors="coerce")
    cancelled_missing = cancelled_raw.isna()
    cancelled_not_int = (~cancelled_missing) & (
        cancelled_num.isna() | (cancelled_num % 1 != 0)
    )
    cancelled_int = cancelled_num.round(0).astype("Int64")
    cancelled_not_01 = (~cancelled_missing) & (~cancelled_not_int) & (~cancelled_int.isin([0, 1]))
    cancelled_bad = cancelled_missing | cancelled_not_int | cancelled_not_01

    report = {
        "cancelled_validation": {
            "missing": int(cancelled_missing.sum()),
            "not_int": int(cancelled_not_int.sum()),
            "not_0_or_1": int(cancelled_not_01.sum()),
            "bad_total": int(cancelled_bad.sum()),
        }
    }

    df["cancelled"] = cancelled_int
    cancelled_one = df["cancelled"].eq(1)
    df.loc[cancelled_one, CANCELLED_RELATED_COLS] = np.nan

    allowed_codes = ["A", "B", "C", "D"]
    cancellation_code = _clean_string(df["cancellation_code"]).str.upper()
    invalid_code_value = cancellation_code.notna() & (~cancellation_code.isin(allowed_codes))
    cancelled0_code_present = df["cancelled"].eq(0) & cancellation_code.notna()
    cancelled1_code_missing = df["cancelled"].eq(1) & cancellation_code.isna()
    cancelled1_code_invalid = df["cancelled"].eq(1) & invalid_code_value

    report["cancellation_code_validation"] = {
        "invalid_code_value": int(invalid_code_value.sum()),
        "cancelled_0_but_code_present": int(cancelled0_code_present.sum()),
        "cancelled_1_but_code_missing": int(cancelled1_code_missing.sum()),
        "cancelled_1_but_code_invalid": int(cancelled1_code_invalid.sum()),
        "bad_total": int(
            (
                invalid_code_value
                | cancelled0_code_present
                | cancelled1_code_missing
                | cancelled1_code_invalid
            ).sum()
        ),
    }

    df["cancellation_code"] = cancellation_code
    return df, report


def part5_clean_elapsed_time_and_distance(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean elapsed-time, air-time, and distance variables."""
    df = df.copy()
    report: dict[str, Any] = {}

    crs_elapsed = pd.to_numeric(_clean_string(df["crs_elapsed_time"]), errors="coerce")
    crs_missing = crs_elapsed.isna()
    crs_negative = crs_elapsed.notna() & (crs_elapsed < 0)
    report["crs_elapsed_time_validation"] = {
        "missing": int(crs_missing.sum()),
        "negative": int(crs_negative.sum()),
        "bad_total": int((crs_missing | crs_negative).sum()),
    }

    # Notebook imputation: manually filled the missing scheduled elapsed time with 292.0.
    df.loc[crs_missing, "crs_elapsed_time"] = 292.0
    df = df.loc[~crs_negative].copy()
    report["rows_dropped_negative_crs_elapsed_time"] = int(crs_negative.sum())

    actual_elapsed = pd.to_numeric(_clean_string(df["actual_elapsed_time"]), errors="coerce")
    actual_missing = actual_elapsed.isna()
    actual_negative = actual_elapsed.notna() & (actual_elapsed < 0)
    report["actual_elapsed_time_validation"] = {
        "missing": int(actual_missing.sum()),
        "negative": int(actual_negative.sum()),
    }

    non_cancelled_missing_actual = df["cancelled"].eq(0) & actual_missing
    df = df.loc[~non_cancelled_missing_actual].copy()
    report["rows_dropped_missing_actual_elapsed_time_when_not_cancelled"] = int(
        non_cancelled_missing_actual.sum()
    )

    air_time = pd.to_numeric(_clean_string(df["air_time"]), errors="coerce")
    air_time_missing = air_time.isna()
    air_time_negative = air_time.notna() & (air_time < 0)
    report["air_time_validation"] = {
        "missing": int(air_time_missing.sum()),
        "negative": int(air_time_negative.sum()),
    }

    distance = pd.to_numeric(_clean_string(df["distance"]), errors="coerce")
    distance_missing = distance.isna()
    distance_negative = distance.notna() & (distance < 0)
    report["distance_validation"] = {
        "missing": int(distance_missing.sum()),
        "negative": int(distance_negative.sum()),
    }

    return df, report


def part6_validate_delay_components(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the delay-component columns against arr_delay and add a flag."""
    df = df.copy()

    neg_mask = (df[DELAY_COMPONENT_COLS] < 0).any(axis=1)
    positive_arr_delay = pd.to_numeric(df["arr_delay"], errors="coerce") > 0
    component_sum = df[DELAY_COMPONENT_COLS].sum(axis=1)
    mismatch_mask = positive_arr_delay & (component_sum != pd.to_numeric(df["arr_delay"], errors="coerce"))
    has_na_component = df[DELAY_COMPONENT_COLS].isna().any(axis=1)

    df["arr_delay_consistent"] = True
    df.loc[positive_arr_delay, "arr_delay_consistent"] = (
        component_sum[positive_arr_delay] == pd.to_numeric(df.loc[positive_arr_delay, "arr_delay"], errors="coerce")
    )

    report = {
        "rows_with_negative_delay_components": int(neg_mask.sum()),
        "arr_delay_sum_mismatch_rows": int(mismatch_mask.sum()),
        "arr_delay_positive_rows": int(positive_arr_delay.sum()),
        "arr_delay_consistent_rows": int((positive_arr_delay & ~mismatch_mask).sum()),
        "arr_delay_inconsistent_rows": int(mismatch_mask.sum()),
        "mismatch_rows_with_missing_component": int((mismatch_mask & has_na_component).sum()),
        "sum_less_than_arr_delay_rows": int((positive_arr_delay & (component_sum < pd.to_numeric(df["arr_delay"], errors="coerce"))).sum()),
        "sum_greater_than_arr_delay_rows": int((positive_arr_delay & (component_sum > pd.to_numeric(df["arr_delay"], errors="coerce"))).sum()),
    }

    return df, report


def clean_flight_data(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full notebook workflow as reusable Python functions.

    Parameters
    ----------
    input_path:
        Path to the raw CSV.
    output_path:
        Optional path where the cleaned CSV should be saved.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Cleaned dataframe and a nested report containing the notebook checks.
    """
    df = load_flight_data(input_path)
    report: dict[str, Any] = {}

    df, report["part1"] = part1_clean_dates(df)
    df, report["part2"] = part2_clean_carrier_and_route(df)
    df, report["part3"] = part3_clean_time_columns(df)
    df, report["part4"] = part4_clean_cancellation_columns(df)
    df, report["part5"] = part5_clean_elapsed_time_and_distance(df)
    df, report["part6"] = part6_validate_delay_components(df)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        report["output"] = {
            "saved_to": str(output_path),
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "size_mib": round(output_path.stat().st_size / (1024 * 1024), 2),
        }

    return df, report


def main() -> None:
    """Example command-line entrypoint.

    Usage
    -----
    python flight_data_cleaning_functions.py data/flight_data_2024.csv data/flights_clean.csv
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Clean the flight data CSV.")
    parser.add_argument("input_path", help="Path to the raw CSV file")
    parser.add_argument(
        "output_path",
        nargs="?",
        default="data/flights_clean.csv",
        help="Path to save the cleaned CSV",
    )
    args = parser.parse_args()

    _, report = clean_flight_data(args.input_path, args.output_path)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
