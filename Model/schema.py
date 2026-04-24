from __future__ import annotations

TARGET_CLF = "is_dep_delayed"
TARGET_REG = "dep_delay"

TARGET_COLS = [TARGET_REG, TARGET_CLF]
REFERENCE_COLS = ["fl_date"]

BASELINE_COLS = [
    "origin_avg_dep_delay",
    "carrier_avg_dep_delay",
    "route_avg_dep_delay",
]

PREDICTOR_COLS = [
    # Calendar / time
    "month",
    "day_of_week",
    "is_weekend",
    "season",
    "scheduled_dep_hour",
    "is_peak_hour",
    # Schedule / route
    "crs_elapsed_time",
    "distance",
    "route_distance_bucket",
    # Categorical
    "op_unique_carrier",
    "origin",
    "dest",
    "route_id",
    # Leakage-safe baselines
    *BASELINE_COLS,
    # Weather — origin
    "origin_tavg",
    "origin_tmin",
    "origin_tmax",
    "origin_prcp",
    "origin_snow",
    "origin_wspd",
    "origin_pres",
    "origin_bad_weather",
    "origin_wind_cat",
    # Weather — destination
    "dest_tavg",
    "dest_prcp",
    "dest_wspd",
    "dest_bad_weather",
    "dest_wind_cat",
]

CATEGORICAL_COLS = [
    "season",
    "route_distance_bucket",
    "op_unique_carrier",
    "origin",
    "dest",
    "route_id",
    "origin_wind_cat",
    "dest_wind_cat",
]

KEEP_COLS = TARGET_COLS + REFERENCE_COLS + PREDICTOR_COLS