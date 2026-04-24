"""
Feature computation for Shiny app at prediction time.
Must match the training pipeline exactly.
"""
import pandas as pd
import numpy as np

PREDICTOR_COLS = ['month', 'day_of_week', 'is_weekend', 'season', 'scheduled_dep_hour', 'is_peak_hour', 'crs_elapsed_time', 'distance', 'route_distance_bucket', 'op_unique_carrier', 'origin', 'dest', 'route_id', 'origin_avg_dep_delay', 'carrier_avg_dep_delay', 'route_avg_dep_delay', 'origin_tavg', 'origin_tmin', 'origin_tmax', 'origin_prcp', 'origin_snow', 'origin_wspd', 'origin_pres', 'origin_bad_weather', 'origin_wind_cat', 'dest_tavg', 'dest_prcp', 'dest_wspd', 'dest_bad_weather', 'dest_wind_cat']

CATEGORICAL_COLS = ['season', 'route_distance_bucket', 'op_unique_carrier', 'origin', 'dest', 'route_id', 'origin_wind_cat', 'dest_wind_cat']


def compute_features(origin, dest, carrier, fl_date, scheduled_dep_time, distance,
                     crs_elapsed_time, baselines, weather_lookup):
    """
    Build a single-row feature dataframe for the model.

    Parameters
    ----------
    origin, dest : str
        IATA airport codes
    carrier : str
        Carrier code (e.g., 'AA', 'DL')
    fl_date : str or datetime
        Flight date
    scheduled_dep_time : int
        Military time, e.g. 1430 for 2:30pm
    distance : int
        Flight distance in miles
    crs_elapsed_time : int
        Scheduled elapsed time in minutes
    baselines : dict
        Loaded from baseline_lookups.pkl
    weather_lookup : pd.DataFrame
        Weather reference with columns [iata, day_of_year, tavg, ...]
    """
    fl_date = pd.to_datetime(fl_date)

    # Calendar
    month = fl_date.month
    day_of_week = fl_date.dayofweek + 1  # 1=Mon
    is_weekend = 1 if day_of_week >= 6 else 0
    season = {12:"winter",1:"winter",2:"winter",
               3:"spring",4:"spring",5:"spring",
               6:"summer",7:"summer",8:"summer",
               9:"fall",10:"fall",11:"fall"}[month]

    # Time
    scheduled_dep_hour = scheduled_dep_time // 100
    is_peak_hour = 1 if (6 <= scheduled_dep_hour <= 8) or (17 <= scheduled_dep_hour <= 19) else 0

    # Route
    route_id = f"{origin}_{dest}"
    if distance < 500:
        route_distance_bucket = "short"
    elif distance < 1500:
        route_distance_bucket = "medium"
    else:
        route_distance_bucket = "long"

    # Baselines
    gm = baselines["global_mean"]
    origin_avg_dep_delay = baselines["origin_map"].get(origin, gm)
    carrier_avg_dep_delay = baselines["carrier_map"].get(carrier, gm)
    route_avg_dep_delay = baselines["route_map"].get(route_id, gm)

    # Weather
    day_of_year = fl_date.dayofyear

    def get_weather(iata, day_of_year):
        row = weather_lookup[(weather_lookup["iata"] == iata) & 
                             (weather_lookup["day_of_year"] == day_of_year)]
        if len(row) == 0:
            return weather_lookup[weather_lookup["iata"] == iata].iloc[0] if \
                   (weather_lookup["iata"] == iata).any() else \
                   weather_lookup.iloc[0]
        return row.iloc[0]

    o_weather = get_weather(origin, day_of_year)
    d_weather = get_weather(dest, day_of_year)

    # Derived weather flags
    origin_bad_weather = int(
        (o_weather["prcp"] > 10) or (o_weather["snow"] > 0) or (o_weather["wspd"] > 30)
    )
    dest_bad_weather = int(
        (d_weather["prcp"] > 10) or (d_weather["snow"] > 0) or (d_weather["wspd"] > 30)
    )

    def wind_cat(wspd):
        if wspd <= 10: return 0
        elif wspd <= 20: return 1
        elif wspd <= 30: return 2
        else: return 3

    origin_wind_cat = wind_cat(o_weather["wspd"])
    dest_wind_cat = wind_cat(d_weather["wspd"])

    # Build the row
    features = {
        "month": month, "day_of_week": day_of_week, "is_weekend": is_weekend,
        "season": season, "scheduled_dep_hour": scheduled_dep_hour, "is_peak_hour": is_peak_hour,
        "crs_elapsed_time": crs_elapsed_time, "distance": distance, "route_distance_bucket": route_distance_bucket,
        "op_unique_carrier": carrier, "origin": origin, "dest": dest, "route_id": route_id,
        "origin_avg_dep_delay": origin_avg_dep_delay,
        "carrier_avg_dep_delay": carrier_avg_dep_delay,
        "route_avg_dep_delay": route_avg_dep_delay,
        "origin_tavg": o_weather["tavg"], "origin_tmin": o_weather["tmin"], "origin_tmax": o_weather["tmax"],
        "origin_prcp": o_weather["prcp"], "origin_snow": o_weather["snow"],
        "origin_wspd": o_weather["wspd"], "origin_pres": o_weather["pres"],
        "origin_bad_weather": origin_bad_weather, "origin_wind_cat": origin_wind_cat,
        "dest_tavg": d_weather["tavg"], "dest_prcp": d_weather["prcp"],
        "dest_wspd": d_weather["wspd"],
        "dest_bad_weather": dest_bad_weather, "dest_wind_cat": dest_wind_cat,
    }

    X = pd.DataFrame([features])
    # Ensure correct column order
    X = X[PREDICTOR_COLS]
    # Set categorical dtypes
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")
    return X
