from __future__ import annotations

import pandas as pd


def _smooth_group_mean(
    stats_df: pd.DataFrame,
    value_col: str,
    count_col: str,
    global_mean: float,
    min_n: int,
) -> pd.Series:
    n = stats_df[count_col]
    return (n * stats_df[value_col] + min_n * global_mean) / (n + min_n)


def compute_baselines_for_fold(
    fold_train_df: pd.DataFrame,
    target_reg_col: str = "dep_delay",
    min_n_origin: int = 50,
    min_n_carrier: int = 200,
    min_n_route: int = 30,
) -> dict[str, object]:
    global_mean = float(fold_train_df[target_reg_col].mean())

    origin_stats = (
        fold_train_df.groupby("origin", observed=True)[target_reg_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "origin_mean", "count": "origin_n"})
    )
    origin_stats["smoothed"] = _smooth_group_mean(
        origin_stats,
        value_col="origin_mean",
        count_col="origin_n",
        global_mean=global_mean,
        min_n=min_n_origin,
    )

    carrier_stats = (
        fold_train_df.groupby("op_unique_carrier", observed=True)[target_reg_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "carrier_mean", "count": "carrier_n"})
    )
    carrier_stats["smoothed"] = _smooth_group_mean(
        carrier_stats,
        value_col="carrier_mean",
        count_col="carrier_n",
        global_mean=global_mean,
        min_n=min_n_carrier,
    )

    route_stats = (
        fold_train_df.groupby("route_id", observed=True)[target_reg_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "route_mean", "count": "route_n"})
    )
    route_stats["smoothed"] = _smooth_group_mean(
        route_stats,
        value_col="route_mean",
        count_col="route_n",
        global_mean=global_mean,
        min_n=min_n_route,
    )

    return {
        "origin_map": origin_stats["smoothed"].to_dict(),
        "carrier_map": carrier_stats["smoothed"].to_dict(),
        "route_map": route_stats["smoothed"].to_dict(),
        "global_mean": global_mean,
    }


def apply_baselines(
    df: pd.DataFrame,
    baselines: dict[str, object],
) -> pd.DataFrame:
    df = df.copy()
    global_mean = float(baselines["global_mean"])

    df["origin_avg_dep_delay"] = (
        df["origin"].astype("string").map(baselines["origin_map"]).astype(float).fillna(global_mean)
    )
    df["carrier_avg_dep_delay"] = (
        df["op_unique_carrier"]
        .astype("string")
        .map(baselines["carrier_map"])
        .astype(float)
        .fillna(global_mean)
    )
    df["route_avg_dep_delay"] = (
        df["route_id"].astype("string").map(baselines["route_map"]).astype(float).fillna(global_mean)
    )

    return df