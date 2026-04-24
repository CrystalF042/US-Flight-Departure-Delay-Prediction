from __future__ import annotations

from typing import Iterable

import pandas as pd

from Model.schema import TARGET_CLF


def make_time_splits(
    df: pd.DataFrame,
    date_col: str = "fl_date",
    train_start: str = "2024-01-01",
    train_end: str = "2024-08-31",
    val_start: str = "2024-09-01",
    val_end: str = "2024-10-31",
    test_start: str = "2024-11-01",
    test_end: str = "2024-12-31",
    debug: bool = False,
    target_col: str = TARGET_CLF,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    train_mask = (df[date_col] >= train_start) & (df[date_col] <= train_end)
    val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)
    test_mask = (df[date_col] >= test_start) & (df[date_col] <= test_end)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more time splits are empty. Check date ranges and input data.")

    assert train_df[date_col].max() < val_df[date_col].min(), "Train/val overlap detected."
    assert val_df[date_col].max() < test_df[date_col].min(), "Val/test overlap detected."

    if debug:
        summarize_time_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            date_col=date_col,
            target_col=target_col,
        )

    return train_df, val_df, test_df


def summarize_time_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = "fl_date",
    target_col: str = TARGET_CLF,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)

    for name, d in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"{name}:")
        print(f"  Rows:       {len(d):,} ({len(d) / total * 100:.1f}%)")
        print(f"  Date range: {d[date_col].min().date()} to {d[date_col].max().date()}")
        if target_col in d.columns:
            print(f"  Delay rate: {d[target_col].mean() * 100:.2f}%")
        print()

    print("No date overlap between splits.")


def find_unseen_categories(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    columns: Iterable[str],
) -> dict[str, set]:
    unseen = {}
    for col in columns:
        if col in train_df.columns and col in other_df.columns:
            train_vals = set(train_df[col].dropna().astype(str).unique())
            other_vals = set(other_df[col].dropna().astype(str).unique())
            unseen[col] = other_vals - train_vals
    return unseen


def make_cv_folds(
    df: pd.DataFrame,
    date_col: str = "fl_date",
    fold_ranges: list[tuple[str, str, str, str]] | None = None,
    debug: bool = False,
) -> list[tuple[pd.Index, pd.Index]]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if fold_ranges is None:
        fold_ranges = [
            ("2024-01-01", "2024-04-30", "2024-05-01", "2024-05-31"),
            ("2024-01-01", "2024-05-31", "2024-06-01", "2024-07-31"),
            ("2024-01-01", "2024-07-31", "2024-08-01", "2024-08-31"),
        ]

    fold_indices: list[tuple[pd.Index, pd.Index]] = []

    for i, (tr_start, tr_end, va_start, va_end) in enumerate(fold_ranges, start=1):
        train_mask = (df[date_col] >= tr_start) & (df[date_col] <= tr_end)
        val_mask = (df[date_col] >= va_start) & (df[date_col] <= va_end)

        tr_idx = df.index[train_mask]
        va_idx = df.index[val_mask]

        if debug:
            print(
                f"Fold {i}: train {tr_start} -> {tr_end} ({len(tr_idx):,} rows) | "
                f"val {va_start} -> {va_end} ({len(va_idx):,} rows)"
            )

        fold_indices.append((tr_idx, va_idx))

    return fold_indices