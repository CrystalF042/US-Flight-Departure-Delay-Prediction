from __future__ import annotations

import pandas as pd


def stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    n_rows: int,
    seed: int = 42,
) -> pd.DataFrame:
    frac = n_rows / len(df)
    if frac >= 1.0:
        return df.copy()

    sampled = (
        df.groupby([target_col, group_col], observed=True, group_keys=False)[df.columns.tolist()]
        .sample(frac=frac, random_state=seed)
    )
    return sampled.reset_index(drop=True)