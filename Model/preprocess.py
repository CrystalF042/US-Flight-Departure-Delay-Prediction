from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def select_existing_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()


def coerce_target_to_int64(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="raise").astype("int64")
    return df


def set_categorical_dtypes(
    df: pd.DataFrame,
    categorical_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def align_categories_to_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df = set_categorical_dtypes(train_df, categorical_cols)
    val_df = set_categorical_dtypes(val_df, categorical_cols)
    test_df = set_categorical_dtypes(test_df, categorical_cols)

    for col in categorical_cols:
        if col in train_df.columns:
            train_cats = train_df[col].cat.categories
            if col in val_df.columns:
                val_df[col] = val_df[col].cat.set_categories(train_cats)
            if col in test_df.columns:
                test_df[col] = test_df[col].cat.set_categories(train_cats)

    return train_df, val_df, test_df


def prepare_model_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    keep_cols: list[str],
    target_col: str,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = select_existing_columns(train_df, keep_cols)
    val_df = select_existing_columns(val_df, keep_cols)
    test_df = select_existing_columns(test_df, keep_cols)

    train_df = coerce_target_to_int64(train_df, target_col)
    val_df = coerce_target_to_int64(val_df, target_col)
    test_df = coerce_target_to_int64(test_df, target_col)

    train_df, val_df, test_df = align_categories_to_train(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        categorical_cols=categorical_cols,
    )

    return train_df, val_df, test_df


def build_preprocessor(
    X_train: pd.DataFrame,
    categorical_cols: list[str],
    min_category_count: int = 100,
) -> ColumnTransformer:
    categorical_cols = [c for c in categorical_cols if c in X_train.columns]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_count,
                    sparse_output=True,
                ),
                categorical_cols,
            ),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    return preprocessor