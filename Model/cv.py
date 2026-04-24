from __future__ import annotations

import pandas as pd

from Model.baselines import apply_baselines, compute_baselines_for_fold
from Model.evaluate import evaluate_predictions


def run_cv(
    model_name: str,
    train_fn,
    predict_fn,
    train_sample: pd.DataFrame,
    cv_folds: list[tuple[pd.Index, pd.Index]],
    predictor_cols: list[str],
    target_col: str,
    verbose: bool = True,
) -> pd.DataFrame:
    fold_metrics = []

    for fold_num, (tr_idx, va_idx) in enumerate(cv_folds, start=1):
        if verbose:
            print(f"\n  Fold {fold_num}...", end=" ")

        fold_train = train_sample.loc[tr_idx].copy()
        fold_val = train_sample.loc[va_idx].copy()

        baselines = compute_baselines_for_fold(fold_train)
        fold_train = apply_baselines(fold_train, baselines)
        fold_val = apply_baselines(fold_val, baselines)

        X_tr = fold_train[predictor_cols]
        y_tr = fold_train[target_col].values
        X_va = fold_val[predictor_cols]
        y_va = fold_val[target_col].values

        model = train_fn(X_tr, y_tr, X_va, y_va)
        y_prob = predict_fn(model, X_va)

        metrics = evaluate_predictions(y_va, y_prob)
        metrics["fold"] = fold_num
        metrics["n_train"] = len(X_tr)
        metrics["n_val"] = len(X_va)
        fold_metrics.append(metrics)

        if verbose:
            print(
                f"ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"PR-AUC: {metrics['pr_auc']:.4f} | "
                f"Brier: {metrics['brier']:.4f}"
            )

    df_metrics = pd.DataFrame(fold_metrics)
    df_metrics["model"] = model_name

    if verbose:
        print(
            f"\n  Mean: ROC-AUC {df_metrics['roc_auc'].mean():.4f} ± {df_metrics['roc_auc'].std():.4f} | "
            f"PR-AUC {df_metrics['pr_auc'].mean():.4f} ± {df_metrics['pr_auc'].std():.4f}"
        )

    return df_metrics