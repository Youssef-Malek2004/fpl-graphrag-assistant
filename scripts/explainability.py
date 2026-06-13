# scripts/explainability.py
"""
Explainability utilities for the FPL project.

Defaults:
- SHAP background = 1024 rows (env override: SHAP_BACKGROUND_N)
- SHAP test points shown = 800 (env override: SHAP_TEST_POINTS)

Outputs:
- SHAP figures/CSV -> reports/figures/shap/
- LIME HTML/PNGs   -> reports/figures/lime/
"""

import os
from typing import List, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SHAP & LIME
import shap
from lime.lime_tabular import LimeTabularExplainer

# Tunables (env overridable)
SHAP_BACKGROUND_N = int(os.getenv("SHAP_BACKGROUND_N", 800))
SHAP_TEST_POINTS  = int(os.getenv("SHAP_TEST_POINTS", 800))

# Detect CatBoost for fast TreeExplainer path (optional)
try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False


# ---------- Path helpers ----------
def _project_root() -> str:
    # parent of scripts/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _dir_reports() -> str:
    return _ensure_dir(os.path.join(_project_root(), "reports"))

def _dir_figures() -> str:
    # default save root: <project>/reports/figures
    # optional override via env var EXPLAIN_OUT_DIR
    base = os.environ.get("EXPLAIN_OUT_DIR", os.path.join(_dir_reports(), "figures"))
    return _ensure_dir(base)

def _dir_shap() -> str:
    return _ensure_dir(os.path.join(_dir_figures(), "shap"))

def _dir_lime() -> str:
    return _ensure_dir(os.path.join(_dir_figures(), "lime"))


# ---------- Small helpers ----------
def _predict_fn(model):
    """Return a callable f(X) -> 1D np.array for any regressor (CatBoost, Keras, sklearn-like)."""
    def f(X):
        if isinstance(X, pd.DataFrame):
            X_ = X.values
        else:
            X_ = X
        y = model.predict(X_)
        return np.asarray(y).reshape(-1)
    return f

def _sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed) if len(df) > n else df.copy()

def _indices_from_names(feature_names: List[str], selected: Iterable[str]) -> List[int]:
    """Map a list of feature names to their column indices; ignore missing with a warning."""
    selected = list(selected)
    name_to_i = {n: i for i, n in enumerate(feature_names)}
    idx = [name_to_i[n] for n in selected if n in name_to_i]
    missing = [n for n in selected if n not in name_to_i]
    if missing:
        print(f"[Explainability] Warning: missing features ignored: {missing}")
    return idx

def _subset_predict_fn(model,
                       all_feature_names: List[str],
                       subset_names: List[str],
                       anchor_full_row: np.ndarray):
    """
    Create a predict_fn(X_subset) that rebuilds full feature vectors by
    starting from 'anchor_full_row' and replacing only the subset columns.
    """
    subset_idx = _indices_from_names(all_feature_names, subset_names)
    full_predict = _predict_fn(model)

    anchor_full_row = np.asarray(anchor_full_row, dtype=float)
    anchor_full_row = np.nan_to_num(anchor_full_row, nan=0.0, posinf=0.0, neginf=0.0)

    def f(X_subset: np.ndarray):
        X_subset = np.asarray(X_subset, dtype=float)
        X_full = np.tile(anchor_full_row, (X_subset.shape[0], 1))
        X_full[:, subset_idx] = X_subset
        return full_predict(X_full)
    return f


# ---------- SHAP (global + local) ----------
def shap_global_summary(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    feature_names: Optional[Iterable[str]] = None,
    max_background: int = SHAP_BACKGROUND_N,
    max_test_points: int = SHAP_TEST_POINTS,
    feature_whitelist: Optional[Iterable[str]] = None,
):
    """
    Creates global SHAP plots (beeswarm + bar) and a CSV of mean |SHAP|.
    If feature_whitelist is provided, explanations are restricted to exactly those features.
    Saves into reports/figures/shap/.
    """
    out_dir = _dir_shap()
    feature_names = list(feature_names or X_train.columns)

    # Background for SHAP (downsampled for speed)
    background = _sample_df(X_train, max_background)

    # Choose efficient explainer
    if _HAS_CATBOOST and isinstance(model, CatBoostRegressor):
        # Interventional SHAP recommended for tabular
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional"
        )
        X_for_shap = _sample_df(X_test, max_test_points)
        shap_values = explainer(X_for_shap)
    else:
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(_predict_fn(model), masker, algorithm="auto")
        X_for_shap = _sample_df(X_test, max_test_points)
        shap_values = explainer(X_for_shap)

    # Restrict to whitelist if provided
    if feature_whitelist:
        selected = [c for c in feature_whitelist if c in feature_names]
        if len(selected) == 0:
            print("[SHAP] feature_whitelist empty/invalid; falling back to all features.")
            selected = feature_names
        idx = _indices_from_names(feature_names, selected)
        shap_values = shap_values[:, idx]  # slice features
        feature_names = selected

    n_feats = len(feature_names)

    # Beeswarm — show the (restricted) features
    plt.figure()
    plt.gcf().set_size_inches(10, max(6, 0.35 * n_feats))
    shap.plots.beeswarm(shap_values, show=False, max_display=n_feats)
    plt.title(f"SHAP Beeswarm — {model_name}")
    beeswarm_path = os.path.join(out_dir, f"shap_beeswarm_{model_name}.png")
    plt.tight_layout(); plt.savefig(beeswarm_path, dpi=150); plt.close()

    # Bar — show the (restricted) features
    plt.figure()
    plt.gcf().set_size_inches(10, max(6, 0.35 * n_feats))
    shap.plots.bar(shap_values, show=False, max_display=n_feats)
    plt.title(f"SHAP Bar — {model_name}")
    bar_path = os.path.join(out_dir, f"shap_bar_{model_name}.png")
    plt.tight_layout(); plt.savefig(bar_path, dpi=150); plt.close()

    # CSV of global importance for the (restricted) features
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    imp = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = os.path.join(out_dir, f"shap_global_importance_{model_name}.csv")
    imp.to_csv(csv_path, index=False)

    print(f"[SHAP] Saved:\n  {beeswarm_path}\n  {bar_path}\n  {csv_path}")


def shap_local_waterfalls(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    indices: List[int],
    max_background: int = SHAP_BACKGROUND_N,
    feature_whitelist: Optional[Iterable[str]] = None,
):
    """
    Saves SHAP waterfall plots for specified test rows into reports/figures/shap/.
    If feature_whitelist is provided, each instance waterfall shows only those features.
    """
    out_dir = _dir_shap()
    background = _sample_df(X_train, max_background)

    if _HAS_CATBOOST and isinstance(model, CatBoostRegressor):
        explainer = shap.TreeExplainer(
            model,
            data=background,
            feature_perturbation="interventional"
        )
    else:
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(_predict_fn(model), masker, algorithm="auto")

    rows = X_test.iloc[indices]
    shap_values = explainer(rows)
    feature_names_full = list(X_train.columns)

    # Restrict to whitelist if provided (consistent order)
    if feature_whitelist:
        selected = [c for c in feature_whitelist if c in feature_names_full]
        if len(selected) == 0:
            print("[SHAP] feature_whitelist empty/invalid; falling back to all features.")
            selected = feature_names_full
        idx = _indices_from_names(feature_names_full, selected)
        shap_values = shap_values[:, idx]
        n_feats = len(selected)
    else:
        n_feats = X_test.shape[1]

    for j, i in enumerate(indices):
        plt.figure()
        plt.gcf().set_size_inches(10, max(6, 0.35 * n_feats))
        shap.plots.waterfall(shap_values[j], show=False, max_display=n_feats)
        fpath = os.path.join(out_dir, f"shap_waterfall_{model_name}_idx{int(i)}.png")
        plt.tight_layout(); plt.savefig(fpath, dpi=150); plt.close()
        print(f"[SHAP] Saved local waterfall for row {int(i)} -> {fpath}")


# ---------- LIME (local) ----------
def lime_local_explanations(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str,
    feature_names: Optional[Iterable[str]] = None,
    indices: List[int] = [0, 1, 2],
    num_features: Optional[int] = None,
    num_samples: int = 5000,
    feature_whitelist: Optional[Iterable[str]] = None,
):
    """
    Generates LIME explanations (HTML + PNG) for selected rows.
    Saves into reports/figures/lime/.
    Robust to zero-variance columns and NaNs/Infs.

    If feature_whitelist is given, LIME perturbs ONLY those features; all others
    are frozen to the instance's original values via a reconstructing predict_fn.
    """
    out_dir = _dir_lime()
    all_feature_names = list(feature_names or X_train.columns)

    # LIME-only sanitized copies (jitter constant cols; handle NaN/Inf) on ALL columns
    Xtr_s_all, Xte_s_all, constant_cols = _sanitize_for_lime(X_train, X_test, eps=1e-6)
    if len(constant_cols) > 0:
        print(f"[LIME] Detected zero-variance features (jittered for sampling): {list(constant_cols)}")

    # Determine subset to use
    if feature_whitelist:
        selected = [c for c in feature_whitelist if c in all_feature_names]
        if len(selected) == 0:
            print("[LIME] feature_whitelist empty/invalid; falling back to all features.")
            selected = all_feature_names
    else:
        selected = all_feature_names

    # Now restrict training/test matrices to the selected subset (order preserved)
    Xtr_s = Xtr_s_all[selected].copy()
    Xte_s = Xte_s_all[selected].copy()
    feature_names_subset = selected

    # num_features governs how many LIME coefficients are displayed
    if num_features is None or num_features <= 0:
        num_features = len(feature_names_subset)

    explainer = LimeTabularExplainer(
        training_data=Xtr_s.values,
        feature_names=feature_names_subset,
        mode="regression",
        discretize_continuous=False,  # avoid truncnorm issues
        sample_around_instance=True,
        random_state=42,
    )

    # If we restricted features, we need a predict_fn that rebuilds the full vector
    full_cols = list(X_train.columns)

    for i in indices:
        i = int(i)
        x0_subset = Xte_s.iloc[i].values.astype(float, copy=False)

        if feature_whitelist:
            predict = _subset_predict_fn(
                model=model,
                all_feature_names=full_cols,
                subset_names=feature_names_subset,
                anchor_full_row=Xte_s_all.iloc[i].values  # sanitized full row
            )
        else:
            predict = _predict_fn(model)

        exp = explainer.explain_instance(
            data_row=x0_subset,
            predict_fn=predict,
            num_features=min(num_features, len(feature_names_subset)),
            num_samples=num_samples,
        )

        html_path = os.path.join(out_dir, f"lime_{model_name}_idx{i}.html")
        exp.save_to_file(html_path)

        fig = exp.as_pyplot_figure()
        fig.set_size_inches(10, max(6, 0.35 * num_features))
        plt.title(f"LIME (regression) — {model_name} — row {i}")
        png_path = os.path.join(out_dir, f"lime_{model_name}_idx{i}.png")
        plt.tight_layout(); plt.savefig(png_path, dpi=150); plt.close(fig)

        print(f"[LIME] Saved:\n  {html_path}\n  {png_path}")


def _sanitize_for_lime(X_train: pd.DataFrame, X_test: pd.DataFrame, eps: float = 1e-6):
    """
    Prepare copies for LIME:
      - replace inf/NaN
      - add tiny jitter to zero-variance columns (so sampling has positive scale)
    Returns: Xtr_s, Xte_s, constant_cols (Index)
    """
    Xtr = X_train.copy()
    Xte = X_test.copy()

    # Replace inf/NaN (LIME will choke on them)
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Find zero-variance columns
    constant_cols = Xtr.columns[Xtr.nunique(dropna=False) <= 1]

    if len(constant_cols) > 0:
        # Add tiny Gaussian jitter ONLY for LIME's synthetic sampling
        rng = np.random.default_rng(42)
        Xtr.loc[:, constant_cols] = (
            Xtr.loc[:, constant_cols].to_numpy(dtype=float)
            + rng.normal(0.0, eps, size=(len(Xtr), len(constant_cols)))
        )
        Xte.loc[:, constant_cols] = (
            Xte.loc[:, constant_cols].to_numpy(dtype=float)
            + rng.normal(0.0, eps, size=(len(Xte), len(constant_cols)))
        )

    return Xtr, Xte, constant_cols


# ---------- Orchestrator ----------
def run_explainability(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
    local_rows: Optional[List[int]] = None,
    do_shap: bool = True,
    do_lime: bool = True,
    feature_whitelist: Optional[Iterable[str]] = None,
):
    """
    One-call convenience wrapper from main.py

    feature_whitelist: if provided, explanations (SHAP + LIME) are restricted
    to exactly these features (order preserved). Useful for "show only my 10".
    """
    feature_names = list(feature_names or X_train.columns)
    local_rows = local_rows or [0, 1, 2]

    if feature_whitelist:
        wl = [c for c in feature_whitelist if c in feature_names]
        print(f"[Explainability] Using feature whitelist ({len(wl)}): {wl}")

    if do_shap:
        shap_global_summary(
            model, X_train, X_test, model_name, feature_names,
            feature_whitelist=feature_whitelist
        )
        shap_local_waterfalls(
            model, X_train, X_test, model_name,
            indices=local_rows, feature_whitelist=feature_whitelist
        )

    if do_lime:
        # num_features=None => show ALL from the (restricted) set
        lime_local_explanations(
            model, X_train, X_test, model_name, feature_names,
            indices=local_rows, num_features=None,
            feature_whitelist=feature_whitelist
        )
