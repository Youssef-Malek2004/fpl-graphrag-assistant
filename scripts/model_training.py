from tf_models.FFNNRegressorModel import FFNNRegressor
from scripts.feature_engineering import season_start_year
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool
from itertools import product
import pandas as pd
import numpy as np
import os
import joblib
from typing import Optional

def train_ffnn(
    X_train, y_train, X_valid, y_valid,
    params: Optional[dict] = None,
    save_path: str = os.path.join("models", "ffnn_model.pkl"),
):
    """
    Initializes, trains, and saves a feed-forward neural network for regression.
    Saves the entire FFNNRegressor object as a .pkl file in the models directory.
    Returns the trained model.
    """
    if params is None:
        params = {
            "hidden_units": (256, 128, 64, 32),
            "dropout": 0.10,
            "l2": 1e-4,
            "lr": 1e-3,
            "epochs": 400,
            "batch_size": 1024,
            "patience": 25,
            "seed": 42,
            "verbose": 1,
        }

    # Initialize and train
    model = FFNNRegressor(**params)
    model.fit(X_train, y_train, X_valid, y_valid)

    # Ensure models/ directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the entire FFNNRegressor object (recommended for reloading with joblib)
    joblib.dump(model, save_path)

    return model




def grid_search_ffnn(
    X_train, y_train, X_valid, y_valid,
    param_grid: Dict[str, List[Any]],
    verbose: bool = True
):
    """
    Runs a grid search over FFNN hyperparameters using your existing train_ffnn(...)
    Returns: best_model, best_params, leaderboard_df (sorted by RMSE asc)
    """
    # Build list of param combinations
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]

    results = []
    best = {"rmse": np.inf, "model": None, "params": None, "mae": None, "r2": None}

    for i, params in enumerate(combos, 1):
        if verbose:
            print(f"\n[{i}/{len(combos)}] Trying params: {params}")

        # Train one FFNN with these params
        model = train_ffnn(X_train, y_train, X_valid, y_valid, params=params)

        # Score on validation set
        preds = model.predict(X_valid)
        mse = mean_squared_error(y_valid, preds)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y_valid, preds))
        r2   = float(r2_score(y_valid, preds))

        results.append({**params, "RMSE": rmse, "MAE": mae, "R2": r2})

        if verbose:
            print(f" -> val RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

        if rmse < best["rmse"]:
            best = {"rmse": rmse, "model": model, "params": params, "mae": mae, "r2": r2}

    leaderboard = pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    if verbose:
        print("\nTop 5 configs by RMSE:")
        print(leaderboard.head(5))

    return best["model"], best["params"], leaderboard

def build_xy(
    df: pd.DataFrame,
    target_col: str = "upcoming_total_points",
    drop_cols: Optional[List[str]] = None,
    keep_player_id: bool = True,
    player_col: str = "name_encoded",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates features (X) and target (y).
    If keep_player_id=True, keeps `name_encoded` in X.
    """
    if drop_cols is None:
        drop_cols = ["season_x", "round"]

    if not keep_player_id:
        drop_cols = drop_cols + [player_col]

    feature_cols = [c for c in df.columns if c not in drop_cols + [target_col]]
    X = df[feature_cols]
    y = df[target_col]
    return X, y

from typing import Tuple, List, Optional, Dict

def per_player_temporal_split(
    df: pd.DataFrame,
    player_col: str = "name_encoded",
    season_col: str = "season_x",
    week_col: str = "round",
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float  = 0.1,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    For each player, sort by (season, round) and split rows:
      first 80% -> train, next 10% -> valid, last 10% -> test.
    Handles small sample sizes gracefully.
    Returns (train_idx, valid_idx, test_idx) as concatenated indices.
    """
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-9, "fractions must sum to 1"

    train_indices: List[pd.Index] = []
    valid_indices: List[pd.Index] = []
    test_indices:  List[pd.Index] = []

    # Sort globally for stable grouping
    df_sorted = df.sort_values([player_col, season_col, week_col], kind="mergesort")

    for pid, g in df_sorted.groupby(player_col, sort=False):
        idx = g.index.to_list()
        n = len(idx)

        if n == 1:
            # only train
            train_indices.append(pd.Index(idx))
            continue
        if n == 2:
            # 1 train, 1 test
            train_indices.append(pd.Index(idx[:1]))
            test_indices.append(pd.Index(idx[1:]))
            continue

        n_train = max(1, int(n * train_frac))
        n_valid = max(1, int(n * valid_frac))
        # ensure at least 1 test if n>=3
        n_test  = max(1, n - n_train - n_valid)

        # if rounding overflowed, fix by stealing from train
        while n_train + n_valid + n_test > n:
            n_train -= 1
        # if deficit (rare), add to test
        while n_train + n_valid + n_test < n:
            n_test += 1

        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train + n_valid]
        test_idx  = idx[n_train + n_valid:]

        train_indices.append(pd.Index(train_idx))
        valid_indices.append(pd.Index(valid_idx))
        test_indices.append(pd.Index(test_idx))

    return (
        pd.Index([]).append(train_indices),
        pd.Index([]).append(valid_indices),
        pd.Index([]).append(test_indices),
    )



def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_features: Optional[List[str]] = None,
    params: Optional[dict] = None,
):
    """
    Initializes and trains a CatBoostRegressor model.
    """
    if params is None:
        params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 6.0,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "early_stopping_rounds": 50,
            "verbose": 200,

            # Regularization via randomness/subsampling
            "subsample": 0.8,  # row sampling
            "rsm": 0.8,  # feature sampling per split
            "random_strength": 1.5,  # adds noise to splits → less overfit
            "bagging_temperature": 0.5,  # softer sampling
        }

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool)
    return model

def evaluate_model(
    model,
    X_test: pd.DataFrame, y_test: pd.Series,
    X_train: pd.DataFrame = None, y_train: pd.Series = None,
    X_valid: pd.DataFrame = None, y_valid: pd.Series = None,
) -> dict:
    def _metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    report = {}

    if X_train is not None and y_train is not None:
        preds_tr = model.predict(X_train)
        report["train"] = _metrics(y_train, preds_tr)

    if X_valid is not None and y_valid is not None:
        preds_val = model.predict(X_valid)
        report["valid"] = _metrics(y_valid, preds_val)

    preds_te = model.predict(X_test)
    report["test"] = _metrics(y_test, preds_te)

    # Pretty print
    print("\n📊 Evaluation Metrics:")
    for split in ["train", "valid", "test"]:
        if split in report:
            m = report[split]
            print(f"  {split.upper()}:  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")

    return report


def auto_global_temporal_split(
    df: pd.DataFrame,
    season_col: str = "season_x",
    week_col: str = "round",
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[pd.Index, pd.Index, pd.Index, List[int]]:
    """
    Split by season start year in chronological order.
    Uses floor for cut points and guarantees non-empty Train/Valid/Test when n>=3.
    """
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-9, "Fractions must sum to 1."

    years = sorted({season_start_year(s) for s in df[season_col].unique()})
    n = len(years)
    if n < 3:
        raise ValueError("Need at least 3 distinct seasons for train/valid/test.")

    # Use floor to avoid exhausting the tail buckets
    cut1 = max(1, int(np.floor(n * train_frac)))
    cut2 = max(cut1 + 1, int(np.floor(n * (train_frac + valid_frac))))

    # Ensure at least one season remains for test
    if cut2 >= n:
        cut2 = n - 1  # leave last season for test

    years_train = set(years[:cut1])
    years_valid = set(years[cut1:cut2])
    years_test  = set(years[cut2:])

    # In rare cases valid could end up empty -> steal one from train
    if len(years_valid) == 0:
        years_train = set(years[:cut1-1])
        years_valid = {years[cut1-1]}
        years_test  = set(years[cut1:])

    df2 = df.copy()
    df2["_year"] = df2[season_col].map(season_start_year)
    df2 = df2.sort_values([season_col, week_col], kind="mergesort")

    train_idx = df2[df2["_year"].isin(years_train)].index
    valid_idx = df2[df2["_year"].isin(years_valid)].index
    test_idx  = df2[df2["_year"].isin(years_test)].index
    return train_idx, valid_idx, test_idx, years

def auto_global_temporal_split_inseason(
    df: pd.DataFrame,
    season_col: str = "season_x",
    week_col: str = "round",
    train_frac: float = 0.8,
    valid_frac: float = 0.1,
    test_frac: float = 0.1,
    split_train_valid: bool = True,
    split_valid_test: bool = True,
    ratio_train_valid: float = 0.5,  # e.g., 0.5 ⇒ first half of boundary season to TRAIN, rest to VALID
    ratio_valid_test: float = 0.5,   # e.g., 0.5 ⇒ first half of boundary season to VALID, rest to TEST
) -> Tuple[pd.Index, pd.Index, pd.Index, List[int]]:
    """
    Global chronological split by season start year, with optional *within-season* (by rounds) splitting
    at the TRAIN↔VALID boundary and/or VALID↔TEST boundary.

    If TRAIN and VALID touch the same season:
        - earlier rounds (<= floor(ratio_train_valid * max_round_in_that_season)) → TRAIN
        - later  rounds  (> ...) → VALID

    If VALID and TEST touch the same season:
        - earlier rounds (<= floor(ratio_valid_test * max_round_in_that_season)) → VALID
        - later  rounds  (> ...) → TEST

    Guarantees non-empty buckets when n_seasons ≥ 3.
    """
    assert abs(train_frac + valid_frac + test_frac - 1.0) < 1e-9, "Fractions must sum to 1."

    years = sorted({season_start_year(s) for s in df[season_col].unique()})
    n = len(years)
    if n < 3:
        raise ValueError("Need at least 3 distinct seasons for train/valid/test.")

    # Base year cuts (use floor; keep at least 1 season per bucket)
    cut1 = max(1, int(np.floor(n * train_frac)))
    cut2 = max(cut1 + 1, int(np.floor(n * (train_frac + valid_frac))))
    if cut2 >= n:  # ensure non-empty test
        cut2 = n - 1

    years_train_list = years[:cut1]
    years_valid_list = years[cut1:cut2]
    years_test_list  = years[cut2:]

    # If valid ended empty, steal one from train
    if len(years_valid_list) == 0:
        years_train_list = years[:cut1-1]
        years_valid_list = [years[cut1-1]]
        years_test_list  = years[cut1:]

    years_train, years_valid, years_test = set(years_train_list), set(years_valid_list), set(years_test_list)

    df2 = df.copy()
    df2["_year"] = df2[season_col].map(season_start_year)
    df2 = df2.sort_values([season_col, week_col], kind="mergesort")

    # Initial whole-season masks
    train_mask = df2["_year"].isin(years_train)
    valid_mask = df2["_year"].isin(years_valid)
    test_mask  = df2["_year"].isin(years_test)

    # Helper: move early/late rounds between masks for a *single* boundary season
    def split_boundary(season_year: int, early_to: str, ratio: float):
        # max round for that season (across all players)
        season_rows = (df2["_year"] == season_year)
        if not season_rows.any():
            return
        max_round = int(df2.loc[season_rows, week_col].max())
        thr = max(1, int(np.floor(ratio * max_round)))  # e.g., 38 * 0.5 ⇒ 19

        early_rows = season_rows & (df2[week_col] <= thr)
        late_rows  = season_rows & (df2[week_col] >  thr)

        # Clear current assignment for that season
        nonlocal train_mask, valid_mask, test_mask
        train_mask = train_mask & (~season_rows)
        valid_mask = valid_mask & (~season_rows)
        test_mask  = test_mask  & (~season_rows)

        # Reassign early/late to the two adjacent sets
        if early_to == "train":         # TRAIN | VALID boundary
            train_mask |= early_rows
            valid_mask |= late_rows
        elif early_to == "valid":       # VALID | TEST boundary
            valid_mask |= early_rows
            test_mask  |= late_rows
        else:
            raise ValueError("early_to must be 'train' or 'valid'.")

    # TRAIN↔VALID boundary season = first VALID season (if any)
    if split_train_valid and len(years_valid_list) > 0:
        b1 = years_valid_list[0]
        # Only split if TRAIN and VALID are adjacent seasons (i.e., boundary not separated by gaps is fine)
        # This always holds by construction; we just apply the half-split.
        split_boundary(b1, early_to="train", ratio=ratio_train_valid)

    # VALID↔TEST boundary season = first TEST season (if any)
    if split_valid_test and len(years_test_list) > 0:
        b2 = years_test_list[0]
        split_boundary(b2, early_to="valid", ratio=ratio_valid_test)

    # Final indices
    train_idx = df2.index[train_mask]
    valid_idx = df2.index[valid_mask]
    test_idx  = df2.index[test_mask]

    # Safety: ensure non-empty
    if len(train_idx) == 0 or len(valid_idx) == 0 or len(test_idx) == 0:
        # Fallback to strict season buckets (last two seasons valid/test)
        years_train = set(years[:-2]); years_valid = {years[-2]}; years_test = {years[-1]}
        base = df2.copy()
        train_idx = base[base["_year"].isin(years_train)].index
        valid_idx = base[base["_year"].isin(years_valid)].index
        test_idx  = base[base["_year"].isin(years_test)].index

    return train_idx, valid_idx, test_idx, years





