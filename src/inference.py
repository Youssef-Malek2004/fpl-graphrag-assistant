import os
import sys
import argparse
import pickle
import json
import joblib
import numpy as np
import pandas as pd

# Make project root (parent of src/) importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.data_cleaning import drop_columns_save_interim, normalize_position_column
from scripts.feature_engineering import (
    one_hot_encode_columns,
    map_bool_to_int,
    add_form,
    add_team_and_opponent_goals,
    add_upcoming_total_points, add_upcoming_total_points_inference,
)
from scripts.model_training import build_xy


# -------------------------------------------------
# Paths / constants
# -------------------------------------------------

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ffnn_model.pkl")
DEFAULT_SCALER_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "dataset_scaler.pkl")
DEFAULT_SCALED_COLS_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "dataset_scaled_columns.json")
DEFAULT_LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "dataset_name_label_encoder.pkl")

COLUMNS_TO_DROP = [
    "selected", "transfers_in", "transfers_out",
    "transfers_balance", "GW", 'element',
    'fixture', 'kickoff_time', 'opponent_team', 'team_a_score',
    'team_h_score', 'influence', 'opp_team_name', 'own_goals', 'creativity',
    'threat', 'team_x'
]

ONE_HOT_COLS = ["position"]

EXPECTED_POSITION_DUMMIES = [
    "position_FWD",
    "position_MID",
    "position_GK",
]

BOOL_INT_COLS = ['was_home', 'position_FWD', 'position_MID', 'position_GK']

SCALE_EXCLUDE = ["round", "upcoming_total_points", "season_x"]


# -------------------------------------------------
# Utilities for loading artifacts
# -------------------------------------------------

def load_label_encoder(le_path: str):
    """Load the saved LabelEncoder for 'name'."""
    with open(le_path, "rb") as f:
        le = pickle.load(f)
    return le

def load_scaler_artifacts(scaler_path: str, scaled_cols_path: str):
    """
    Returns (scaler, cols_to_scale)
    scaler_path: dataset_scaler.pkl
    scaled_cols_path: dataset_scaled_columns.json
    """
    with open(scaler_path, "rb") as f:
        scaler_bundle = pickle.load(f)
    scaler = scaler_bundle["scaler"]
    with open(scaled_cols_path, "r", encoding="utf-8") as f:
        cols_json = json.load(f)
    cols_to_scale = cols_json["columns"]
    return scaler, cols_to_scale

def load_model_ffnn(model_path: str):
    """
    Load the trained FFNNRegressor (joblib).
    Must return an object with .predict(X_numpy) -> (N,) or (N,1)
    """
    model = joblib.load(model_path)
    return model


# -------------------------------------------------
# Step 1: minimal clone of label_encode_column BUT using saved encoder
# -------------------------------------------------

def apply_saved_label_encoder(df: pd.DataFrame, column: str, le, drop_original: bool = True) -> pd.DataFrame:
    """
    Use an already-fitted LabelEncoder 'le' to transform df[column] -> df[f"{column}_encoded"].
    If df has a player not seen in training, this will raise. You can decide how you want to handle that.
    """
    # transform using existing encoder
    df[f"{column}_encoded"] = le.transform(df[column].astype(str))
    if drop_original:
        df = df.drop(columns=[column], errors="ignore")
    return df


# -------------------------------------------------
# Step 2: scaling step for inference
# -------------------------------------------------

def apply_saved_scaler(df: pd.DataFrame,
                       scaler,
                       cols_to_scale,
                       exclude_cols=SCALE_EXCLUDE) -> pd.DataFrame:
    """
    We DO NOT fit, we ONLY transform using the saved scaler.
    We respect the same numeric columns that were scaled during training.
    """
    df_out = df.copy()

    # restrict to the intersection of cols_to_scale and df columns,
    # minus excluded columns (safety if training vs inference differ slightly)
    cols_final = [c for c in cols_to_scale if c in df_out.columns and c not in exclude_cols]

    if len(cols_final) == 0:
        # nothing to scale
        return df_out

    # scaler was fit on these same columns in this order.
    # we transform in that order:
    df_out[cols_final] = scaler.transform(df_out[cols_final].astype(float).values)
    return df_out


# -------------------------------------------------
# Core preprocessing pipeline for inference
# -------------------------------------------------

def ensure_expected_dummies(df: pd.DataFrame,
                            expected_cols: list[str]) -> pd.DataFrame:
    """
    Make sure all expected one-hot dummy columns exist.
    If any is missing (because that class wasn't present in this batch),
    create it and fill with 0.
    """
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df


def preprocess_for_inference(
    raw_df: pd.DataFrame,
    label_encoder_path: str = DEFAULT_LABEL_ENCODER_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    scaled_cols_path: str = DEFAULT_SCALED_COLS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Mirror training pipeline up to the point where we have model-ready X.
    Returns:
        X_model      -> what the model expects (no name_encoded, no total_points)
        meta_df      -> identifier columns we want to keep for output (season_x, round, name_encoded)
        y_true       -> upcoming_total_points if available
    """

    # 1. team + opponent goals
    df = add_team_and_opponent_goals(raw_df)

    # 2. drop unused cols
    df = drop_columns_save_interim(df, COLUMNS_TO_DROP, filename="inference_batch")

    # 3. normalize position
    df = normalize_position_column(df)

    # 4. label encode "name" using the SAVED encoder
    le = load_label_encoder(label_encoder_path)
    df = apply_saved_label_encoder(df, column="name", le=le, drop_original=True)

    # 5. one-hot encode e.g. "position"
    df = one_hot_encode_columns(df, ONE_HOT_COLS)

    # 5.1. make sure all position_* columns that existed in training exist now
    df = ensure_expected_dummies(df, EXPECTED_POSITION_DUMMIES)

    # 6. map bools to int (was_home + position_* we care about)
    df = map_bool_to_int(df, BOOL_INT_COLS)

    # 6. map bools to int
    df = map_bool_to_int(df, BOOL_INT_COLS)

    # 7. add rolling form features
    df = add_form(df)

    # 8. add upcoming_total_points column (target/label)
    df = add_upcoming_total_points_inference(df)

    # 9. apply saved scaler
    scaler, cols_to_scale = load_scaler_artifacts(scaler_path, scaled_cols_path)
    df_scaled = apply_saved_scaler(df, scaler, cols_to_scale, exclude_cols=SCALE_EXCLUDE)

    # 10. build XY the SAME way as training
    X, y = build_xy(df_scaled, keep_player_id=True, player_col="name_encoded")

    # drop training-only col
    if 'total_points' in X.columns:
        X = X.drop(columns=['total_points'])

    # keep metadata before we strip name_encoded
    meta_cols = []
    for c in ['season_x', 'round', 'name_encoded']:
        if c in X.columns:
            meta_cols.append(c)
    meta_df = X[meta_cols].copy()

    # model input drops name_encoded
    X_model = X.drop(columns=["name_encoded"], errors="ignore")

    # y is upcoming_total_points
    y_true = y.copy() if y is not None else None

    return X_model, meta_df, y_true


def inference(
    df_raw: pd.DataFrame,
    model_path: str = DEFAULT_MODEL_PATH,
    label_encoder_path: str = DEFAULT_LABEL_ENCODER_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    scaled_cols_path: str = DEFAULT_SCALED_COLS_PATH,
) -> pd.DataFrame:
    """
    Run inference on a raw dataframe that looks like:
    season_x,name,position,team_x,assists,bonus,bps,clean_sheets,creativity,element,fixture,...
    ...,value,was_home,yellow_cards,GW

    Returns dataframe with:
      season_x, round, name_encoded, prediction_upcoming_total_points
    """

    # Step A. preprocess to get model-ready features
    X_model, meta_df, y_true = preprocess_for_inference(
        df_raw,
        label_encoder_path=label_encoder_path,
        scaler_path=scaler_path,
        scaled_cols_path=scaled_cols_path,
    )

    model = load_model_ffnn(model_path)

    y_pred = np.asarray(model.predict(X_model.values)).reshape(-1)

    result = meta_df.copy()
    result["prediction_upcoming_total_points"] = y_pred

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on raw FPL-like rows.")
    p.add_argument(
        "--csv",
        help="Path to raw CSV with columns like season_x,name,position,... as provided.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to saved FFNN model (joblib .pkl).",
    )
    p.add_argument(
        "--le",
        default=DEFAULT_LABEL_ENCODER_PATH,
        help="Path to saved LabelEncoder pickle for 'name'.",
    )
    p.add_argument(
        "--scaler",
        default=DEFAULT_SCALER_PATH,
        help="Path to saved scaler pickle.",
    )
    p.add_argument(
        "--scaled-cols",
        default=DEFAULT_SCALED_COLS_PATH,
        help="Path to JSON listing scaled columns.",
    )
    return p.parse_args()


def main():
    DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "test_inference.csv")
    DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "ffnn_model.pkl")
    DEFAULT_LE = os.path.join(PROJECT_ROOT, "data", "interim", "dataset_name_label_encoder.pkl")
    DEFAULT_SCALER = os.path.join(PROJECT_ROOT, "data", "interim", "dataset_scaler.pkl")
    DEFAULT_SCALED_COLS = os.path.join(PROJECT_ROOT, "data", "interim", "dataset_scaled_columns.json")

    args = parse_args()

    # Apply defaults if not provided
    csv_path = args.csv or DEFAULT_CSV
    model_path = args.model or DEFAULT_MODEL
    le_path = args.le or DEFAULT_LE
    scaler_path = args.scaler or DEFAULT_SCALER
    scaled_cols_path = args.scaled_cols or DEFAULT_SCALED_COLS

    # -------- Validate Files --------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Input CSV not found: {csv_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"❌ Label encoder not found: {le_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Scaler pickle not found: {scaler_path}")
    if not os.path.exists(scaled_cols_path):
        raise FileNotFoundError(f"❌ Scaled columns JSON not found: {scaled_cols_path}")

    # -------- Run Inference --------
    raw_df = pd.read_csv(csv_path, low_memory=False)
    preds_df = inference(
        df_raw=raw_df,
        model_path=model_path,
        label_encoder_path=le_path,
        scaler_path=scaler_path,
        scaled_cols_path=scaled_cols_path,
    )

    # -------- Output --------
    print("Inference completed successfully!\n")
    print(preds_df.head())
    output_path = os.path.join(PROJECT_ROOT, "data", "predictions", "inference_output.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    preds_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")



if __name__ == "__main__":
    main()
