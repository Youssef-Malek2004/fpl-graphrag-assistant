import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import pickle
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler



def one_hot_encode_columns(
    df: pd.DataFrame,
    columns_to_encode: list,
    filename: str = "dataset",
    output_subdir: str = "interim",
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    One-hot encodes specified categorical columns in the given DataFrame.
    Saves both the encoded DataFrame and a record of encoded column names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns.
    columns_to_encode : list
        List of column names to one-hot encode.
    filename : str, optional
        Base name for saved CSVs (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs will be saved (default is 'interim').
    drop_first : bool, optional
        Whether to drop the first level of each encoded variable
        (useful for regression models to avoid dummy-variable trap).

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with one-hot encoded columns.
    """

    # --- Setup directories ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- One-hot encode ---
    encoded_df = pd.get_dummies(df, columns=columns_to_encode, drop_first=drop_first)

    # --- Save outputs ---
    encoded_path = os.path.join(output_folder, f"{filename}_encoded.csv")

    encoded_df.to_csv(encoded_path, index=False)

    return encoded_df


def label_encode_column(
    df: pd.DataFrame,
    column: str,
    filename: str = "dataset",
    output_subdir: str = "interim",
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Label-encodes a single categorical column (e.g., player names) and saves
    both the encoded DataFrame and the fitted LabelEncoder object for reuse.

    Artifacts saved under ../data/<output_subdir>/:
      - <filename>_label_encoded.csv
      - <filename>_<column>_label_encoder.pkl

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to label-encode.
    filename : str, optional
        Base name for saved files (default: 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs are saved (default: 'interim').

    Returns
    -------
    Tuple[pd.DataFrame, LabelEncoder]
        - DataFrame with a new column '<column>_encoded'
        - The fitted LabelEncoder (for reverse mapping)
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", output_subdir))
    os.makedirs(data_dir, exist_ok=True)

    # --- Fit and encode ---
    le = LabelEncoder()
    df[f"{column}_encoded"] = le.fit_transform(df[column].astype(str))
    df = df.drop(columns=[column])

    # --- Save the encoded dataframe ---
    encoded_csv_path = os.path.join(data_dir, f"{filename}_label_encoded.csv")
    df.to_csv(encoded_csv_path, index=False)

    encoder_pkl_path = os.path.join(data_dir, f"{filename}_{column}_label_encoder.pkl")
    with open(encoder_pkl_path, "wb") as f:
        pickle.dump(le, f)

    return df, le


def map_bool_to_int(
    df: pd.DataFrame,
    columns_to_map: list,
    filename: str = "dataset",
    output_subdir: str = "interim",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns.
    columns_to_map : list
        List of column names to map to int.
    filename : str, optional
        Base name for saved CSVs (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs will be saved (default is 'interim').

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with mapped columns.
    """

    # --- Setup directories ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- Map bool values to int ---
    mapped_df = df.copy()
    for col in columns_to_map:
        mapped_df[col] = mapped_df[col].map(lambda x: 1 if str(x) == "True" else 0)

    # --- Save outputs ---
    mapped_path = os.path.join(output_folder, f"{filename}_mapped.csv")

    mapped_df.to_csv(mapped_path, index=False)

    return mapped_df


def add_form(
    df: pd.DataFrame,
    filename: str = "dataset",
    output_subdir: str = "interim",
    name_column: str = "name_encoded",
) -> pd.DataFrame:
    """
    Adds 'form' for each (name, season_x) as the average of the PREVIOUS `window`
    gameweeks' total_points, divided by `divisor`, using up to `min_periods` available
    past GWs (no leakage). Saves to ../data/<output_subdir>/<filename>.csv.

    Expects columns: ['name'/'name_encoded', 'season_x', 'round', 'total_points'] exactly.
    """
    window = 4
    divisor = 10.0
    min_periods = 1
    fill_strategy = "zero"

    required = [name_column, "season_x", "round", "total_points"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["round"] = pd.to_numeric(out["round"], errors="coerce")
    out = out.sort_values([name_column, "season_x", "round"])

    form = (
        out.groupby([name_column, "season_x"])["total_points"]
           .apply(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean() / divisor)
           .reset_index(level=[0, 1], drop=True)
    )

    if fill_strategy == "zero":
        form = form.fillna(0.0)

    out["form"] = form

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", output_subdir))
    os.makedirs(data_dir, exist_ok=True)

    out_path = os.path.join(data_dir, f"{filename}.csv")
    out.to_csv(out_path, index=False)
    print(f"Form-added file saved to: {out_path}")

    return out

def add_team_and_opponent_goals(
    df: pd.DataFrame,
    filename: str = "dataset",
    output_subdir: str = "interim",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns.
    filename : str, optional
        Base name for saved CSVs (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where outputs will be saved (default is 'interim').

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame with added features columns.
    """

    # --- Setup directories ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- Add Features ---
    df_with_features = df.copy()

    df_with_features['ally_goals'] = df_with_features.apply(
        lambda x: x['team_h_score'] if x['was_home'] == True else x['team_a_score'],
        axis=1
    )

    df_with_features['opponent_goals'] = df_with_features.apply(
        lambda x: x['team_a_score'] if x['was_home'] == True else x['team_h_score'],
        axis=1
    )

    # --- Save outputs ---
    mapped_path = os.path.join(output_folder, f"{filename}_mapped.csv")
    df_with_features.to_csv(mapped_path, index=False)

    return df_with_features

import pandas as pd
import os

def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int] = [1, 2],
    filename: str = "dataset",
    output_subdir: str = "interim"
) -> pd.DataFrame:
    """
    Adds lag features (e.g., lag 1 and lag 2) for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, typically time-sorted.
    columns : list of str
        Columns to generate lag features for.
    lags : list of int, optional
        Lag steps to apply (default is [1, 2]).
    filename : str, optional
        Base name for saved CSV (default is 'dataset').
    output_subdir : str, optional
        Folder under /data where output is saved (default is 'interim').

    Returns
    -------
    pd.DataFrame
        DataFrame with new lag columns added.
    """
    # --- Setup directories ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- Add lag features ---
    df_with_lags = df.copy()
    for col in columns:
        if col not in df.columns:
            print(f"Skipping '{col}' — not found in DataFrame.")
            continue
        for lag in lags:
            df_with_lags[f"{col}_lag{lag}"] = df_with_lags[col].shift(lag)

    # --- Save outputs ---
    lagged_path = os.path.join(output_folder, f"{filename}_lagged.csv")
    df_with_lags.to_csv(lagged_path, index=False)

    return df_with_lags

def add_upcoming_total_points_inference(
    df: pd.DataFrame,
    player_col: str = "name_encoded",
    season_col: str = "season_x",
    week_col: str = "round",
    points_col: str = "total_points",
) -> pd.DataFrame:
    """
    Like add_upcoming_total_points, but DOES NOT drop rows with missing future points.
    We keep upcoming_total_points even if it's NaN (model target at training time).
    This is safe for inference because we only need features to predict, not ground truth.
    """
    df_sorted = df.sort_values([player_col, season_col, week_col]).copy()

    df_sorted["upcoming_total_points"] = (
        df_sorted.groupby([player_col, season_col])[points_col].shift(-1)
    )

    return df_sorted.reset_index(drop=True)


def add_upcoming_total_points(
    df: pd.DataFrame,
    player_col: str = "name_encoded",
    season_col: str = "season_x",
    week_col: str = "round",
    points_col: str = "total_points",
) -> pd.DataFrame:
    """
    Adds a new column `upcoming_total_points` representing next week's points
    for each player-season, shifted by -1 in chronological order.
    """
    df_sorted = df.sort_values([player_col, season_col, week_col])
    df_sorted["upcoming_total_points"] = (
        df_sorted.groupby([player_col, season_col])[points_col].shift(-1)
    )
    df_sorted = df_sorted.dropna(subset=["upcoming_total_points"]).reset_index(drop=True)
    return df_sorted

def season_start_year(season_str: str) -> int:
    s = str(season_str)
    try:
        return int(s.split("-")[0])
    except Exception:
        return int(float(s))

def scale_all_numeric(
    df: pd.DataFrame,
    filename: str = "dataset",
    output_subdir: str = "interim",
    columns: Optional[List[str]] = None,      # None -> auto-detect numeric
    exclude: Optional[List[str]] = None,      # columns to leave untouched
    scaler_type: str = "standard",            # "standard" | "minmax" | "robust" | "maxabs"
    feature_range: Tuple[float, float] = (0.0, 1.0),  # only for "minmax"
    fit_on: Optional[pd.DataFrame] = None,    # fit scaler on this DF, then transform `df` (avoid leakage)
    save_scaler: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, object, List[str]]:
    """
    Scales numeric columns and writes artifacts under ../data/<output_subdir>/ :
      - <filename>_scaled.csv
      - <filename>_scaler.pkl
      - <filename>_scaled_columns.json

    Returns (scaled_df, fitted_scaler, scaled_columns).
    """
    # --- Setup directories (same convention as other helpers) ---
    root_data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_folder = os.path.join(root_data_dir, output_subdir)
    os.makedirs(output_folder, exist_ok=True)

    # --- Decide which columns to scale ---
    if columns is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        numeric_cols = list(columns)

    if exclude:
        ex = set(exclude)
        numeric_cols = [c for c in numeric_cols if c not in ex]

    if not numeric_cols:
        if verbose:
            print("No numeric columns selected for scaling. Saving original DataFrame.")
        out_path = os.path.join(output_folder, f"{filename}_scaled.csv")
        df.to_csv(out_path, index=False)
        return df.copy(), None, []

    # --- Choose scaler ---
    st = scaler_type.lower().strip()
    if st == "standard":
        scaler = StandardScaler()
    elif st == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif st == "robust":
        scaler = RobustScaler()
    elif st == "maxabs":
        scaler = MaxAbsScaler()
    else:
        raise ValueError("Unknown scaler_type. Use one of: 'standard' | 'minmax' | 'robust' | 'maxabs'.")

    # --- Fit on specified data (e.g., your train split) ---
    fit_df = fit_on if fit_on is not None else df
    scaler.fit(fit_df[numeric_cols].astype(float).values)

    # --- Transform and save ---
    out = df.copy()
    out[numeric_cols] = scaler.transform(out[numeric_cols].astype(float).values)

    scaled_path = os.path.join(output_folder, f"{filename}_scaled.csv")
    out.to_csv(scaled_path, index=False)

    if save_scaler:
        pkl_path = os.path.join(output_folder, f"{filename}_scaler.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"scaler": scaler, "columns": numeric_cols, "scaler_type": st}, f)

        cols_json_path = os.path.join(output_folder, f"{filename}_scaled_columns.json")
        with open(cols_json_path, "w", encoding="utf-8") as f:
            json.dump({"columns": numeric_cols}, f, indent=2)

        if verbose:
            print(f"Saved scaler -> {pkl_path}")
            print(f"Saved scaled columns -> {cols_json_path}")

    if verbose:
        print(f"Scaled DataFrame saved to: {scaled_path}")
        print(f"Columns scaled ({len(numeric_cols)}): {numeric_cols}")

    return out, scaler, numeric_cols
