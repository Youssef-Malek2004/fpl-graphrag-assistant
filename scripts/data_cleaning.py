import os
import pandas as pd

def drop_columns_save_interim(
    df: pd.DataFrame,
    cols_to_drop: list = None,
    filename: str = "dataset",
    output_subdir: str = "interim",
) -> pd.DataFrame:
    """
    Drops specified columns from a DataFrame.
    Saves the cleaned DataFrame in ../data/interim relative to this script.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols_to_drop : list, optional
        Columns to Drop, by default None
    filename : str, optional
        Base name for saved CSV files (default is 'dataset').
    output_subdir : str, optional
        Subdirectory inside /data (default is 'interim').

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame with specified columns removed.
    """

    # Determine absolute path: ../data/interim relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", output_subdir))

    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)

    # Drop them from the main DataFrame
    df_reduced = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Build save paths
    cleaned_path = os.path.join(data_dir, f"{filename}_post_drop.csv")

    # Save both files
    df_reduced.to_csv(cleaned_path, index=False)

    print(f"Cleaned file saved to: {cleaned_path}")

    return df_reduced

def normalize_position_column(
    df: pd.DataFrame,
    column: str = "position",
    filename: str = "dataset",
    output_subdir: str = "interim",
) -> pd.DataFrame:
    """
    Normalizes position values — e.g., replaces 'GKP' with 'GK' — and saves the cleaned dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str, optional
        The column to normalize (default is 'position').
    filename : str, optional
        Base name for saved file (default is 'dataset').
    output_subdir : str, optional
        Subdirectory under /data to save (default is 'interim').

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with normalized position values.
    """

    # Determine save path (../data/interim)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "..", "data", output_subdir))
    os.makedirs(data_dir, exist_ok=True)

    df = df.copy()
    df[column] = df[column].apply(lambda x: "GK" if x == "GKP" else x)

    # Save cleaned file
    cleaned_path = os.path.join(data_dir, f"{filename}_normalized.csv")
    df.to_csv(cleaned_path, index=False)

    return df
