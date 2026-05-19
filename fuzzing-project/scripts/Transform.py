import gc
import logging
import re  # Import regex library for cleaning names
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # NEW: For multiclass target encoding
from tqdm import tqdm

# ======================== CONFIGURATION ========================
INPUT_FILE = "requests_responses.csv"
OUTPUT_FILE = "transformed_data_sampled.csv"
TARGET_SAMPLE_SIZE = 150_000
RANDOM_STATE = 42
MIN_RUN_ID = 10
EXCLUDED_STATUS = 429
TARGET_COL = "status"  # NEW: Explicit target column configuration

# Columns that must be completely dropped
COLUMNS_TO_DROP: List[str] = [
    "id.1",
    "url",
    "testcase",
    "data",
    "error",
    "timestamp.1",
    "reqid",
    "timestamp",
    "data.1", 
    "path"
]

# Columns for one-hot encoding
ONE_HOT_COLUMNS: List[str] = ["body", "type"]

# Maximum number of unique values per categorical column (prevents explosion)
MAX_CATEGORY_VALUES: int = 250

# Stratification column candidates (if available in the data)
STRATIFY_COLUMN_CANDIDATES: List[str] = ["type", "status"]

# Chunk size for incremental reading (Set to None to read everything at once)
CHUNK_SIZE: Optional[int] = 100_000  # Adjust based on available RAM
# ===============================================================


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Makes tqdm accessible for pandas operations (progress bars)
tqdm.pandas()


def load_data(filepath: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """Loads the CSV file with optional chunking and progress tracking.

    Args:
        filepath (str): The path to the input CSV file.
        chunk_size (Optional[int]): The size of chunks to read. If None, reads
          all at once.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    logger.info(f"Loading data from {filepath}...")
    if chunk_size:
        chunks = []
        try:
            # Estimate total rows for accurate progress bar
            with open(filepath, "rb") as f:
                total_lines = sum(1 for _ in f) - 1  # minus header
            logger.info(f"Estimated total rows: {total_lines:,}")
        except FileNotFoundError:
            logger.error(f"Error: Input file not found at {filepath}")
            return pd.DataFrame()

        # Loop through the chunks of the file (each chunk contains a set of rows from the data)
        with tqdm(
            total=total_lines, desc="Reading chunks", unit="rows"
        ) as pbar:
            for chunk in pd.read_csv(
                filepath, chunksize=chunk_size, low_memory=False
            ):
                chunks.append(chunk)
                pbar.update(len(chunk))
        df = pd.concat(chunks, ignore_index=True)
    else:
        # Reading in a single batch (Caution for very large files)
        logger.info(
            "Reading file in single batch (Warning: Use chunking for multi-GB datasets)."
        )
        df = pd.read_csv(filepath, low_memory=False)

    logger.info(
        f"Data loading complete. Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns."
    )
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Converts columns to more efficient data types (e.g., 'category' for low

    cardinality, downcasting integers/floats) to significantly reduce memory
    footprint.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The memory-optimized DataFrame.
    """
    logger.info("--- Memory Optimization ---")
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == "object":
            num_unique = df[col].nunique()
            # Convert to category if cardinality is low enough (< 50% unique ratio) and not too large
            if num_unique / len(df) < 0.5 and num_unique > 1:
                try:
                    df[col] = df[col].astype("category")
                except Exception as e:
                    logger.warning(
                        f"Skipping category conversion for '{col}': {e}"
                    )

        elif col_type == "int64":
            # Downcast integer types
            try:
                df[col] = pd.to_numeric(df[col], downcast="integer")
            except Exception:
                pass
        elif col_type == "float64":
            # Downcast float types
            try:
                df[col] = pd.to_numeric(df[col], downcast="float")
            except Exception:
                pass

    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(
        f"Memory optimization complete. Original size: {initial_memory:.2f} MB, Optimized size: {final_memory:.2f} MB"
    )
    return df


def drop_unwanted_columns(
    df: pd.DataFrame, columns_to_drop: List[str]
) -> pd.DataFrame:
    """Drops specified columns from the DataFrame if they exist.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (List[str]): A list of column names to remove.

    Returns:
        pd.DataFrame: The DataFrame with unwanted columns removed.
    """
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    if existing_cols:
        logger.info(f"Dropping columns: {', '.join(existing_cols)}")
        df = df.drop(columns=existing_cols, errors="ignore")
    else:
        logger.warning(
            "None of the specified columns were found for dropping."
        )
    return df


def filter_min_runs(df: pd.DataFrame, min_run: int) -> pd.DataFrame:
    """Filters the DataFrame to keep only records where 'runid' is greater than

    or equal to a specified minimum value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        min_run (int): The minimum required run ID.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Search for the 'runid' column (case insensitive)
    runid_col = next(
        (col for col in df.columns if str(col).lower() == "runid"), None
    )
    if runid_col is None:
        logger.warning(
            "Column 'runid' not found. Filtering by minimum run ID skipped."
        )
        return df

    initial_count = len(df)
    try:
        # Use safe numeric conversion for filtering
        mask = (
            pd.to_numeric(df[runid_col], errors="coerce").astype("Int64")
            >= min_run
        )
        df = df[mask]
        logger.info(
            f"Filtering applied (Run ID >= {min_run}). Rows removed: {initial_count:,} -> {len(df):,} rows remaining."
        )
    except Exception as e:
        logger.error(
            f"Error during runid filtering: {e}. Skipping this filter step."
        )
        return df

    return df


def filter_status_code(df: pd.DataFrame, exclude_status: int) -> pd.DataFrame:
    """Filters the DataFrame by removing rows that match a specified status code

    (e.g., 429).

    Args:
        df (pd.DataFrame): The input DataFrame.
        exclude_status (int): The status code to exclude from the dataset.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Search for the 'status' column (case insensitive)
    status_col = next(
        (col for col in df.columns if str(col).lower() == "status"), None
    )
    if status_col is None:
        logger.warning("Column 'status' not found. Status filtering skipped.")
        return df

    initial_count = len(df)
    try:
        # Safe numeric conversion for comparison
        mask = pd.to_numeric(df[status_col], errors="coerce") != exclude_status
        df = df[mask]
        logger.info(
            f"Filtering applied (Excluding status {exclude_status}). Rows removed: {initial_count:,} -> {len(df):,} rows remaining."
        )
    except Exception as e:
        logger.error(
            f"Error during status filtering: {e}. Skipping this filter step."
        )
        return df
    return df


def one_hot_encode_columns(
    df: pd.DataFrame,
    encode_columns: List[str],
    max_unique: int = MAX_CATEGORY_VALUES,
) -> pd.DataFrame:
    """Applies one-hot encoding to specified categorical columns while

    implementing a cardinality reduction technique to prevent memory explosion
    (by mapping rare values).

    Args:
        df (pd.DataFrame): The input DataFrame.
        encode_columns (List[str]): List of columns to encode.
        max_unique (int): Maximum number of unique categories to retain per
          column.

    Returns:
        pd.DataFrame: The DataFrame with original categorical columns replaced
        by dummy variables.
    """
    existing_cols = [col for col in encode_columns if col in df.columns]
    if not existing_cols:
        logger.warning("No specified columns found for one-hot encoding.")
        return df

    logger.info(
        f"--- Starting One-Hot Encoding for columns: {', '.join(existing_cols)} ---"
    )

    for col in tqdm(existing_cols, desc="Preprocessing categorical columns"):
        # 1. Fill NaN and convert to string
        df[col] = df[col].fillna("MISSING").astype(str)

        # 2. Cardinality reduction (Grouping rare/unknown values)
        value_counts = df[col].value_counts()
        top_values = value_counts.nlargest(max_unique).index
        df[col] = df[col].apply(
            lambda x: x if x in top_values else "OTHER_VALUE"
        )

    # 3. Generate dummy variables
    logger.info("Generating dummy variables...")
    for col in tqdm(existing_cols, desc="Creating dummy columns"):
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)

    # 4. Remove original source columns
    df = df.drop(columns=existing_cols)

    logger.info(f"One-hot encoding complete. New dimensions: {df.shape}")
    return df


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """CLEANS COLUMN NAMES to ensure they are safe for use in machine learning

    libraries (like XGBoost), removing special characters that cause
    ValueErrors.

    It replaces unsafe characters with underscores.

    Args:
        df (pd.DataFrame): The DataFrame with potentially unsafe column names.

    Returns:
        pd.DataFrame: The DataFrame with sanitized, ML-safe column names.
    """
    logger.warning("--- Running Column Name Sanitization Check ---")
    initial_cols = list(df.columns)
    new_cols = []

    # Regex to find any character that is NOT a letter, number, or underscore
    pattern = re.compile(r"[^\w]")

    for col in initial_cols:
        sanitized_col = pattern.sub("", col)  # Replace unsafe characters with nothing
        new_cols.append(sanitized_col)

    # Check if any changes were made
    if new_cols != initial_cols:
        logger.info("Column names sanitized successfully.")
        df.columns = new_cols
    else:
        logger.info(
            "No illegal characters detected in column names; no sanitization needed."
        )

    return df


def encode_target_column(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """NEW: Encodes the multiclass target column (status codes) into consecutive

    integers starting from 0 (e.g., 200->0, 404->1, 500->2). This is strictly
    required by XGBoost for multiclass classification objectives.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column to encode.

    Returns:
        pd.DataFrame: The DataFrame with the encoded target column.
    """
    if target_col not in df.columns:
        logger.error(
            f"Target column '{target_col}' not found. Target encoding skipped."
        )
        return df

    logger.info(f"--- Encoding Target Column '{target_col}' ---")
    initial_distribution = df[target_col].value_counts()
    logger.info(
        f"Original classes and counts:\n{initial_distribution.to_string()}"
    )

    try:
        label_encoder = LabelEncoder()
        # Transform labels to 0, 1, 2...
        df[target_col] = label_encoder.fit_transform(df[target_col])

        # Log mapping details for transparency
        mapping = {
            index: label for index, label in enumerate(label_encoder.classes_)
        }
        logger.info(f"Target encoding successful! Class mapping: {mapping}")
    except Exception as e:
        logger.error(f"Error during target encoding: {e}")

    return df


def stratified_sample(
    df: pd.DataFrame,
    target_size: int,
    stratify_candidates: List[str],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Performs a stratified random sample on the dataset.

    If successful stratification column is found, sampling ensures the
    proportional representation of categories defined by that column.
    Otherwise, it defaults to a simple random sample.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_size (int): The desired size of the sampled dataset.
        stratify_candidates (List[str]): List of potential columns for
          stratification.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: The sampled or original DataFrame.
    """
    initial_count = len(df)
    if initial_count <= target_size:
        logger.info(
            f"Dataset size ({initial_count:,}) is less than or equal to target sample size ({target_size:,}). No sampling required."
        )
        return df.copy()

    stratify_col = next(
        (c for c in stratify_candidates if c in df.columns), None
    )

    if stratify_col is None:
        logger.warning(
            "No suitable stratification column found. Taking a simple random sample."
        )
        return (
            df.sample(n=target_size, random_state=random_state)
            .reset_index(drop=True)
        )

    logger.info(
        f"Performing stratified sampling on '{stratify_col}' down to {target_size:,} rows..."
    )
    try:
        # This block requires scikit-learn
        from sklearn.model_selection import train_test_split

        df_sampled, _ = train_test_split(
            df,
            train_size=target_size,
            stratify=df[stratify_col],
            random_state=random_state,
        )
        logger.info(
            f"Sampling complete. Sample size: {len(df_sampled):,} rows."
        )
        return df_sampled.reset_index(drop=True)

    except NameError as e:
        logger.error(
            f"Dependency Error encountered during stratification ({e}). This usually means scikit-learn is missing or scoped incorrectly."
        )
        logger.warning("Falling back to simple random sample instead.")
        return (
            df.sample(n=target_size, random_state=random_state)
            .reset_index(drop=True)
        )
    except Exception as e:
        logger.error(
            f"General Error encountered during stratified sampling ({type(e).__name__}: {e}). Falling back to simple random sample."
        )
        return (
            df.sample(n=target_size, random_state=random_state)
            .reset_index(drop=True)
        )


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Saves the DataFrame to a CSV file with progress indication during

    writing.

    Args:
        df (pd.DataFrame): The final processed DataFrame.
        filepath (str): The path where the data should be saved.
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("--- Saving Data ---")

    total_rows = len(df)
    chunk_size = 50_000
    temp_df = pd.DataFrame()  # Initialize temporary storage for concatenation

    # Loop through the rows of the dataframe in steps (chunks) to display progress
    with tqdm(
        total=total_rows, desc="Writing data chunks", unit="rows"
    ) as pbar:
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = df.iloc[start:end]

            if start == 0:
                temp_df = chunk
            else:
                # Concatenate to temporary storage (this is safer than mode='a')
                temp_df = pd.concat([temp_df, chunk], ignore_index=True)

            pbar.update(len(chunk))

    # Final single write operation for efficiency
    temp_df.to_csv(filepath, index=False)

    logger.info(
        f"Data successfully saved to {filepath}. Total rows: {df.shape[0]:,}, Columns: {df.shape[1]}, Memory usage estimate: {memory_mb:.2f} MB."
    )


def main():
    """Main pipeline function orchestrating the entire data transformation

    process.

    Includes target column encoding to support clean multiclass training in
    XGBoost.
    """
    logger.info("=========================================")
    logger.info(" STARTING DATA TRANSFORMATION PIPELINE")
    logger.info("=========================================\n")

    # Step 1: Data loading
    df = load_data(INPUT_FILE, chunk_size=CHUNK_SIZE)
    if df.empty:
        logger.critical(
            "Pipeline aborted due to failure in data loading or empty dataset."
        )
        return

    # Step 2: Memory optimization (Type Casting)
    df = optimize_dtypes(df)

    # Step 3: Column Removal
    df = drop_unwanted_columns(df, COLUMNS_TO_DROP)

    # Step 4: Filter minimum runs
    df = filter_min_runs(df, MIN_RUN_ID)

    # Step 5: Remove excluded status codes (like 429)
    df = filter_status_code(df, EXCLUDED_STATUS)

    # Step 6: One-hot encoding (Feature Engineering)
    df = one_hot_encode_columns(df, ONE_HOT_COLUMNS)

    # === NEW CRITICAL STEP FOR MULTICLASS XGBOOST ===
    # Encodes target status codes (e.g., 200, 404, 500) into sequential integers (0, 1, 2)
    df = encode_target_column(df, TARGET_COL)

    # Step 7: Sanitizes column names to ensure compatibility with ML libraries
    df = sanitize_column_names(df)

    # Step 8: Stratified sampling
    df = stratified_sample(df, TARGET_SAMPLE_SIZE, STRATIFY_COLUMN_CANDIDATES)

    # Step 9: Saving the final result
    save_dataframe(df, OUTPUT_FILE)

    logger.info("\n=========================================")
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=========================================")


if __name__ == "__main__":
    main()