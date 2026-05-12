import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional, List
from tqdm import tqdm
import numpy as np


# ======================== CONFIGURATION ========================
INPUT_FILE = "requests_responses.csv"
OUTPUT_FILE = "transformed_data_sampled.csv"
TARGET_SAMPLE_SIZE = 150_000
RANDOM_STATE = 42
MIN_RUN_ID = 10
EXCLUDED_STATUS = 429


# Columns that must be completely dropped
COLUMNS_TO_DROP = ['id.1', 'url', 'testcase', 'data', 'error', 'timestamp.1', 'reqid']

# Columns for one-hot encoding
ONE_HOT_COLUMNS = ['body', 'type', 'data.1', "path"]

# Maximum number of unique values per categorical column (prevents explosion)
MAX_CATEGORY_VALUES = 250

# Stratification column candidates (if available in the data)
STRATIFY_COLUMN_CANDIDATES = ['type', 'status']

# Chunk size for incremental reading (Set to None to read everything at once)
CHUNK_SIZE = 100_000 # Adjust based on available RAM


# ===============================================================


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Makes tqdm accessible for pandas operations (progress bars)
tqdm.pandas()


def load_data(filepath: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Loads the CSV file with optional chunking and progress tracking.

    If chunk_size is provided, the file is read in parts and concatenated.
    This function includes logic for estimating total rows for accurate progress display.

    Args:
        filepath (str): The path to the input CSV file.
        chunk_size (Optional[int]): The size of chunks to read. If None, reads all at once.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    logger.info(f"Loading data from {filepath}...")
    # If a chunk_size is provided (means we read the file in small parts to save memory)
    if chunk_size:
        chunks = []
        try:
            # Estimate total rows for accurate progress bar
            with open(filepath, 'rb') as f:
                total_lines = sum(1 for _ in f) - 1  # minus header
            logger.info(f"Estimated total rows: {total_lines:,}")
        except FileNotFoundError:
            logger.error(f"Error: Input file not found at {filepath}")
            return pd.DataFrame()

        # Loop through the chunks of the file (each chunk contains a set of rows from the data)
        with tqdm(total=total_lines, desc="Reading chunks", unit="rows") as pbar:
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                pbar.update(len(chunk))
        df = pd.concat(chunks, ignore_index=True)
    else:
        # Reading in a single batch (Caution for very large files)
        logger.info("Reading file in single batch (Warning: Use chunking for multi-GB datasets).")
        df = pd.read_csv(filepath, low_memory=False)

    logger.info(f"Data loading complete. Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts columns to more efficient data types (e.g., 'category' for low cardinality, 
    downcasting integers/floats) to significantly reduce memory footprint.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The memory-optimized DataFrame.
    """
    logger.info("--- Memory Optimization ---")
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Loop over every column in the dataframe to check its data type
    for col in df.columns:
        col_type = df[col].dtype
        
        # If the column type is an 'object' (usually indicates strings or mixed data)
        if col_type == 'object':
            # Convert to category if cardinality is low enough (< 50% unique ratio)
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:
                try:
                    df[col] = df[col].astype('category')
                except KeyError:
                     pass
        # If the column contains whole numbers (integers)
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        # If the column contains decimal numbers (floats)
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')

    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory optimization complete. Original size: {initial_memory:.2f} MB, Optimized size: {final_memory:.2f} MB")
    return df


def drop_unwanted_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drops specified columns from the DataFrame if they exist.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (List[str]): A list of column names to remove.

    Returns:
        pd.DataFrame: The DataFrame with unwanted columns removed.
    """
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    # If columns were found that are on the exclusion list
    if existing_cols:
        logger.info(f"Dropping columns: {', '.join(existing_cols)}")
        df = df.drop(columns=existing_cols, errors='ignore')
    else:
        logger.warning("None of the specified columns were found for dropping.")
    return df


def filter_min_runs(df: pd.DataFrame, min_run: int) -> pd.DataFrame:
    """
    Filters the DataFrame to keep only records where 'runid' is greater than or equal 
    to a specified minimum value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        min_run (int): The minimum required run ID.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Search for the 'runid' column (case insensitive)
    runid_col = next((col for col in df.columns if str(col).lower() == 'runid'), None)
    # If the 'runid' column is not present in the dataset
    if runid_col is None:
        logger.warning("Column 'runid' not found. Filtering by minimum run ID skipped.")
        return df

    initial_count = len(df)
    try:
        mask = pd.to_numeric(df[runid_col], errors='coerce').astype('Int64') >= min_run
        df = df[mask]
        logger.info(f"Filtering applied (Run ID >= {min_run}). Rows removed: {initial_count:,} -> {len(df):,} rows remaining.")
    except Exception as e:
         logger.error(f"Error during runid filtering: {e}. Skipping this filter step.")
         return df

    return df


def filter_status_code(df: pd.DataFrame, exclude_status: int) -> pd.DataFrame:
    """
    Filters the DataFrame by removing rows that match a specified status code (e.g., 429).

    Args:
        df (pd.DataFrame): The input DataFrame.
        exclude_status (int): The status code to exclude from the dataset.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    # Search for the 'status' column (case insensitive)
    status_col = next((col for col in df.columns if str(col).lower() == 'status'), None)
    # If the 'status' column is not found in the data
    if status_col is None:
        logger.warning("Column 'status' not found. Status filtering skipped.")
        return df

    initial_count = len(df)
    try:
        mask = pd.to_numeric(df[status_col], errors='coerce') != exclude_status
        df = df[mask]
        logger.info(f"Filtering applied (Excluding status {exclude_status}). Rows removed: {initial_count:,} -> {len(df):,} rows remaining.")
    except Exception as e:
         logger.error(f"Error during status filtering: {e}. Skipping this filter step.")
         return df
    return df


def one_hot_encode_columns(
    df: pd.DataFrame,
    encode_columns: List[str],
    max_unique: int = MAX_CATEGORY_VALUES
) -> pd.DataFrame:
    """
    Applies one-hot encoding to specified categorical columns while implementing 
    a cardinality reduction technique to prevent memory explosion (by mapping rare values).

    The process involves:
    1. Filling NaN values with 'MISSING'.
    2. Identifying the top N most frequent values based on `max_unique`.
    3. Mapping all other infrequent values to 'OTHER_VALUE'.
    4. Generating dummy variables using pd.get_dummies().

    Args:
        df (pd.DataFrame): The input DataFrame.
        encode_columns (List[str]): List of columns to encode.
        max_unique (int): Maximum number of unique categories to retain per column.

    Returns:
        pd.DataFrame: The DataFrame with original categorical columns replaced by dummy variables.
    """
    existing_cols = [col for col in encode_columns if col in df.columns]
    # If none of the specified columns for encoding are present in the dataset
    if not existing_cols:
        logger.warning("No specified columns found for one-hot encoding.")
        return df

    logger.info(f"--- Starting One-Hot Encoding for columns: {', '.join(existing_cols)} ---")

    # Loop through each column that we want to transform into binary (0/1) columns
    for col in tqdm(existing_cols, desc="Preprocessing categorical columns"):
        df[col] = df[col].fillna('MISSING').astype(str)

        # Cardinality reduction: Keep only the top N values
        top_values = df[col].value_counts().nlargest(max_unique).index
        df[col] = df[col].apply(lambda x: x if x in top_values else 'OTHER_VALUE')

    # Loop through the prepared columns to generate the actual dummy variables
    logger.info("Generating dummy variables...")
    for col in tqdm(existing_cols, desc="Creating dummy columns"):
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)

    # Remove original source columns after encoding
    df = df.drop(columns=existing_cols)

    logger.info(f"One-hot encoding complete. New dimensions: {df.shape}")
    return df


def stratified_sample(
    df: pd.DataFrame,
    target_size: int,
    stratify_candidates: List[str],
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Performs a stratified random sample on the dataset.

    If successful stratification column is found, sampling ensures the proportional 
    representation of categories defined by that column. Otherwise, it defaults to 
    a simple random sample.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_size (int): The desired size of the sampled dataset.
        stratify_candidates (List[str]): List of potential columns for stratification.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: The sampled or original DataFrame.
    """
    # If the dataset is already smaller than the desired sample size
    if len(df) <= target_size:
        logger.info(f"Dataset size ({len(df):,}) is less than or equal to target sample size ({target_size:,}). No sampling required.")
        return df

    # Find the best candidate for stratification
    stratify_col = next((c for c in stratify_candidates if c in df.columns), None)
    
    # If no suitable column is found to base the proportional distribution (stratification) on
    if stratify_col is None:
        logger.warning("No suitable stratification column found. Taking a simple random sample.")
        return df.sample(n=target_size, random_state=random_state).reset_index(drop=True)

    logger.info(f"Performing stratified sampling on '{stratify_col}' down to {target_size:,} rows...")
    try:
        df_sampled, _ = train_test_split(
            df,
            train_size=target_size,
            stratify=df[stratify_col],
            random_state=random_state
        )
        logger.info(f"Sampling complete. Sample size: {len(df_sampled):,} rows.")
        return df_sampled.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error during stratified sampling ({e}). Falling back to simple random sample.")
        return df.sample(n=target_size, random_state=random_state).reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the DataFrame to a CSV file with progress indication during writing.

    Args:
        df (pd.DataFrame): The final processed DataFrame.
        filepath (str): The path where the data should be saved.
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("--- Saving Data ---")

    chunk_size = 50_000
    total_rows = len(df)
    
    # Loop through the rows of the dataframe in steps (chunks) to display progress
    with tqdm(total=total_rows, desc="Writing data chunks", unit="rows") as pbar:
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = df.iloc[start:end]

            # If we are at the first set of rows, create the temporary object
            if start == 0:
                temp_df = chunk
            # Otherwise, append the new rows to the existing object
            else:
                temp_df = pd.concat([temp_df, chunk], ignore_index=True)

            pbar.update(end - start)

    # Final single write operation for efficiency
    temp_df.to_csv(filepath, index=False)

    logger.info(f"Data successfully saved to {filepath}. Total rows: {df.shape[0]:,}, Columns: {df.shape[1]}, Memory usage estimate: {memory_mb:.2f} MB.")


def main():
    """
    Main pipeline function orchestrating the entire data transformation process.

    The steps include loading, cleaning, optimizing memory, encoding categorical 
    features, and sampling down to a manageable size.
    """
    logger.info("=========================================")
    logger.info(" STARTING DATA TRANSFORMATION PIPELINE")
    logger.info("=========================================")

    # Step 1: Data loading
    df = load_data(INPUT_FILE, chunk_size=CHUNK_SIZE)
    # If the loading failed and the dataframe remains empty
    if df.empty:
        logger.error("Pipeline aborted due to failure in data loading.")
        return

    # Step 2: Memory optimization
    df = optimize_dtypes(df)

    # Step 3: Drop unwanted columns
    df = drop_unwanted_columns(df, COLUMNS_TO_DROP)

    # Step 4: Filter minimum runs
    df = filter_min_runs(df, MIN_RUN_ID)

    # Step 5: Remove excluded status codes
    df = filter_status_code(df, EXCLUDED_STATUS)

    # Step 6: One-hot encoding
    df = one_hot_encode_columns(df, ONE_HOT_COLUMNS)

    # Step 7: Stratified sampling
    df = stratified_sample(df, TARGET_SAMPLE_SIZE, STRATIFY_COLUMN_CANDIDATES)

    # Step 8: Saving the final result
    save_dataframe(df, OUTPUT_FILE)

    logger.info("=========================================")
    logger.info(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=========================================")


if __name__ == "__main__":
    main()