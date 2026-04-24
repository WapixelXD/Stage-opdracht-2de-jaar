import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional, List
from tqdm import tqdm
import numpy as np

# ======================== CONFIGURATIE ========================
INPUT_FILE = "requests_responses.csv"
OUTPUT_FILE = "transformed_data_sampled.csv"
TARGET_SAMPLE_SIZE = 150_000
RANDOM_STATE = 42
MIN_RUN_ID = 10
EXCLUDED_STATUS = 429

# Kolommen die volledig verwijderd moeten worden
COLUMNS_TO_DROP = ['id.1', 'url', 'testcase', 'data', 'error', 'timestamp.1', 'reqid']

# Kolommen voor one-hot encoding
ONE_HOT_COLUMNS = ['body', 'type', 'data.1', "path"]

# Maximaal aantal unieke waarden per categorische kolom (voorkomt explosie)
MAX_CATEGORY_VALUES = 250

# Stratificatiekolom (indien beschikbaar in de data)
STRATIFY_COLUMN_CANDIDATES = ['type', 'status']

# Chunk-grootte voor incrementeel inlezen (zet op None om alles in één keer te lezen)
CHUNK_SIZE = 100_000  # Pas aan op basis van beschikbaar RAM

# ===============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Maak tqdm beschikbaar voor pandas
tqdm.pandas()

def load_data(filepath: str, chunk_size: Optional[int] = None) -> pd.DataFrame:
    """
    Laad het CSV-bestand met optionele chunking en een voortgangsbalk.
    Als chunk_size is ingesteld, wordt het bestand in delen ingelezen en samengevoegd.
    """
    logger.info(f"📂 Inlezen van {filepath}...")
    if chunk_size:
        chunks = []
        # Tel eerst het aantal regels voor een accurate progress bar (optioneel)
        with open(filepath, 'rb') as f:
            # Snelle schatting van totaal aantal regels
            total_lines = sum(1 for _ in f) - 1  # minus header
        logger.info(f"   Geschat totaal regels: {total_lines:,}")

        with tqdm(total=total_lines, desc="Regels inlezen", unit="regels") as pbar:
            for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                pbar.update(len(chunk))
        df = pd.concat(chunks, ignore_index=True)
    else:
        # In één keer lezen met tqdm (tqdm kan geen directe voortgang tonen, dus we doen een schatting)
        df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"✅ Dataset geladen: {df.shape[0]:,} rijen, {df.shape[1]} kolommen.")
    return df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converteer kolommen naar efficiëntere datatypes om geheugen te besparen.
    """
    logger.info("⚡ Optimaliseren van datatypes...")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object':
            # Converteer object naar category als het aantal unieke waarden klein is
            num_unique = df[col].nunique()
            if num_unique / len(df) < 0.5:  # alleen als minder dan 50% uniek
                df[col] = df[col].astype('category')
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    logger.info(f"   Geheugen na optimalisatie: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df


def drop_unwanted_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """Verwijder opgegeven kolommen als ze bestaan."""
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    if existing_cols:
        logger.info(f"🗑️ Verwijderen van kolommen: {existing_cols}")
        df = df.drop(columns=existing_cols)
    else:
        logger.info("ℹ️ Geen van de opgegeven kolommen gevonden om te verwijderen.")
    return df


def filter_min_runs(df: pd.DataFrame, min_run: int) -> pd.DataFrame:
    """Behoud alleen rijen waarvan 'runid' >= min_run."""
    runid_col = next((col for col in df.columns if col.lower() == 'runid'), None)
    if runid_col is None:
        logger.warning("⚠️ Kolom 'runid' niet gevonden. Filteren op runs overgeslagen.")
        return df

    initial_count = len(df)
    mask = df[runid_col] >= min_run
    df = df[mask]
    logger.info(f"🔽 Runs < {min_run} verwijderd: {initial_count:,} → {len(df):,} rijen.")
    return df


def filter_status_code(df: pd.DataFrame, exclude_status: int) -> pd.DataFrame:
    """Verwijder rijen met een specifieke statuscode."""
    status_col = next((col for col in df.columns if col.lower() == 'status'), None)
    if status_col is None:
        logger.warning("⚠️ Kolom 'status' niet gevonden. Filteren op status overgeslagen.")
        return df

    initial_count = len(df)
    mask = df[status_col] != exclude_status
    df = df[mask]
    logger.info(f"🚫 Status {exclude_status} verwijderd: {initial_count:,} → {len(df):,} rijen.")
    return df


def one_hot_encode_columns(
    df: pd.DataFrame,
    encode_columns: List[str],
    max_unique: int = MAX_CATEGORY_VALUES
) -> pd.DataFrame:
    """
    Pas one-hot encoding toe op opgegeven kolommen.
    Om geheugenexplosie te voorkomen worden alleen de meest voorkomende
    waarden behouden; de rest wordt 'OTHER_VALUE'.
    Toont een voortgangsbalk tijdens het encoderen.
    """
    existing_cols = [col for col in encode_columns if col in df.columns]
    if not existing_cols:
        logger.warning("⚠️ Geen van de opgegeven one-hot kolommen gevonden.")
        return df

    logger.info(f"🔧 One-hot encoding voor kolommen: {existing_cols}")

    for col in tqdm(existing_cols, desc="One-hot encoding voorbereiden"):
        # Vul missende waarden en converteer naar string
        df[col] = df[col].fillna('MISSING').astype(str)

        # Beperk cardinaliteit
        top_values = df[col].value_counts().nlargest(max_unique).index
        df[col] = df[col].where(df[col].isin(top_values), 'OTHER_VALUE')

    # pd.get_dummies met voortgangsindicatie (doen we door eerst dummy kolommen te maken en dan te concat)
    logger.info("   Creëren van dummy variabelen...")
    for col in tqdm(existing_cols, desc="Dummy kolommen aanmaken"):
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=existing_cols)  # verwijder originele kolommen

    logger.info(f"✅ One-hot encoding voltooid. Nieuwe dimensies: {df.shape}")
    return df


def stratified_sample(
    df: pd.DataFrame,
    target_size: int,
    stratify_candidates: List[str],
    random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Neem een gestratificeerde steekproef van de dataset."""
    if len(df) <= target_size:
        logger.info(f"ℹ️ Dataset heeft slechts {len(df):,} rijen (≤ {target_size:,}), geen sampling nodig.")
        return df

    stratify_col = next((c for c in stratify_candidates if c in df.columns), None)
    if stratify_col is None:
        logger.warning("⚠️ Geen geschikte stratificatiekolom gevonden. Er wordt een gewone random sample genomen.")
        return df.sample(n=target_size, random_state=random_state)

    logger.info(f"🎲 Gestratificeerde sampling op '{stratify_col}' naar {target_size:,} rijen...")
    df_sampled, _ = train_test_split(
        df,
        train_size=target_size,
        stratify=df[stratify_col],
        random_state=random_state
    )
    logger.info(f"✅ Sample genomen: {len(df_sampled):,} rijen.")
    return df_sampled


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Sla het DataFrame op als CSV met voortgangsindicatie."""
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"💾 Opslaan als {filepath}...")

    # tqdm voor schrijven: we kunnen de chunks gebruiken
    chunk_size = 50_000
    total_rows = len(df)
    with tqdm(total=total_rows, desc="Regels wegschrijven", unit="regels") as pbar:
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            df.iloc[start:end].to_csv(
                filepath,
                index=False,
                mode='a' if start > 0 else 'w',
                header=(start == 0)
            )
            pbar.update(end - start)

    logger.info(f"📊 Dataset opgeslagen: {df.shape[0]:,} rijen, {df.shape[1]} kolommen, {memory_mb:.2f} MB")


def main():
    """Hoofd pipeline voor datatransformatie."""
    logger.info("=== START DATA TRANSFORMATIE PIPELINE ===")

    # Stap 1: Data laden
    df = load_data(INPUT_FILE, chunk_size=CHUNK_SIZE)

    # Stap 2: Geheugen optimaliseren
    df = optimize_dtypes(df)

    # Stap 3: Ongewenste kolommen verwijderen
    df = drop_unwanted_columns(df, COLUMNS_TO_DROP)

    # Stap 4: Filteren op runs >= 10
    df = filter_min_runs(df, MIN_RUN_ID)

    # Stap 5: Verwijder status 429
    df = filter_status_code(df, EXCLUDED_STATUS)

    # Stap 6: One-hot encoding op body, type, data.1
    df = one_hot_encode_columns(df, ONE_HOT_COLUMNS)

    # Stap 7: Gestratificeerde steekproef van 150.000 rijen
    df = stratified_sample(df, TARGET_SAMPLE_SIZE, STRATIFY_COLUMN_CANDIDATES)

    # Stap 8: Opslaan
    save_dataframe(df, OUTPUT_FILE)

    logger.info("=== PIPELINE SUCCESVOL AFGEROND ===")


if __name__ == "__main__":
    main()