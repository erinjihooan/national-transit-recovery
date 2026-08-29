"""
Ingest raw FTA NTD monthly ridership data.
Assumes you've manually downloaded the 'Monthly Ridership' Excel/CSV
release into data/raw/ (NTD publishes as .xlsx with multiple sheets).
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def load_ntd_upt(filepath: Path, sheet_name: str = "UPT") -> pd.DataFrame:
    """
    Load the Unlinked Passenger Trips (UPT) sheet from the NTD monthly
    ridership workbook. NTD stores each month as a separate column,
    so this loads in 'wide' format initially.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df


if __name__ == "__main__":
    ntd_path = RAW_DIR / "ntd_monthly_ridership.xlsx"
    df_wide = load_ntd_upt(ntd_path)
    df_wide.to_parquet(INTERIM_DIR / "ntd_upt_raw.parquet")
    print(f"Loaded {len(df_wide)} agency-mode rows.")