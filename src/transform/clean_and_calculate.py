"""
Clean NTD data and calculate city-level recovery metrics vs. a 2019 baseline.
"""
import pandas as pd
import numpy as np
from pathlib import Path

INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MIN_BASELINE_UPT = 1000

# NTD 2-letter mode codes -> readable names
MODE_NAME_MAP = {
    "DR": "Demand Response", "FB": "Ferryboat", "LR": "Light Rail",
    "MB": "Bus", "SR": "Streetcar Rail", "TB": "Trolleybus",
    "VP": "Vanpool", "CB": "Commuter Bus", "RB": "Bus Rapid Transit",
    "CR": "Commuter Rail", "YR": "Hybrid Rail", "MG": "Monorail/Automated Guideway",
    "AR": "Alaska Railroad", "TR": "Aerial Tramway", "HR": "Heavy Rail",
    "OR": "Other Rail", "IP": "Inclined Plane", "PB": "Publico", "CC": "Cable Car",
}

# Manually mapped major metros where the default parsing (splitting on "--"/",")
# would produce an awkward or misleading name. Extend as you spot-check output.
CITY_NAME_MAP = {
    "New York--Jersey City--Newark, NY--NJ": "New York",
    "Los Angeles--Long Beach--Anaheim, CA": "Los Angeles",
    "Chicago, IL--IN": "Chicago",
    "San Francisco--Oakland, CA": "San Francisco",
    "Washington--Arlington, DC--VA--MD": "Washington DC",  # fixed: real UZA string has "--Arlington"
    "Boston, MA--NH--RI": "Boston",
    "Philadelphia, PA--NJ--DE--MD": "Philadelphia",
    "Seattle--Tacoma, WA": "Seattle",
    # Optional readability simplification for consolidated city-county govts.
    # Comment out any of these if you'd rather keep the official name as-is.
    "Nashville-Davidson, TN": "Nashville",
    "Lexington-Fayette, KY": "Lexington",
    "Louisville/Jefferson County, KY--IN": "Louisville",
    "Athens-Clarke County, GA": "Athens",
    "Augusta-Richmond County, GA--SC": "Augusta",
}

# NTD's catch-all buckets for rural/non-urbanized-area service within a state
# (e.g. "Wisconsin Non-UZA"). Not real cities -- exclude from a city-level analysis.
NON_UZA_SUFFIX = "Non-UZA"


def melt_to_tidy(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide (one column per month) into tidy long format."""
    id_cols = [
        "NTD ID", "Agency", "Mode/Type of Service Status",
        "Reporter Type", "UZA Name", "Mode", "TOS",
    ]
    month_cols = [c for c in df_wide.columns if "/" in c]  # e.g. '1/2002'

    return df_wide.melt(
        id_vars=id_cols,
        value_vars=month_cols,
        var_name="period_raw",
        value_name="upt",
    )


def parse_period(df: pd.DataFrame) -> pd.DataFrame:
    """Parse 'M/YYYY' labels (e.g. '6/2026') into real dates."""
    df["date"] = pd.to_datetime(df["period_raw"], format="%m/%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize city (UZA) names and map mode codes to readable labels."""
    df = df.dropna(subset=["UZA Name"]).copy()
    df["uza_name_clean"] = df["UZA Name"].str.strip()

    # Drop NTD's rural "Non-UZA" catch-all rows -- these represent statewide
    # rural service, not a specific city, and would show up as junk entries
    # like "Wisconsin Non-UZA" in a city-level dashboard filter.
    df = df[~df["uza_name_clean"].str.contains(NON_UZA_SUFFIX, na=False)].copy()

    df["city"] = df["uza_name_clean"].replace(CITY_NAME_MAP)
    # Fallback for UZAs not manually mapped: take the first city name
    # before the "--" (multi-city UZA) and "," (state suffix) delimiters.
    unmapped = ~df["uza_name_clean"].isin(CITY_NAME_MAP.keys())
    df.loc[unmapped, "city"] = (
        df.loc[unmapped, "uza_name_clean"].str.split("--").str[0].str.split(",").str[0].str.strip()
    )

    df["mode_name"] = df["Mode"].map(MODE_NAME_MAP).fillna(df["Mode"])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop inactive services, coerce ridership to numeric, validate each
    AGENCY's own 2019 baseline (not the city's combined total), then
    aggregate the surviving agencies to the city + mode + month grain.

    Validating before aggregation matters: if we summed agencies first and
    checked validity second, an agency with zero 2019 data (like a
    university bus system that started reporting years later) could ride
    along inside a city+mode total that looks valid only because a
    different, established agency shares that same mode -- e.g. Texas A&M's
    multi-hundred-thousand-rider bus service getting silently averaged
    against Brazos Transit's much smaller, complete 2019 baseline.

    We still filter at the mode/agency grain here even though the final
    output is city-level only -- excluding a bad series has to happen
    before it gets summed into the city total, not after.
    """
    df["upt"] = pd.to_numeric(df["upt"], errors="coerce")
    df = df[df["Mode/Type of Service Status"] == "Active"]

    # --- Validate at (city, mode_name, Agency) grain, BEFORE cross-agency
    # aggregation.
    valid_2019 = (
        df[df["year"] == 2019]
        .groupby(["city", "mode_name", "Agency"])["upt"]
        .agg(["count", "sum"])
    )
    valid_2019["avg"] = valid_2019["sum"] / valid_2019["count"]

    # Require: (a) at least 6 reported months in 2019, so a single stray
    # month can't pass as a "baseline"; (b) non-zero total; (c) an average
    # monthly baseline above MIN_BASELINE_UPT, so thin-but-technically-valid
    # series don't quietly inflate the city total once they scale up.
    valid_keys = valid_2019[
        (valid_2019["count"] >= 6)
        & (valid_2019["sum"] > 0)
        & (valid_2019["avg"] >= MIN_BASELINE_UPT)
    ].index

    df = df.set_index(["city", "mode_name", "Agency"])
    df = df.loc[df.index.isin(valid_keys)].reset_index()

    # Aggregate to the city + mode + month grain across the *validated*
    # agencies only.
    grouped = (
        df.groupby(["city", "mode_name", "date", "year", "month"], as_index=False)
        .agg(upt=("upt", "sum"), n_agencies=("Agency", "nunique"))
    )
    return grouped


def calculate_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """
    City-level recovery % = total current-month ridership (summed across
    all modes) / average 2019 monthly ridership (also summed across all
    modes first, then divided by 12).

    Summing before dividing makes this ridership-weighted by construction:
    a mode with 200,000 riders/month naturally dominates a mode with 300
    riders/month, with no separate weighting step required. This avoids
    the failure mode of averaging each mode's own recovery percentage,
    where a single small, volatile mode can distort the whole city's
    number even though it represents a tiny share of actual ridership.
    """
    city_baseline_total_2019 = (
        df[df["year"] == 2019]
        .groupby("city")["upt"]
        .sum()
        .rename("baseline_2019_avg_upt")
        / 12  # convert full-year total to an average month, for a fair comparison
    )

    city_monthly = (
        df.groupby(["city", "date", "year", "month"], as_index=False)["upt"]
        .sum()
    )

    city_monthly = city_monthly.merge(city_baseline_total_2019, on="city", how="left")
    city_monthly["recovery_pct"] = (
        city_monthly["upt"] / city_monthly["baseline_2019_avg_upt"]
    ) * 100

    return city_monthly


def build_mode_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ridership by city + mode + month, for composition/mix views like a
    "% of Total UPT by mode" stacked bar chart.

    This is NOT a recovery calculation, so it doesn't hit the averaging
    problem that motivated summing before dividing in calculate_recovery().
    Raw ridership counts by mode can be safely compared and stacked --
    the issue earlier was specifically averaging RATIOS (each mode's own
    recovery %) across modes of very different sizes, not looking at modes'
    raw ridership side by side, which is what a mode-share chart does.
    """
    mode_monthly = (
        df.groupby(["city", "mode_name", "date", "year", "month"], as_index=False)["upt"]
        .sum()
        .rename(columns={"mode_name": "mode"})
    )
    return mode_monthly


def build_pipeline(raw_parquet: Path) -> pd.DataFrame:
    df_wide = pd.read_parquet(raw_parquet)
    df = melt_to_tidy(df_wide)
    df = parse_period(df)
    df = clean_names(df)
    df = handle_missing_values(df)

    mode_breakdown = build_mode_breakdown(df)
    city_recovery = calculate_recovery(df)

    return city_recovery, mode_breakdown


if __name__ == "__main__":
    city_recovery, mode_breakdown = build_pipeline(INTERIM_DIR / "ntd_upt_raw.parquet")

    # City-level recovery: same filename and column set as before -- your
    # existing City Rankings sheet needs no reconnection, just Refresh /
    # Extract Refresh after re-running this.
    out_cols = ["city", "date", "year", "month",
                "upt", "baseline_2019_avg_upt", "recovery_pct"]
    city_recovery = city_recovery[out_cols]
    city_recovery.to_csv(PROCESSED_DIR / "transit_recovery_tidy.csv", index=False)
    city_recovery.to_parquet(PROCESSED_DIR / "transit_recovery_tidy.parquet", index=False)

    # NEW, separate file for mode-share/composition views (e.g. the
    # "Mode Breakdown" % of Total UPT stacked bar chart). Add this as a
    # second data source in Tableau -- it doesn't touch the existing one.
    mode_cols = ["city", "mode", "date", "year", "month", "upt"]
    mode_breakdown = mode_breakdown[mode_cols]
    mode_breakdown.to_csv(PROCESSED_DIR / "transit_mode_breakdown.csv", index=False)
    mode_breakdown.to_parquet(PROCESSED_DIR / "transit_mode_breakdown.parquet", index=False)

    print(f"City recovery: {len(city_recovery):,} rows across {city_recovery['city'].nunique()} cities.")
    print(f"Mode breakdown: {len(mode_breakdown):,} rows across {mode_breakdown['city'].nunique()} cities.")