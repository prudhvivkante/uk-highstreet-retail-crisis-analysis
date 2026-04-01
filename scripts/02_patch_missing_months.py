"""
02_patch_missing_months.py
==========================
Patch Eight Rate-Limited Months from Manual Companies House Downloads
Paper: "Why Your High Street Is Emptying" — Kante (2026)

PURPOSE
-------
During bulk scraping, eight months returned zero counts due to Companies
House rate-limiting (HTTP 403 across all SIC codes simultaneously).
This script patches those months using manual CSV downloads, then
linearly interpolates June 2018 which could not be downloaded directly.

MONTHS PATCHED FROM MANUAL DOWNLOADS
--------------------------------------
  2016-10  →  909 dissolutions   (from Companies-House-search-results__3_.csv)
  2017-05  →  1,900              (from Companies-House-search-results__5_.csv)
  2017-11  →  1,455              (from Companies-House-search-results__6_.csv)
  2018-12  →  1,820              (from Companies-House-search-results__7_.csv)
  2019-01  →  3,699              (from Companies-House-search-results__8_.csv)
  2019-07  →  3,715              (from Companies-House-search-results__9_.csv)
  2025-02  →  5,000 (lower bound, cap reached) (from Companies-House-search-results__10_.csv)

MONTH INTERPOLATED
------------------
  2018-06  →  linearly interpolated between 2018-05 (579) and 2018-07 (2,552)
              = round((579 + 2552) / 2) = 1,566  → actual stored value may differ

INPUT
-----
data/processed/ch_dissolution_final.csv  (output of 01_ch_dissolution_scraper.py,
                                           may have zeros for patched months)
data/raw/Companies-House-search-results*.csv  (manual downloads)

OUTPUT
------
data/processed/ch_dissolution_final.csv  (in-place update with zeros replaced)

USAGE
-----
    python 02_patch_missing_months.py
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
RAW       = os.path.join(ROOT, "data", "raw")

# ── MANUAL DOWNLOAD MAPPINGS ──────────────────────────────────────────────────
# Each entry: (year_month_string, raw_csv_filename)
MANUAL_PATCHES = [
    ("2016-10", "Companies-House-search-results__3_.csv"),
    ("2017-05", "Companies-House-search-results__5_.csv"),
    ("2017-11", "Companies-House-search-results__6_.csv"),
    ("2018-12", "Companies-House-search-results__7_.csv"),
    ("2019-01", "Companies-House-search-results__8_.csv"),
    ("2019-07", "Companies-House-search-results__9_.csv"),
    ("2025-02", "Companies-House-search-results__10_.csv"),
]


def count_from_csv(path: str) -> int:
    """Count rows in a raw CH download CSV (each row = one company)."""
    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception as e:
        logger.error(f"Could not read {path}: {e}")
        return 0


def patch_panel(panel_path: str) -> pd.DataFrame:
    """Load panel, apply patches, return corrected DataFrame."""
    df = pd.read_csv(panel_path)
    df['date'] = pd.to_datetime(df['year_month'] + '-01')
    df = df.sort_values('date').reset_index(drop=True)

    logger.info(f"Loaded panel: {len(df)} months")
    logger.info(f"Zero-count months before patching: {(df['dissolution_count'] == 0).sum()}")

    # ── Apply manual patches ──────────────────────────────────────────────────
    for ym, csv_file in MANUAL_PATCHES:
        csv_path = os.path.join(RAW, csv_file)
        if not os.path.exists(csv_path):
            logger.warning(f"  {ym}: raw file not found at {csv_path}, skipping")
            continue

        count = count_from_csv(csv_path)
        mask = df['year_month'] == ym
        if mask.sum() == 0:
            logger.warning(f"  {ym}: month not found in panel")
            continue

        old_val = df.loc[mask, 'dissolution_count'].values[0]
        df.loc[mask, 'dissolution_count'] = count
        logger.info(f"  Patched {ym}: {old_val:,} → {count:,} (from {csv_file})")

    # ── Linear interpolation for 2018-06 ─────────────────────────────────────
    ym_interp = "2018-06"
    mask = df['year_month'] == ym_interp
    if mask.sum() > 0:
        idx = df[mask].index[0]
        if df.loc[idx, 'dissolution_count'] == 0 and idx > 0 and idx < len(df) - 1:
            prev_val = float(df.loc[idx-1, 'dissolution_count'])
            next_val = float(df.loc[idx+1, 'dissolution_count'])
            interp_val = int(round((prev_val + next_val) / 2))
            df.loc[idx, 'dissolution_count'] = interp_val
            logger.info(
                f"  Interpolated {ym_interp}: "
                f"({prev_val:.0f} + {next_val:.0f}) / 2 = {interp_val:,}"
            )

    zeros_remaining = (df['dissolution_count'] == 0).sum()
    logger.info(f"Zero-count months after patching: {zeros_remaining}")
    if zeros_remaining > 0:
        logger.warning("Remaining zero months:")
        logger.warning(df[df['dissolution_count'] == 0]['year_month'].tolist())

    return df[['year_month', 'dissolution_count']]


if __name__ == "__main__":
    panel_path = os.path.join(PROCESSED, "ch_dissolution_final.csv")

    if not os.path.exists(panel_path):
        logger.error(f"Panel file not found: {panel_path}")
        logger.error("Run 01_ch_dissolution_scraper.py first.")
        raise SystemExit(1)

    df_patched = patch_panel(panel_path)
    df_patched.to_csv(panel_path, index=False)

    logger.info(f"Panel saved to {panel_path}")
    logger.info(f"Total dissolutions in panel: {df_patched['dissolution_count'].sum():,}")
    logger.info("Sample (first 5 and last 5 rows):")
    print(pd.concat([df_patched.head(), df_patched.tail()]).to_string(index=False))
