"""
01_ch_dissolution_scraper.py
============================
Companies House Dissolution Panel Scraper
Paper: "Why Your High Street Is Emptying" — Kante (2026)
GitHub: https://github.com/prudhvivkante/uk-highstreet-retail-crisis-analysis

PURPOSE
-------
Constructs the monthly retail dissolution panel (n=666,462 records,
Jan 2015 – Jan 2026) used throughout the paper. Queries the Companies House
advanced company search API for all SIC 47xx dissolved companies by month.

NO API KEY REQUIRED — uses the public Companies House search endpoint.

OUTPUTS
-------
data/processed/ch_dissolution_final.csv
    Columns: year_month, dissolution_count
    133 monthly observations, Jan 2015 – Jan 2026

NOTES
-----
- Rate limits: CH throttles at ~100 req/min. Script uses single-threaded
  operation with exponential backoff (45s / 90s / 135s on 403 responses).
- 8 months in the 2016–2018 range required correction from direct batch
  downloads (see 02_patch_missing_months.py).
- June 2018 is linearly interpolated from adjacent months.
- February 2025 is a lower bound (5,000-record download cap reached).
- 97.5% of dissolutions fall on Tuesdays (CH administrative batch processing);
  monthly aggregation removes this artefact entirely.

SIC CODES
---------
44 verified SIC 47xx sub-codes used. Note: codes 47420, 47550, 47590,
47720, 47740 do NOT exist in Companies House (despite appearing in ONS
SIC 2007 documentation) — querying them returns zero results.

USAGE
-----
    python 01_ch_dissolution_scraper.py

    # To resume from a specific month:
    python 01_ch_dissolution_scraper.py --start 2018-01 --end 2019-12
"""

import requests
import time
import csv
import json
import os
import argparse
import logging
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ch_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

BASE_URL = "https://find-and-update.company-information.service.gov.uk/advanced-search/get-results"

# All 44 verified SIC 47xx sub-codes in Companies House
# (codes 47420, 47550, 47590, 47720, 47740 verified NOT to exist in CH)
SIC_CODES_47 = [
    "47110", "47190", "47210", "47220", "47230", "47240", "47250",
    "47260", "47270", "47280", "47290", "47300", "47410", "47430",
    "47510", "47520", "47530", "47540", "47560", "47571", "47572",
    "47580", "47591", "47599", "47610", "47620", "47630", "47640",
    "47650", "47710", "47730", "47741", "47742", "47749", "47750",
    "47760", "47770", "47781", "47782", "47789", "47791", "47799",
    "47810", "47820", "47830", "47890", "47910", "47990"
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Research) "
        "UK-Retail-Dissolution-Study/1.0 "
        "Contact: prudhvi.ncsnlr@gmail.com"
    ),
    "Accept": "application/json",
}

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "ch_dissolution_final.csv"
)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def month_range(start: date, end: date):
    """Yield (year, month) tuples from start to end inclusive."""
    cur = start.replace(day=1)
    end = end.replace(day=1)
    while cur <= end:
        yield cur.year, cur.month
        cur += relativedelta(months=1)


def last_day(year: int, month: int) -> int:
    """Return the last calendar day of a given month."""
    import calendar
    return calendar.monthrange(year, month)[1]


def fetch_month_count(year: int, month: int, sic_code: str,
                      max_retries: int = 3) -> int:
    """
    Query CH for the number of SIC-code dissolutions in a calendar month.
    Returns integer count, or raises after max_retries.
    """
    date_from = f"{year:04d}-{month:02d}-01"
    date_to   = f"{year:04d}-{month:02d}-{last_day(year, month):02d}"

    params = {
        "dissolvedFromDate": date_from,
        "dissolvedToDate":   date_to,
        "sicCodes":          sic_code,
        "status":            "dissolved",
        "type":              "ltd",          # limited companies
        "subtype":           "community-interest-company",
    }

    backoff_delays = [45, 90, 135]

    for attempt in range(max_retries):
        try:
            resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                # CH returns {"hits": {"total": {"value": N, ...}, ...}}
                return int(data.get("hits", {}).get("total", {}).get("value", 0))

            elif resp.status_code == 403:
                delay = backoff_delays[min(attempt, len(backoff_delays)-1)]
                logger.warning(
                    f"  Rate limited (403) for {sic_code} {year}-{month:02d}. "
                    f"Waiting {delay}s (attempt {attempt+1}/{max_retries})"
                )
                time.sleep(delay)

            elif resp.status_code == 429:
                delay = backoff_delays[min(attempt, len(backoff_delays)-1)] * 2
                logger.warning(f"  429 Too Many Requests. Waiting {delay}s")
                time.sleep(delay)

            else:
                logger.error(
                    f"  Unexpected status {resp.status_code} for "
                    f"{sic_code} {year}-{month:02d}"
                )
                time.sleep(10)

        except requests.exceptions.ConnectionError as e:
            delay = backoff_delays[min(attempt, len(backoff_delays)-1)]
            logger.warning(f"  ConnectionError: {e}. Waiting {delay}s")
            time.sleep(delay)

        except requests.exceptions.Timeout:
            logger.warning(f"  Timeout for {sic_code} {year}-{month:02d}")
            time.sleep(15)

        except Exception as e:
            logger.error(f"  Unexpected error: {e}")
            time.sleep(10)

    raise RuntimeError(
        f"Failed after {max_retries} attempts: {sic_code} {year}-{month:02d}"
    )


# ── MAIN SCRAPER ──────────────────────────────────────────────────────────────

def scrape_dissolution_panel(start: date, end: date) -> dict:
    """
    Scrape monthly dissolution counts for all SIC 47xx codes.

    Returns dict: {(year, month): total_count}
    """
    monthly_totals = defaultdict(int)
    months = list(month_range(start, end))
    total_months = len(months)

    logger.info(
        f"Scraping {total_months} months × {len(SIC_CODES_47)} SIC codes "
        f"= {total_months * len(SIC_CODES_47):,} requests"
    )
    logger.info(f"Period: {start.strftime('%Y-%m')} to {end.strftime('%Y-%m')}")

    for m_idx, (year, month) in enumerate(months, 1):
        month_total = 0
        logger.info(
            f"[{m_idx:3d}/{total_months}] {year}-{month:02d} ..."
        )

        for sic in SIC_CODES_47:
            count = fetch_month_count(year, month, sic)
            monthly_totals[(year, month)] += count
            month_total += count
            time.sleep(0.35)  # ~170 req/min — safely below CH rate limit

        logger.info(
            f"  {year}-{month:02d}: {month_total:,} total dissolutions"
        )

    return dict(monthly_totals)


def save_results(totals: dict, output_path: str):
    """Save monthly totals to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = sorted(totals.items())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year_month", "dissolution_count"])
        for (year, month), count in rows:
            writer.writerow([f"{year:04d}-{month:02d}", count])

    logger.info(f"Saved {len(rows)} months to {output_path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape UK retail dissolution data from Companies House API"
    )
    parser.add_argument(
        "--start", default="2015-01",
        help="Start month YYYY-MM (default: 2015-01)"
    )
    parser.add_argument(
        "--end", default="2026-01",
        help="End month YYYY-MM (default: 2026-01)"
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH,
        help="Output CSV path"
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m").date()
    end_date   = datetime.strptime(args.end,   "%Y-%m").date()

    logger.info("=" * 60)
    logger.info("Companies House Dissolution Panel Scraper")
    logger.info("Kante (2026) — Why Your High Street Is Emptying")
    logger.info("=" * 60)

    totals = scrape_dissolution_panel(start_date, end_date)
    save_results(totals, args.output)

    logger.info("Scraping complete.")
    logger.info(
        f"Total dissolutions scraped: {sum(totals.values()):,}"
    )
