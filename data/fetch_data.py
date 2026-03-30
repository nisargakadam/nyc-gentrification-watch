"""
fetch_data.py
Pulls raw data from NYC Open Data APIs and Zillow Research.
Run: python fetch_data.py
Outputs CSVs to data/raw/
"""

import os
import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

# NYC Open Data app token (free at data.cityofnewyork.us)
APP_TOKEN = os.getenv("NYC_OPEN_DATA_TOKEN", "")

HEADERS = {"X-App-Token": APP_TOKEN} if APP_TOKEN else {}


def fetch_dob_permits(limit=50000):
    """
    DOB Building Permits — proxy for new construction / gut rehabs
    https://data.cityofnewyork.us/Housing-Development/DOB-Permit-Issuance/ipu4-2q9a
    """
    print("Fetching DOB permits...")
    url = "https://data.cityofnewyork.us/resource/ipu4-2q9a.json"
    params = {
        "$limit": limit,
        "$where": "permit_type IN ('NB','A1','A2') AND issuance_date >= '2015-01-01'",
        "$select": "issuance_date,community_board,permit_type,job_type,latitude,longitude",
        "$order": "issuance_date DESC",
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df.to_csv(RAW_DIR / "dob_permits.csv", index=False)
    print(f"  Saved {len(df)} permit records.")
    return df


def fetch_311_complaints(limit=50000):
    """
    311 Service Requests — noise, housing maintenance, heat complaints
    https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9
    """
    print("Fetching 311 complaints...")
    url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
    params = {
        "$limit": limit,
        "$where": "complaint_type IN ('HEAT/HOT WATER','Noise - Residential','Housing') AND created_date >= '2015-01-01'",
        "$select": "created_date,complaint_type,community_board,borough,latitude,longitude",
        "$order": "created_date DESC",
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df.to_csv(RAW_DIR / "complaints_311.csv", index=False)
    print(f"  Saved {len(df)} complaint records.")
    return df


def fetch_liquor_licenses(limit=20000):
    """
    SLA Liquor License Applications — a classic gentrification signal
    https://data.ny.gov/Economic-Development/Liquor-Authority-Quarterly-List-of-Active-Licenses/hrvs-fxs2
    """
    print("Fetching liquor licenses...")
    url = "https://data.ny.gov/resource/hrvs-fxs2.json"
    params = {
        "$limit": limit,
        "$where": "county IN ('NEW YORK','KINGS','BRONX','QUEENS','RICHMOND')",
        "$select": "effective_date,license_type_name,county,zip_code,premises_name",
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df.to_csv(RAW_DIR / "liquor_licenses.csv", index=False)
    print(f"  Saved {len(df)} license records.")
    return df


def generate_synthetic_rent_data():
    """
    Synthetic monthly median rent by neighborhood (2015–2024).
    Replace with real Zillow ZORI data from:
    https://www.zillow.com/research/data/ → ZORI (Smoothed): All Homes Plus Multifamily
    """
    print("Generating synthetic rent data (replace with Zillow ZORI CSV)...")
    import numpy as np

    neighborhoods = [
        "Bushwick", "East New York", "Crown Heights", "Bedford-Stuyvesant",
        "Flatbush", "Sunset Park", "Ridgewood", "Jackson Heights",
        "Astoria", "Long Island City", "Harlem", "Washington Heights",
        "South Bronx", "Mott Haven", "Williamsburg", "Greenpoint",
        "Park Slope", "Fort Greene", "Clinton Hill", "Prospect Heights",
    ]

    np.random.seed(42)
    records = []
    base_rents = {n: np.random.uniform(1200, 2200) for n in neighborhoods}
    # Some neighborhoods gentrify faster
    hot_hoods = ["Bushwick", "Mott Haven", "Ridgewood", "East New York", "Jackson Heights"]

    for hood in neighborhoods:
        rent = base_rents[hood]
        for year in range(2015, 2025):
            for month in range(1, 13):
                growth = 0.004 if hood in hot_hoods else 0.002
                noise = np.random.normal(0, 30)
                rent = rent * (1 + growth) + noise
                records.append({
                    "neighborhood": hood,
                    "year": year,
                    "month": month,
                    "median_rent": max(800, rent),
                })

    df = pd.DataFrame(records)
    df.to_csv(RAW_DIR / "rent_by_neighborhood.csv", index=False)
    print(f"  Saved {len(df)} rent records.")
    return df


if __name__ == "__main__":
    generate_synthetic_rent_data()

    # Uncomment to fetch live data (requires internet + optional token):
    # fetch_dob_permits()
    # fetch_311_complaints()
    # fetch_liquor_licenses()

    print("\nDone! Raw data saved to data/raw/")
    print("For live NYC Open Data, set NYC_OPEN_DATA_TOKEN env var and uncomment fetchers.")
