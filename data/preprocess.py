"""
preprocess.py
Builds the feature matrix used for LSTM training.
Each row = (neighborhood, year, month) with engineered features.
Run: python preprocess.py
Outputs: data/processed/features.csv, data/processed/labels.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
PROC_DIR = Path(__file__).parent / "processed"
PROC_DIR.mkdir(exist_ok=True)

SEQUENCE_LENGTH = 12  # 12 months of history per prediction


def load_rent_data():
    df = pd.read_csv(RAW_DIR / "rent_by_neighborhood.csv")
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df


def compute_rent_features(df):
    """YoY rent growth, 3-month momentum, z-score relative to NYC median."""
    df = df.sort_values(["neighborhood", "date"])

    df["rent_yoy"] = df.groupby("neighborhood")["median_rent"].pct_change(12)
    df["rent_mom"] = df.groupby("neighborhood")["median_rent"].pct_change(1)
    df["rent_3m_momentum"] = df.groupby("neighborhood")["rent_mom"].transform(
        lambda x: x.rolling(3).mean()
    )

    nyc_median = df.groupby("date")["median_rent"].transform("median")
    df["rent_vs_nyc"] = df["median_rent"] / nyc_median

    return df


def generate_synthetic_permit_features(neighborhoods, dates):
    """
    Synthetic permit + license + income features.
    Replace columns here if you fetched real data in fetch_data.py.
    """
    np.random.seed(123)
    hot_hoods = ["Bushwick", "Mott Haven", "Ridgewood", "East New York", "Jackson Heights"]
    records = []

    for hood in neighborhoods:
        is_hot = hood in hot_hoods
        for date in dates:
            year_offset = (date.year - 2015) + date.month / 12
            records.append({
                "neighborhood": hood,
                "date": date,
                # New construction permits (normalized per 1000 residents)
                "permit_intensity": np.clip(
                    (0.8 if is_hot else 0.3) * (1 + 0.1 * year_offset)
                    + np.random.normal(0, 0.15), 0, 5
                ),
                # New bar/restaurant license applications per quarter
                "new_licenses": np.clip(
                    (5 if is_hot else 2) + np.random.poisson(2 if is_hot else 1), 0, 20
                ),
                # 311 housing complaints (inverse signal — drops as tenants leave)
                "housing_complaints": np.clip(
                    (3 if is_hot else 6) - 0.05 * year_offset
                    + np.random.normal(0, 1), 0, 15
                ),
                # Demographic shift index (0-1, higher = more change)
                "demo_shift": np.clip(
                    (0.6 if is_hot else 0.2) + 0.02 * year_offset
                    + np.random.normal(0, 0.05), 0, 1
                ),
                # Income index relative to borough median
                "income_index": np.clip(
                    0.85 + (0.015 if is_hot else 0.005) * year_offset
                    + np.random.normal(0, 0.05), 0.5, 2
                ),
            })

    return pd.DataFrame(records)


def build_labels(rent_df, threshold_yoy=0.08):
    """
    Label = 1 if YoY rent growth > threshold in NEXT 6 months.
    This frames it as: 'is this neighborhood about to gentrify?'
    """
    df = rent_df[["neighborhood", "date", "rent_yoy"]].copy()
    df = df.sort_values(["neighborhood", "date"])
    df["future_yoy"] = df.groupby("neighborhood")["rent_yoy"].shift(-6)
    df["label"] = (df["future_yoy"] > threshold_yoy).astype(int)
    return df[["neighborhood", "date", "label"]]


def build_sequences(features_df, labels_df, seq_len=SEQUENCE_LENGTH):
    """
    Build (X, y) where X.shape = (N, seq_len, n_features), y.shape = (N,)
    """
    feature_cols = [
        "rent_yoy", "rent_3m_momentum", "rent_vs_nyc",
        "permit_intensity", "new_licenses", "housing_complaints",
        "demo_shift", "income_index",
    ]

    merged = features_df.merge(labels_df, on=["neighborhood", "date"])
    merged = merged.sort_values(["neighborhood", "date"]).dropna(subset=feature_cols)

    X_list, y_list, meta_list = [], [], []

    for hood, group in merged.groupby("neighborhood"):
        group = group.reset_index(drop=True)
        feat = group[feature_cols].values
        labels = group["label"].values
        dates = group["date"].values

        for i in range(seq_len, len(group)):
            X_list.append(feat[i - seq_len:i])
            y_list.append(labels[i])
            meta_list.append({"neighborhood": hood, "date": str(dates[i])})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y, meta_list, feature_cols


def run():
    print("Loading rent data...")
    rent_df = load_rent_data()
    rent_df = compute_rent_features(rent_df)

    neighborhoods = rent_df["neighborhood"].unique().tolist()
    dates = sorted(rent_df["date"].unique())

    print("Building synthetic auxiliary features...")
    aux_df = generate_synthetic_permit_features(neighborhoods, dates)

    features_df = rent_df.merge(aux_df, on=["neighborhood", "date"], how="left")

    print("Computing labels...")
    labels_df = build_labels(rent_df)

    print("Building sequences...")
    X, y, meta, feature_cols = build_sequences(features_df, labels_df)

    # Save
    np.save(PROC_DIR / "X.npy", X)
    np.save(PROC_DIR / "y.npy", y)
    pd.DataFrame(meta).to_csv(PROC_DIR / "meta.csv", index=False)
    pd.DataFrame({"feature": feature_cols}).to_csv(PROC_DIR / "feature_names.csv", index=False)

    print(f"\nDone!")
    print(f"  X shape: {X.shape}  (samples, seq_len, features)")
    print(f"  y shape: {y.shape}")
    print(f"  Label balance: {y.mean():.2%} positive")
    print(f"  Saved to data/processed/")


if __name__ == "__main__":
    run()
