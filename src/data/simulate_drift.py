import os

import numpy as np
import pandas as pd
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def simulate_data_drift(
    input_path=None, output_path=None, drift_percentage=0.3, config_path="config.yaml"
):
    config = load_config(config_path)

    if input_path is None:
        # Update to correct processed data path
        input_path = config["data"].get(
            "processed_path", "data/processed/train_data.parquet"
        )

    if output_path is None:
        output_path = config["data"].get(
            "current_path", "data/current/drifted_data.parquet"
        )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)
    df_drift = df.copy()

    # Numerical columns
    exclude_cols = ["Booking ID", "Customer ID", "Is_Cancelled"]
    num_cols = [
        c
        for c in df_drift.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]
    for col in num_cols:
        df_drift[col] += np.random.normal(0, df_drift[col].std() * 0.2, len(df_drift))

    # Categorical drift
    categorical_cols = ["Vehicle Type", "Payment Method"]
    for col in categorical_cols:
        if col in df_drift.columns:
            n_changes = int(len(df_drift) * drift_percentage)
            indices = np.random.choice(len(df_drift), n_changes, replace=False)
            df_drift.loc[indices, col] = np.random.choice(
                df_drift[col].unique(), n_changes
            )

    # Concept drift: flip 10% of labels
    n_flips = int(len(df_drift) * 0.1)
    flip_indices = np.random.choice(len(df_drift), n_flips, replace=False)
    df_drift.loc[flip_indices, "Is_Cancelled"] = (
        1 - df_drift.loc[flip_indices, "Is_Cancelled"]
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_drift.to_parquet(output_path, index=False)

    print(f"Simulated drift saved to: {output_path}")
    print(
        f"Modified {drift_percentage * 100}% of records, flipped {n_flips} target labels."
    )

    return output_path
