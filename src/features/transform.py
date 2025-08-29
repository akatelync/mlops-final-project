import os

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as file:
        return yaml.safe_load(file)


def create_preprocessor():
    """Create preprocessing pipeline."""
    low_cardinality_cols = [
        "Vehicle Type",
        "Reason for cancelling by Customer",
        "Driver Cancellation Reason",
        "Incomplete Rides Reason",
        "Payment Method",
    ]
    high_cardinality_cols = ["Pickup Location", "Drop Location"]

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    low_card_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    high_card_pipeline = Pipeline(
        [
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            )
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipeline, None),
            ("low_cat", low_card_pipeline, low_cardinality_cols),
            ("high_cat", high_card_pipeline, high_cardinality_cols),
        ]
    )


def transform_data(
    input_path: str = "data/processed/ingested_data.parquet",
    config_path: str = "config.yaml",
) -> str:
    """
    Transform data for model training.

    Args:
        input_path: Path to ingested data
        config_path: Path to configuration file

    Returns:
        str: Path to the transformed data file
    """
    config = load_config(config_path)

    df = pd.read_parquet(input_path)

    X = df.drop(columns=["Booking ID", "Customer ID", "Is_Cancelled"])
    y = df["Is_Cancelled"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = create_preprocessor()
    preprocessor.transformers[0] = ("num", preprocessor.transformers[0][1], num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
    )

    os.makedirs("data/processed", exist_ok=True)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_path = "data/processed/train_data.parquet"
    test_path = "data/processed/test_data.parquet"
    preprocessor_path = "data/processed/preprocessor.pkl"

    train_data.to_parquet(train_path, index=False)
    test_data.to_parquet(test_path, index=False)
    joblib.dump(preprocessor, preprocessor_path)

    return train_path


if __name__ == "__main__":
    output_path = transform_data()
    print(f"Transformed training data saved to: {output_path}")
