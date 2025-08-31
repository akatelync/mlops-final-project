"""
Shared fixtures and test utilities for the MLOps project tests.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import yaml


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "data": {
            "raw_path": "data/raw/dataset.csv",
            "ingested_path": "data/processed/ingested_data.parquet",
            "processed_path": "data/processed/train_data.parquet",
            "reference_path": "data/processed/train_data.parquet",
            "current_path": "data/current/drifted_data.parquet",
        },
        "features": {
            "feature_cols": ["avg_vtat", "timestamp"],
            "numerical_cols": ["avg_vtat"],
            "datetime_cols": ["timestamp"],
        },
        "model": {
            "target_column": "is_cancelled",
            "random_state": 42,
            "test_size": 0.2,
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "experiment_name": "test-experiment",
            "model_name": "test-model",
        },
        "thresholds": {"min_accuracy": 0.7},
        "deployment": {
            "api_endpoint": "http://localhost:8000",
            "health_check_timeout": 30,
        },
    }


@pytest.fixture
def sample_raw_data():
    """Sample raw data for testing."""
    return pd.DataFrame(
        {
            "avg_vtat": [10.5, 15.2, 8.7, 20.1, 12.3],
            "timestamp": [
                "2023-01-01T10:00:00",
                "2023-01-01T14:30:00",
                "2023-01-01T09:15:00",
                "2023-01-01T18:45:00",
                "2023-01-01T12:00:00",
            ],
            "is_cancelled": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def sample_processed_data():
    """Sample processed data for testing."""
    return pd.DataFrame(
        {
            "avg_vtat": [10.5, 15.2, 8.7, 20.1, 12.3],
            "timestamp_hour": [10, 14, 9, 18, 12],
            "timestamp_day_of_week": [6, 6, 6, 6, 6],  # Sunday
            "timestamp_month": [1, 1, 1, 1, 1],  # January
            "is_cancelled": [0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def sample_inference_data():
    """Sample data for inference testing."""
    return {"avg_vtat": 15.5, "timestamp": "2023-01-01T12:00:00"}


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_data_file(sample_raw_data):
    """Create a temporary data file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_path = f.name

    sample_raw_data.to_parquet(temp_path, index=False)
    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.set_experiment"),
        patch("mlflow.start_run") as mock_run,
        patch("mlflow.log_params"),
        patch("mlflow.log_metric"),
        patch("mlflow.sklearn.log_model"),
    ):
        # Mock the run context
        mock_run_info = Mock()
        mock_run_info.run_id = "test_run_id"
        mock_run.return_value.__enter__.return_value.info = mock_run_info
        yield mock_run


@pytest.fixture
def mock_model():
    """Mock trained model for testing."""
    model = Mock()
    model.predict.return_value = [0, 1, 0]
    model.predict_proba.return_value = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
    model.coef_ = [[0.5, -0.3, 0.8, -0.2]]
    return model


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def invalid_data():
    """Invalid data for error testing."""
    return pd.DataFrame({"invalid_col": [1, 2, 3]})


@pytest.fixture
def boundary_data():
    """Boundary condition data for testing."""
    return pd.DataFrame(
        {
            "avg_vtat": [0.0, 999.9, -1.0],  # Edge values
            "timestamp": [
                "2023-01-01T00:00:00",  # Start of day
                "2023-12-31T23:59:59",  # End of year
                "invalid_timestamp",  # Invalid format
            ],
            "is_cancelled": [0, 1, 0],
        }
    )
