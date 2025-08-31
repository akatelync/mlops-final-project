"""
Unit tests for data transformation and preprocessing functions.

Tests cover:
- Data loading and configuration
- Datetime transformations
- Feature engineering
- Data validation
- Edge cases and error handling
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.transform import load_config, transform_data  # ruff: noqa: E402


class TestLoadConfig:
    """Test configuration loading functionality."""

    def test_load_config_success(self, temp_config_file):
        """Test successful config loading."""
        config = load_config(temp_config_file)

        assert isinstance(config, dict)
        assert "data" in config
        assert "features" in config
        assert "model" in config

    def test_load_config_missing_file(self):
        """Test config loading with missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_load_config_invalid_yaml(self):
        """Test config loading with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)


class TestTransformData:
    """Test data transformation functionality."""

    def test_transform_data_training_mode(
        self, sample_raw_data, temp_config_file, temp_data_file
    ):
        """Test data transformation in training mode."""
        with patch("os.makedirs"), patch("pandas.DataFrame.to_parquet"):
            result = transform_data(
                input_path=temp_data_file,
                config_path=temp_config_file,
                for_inference=False,
            )

            assert isinstance(result, str)
            assert "train_data.parquet" in result

    def test_transform_data_inference_mode(self, sample_raw_data, temp_config_file):
        """Test data transformation in inference mode."""
        result = transform_data(
            input_df=sample_raw_data, config_path=temp_config_file, for_inference=True
        )

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # Check that datetime columns are transformed
        expected_cols = [
            "avg_vtat",
            "timestamp_hour",
            "timestamp_day_of_week",
            "timestamp_month",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_datetime_transformations(self, sample_raw_data, temp_config_file):
        """Test datetime feature engineering."""
        result = transform_data(
            input_df=sample_raw_data, config_path=temp_config_file, for_inference=True
        )

        # Check datetime-derived features
        assert "timestamp_hour" in result.columns
        assert "timestamp_day_of_week" in result.columns
        assert "timestamp_month" in result.columns

        # Check value ranges
        assert result["timestamp_hour"].min() >= 0
        assert result["timestamp_hour"].max() <= 23
        assert result["timestamp_day_of_week"].min() >= 0
        assert result["timestamp_day_of_week"].max() <= 6
        assert result["timestamp_month"].min() >= 1
        assert result["timestamp_month"].max() <= 12

    def test_missing_features_config(self, sample_raw_data):
        """Test handling of missing features configuration."""
        config_without_features = {
            "data": {"processed_path": "test.parquet"},
            "model": {"target_column": "is_cancelled"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_without_features, f)
            temp_path = f.name

        try:
            with pytest.raises(KeyError, match="'features' key missing from config"):
                transform_data(
                    input_df=sample_raw_data, config_path=temp_path, for_inference=True
                )
        finally:
            os.unlink(temp_path)

    def test_empty_dataframe(self, temp_config_file, empty_dataframe):
        """Test transformation with empty DataFrame."""
        result = transform_data(
            input_df=empty_dataframe, config_path=temp_config_file, for_inference=True
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_datetime_columns(self, temp_config_file):
        """Test transformation when datetime columns are missing."""
        data_without_timestamp = pd.DataFrame(
            {"avg_vtat": [10.5, 15.2, 8.7], "is_cancelled": [0, 1, 0]}
        )

        result = transform_data(
            input_df=data_without_timestamp,
            config_path=temp_config_file,
            for_inference=True,
        )

        # Should still work, just without datetime features
        assert isinstance(result, pd.DataFrame)
        assert "avg_vtat" in result.columns

    def test_invalid_datetime_format(self, temp_config_file):
        """Test handling of invalid datetime formats."""
        data_with_invalid_timestamp = pd.DataFrame(
            {
                "avg_vtat": [10.5, 15.2, 8.7],
                "timestamp": ["invalid", "2023-01-01T10:00:00", "also_invalid"],
                "is_cancelled": [0, 1, 0],
            }
        )

        # Should handle invalid timestamps gracefully
        result = transform_data(
            input_df=data_with_invalid_timestamp,
            config_path=temp_config_file,
            for_inference=True,
        )

        assert isinstance(result, pd.DataFrame)

    def test_numerical_feature_preservation(self, sample_raw_data, temp_config_file):
        """Test that numerical features are preserved correctly."""
        result = transform_data(
            input_df=sample_raw_data, config_path=temp_config_file, for_inference=True
        )

        # Check that avg_vtat values are preserved
        original_values = sample_raw_data["avg_vtat"].values
        result_values = result["avg_vtat"].values

        assert len(original_values) == len(result_values)
        for orig, res in zip(original_values, result_values):
            assert orig == res

    def test_feature_column_filtering(self, temp_config_file):
        """Test that only specified feature columns are included."""
        data_with_extra_cols = pd.DataFrame(
            {
                "avg_vtat": [10.5, 15.2, 8.7],
                "timestamp": [
                    "2023-01-01T10:00:00",
                    "2023-01-01T14:30:00",
                    "2023-01-01T09:15:00",
                ],
                "extra_col": [1, 2, 3],  # This should be filtered out
                "is_cancelled": [0, 1, 0],
            }
        )

        result = transform_data(
            input_df=data_with_extra_cols,
            config_path=temp_config_file,
            for_inference=True,
        )

        # Should not contain extra_col
        assert "extra_col" not in result.columns
        # Should contain expected feature columns
        expected_cols = [
            "avg_vtat",
            "timestamp_hour",
            "timestamp_day_of_week",
            "timestamp_month",
        ]
        for col in expected_cols:
            assert col in result.columns

    def test_boundary_values(self, temp_config_file):
        """Test transformation with boundary values."""
        boundary_data = pd.DataFrame(
            {
                "avg_vtat": [0.0, 999.9, -1.0],
                "timestamp": [
                    "2023-01-01T00:00:00",  # Start of day
                    "2023-12-31T23:59:59",  # End of year
                    "2023-06-15T12:30:45",  # Mid-year
                ],
                "is_cancelled": [0, 1, 0],
            }
        )

        result = transform_data(
            input_df=boundary_data, config_path=temp_config_file, for_inference=True
        )

        # Check that boundary values are handled correctly
        assert result["avg_vtat"].min() == -1.0
        assert result["avg_vtat"].max() == 999.9

        # Check datetime boundary transformations
        assert 0 in result["timestamp_hour"].values  # Midnight
        assert 23 in result["timestamp_hour"].values  # 11 PM
        assert 1 in result["timestamp_month"].values  # January
        assert 12 in result["timestamp_month"].values  # December

    def test_data_types_after_transformation(self, sample_raw_data, temp_config_file):
        """Test that data types are appropriate after transformation."""
        result = transform_data(
            input_df=sample_raw_data, config_path=temp_config_file, for_inference=True
        )

        # Check data types
        assert pd.api.types.is_numeric_dtype(result["avg_vtat"])
        assert pd.api.types.is_integer_dtype(result["timestamp_hour"])
        assert pd.api.types.is_integer_dtype(result["timestamp_day_of_week"])
        assert pd.api.types.is_integer_dtype(result["timestamp_month"])

    def test_no_data_leakage(self, sample_raw_data, temp_config_file):
        """Test that target column is not included in inference mode."""
        result = transform_data(
            input_df=sample_raw_data, config_path=temp_config_file, for_inference=True
        )

        # Target column should not be in the result for inference
        assert "is_cancelled" not in result.columns
