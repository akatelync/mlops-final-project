"""
Unit tests for model training pipeline.

Tests cover:
- Model training functionality
- MLflow integration
- Metrics calculation
- Configuration handling
- Error scenarios
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.train import load_config, train_model  # ruff: noqa: E402


class TestLoadConfig:
    """Test configuration loading for training module."""

    def test_load_config_success(self, temp_config_file):
        """Test successful config loading."""
        config = load_config(temp_config_file)

        assert isinstance(config, dict)
        assert "mlflow" in config
        assert "model" in config
        assert config["model"]["target_column"] == "is_cancelled"

    def test_load_config_missing_file(self):
        """Test config loading with missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestTrainModel:
    """Test model training functionality."""

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_success(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test successful model training."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training
        result = train_model(config_path=temp_config_file)

        # Assertions
        assert result == "runs:/test_run_id/model"
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.sklearn.log_model.assert_called_once()

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_metrics_logged(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test that training metrics are properly logged."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training
        train_model(config_path=temp_config_file)

        # Check that metrics were logged
        assert mock_mlflow.log_metric.call_count == 4  # accuracy, precision, recall, f1

        # Get the metric names that were logged
        logged_metrics = [call[0][0] for call in mock_mlflow.log_metric.call_args_list]
        expected_metrics = [
            "train_accuracy",
            "train_precision",
            "train_recall",
            "train_f1_score",
        ]

        for metric in expected_metrics:
            assert metric in logged_metrics

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_parameters_logged(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test that training parameters are properly logged."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training
        train_model(config_path=temp_config_file)

        # Check that parameters were logged
        mock_mlflow.log_params.assert_called_once()
        logged_params = mock_mlflow.log_params.call_args[0][0]

        assert "model_type" in logged_params
        assert "random_state" in logged_params
        assert "test_size" in logged_params
        assert logged_params["model_type"] == "LogisticRegression"

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_with_custom_input_path(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test training with custom input path."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        custom_path = "custom/path/data.parquet"

        # Run training
        train_model(input_path=custom_path, config_path=temp_config_file)

        # Check that custom path was used
        mock_read_parquet.assert_called_once_with(custom_path)

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_experiment_creation(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test MLflow experiment creation."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_mlflow.get_experiment_by_name.return_value = (
            None  # Experiment doesn't exist
        )
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training
        train_model(config_path=temp_config_file)

        # Check that experiment was created
        mock_mlflow.create_experiment.assert_called_once_with("test-experiment")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_existing_experiment(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test with existing MLflow experiment."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_experiment = Mock()
        mock_mlflow.get_experiment_by_name.return_value = (
            mock_experiment  # Experiment exists
        )
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training
        train_model(config_path=temp_config_file)

        # Check that experiment was not created but was set
        mock_mlflow.create_experiment.assert_not_called()
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_mlflow_exception_handling(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test handling of MLflow exceptions."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_mlflow.get_experiment_by_name.side_effect = Exception("MLflow error")
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training - should not raise exception
        result = train_model(config_path=temp_config_file)

        # Should still return a valid result
        assert result == "runs:/test_run_id/model"

    @patch("src.models.train.mlflow")
    def test_train_model_missing_data_file(self, mock_mlflow, temp_config_file):
        """Test training with missing data file."""
        # Setup mocks
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training with non-existent file
        with pytest.raises(FileNotFoundError):
            train_model(input_path="nonexistent.parquet", config_path=temp_config_file)

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_empty_dataset(
        self, mock_read_parquet, mock_mlflow, temp_config_file
    ):
        """Test training with empty dataset."""
        # Setup mocks - empty dataframe
        empty_df = pd.DataFrame(columns=["avg_vtat", "is_cancelled"])
        mock_read_parquet.return_value = empty_df
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Should raise an error due to empty dataset
        with pytest.raises(ValueError):
            train_model(config_path=temp_config_file)

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_missing_target_column(
        self, mock_read_parquet, mock_mlflow, temp_config_file
    ):
        """Test training with missing target column."""
        # Setup mocks - dataframe without target column
        df_no_target = pd.DataFrame(
            {"avg_vtat": [10.5, 15.2, 8.7], "timestamp_hour": [10, 14, 9]}
        )
        mock_read_parquet.return_value = df_no_target
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Should raise KeyError for missing target column
        with pytest.raises(KeyError):
            train_model(config_path=temp_config_file)

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_single_class_target(
        self, mock_read_parquet, mock_mlflow, temp_config_file
    ):
        """Test training with single class in target variable."""
        # Setup mocks - all same class
        single_class_df = pd.DataFrame(
            {
                "avg_vtat": [10.5, 15.2, 8.7, 20.1, 12.3],
                "timestamp_hour": [10, 14, 9, 18, 12],
                "timestamp_day_of_week": [6, 6, 6, 6, 6],
                "timestamp_month": [1, 1, 1, 1, 1],
                "is_cancelled": [0, 0, 0, 0, 0],  # All same class
            }
        )
        mock_read_parquet.return_value = single_class_df
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Should handle single class gracefully (though metrics might be affected)
        result = train_model(config_path=temp_config_file)
        assert result == "runs:/test_run_id/model"

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_data_types(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test that model handles different data types correctly."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training
        result = train_model(config_path=temp_config_file)

        # Should complete successfully
        assert result == "runs:/test_run_id/model"

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_random_state_consistency(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test that random state is used consistently."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training
        train_model(config_path=temp_config_file)

        # Check that random_state parameter was logged
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["random_state"] == 42  # From sample_config

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_signature_and_example(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test that model signature and input example are logged."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training
        train_model(config_path=temp_config_file)

        # Check that log_model was called with signature and input_example
        log_model_call = mock_mlflow.sklearn.log_model.call_args
        assert "signature" in log_model_call[1]
        assert "input_example" in log_model_call[1]
        assert log_model_call[1]["artifact_path"] == "model"

    @patch("src.models.train.mlflow")
    @patch("pandas.read_parquet")
    def test_train_model_tags(
        self, mock_read_parquet, mock_mlflow, sample_processed_data, temp_config_file
    ):
        """Test that appropriate tags are set."""
        # Setup mocks
        mock_read_parquet.return_value = sample_processed_data
        mock_run = Mock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        # Run training
        train_model(config_path=temp_config_file)

        # Check that model type tag was set
        mock_mlflow.set_tag.assert_called_once_with(
            "model_type", "LogisticRegressionPipeline"
        )
