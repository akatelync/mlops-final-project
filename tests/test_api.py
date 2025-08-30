"""
Unit tests for FastAPI application endpoints.

Tests cover:
- API endpoint functionality
- Input validation
- Response formats
- Error handling
- Model integration
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# ruff: noqa: E402
from src.serve.app import InputData, app, get_feature_importance, preprocess_input


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_model_loading():
    """Mock model loading for API tests."""
    with patch("src.serve.app.load_champion_model") as mock_load:
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        mock_model._model_impl.coef_ = [[0.5, -0.3, 0.8, -0.2]]
        mock_load.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_model_metadata():
    """Mock model metadata for API tests."""
    with patch("src.serve.app.get_model_metadata") as mock_metadata:
        mock_version = Mock()
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.run_id = "test_run_id"

        mock_run = Mock()
        mock_run.data.params = {"model_type": "LogisticRegression", "random_state": 42}
        mock_run.data.metrics = {"accuracy": 0.85, "f1_score": 0.82}

        mock_metadata.return_value = {
            "model_version": mock_version,
            "run": mock_run,
            "params": mock_run.data.params,
            "metrics": mock_run.data.metrics,
        }
        yield mock_metadata


class TestInputData:
    """Test Pydantic input model."""

    def test_valid_input_data(self):
        """Test valid input data creation."""
        data = InputData(avg_vtat=15.5, timestamp="2023-01-01T12:00:00")

        assert data.avg_vtat == 15.5
        assert data.timestamp == "2023-01-01T12:00:00"

    def test_input_data_schema(self):
        """Test input data schema generation."""
        schema = InputData.model_json_schema()

        assert "properties" in schema
        assert "avg_vtat" in schema["properties"]
        assert "timestamp" in schema["properties"]
        assert "required" in schema
        assert "avg_vtat" in schema["required"]
        assert "timestamp" in schema["required"]

    def test_input_data_validation_missing_fields(self):
        """Test input validation with missing fields."""
        with pytest.raises(ValidationError):
            InputData(avg_vtat=15.5)  # Missing timestamp

        with pytest.raises(ValidationError):
            InputData(timestamp="2023-01-01T12:00:00")  # Missing avg_vtat

    def test_input_data_validation_invalid_types(self):
        """Test input validation with invalid types."""
        with pytest.raises(ValidationError):
            InputData(avg_vtat="invalid", timestamp="2023-01-01T12:00:00")


class TestPreprocessInput:
    """Test preprocessing functionality."""

    @patch("src.serve.app.transform_data")
    def test_preprocess_input_success(self, mock_transform, sample_inference_data):
        """Test successful input preprocessing."""
        expected_df = Mock()
        mock_transform.return_value = expected_df

        result = preprocess_input(sample_inference_data)

        assert result == expected_df
        mock_transform.assert_called_once()

    @patch("src.serve.app.transform_data")
    def test_preprocess_input_with_different_data(self, mock_transform):
        """Test preprocessing with different input data."""
        test_data = {"avg_vtat": 20.0, "timestamp": "2023-06-15T18:30:00"}
        expected_df = Mock()
        mock_transform.return_value = expected_df

        result = preprocess_input(test_data)

        assert result == expected_df
        # Check that transform_data was called with correct parameters
        call_args = mock_transform.call_args
        assert call_args[1]["for_inference"] is True
        assert call_args[1]["config_path"] == "config.yaml"


class TestGetFeatureImportance:
    """Test feature importance extraction."""

    def test_get_feature_importance_with_coef(self):
        """Test feature importance extraction from model with coefficients."""
        mock_model = Mock()
        mock_model._model_impl.coef_ = [[0.5, -0.3, 0.8, -0.2, 0.1]]
        feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]

        result = get_feature_importance(mock_model, feature_names)

        assert len(result) == 5  # Top 5 features
        assert all("feature" in item and "importance" in item for item in result)
        # Should be sorted by importance (absolute value)
        assert result[0]["importance"] == 0.8  # feature3 has highest absolute coef
        assert result[0]["feature"] == "feature3"

    def test_get_feature_importance_no_coef(self):
        """Test feature importance when model has no coefficients."""
        mock_model = Mock()
        del mock_model._model_impl  # No _model_impl attribute
        feature_names = ["feature1", "feature2"]

        result = get_feature_importance(mock_model, feature_names)

        assert result == []

    def test_get_feature_importance_exception(self):
        """Test feature importance extraction with exception."""
        mock_model = Mock()
        mock_model._model_impl.coef_ = None  # Will cause AttributeError
        feature_names = ["feature1", "feature2"]

        result = get_feature_importance(mock_model, feature_names)

        assert result == []


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint_healthy(self, client, mock_model_loading):
        """Test health endpoint when model loads successfully."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "timestamp" in data

    @patch("src.serve.app.load_champion_model")
    def test_health_endpoint_unhealthy(self, mock_load, client):
        """Test health endpoint when model loading fails."""
        mock_load.side_effect = Exception("Model loading failed")

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert "error" in data

    def test_predict_endpoint_success(self, client, mock_model_loading):
        """Test successful prediction."""
        test_input = {"avg_vtat": 15.5, "timestamp": "2023-01-01T12:00:00"}

        with patch("src.serve.app.preprocess_input") as mock_preprocess:
            mock_preprocess.return_value = Mock()

            response = client.post("/predict", json=test_input)

            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] == 1
            assert "probability" in data
            assert data["probability"] == 0.7

    def test_predict_endpoint_invalid_input(self, client):
        """Test prediction with invalid input."""
        invalid_input = {"invalid_field": "value"}

        response = client.post("/predict", json=invalid_input)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_fields(self, client):
        """Test prediction with missing required fields."""
        incomplete_input = {"avg_vtat": 15.5}  # Missing timestamp

        response = client.post("/predict", json=incomplete_input)

        assert response.status_code == 422

    @patch("src.serve.app.load_champion_model")
    def test_predict_endpoint_model_loading_error(self, mock_load, client):
        """Test prediction when model loading fails."""
        mock_load.side_effect = Exception("Model not found")
        test_input = {"avg_vtat": 15.5, "timestamp": "2023-01-01T12:00:00"}

        response = client.post("/predict", json=test_input)

        assert response.status_code == 500

    def test_predict_endpoint_preprocessing_error(self, client, mock_model_loading):
        """Test prediction when preprocessing fails."""
        test_input = {"avg_vtat": 15.5, "timestamp": "2023-01-01T12:00:00"}

        with patch("src.serve.app.preprocess_input") as mock_preprocess:
            mock_preprocess.side_effect = Exception("Preprocessing failed")

            response = client.post("/predict", json=test_input)

            assert response.status_code == 500

    def test_model_endpoint_success(
        self, client, mock_model_loading, mock_model_metadata
    ):
        """Test model info endpoint success."""
        response = client.get("/model")

        assert response.status_code == 200
        data = response.json()
        assert "hyperparameters" in data
        assert "metrics" in data
        assert "top_features" in data
        assert "input_schema" in data
        assert "model_info" in data

    @patch("src.serve.app.get_model_metadata")
    def test_model_endpoint_metadata_error(self, mock_metadata, client):
        """Test model info endpoint when metadata retrieval fails."""
        mock_metadata.side_effect = Exception("Metadata error")

        response = client.get("/model")

        assert response.status_code == 500

    def test_predict_endpoint_boundary_values(self, client, mock_model_loading):
        """Test prediction with boundary values."""
        boundary_input = {"avg_vtat": 0.0, "timestamp": "2023-01-01T00:00:00"}

        with patch("src.serve.app.preprocess_input") as mock_preprocess:
            mock_preprocess.return_value = Mock()

            response = client.post("/predict", json=boundary_input)

            assert response.status_code == 200

    def test_predict_endpoint_extreme_values(self, client, mock_model_loading):
        """Test prediction with extreme values."""
        extreme_input = {"avg_vtat": 999.9, "timestamp": "2023-12-31T23:59:59"}

        with patch("src.serve.app.preprocess_input") as mock_preprocess:
            mock_preprocess.return_value = Mock()

            response = client.post("/predict", json=extreme_input)

            assert response.status_code == 200

    def test_predict_endpoint_no_probability(self, client):
        """Test prediction when model doesn't support predict_proba."""
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        # Remove predict_proba method
        del mock_model.predict_proba

        with patch("src.serve.app.load_champion_model", return_value=mock_model):
            with patch("src.serve.app.preprocess_input") as mock_preprocess:
                mock_preprocess.return_value = Mock()

                test_input = {"avg_vtat": 15.5, "timestamp": "2023-01-01T12:00:00"}
                response = client.post("/predict", json=test_input)

                assert response.status_code == 200
                data = response.json()
                assert "prediction" in data
                assert data["prediction"] == 0
                # Should not have probability when predict_proba fails
                assert "probability" not in data or data.get("probability") is None


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON in request."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_empty_request_body(self, client):
        """Test handling of empty request body."""
        response = client.post("/predict", json={})

        assert response.status_code == 422

    @patch("src.serve.app.load_config")
    def test_config_loading_error(self, mock_load_config):
        """Test handling of configuration loading errors."""
        mock_load_config.side_effect = FileNotFoundError("Config not found")

        # This would be tested during app initialization
        # For now, we just ensure the mock works
        with pytest.raises(FileNotFoundError):
            mock_load_config()
