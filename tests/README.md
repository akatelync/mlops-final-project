# MLOps Project Test Suite

This directory contains comprehensive unit tests for the MLOps Final Project, covering data preprocessing, model training, and API functionality.

## Test Structure

```
tests/
├── __init__.py          # Test package initialization
├── conftest.py          # Shared fixtures and test utilities
├── test_transform.py    # Data preprocessing/feature engineering tests
├── test_train.py        # Model training pipeline tests
├── test_api.py          # FastAPI endpoint tests
├── run_tests.py         # Simple test runner (no pytest required)
└── README.md           # This file
```

## Test Coverage

### Data Preprocessing Tests (`test_transform.py`)
- **Missing value handling**: Tests for handling missing/invalid data
- **Data type conversions**: Validates proper data type transformations
- **Feature engineering**: Tests datetime feature extraction (hour, day_of_week, month)
- **Output validation**: Ensures transformed features are within expected ranges
- **Edge cases**: Empty DataFrames, missing columns, invalid timestamps
- **Configuration handling**: Tests config loading and error scenarios

### Model Training Tests (`test_train.py`)
- **Training pipeline**: End-to-end training process validation
- **MLflow integration**: Logging of parameters, metrics, and models
- **Metrics calculation**: Accuracy, precision, recall, F1-score validation
- **Configuration handling**: Parameter loading and validation
- **Error scenarios**: Missing data, invalid configurations, MLflow failures
- **Data validation**: Empty datasets, missing target columns, single-class scenarios

### API Tests (`test_api.py`)
- **Endpoint functionality**: All API endpoints (`/predict`, `/model`, `/health`, `/`)
- **Input validation**: Pydantic model validation and error handling
- **Response formats**: Correct JSON response structure validation
- **Error handling**: Invalid inputs, model loading failures, preprocessing errors
- **Edge cases**: Boundary values, extreme inputs, missing model components

## Running Tests

### Option 1: Using pytest (Recommended)

First, install pytest:
```bash
pip install pytest
```

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test files:
```bash
python -m pytest tests/test_transform.py -v
python -m pytest tests/test_train.py -v
python -m pytest tests/test_api.py -v
```

Run tests with coverage:
```bash
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
```

### Option 2: Using the Simple Test Runner

If pytest is not available, use the included test runner:
```bash
python tests/run_tests.py
```

This will run basic import and functionality tests without requiring pytest.

## Test Fixtures

The `conftest.py` file provides shared fixtures used across all tests:

- `sample_config`: Sample configuration for testing
- `sample_raw_data`: Sample raw data DataFrame
- `sample_processed_data`: Sample processed data DataFrame
- `sample_inference_data`: Sample data for inference testing
- `temp_config_file`: Temporary config file for testing
- `temp_data_file`: Temporary data file for testing
- `mock_mlflow`: MLflow mocking for training tests
- `mock_model`: Mock trained model for API tests
- `empty_dataframe`: Empty DataFrame for edge case testing
- `invalid_data`: Invalid data for error testing
- `boundary_data`: Boundary condition data for testing

## Key Test Scenarios

### Data Preprocessing
- ✅ Valid data transformation
- ✅ Datetime feature engineering
- ✅ Missing value handling
- ✅ Invalid timestamp formats
- ✅ Empty DataFrames
- ✅ Configuration errors
- ✅ Feature column filtering
- ✅ Boundary value handling

### Model Training
- ✅ Successful training pipeline
- ✅ MLflow experiment management
- ✅ Parameter and metric logging
- ✅ Model artifact saving
- ✅ Error handling (missing data, config errors)
- ✅ Edge cases (empty datasets, single-class targets)

### API Endpoints
- ✅ `/predict` - Valid and invalid predictions
- ✅ `/model` - Model metadata retrieval
- ✅ `/health` - Health check functionality
- ✅ `/` - Root endpoint information
- ✅ Input validation and error responses
- ✅ Model loading error handling
- ✅ Preprocessing error handling

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. They use mocking to avoid dependencies on external services like MLflow servers or trained models.

## Adding New Tests

When adding new functionality:

1. Add corresponding tests to the appropriate test file
2. Use existing fixtures from `conftest.py` when possible
3. Follow the existing test naming convention: `test_<functionality>_<scenario>`
4. Include both positive and negative test cases
5. Test edge cases and error conditions
6. Update this README if adding new test categories

## Dependencies

The tests require the following packages (already included in project dependencies):
- pandas
- scikit-learn
- fastapi
- pydantic
- unittest.mock (built-in)

Optional for enhanced testing:
- pytest
- pytest-cov (for coverage reports)
