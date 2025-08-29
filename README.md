# MLOps Final Project: Uber Ride Cancellation Prediction

A comprehensive end-to-end MLOps pipeline for predicting Uber ride cancellations using machine learning, featuring automated training, deployment, monitoring, and drift detection.

## 🎯 Project Overview

This project implements a complete MLOps solution that predicts whether an Uber ride will be cancelled based on various booking features. The system includes:

- **Machine Learning Pipeline**: Automated data ingestion, preprocessing, training, and validation
- **Model Serving**: FastAPI-based REST API for real-time predictions
- **Orchestration**: Apache Airflow DAGs for workflow automation
- **Experiment Tracking**: MLflow for model versioning and artifact management
- **Drift Detection**: Evidently AI for monitoring data and model drift
- **Model Registry**: Automated model promotion based on performance thresholds

## 🏗️ Architecture
```mermaid
graph TD
    A[Data Sources<br>Kaggle] --> B[Airflow DAGs]
    B --> C[MLflow Tracking]
    B --> D[ML Pipeline<br>(Train/Val)]
    B --> E[Drift Detection]
    C --> F[Model Registry]
    D --> G[FastAPI Serving]
    F --> G
    G --> H[Production Model]
```

## 📊 Dataset

The project uses the **Uber Ride Analytics** dataset from Kaggle, containing ride booking information with features such as:

- **Booking Details**: Date, Time, Booking ID, Customer ID
- **Ride Information**: Vehicle Type, Pickup/Drop Locations, Distance
- **Performance Metrics**: VTAT (Vehicle Time to Arrival), CTAT (Customer Trip Time)
- **Ratings**: Driver and Customer ratings
- **Payment**: Payment methods and booking values
- **Target Variable**: `Is_Cancelled` (derived from Booking Status)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/akatelync/mlops-final-project.git
   cd mlops-final-project
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

3. **Set up configuration**
   ```bash
   # Review and modify config.yaml as needed
   cat config.yaml
   ```

4. **Initialize data**
   ```bash
   python src/data/get_data.py
   ```

### Running the Pipeline

#### Option 1: Local Development

1. **Start MLflow server**
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```

2. **Start Airflow**
   ```bash
   airflow standalone
   ```
   Access Airflow UI at: http://localhost:8080

3. **Run training pipeline**
   - Trigger the `training_pipeline` DAG in Airflow UI
   - Monitor progress and logs

4. **Deploy model**
   - Trigger the `deployment_pipeline` DAG
   - This promotes the best model to Production stage

5. **Start API server**
   ```bash
   uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
   ```
   Access API docs at: http://localhost:8000/docs

#### Option 2: Docker Compose (Coming Soon)

```bash
docker-compose up -d
```

## 📁 Project Structure

```
mlops-final-project/
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   ├── get_data.py          # Data acquisition
│   │   ├── ingest.py            # Data ingestion
│   │   └── simulate_drift.py    # Drift simulation
│   ├── features/                 # Feature engineering
│   │   └── transform.py         # Data preprocessing
│   ├── models/                   # Model training & validation
│   │   ├── train.py             # Model training
│   │   └── validate.py          # Model validation
│   ├── serve/                    # Model serving
│   │   └── app.py               # FastAPI application
│   ├── monitoring/               # Monitoring & drift detection
│   │   └── generate_drift.py    # Evidently AI reports
│   └── deployment/               # Model deployment
│       └── promote.py           # Model promotion logic
├── dags/                         # Airflow DAGs
│   ├── training_dag.py          # Training pipeline
│   ├── deployment_dag.py        # Deployment pipeline
│   └── drift_dag.py             # Drift detection pipeline
├── data/                         # Data storage
│   ├── raw/                     # Raw datasets
│   ├── processed/               # Processed datasets
│   └── current/                 # Current/drifted data
├── docs/                         # Documentation
│   ├── data_dictionary.md       # Dataset documentation
│   └── drift_plan.md           # Drift detection strategy
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── results/                      # Model outputs & reports
├── config.yaml                   # Configuration file
└── pyproject.toml               # Project dependencies
```

## 🔄 Workflows

### 1. Training Pipeline (`training_dag.py`)

Automated daily training workflow:

1. **Data Ingestion**: Load and validate raw data
2. **Data Transformation**: Feature engineering and preprocessing
3. **Model Training**: Train XGBoost classifier with hyperparameters from config
4. **Model Validation**: Evaluate model performance and log metrics to MLflow

### 2. Deployment Pipeline (`deployment_dag.py`)

Model promotion workflow:

1. **Model Evaluation**: Check if latest model meets performance thresholds
2. **Model Promotion**: Promote qualifying models to "Production" stage
3. **Registry Update**: Update MLflow Model Registry

### 3. Drift Detection Pipeline (`drift_dag.py`)

Data quality monitoring:

1. **Data Drift Detection**: Compare current data against reference dataset
2. **Target Drift Analysis**: Monitor target variable distribution changes
3. **Report Generation**: Create Evidently AI HTML reports
4. **Artifact Logging**: Store reports in MLflow for tracking

## 🛠️ API Usage

### Model Information
```bash
curl -X GET "http://localhost:8000/model"
```

### Make Predictions
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "BookingID": 12345,
    "CustomerID": 54321,
    "VehicleType": "Go Sedan",
    "PaymentMethod": "Credit Card",
    "Distance": 10.5
  }'
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.23
}
```

## 📊 Monitoring & Observability

### MLflow Tracking

- **Experiments**: Track model training runs and hyperparameters
- **Metrics**: Monitor accuracy, F1-score, precision, recall
- **Artifacts**: Store trained models, confusion matrices, SHAP plots
- **Model Registry**: Manage model versions and stages

Access MLflow UI at: http://localhost:5000

### Drift Detection

The system monitors for:

- **Data Drift**: Changes in feature distributions
- **Target Drift**: Changes in target variable distribution
- **Model Performance**: Degradation in prediction quality

Reports are generated using Evidently AI and stored as MLflow artifacts.

## ⚙️ Configuration

Key configuration options in `config.yaml`:

```yaml
data:
  raw_path: "data/raw/dataset.csv"
  processed_path: "data/processed/train_data.parquet"
  reference_path: "data/processed/train_data.parquet"
  current_path: "data/current/drifted_data.parquet"

model:
  target_column: "Is_Cancelled"
  random_state: 10
  test_size: 0.2
  n_estimators: 100
  max_depth: 10

mlflow:
  tracking_uri: "http://127.0.0.1:5000"
  experiment_name: "uber-ride-prediction"
  model_name: "uber-ride-prediction-model"

thresholds:
  min_accuracy: 0.0
  min_f1_score: 0.0
```

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

## 📈 Model Performance

The XGBoost classifier achieves:

- **Accuracy**: ~91% (based on dataset success rate)
- **Features**: Booking details, vehicle type, distance, ratings, payment method
- **Target**: Binary classification (cancelled vs. completed rides)

Key insights from the dataset:
- **Vehicle Performance**: UberXL has highest success rate (92.2%)
- **Payment Preferences**: UPI dominates (~40% of revenue)
- **Cancellation Patterns**: Balanced between customer and driver cancellations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
ruff format .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Uber Ride Analytics from Kaggle
- **MLOps Stack**: MLflow, Airflow, FastAPI, Evidently AI
- **ML Framework**: Scikit-learn, XGBoost
- **Infrastructure**: Docker, Python 3.11+

## 📞 Support

For questions or issues:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/akatelync/mlops-final-project/issues)
3. Create a new issue with detailed information

---

**Built with ❤️ for MLOps learning and production-ready ML systems**
