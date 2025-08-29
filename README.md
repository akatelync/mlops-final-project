# Uber Data Analytics


I'll help you break down this MLOps project into a logical sequence of tasks. This is a comprehensive end-to-end ML system that involves multiple components working together. Let me organize the requirements in the order you should tackle them:

## Phase 1: Project Setup & Planning
**Requirement 1: Choose Your ML Problem and Dataset**
- Select a supervised ML problem (classification or regression)
- Keep it simple to focus on MLOps, not model complexity
- Find/create a dataset with 500+ rows (preferably 1000+)
- Ensure the dataset can simulate drift for later requirements
- Document your choice in README.md

**Requirement 2: Set Up Project Structure**
Create this directory structure:
```
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── serve/
│   ├── monitoring/
│   └── deployment/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── reference/
│   └── current/
├── dags/
├── tests/
├── docs/
├── docker/
├── .github/workflows/
├── notebooks/
├── config.yaml
└── pyproject.toml
```

**Requirement 3: Create Data Acquisition Script**
- Write `src/data/get_data.py` to fetch/generate your data
- Save to `data/raw/dataset.csv` or `.parquet`
- Create data dictionary in `docs/data_dictionary.md`
- Document drift simulation plan in `docs/drift_plan.md`

## Phase 2: Core ML Pipeline Development
**Requirement 4: Data Processing Pipeline**
- Create `src/data/ingest.py` for data loading
- Create `src/features/transform.py` for preprocessing/feature engineering
- Create `src/models/train.py` for model training
- Create `src/models/validate.py` for model evaluation
- Use config.yaml for all parameters (hyperparameters, file paths, etc.)

**Requirement 5: Set Up MLflow Tracking**
- Set up MLflow server (initially local, later in Docker)
- Implement logging in your training pipeline:
  - Parameters (hyperparameters)
  - Metrics (accuracy, RMSE, etc.)
  - Artifacts (trained model, SHAP plots)
- Test that everything logs correctly

## Phase 3: Orchestration with Airflow
**Requirement 6: Create Training DAG**
- Write `dags/training_dag.py` with tasks:
  - Data Ingestion
  - Data Transformation
  - Model Training
  - Model Validation
- Use PythonOperator for each task
- Test the DAG runs successfully

**Requirement 7: Create Deployment DAG**
- Write `dags/deployment_dag.py`
- Implement model promotion logic in `src/deployment/promote.py`
- Check metrics against thresholds
- Register "champion" model in MLflow Model Registry

**Requirement 8: Create Drift Detection DAG**
- Write `dags/drift_dag.py`
- Create `src/monitoring/generate_drift.py` using Evidently AI
- Generate data drift and target drift reports
- Log reports as MLflow artifacts

## Phase 4: Model Serving API
**Requirement 9: Build FastAPI Application**
- Create `src/serve/app.py` with endpoints:
  - `/predict` - serve predictions
  - `/model` - return model info
- Load "champion" model from MLflow
- Use Pydantic for input validation
- Implement proper error handling

**Requirement 10: Test API Locally**
- Test both endpoints work correctly
- Ensure API documentation is available at `/docs`
- Verify model loading and prediction pipeline

## Phase 5: Containerization
**Requirement 11: Create Dockerfiles**
- `docker/airflow.Dockerfile`
- `docker/mlflow.Dockerfile`
- `docker/fastapi.Dockerfile`
- Each with proper base images and dependencies

**Requirement 12: Docker Compose Setup**
- Create `docker-compose.yml` orchestrating all services
- Configure networking between services
- Set up volumes for data persistence
- Test entire system runs with `docker-compose up`

## Phase 6: Drift Detection Implementation
**Requirement 13: Implement Evidently AI Integration**
- Create reference dataset (`data/reference.parquet`)
- Create drift simulation script (`src/data/simulate_drift.py`)
- Generate HTML drift reports
- Integrate with Airflow DAG and MLflow logging

## Phase 7: Testing & CI/CD
**Requirement 14: Write Unit Tests**
- Create tests in `tests/` directory using pytest
- Focus on data processing and feature engineering functions
- Test edge cases and error handling
- Use pytest fixtures for test data

**Requirement 15: Set Up GitHub Actions**
- Create `.github/workflows/ci.yml`
- Automate test running on push/PR
- Set up pre-commit hooks with Ruff for code quality

## Phase 8: Documentation & Demo
**Requirement 16: Create System Architecture Documentation**
- Create `docs/architecture.md` and `docs/architecture.png`
- Document how all components interact
- Explain the end-to-end flow

**Requirement 17: Build Jupyter Notebook Demo**
- Create `notebooks/demo.ipynb`
- Demonstrate system usage (happy path)
- Show drift detection reports
- Include clear explanations and setup instructions

**Requirement 18: Final Documentation**
- Complete README.md with setup instructions
- Document port mappings and access URLs
- Ensure all components are documented

Would you like me to elaborate on any of these requirements or help you get started with a specific phase? I recommend starting with Phase 1 (project setup) and working through them sequentially, as later phases depend on earlier ones.
