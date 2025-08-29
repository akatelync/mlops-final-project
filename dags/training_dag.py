import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 29),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "training_pipeline",
    default_args=default_args,
    description="ML Training Pipeline DAG",
    schedule=timedelta(days=1),
    catchup=False,
    max_active_runs=1,
)


def run_data_ingestion(**context):
    """Airflow task wrapper for data ingestion."""
    from src.data.ingest import ingest_data

    result = ingest_data()
    return result


def run_data_transformation(**context):
    """Airflow task wrapper for data transformation."""
    from src.features.transform import transform_data

    ti = context["ti"]
    ingested_path = ti.xcom_pull(task_ids="data_ingestion")
    result = transform_data(ingested_path)
    return result


def run_model_training(**context):
    """Airflow task wrapper for model training."""
    from src.models.train import train_model

    ti = context["ti"]
    train_path = ti.xcom_pull(task_ids="data_transformation")
    result = train_model(train_path)
    return result


def run_model_validation(**context):
    """Airflow task wrapper for model validation."""
    from src.models.validate import validate_model

    ti = context["ti"]
    model_path = ti.xcom_pull(task_ids="model_training")
    result = validate_model(model_path)
    return result


data_ingestion_task = PythonOperator(
    task_id="data_ingestion",
    python_callable=run_data_ingestion,
    dag=dag,
)

data_transformation_task = PythonOperator(
    task_id="data_transformation",
    python_callable=run_data_transformation,
    dag=dag,
)

model_training_task = PythonOperator(
    task_id="model_training",
    python_callable=run_model_training,
    dag=dag,
)

model_validation_task = PythonOperator(
    task_id="model_validation",
    python_callable=run_model_validation,
    dag=dag,
)

(
    data_ingestion_task
    >> data_transformation_task
    >> model_training_task
    >> model_validation_task
)
