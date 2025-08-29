import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# Add project root to path
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
    "drift_monitoring_pipeline",
    default_args=default_args,
    description="Data Drift Monitoring Pipeline",
    schedule=timedelta(hours=6),
    catchup=False,
    max_active_runs=1,
)


def simulate_drift(**context):
    from src.data.simulate_drift import simulate_data_drift

    return simulate_data_drift()


def detect_and_log_drift(**context):
    from src.monitoring.generate_drift import log_drift_reports_to_mlflow

    ti = context["ti"]
    drifted_path = ti.xcom_pull(task_ids="simulate_drift")
    return log_drift_reports_to_mlflow(current_path=drifted_path)


def alert_on_drift(**context):
    ti = context["ti"]
    drift_results = ti.xcom_pull(task_ids="detect_and_log_drift")
    if drift_results["data_drift_detected"]:
        print("ALERT: Data drift detected!")
    else:
        print("No drift detected - system is stable.")
    return drift_results


simulate_task = PythonOperator(
    task_id="simulate_drift",
    python_callable=simulate_drift,
    dag=dag,
)

detect_task = PythonOperator(
    task_id="detect_and_log_drift",
    python_callable=detect_and_log_drift,
    dag=dag,
)

alert_task = PythonOperator(
    task_id="alert_on_drift",
    python_callable=alert_on_drift,
    dag=dag,
)

simulate_task >> detect_task >> alert_task
