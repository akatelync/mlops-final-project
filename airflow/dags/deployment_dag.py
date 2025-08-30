import os
import sys
from datetime import datetime, timedelta

from airflow.providers.standard.operators.python import PythonOperator

from airflow import DAG

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 31),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "deployment_pipeline",
    default_args=default_args,
    description="Model Deployment Pipeline DAG",
    schedule=None,
    catchup=False,
    max_active_runs=1,
)


def run_model_promotion(**context):
    """Run model promotion logic."""
    from src.deployment.promote import promote_model

    result = promote_model()

    if result.get("error"):
        raise Exception(f"Model promotion failed: {result['error']}")

    if result["promoted"]:
        print("Model successfully promoted to Production!")
        if "model_name" in result:
            print(
                f"Model: {result['model_name']} v{result.get('model_version', 'unknown')}"
            )
    else:
        print("Model promotion failed - thresholds not met")
        print(
            f"Current metrics: accuracy={result.get('current_accuracy', 'N/A')}, f1_score={result.get('current_f1_score', 'N/A')}"
        )
        print(
            f"Required: accuracy>={result.get('min_accuracy', 'N/A')}, f1_score>={result.get('min_f1_score', 'N/A')}"
        )

    return result


def deploy_model(**context):
    """Deploy the promoted model."""
    ti = context["ti"]
    promotion_result = ti.xcom_pull(task_ids="model_promotion")

    print(f"Promotion result: {promotion_result}")

    if not promotion_result:
        print("No promotion result found")
        return {"deployment_status": "failed", "reason": "no_promotion_result"}

    if promotion_result.get("promoted", False):
        print("Deploying model to production environment...")

        # Check if model was successfully registered
        if "model_name" in promotion_result:
            print(
                f"Model: {promotion_result['model_name']} v{promotion_result.get('model_version', 'unknown')}"
            )
        else:
            print(f"Model from run: {promotion_result['run_id']}")

        # Signal FastAPI to reload the new champion model
        print("Triggering FastAPI to reload champion model...")
        print(
            "FastAPI will automatically load the latest Production model on next request"
        )

        print("Model deployment completed successfully!")
        return {
            "deployment_status": "success",
            "model_info": {
                "run_id": promotion_result["run_id"],
                "accuracy": promotion_result["current_accuracy"],
                "f1_score": promotion_result["current_f1_score"],
            },
        }
    else:
        reason = "thresholds_not_met"
        if "error" in promotion_result:
            reason = f"promotion_error: {promotion_result['error']}"

        print(f"Skipping deployment - {reason}")
        return {"deployment_status": "skipped", "reason": reason}


# wait_for_training = ExternalTaskSensor(
#     task_id="wait_for_training_completion",
#     external_dag_id="training_pipeline",
#     external_task_id=None,  # Wait for entire DAG completion
#     allowed_states=["success"],  # Add this
#     failed_states=["failed", "upstream_failed", "skipped"],  # Add this
#     timeout=3600,
#     poke_interval=60,
#     mode="reschedule",
#     dag=dag,
# )

# Model promotion task
model_promotion_task = PythonOperator(
    task_id="model_promotion",
    python_callable=run_model_promotion,
    dag=dag,
)

# Model deployment task
model_deployment_task = PythonOperator(
    task_id="model_deployment",
    python_callable=deploy_model,
    dag=dag,
)

# Define task dependencies
# wait_for_training >> model_promotion_task >> model_deployment_task
model_promotion_task >> model_deployment_task
