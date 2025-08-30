FROM apache/airflow:3.0.3

USER airflow

RUN pip install kaggle scikit-learn

COPY --chown=airflow:airflow src/ /opt/airflow/src/

RUN mkdir -p /opt/airflow/models
