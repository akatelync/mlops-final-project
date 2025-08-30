FROM apache/airflow:3.0.3

USER airflow

WORKDIR /opt/airflow

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY --chown=airflow:airflow pyproject.toml /opt/airflow/
COPY --chown=airflow:airflow src/ /opt/airflow/src/
COPY --chown=airflow:airflow config.yaml /opt/airflow/config.yaml

RUN cat /opt/airflow/config.yaml

RUN uv pip install .

RUN mkdir -p /opt/airflow/models
