FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
COPY config.yaml /app/config.yaml
# Remove this line: COPY data/ /app/data/

RUN pip install .

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db"]
