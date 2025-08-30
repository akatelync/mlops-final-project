FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
RUN pip install .

COPY src/serve/ /app/src/serve/
COPY config.yaml /app/config.yaml

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
