FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
COPY config.yaml /app/config.yaml

RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
