FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PFAS_LCA_DATA_DIR=/app/deploy_data \
    OPENLCA_EVALUATOR_MODE=proxy \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-openlca-evaluator.txt /app/requirements-openlca-evaluator.txt
RUN pip install --no-cache-dir -r /app/requirements-openlca-evaluator.txt

COPY generated_outputs/lca_lcc_evaluator.py /app/generated_outputs/lca_lcc_evaluator.py
COPY generated_outputs/openlca_evaluator_service /app/generated_outputs/openlca_evaluator_service
COPY deploy_data /app/deploy_data

EXPOSE 8000

CMD ["sh", "-c", "uvicorn generated_outputs.openlca_evaluator_service.service:app --host 0.0.0.0 --port ${PORT:-8000}"]
