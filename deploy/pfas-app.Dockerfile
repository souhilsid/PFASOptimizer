FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PFAS_LCA_DATA_DIR=/app/deploy_data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-pfas-app.txt /app/requirements-pfas-app.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
RUN pip install --no-cache-dir -r /app/requirements-pfas-app.txt

COPY generated_outputs/predictor_app /app/generated_outputs/predictor_app
COPY generated_outputs/pkl_models /app/generated_outputs/pkl_models
COPY generated_outputs/lca_lcc_evaluator.py /app/generated_outputs/lca_lcc_evaluator.py
COPY generated_outputs/inverse_design_engine.py /app/generated_outputs/inverse_design_engine.py
COPY generated_outputs/run_pso_optimization.py /app/generated_outputs/run_pso_optimization.py
COPY generated_outputs/pfas_smiles_lookup.csv /app/generated_outputs/pfas_smiles_lookup.csv
COPY deploy_data /app/deploy_data

RUN mkdir -p /app/generated_outputs/optimization

EXPOSE 8057

CMD ["python", "generated_outputs/predictor_app/app.py"]
