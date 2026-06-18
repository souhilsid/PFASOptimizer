FROM maven:3.9.3-eclipse-temurin-17 AS olca_ipc
WORKDIR /olca-ipc
COPY olca-ipc-pom.xml /olca-ipc/pom.xml
RUN mvn package

FROM ghcr.io/greendelta/gdt-server-native AS gdt_native

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    PFAS_LCA_DATA_DIR=/srv/pfas/deploy_data \
    PFAS_ENVIRONMENTAL_MODE=openlca_ipc \
    OPENLCA_IPC_PORT=8080 \
    OPENLCA_FALLBACK_TO_PROXY=true \
    OPENLCA_PRODUCT_SYSTEM="PFAS Adsorption Treatment with Biochar" \
    OPENLCA_IMPACT_METHOD="ReCiPe 2016 Midpoint (H)" \
    OPENLCA_DATA_DIR=/srv/openlca-data \
    OPENLCA_DB_NAME=Biochar

WORKDIR /srv/pfas

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        openjdk-21-jre-headless \
        procps \
        tini \
        unzip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=olca_ipc /olca-ipc/target/lib /srv/olca-ipc/lib
COPY --from=gdt_native /app/native /app/native

COPY requirements-pfas-app.txt /srv/pfas/requirements-pfas-app.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
RUN pip install --no-cache-dir -r /srv/pfas/requirements-pfas-app.txt

COPY generated_outputs/predictor_app /srv/pfas/generated_outputs/predictor_app
COPY generated_outputs/pkl_models /srv/pfas/generated_outputs/pkl_models
COPY generated_outputs/lca_lcc_evaluator.py /srv/pfas/generated_outputs/lca_lcc_evaluator.py
COPY generated_outputs/inverse_design_engine.py /srv/pfas/generated_outputs/inverse_design_engine.py
COPY generated_outputs/run_pso_optimization.py /srv/pfas/generated_outputs/run_pso_optimization.py
COPY generated_outputs/pfas_smiles_lookup.csv /srv/pfas/generated_outputs/pfas_smiles_lookup.csv
COPY deploy_data /srv/pfas/deploy_data
COPY start.sh /srv/start.sh

RUN chmod +x /srv/start.sh \
    && mkdir -p /srv/pfas/generated_outputs/optimization /srv/openlca-data

EXPOSE 7860

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/srv/start.sh"]
