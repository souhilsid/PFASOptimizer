#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"
OPENLCA_DATA_DIR="${OPENLCA_DATA_DIR:-/srv/openlca-data}"
OPENLCA_DB_NAME="${OPENLCA_DB_NAME:-Biochar}"
OPENLCA_IPC_URL="${OPENLCA_IPC_URL:-http://127.0.0.1:8080}"
OPENLCA_WAIT_SECONDS="${OPENLCA_WAIT_SECONDS:-90}"

mkdir -p "${OPENLCA_DATA_DIR}/databases" /tmp/pfas-runtime

database_ready() {
  [ -d "${OPENLCA_DATA_DIR}/databases/${OPENLCA_DB_NAME}" ]
}

download_database_zip() {
  local target_zip="/tmp/pfas-runtime/openlca-data.zip"
  if [ -f "/srv/pfas/openlca-data-${OPENLCA_DB_NAME}.zip" ]; then
    echo "Using openLCA database ZIP bundled in the Space repository."
    cp "/srv/pfas/openlca-data-${OPENLCA_DB_NAME}.zip" "${target_zip}"
  elif [ -n "${OPENLCA_DB_ZIP_URL:-}" ]; then
    echo "Downloading openLCA database from OPENLCA_DB_ZIP_URL."
    if [ -n "${OPENLCA_DB_BEARER_TOKEN:-}" ]; then
      curl -fL --retry 3 -H "Authorization: Bearer ${OPENLCA_DB_BEARER_TOKEN}" "${OPENLCA_DB_ZIP_URL}" -o "${target_zip}"
    elif [ -n "${HF_TOKEN:-}" ]; then
      curl -fL --retry 3 -H "Authorization: Bearer ${HF_TOKEN}" "${OPENLCA_DB_ZIP_URL}" -o "${target_zip}"
    else
      curl -fL --retry 3 "${OPENLCA_DB_ZIP_URL}" -o "${target_zip}"
    fi
  elif [ -n "${HF_OPENLCA_DATASET_REPO:-}" ] && [ -n "${HF_OPENLCA_DATASET_FILE:-}" ]; then
    local hf_url="https://huggingface.co/datasets/${HF_OPENLCA_DATASET_REPO}/resolve/main/${HF_OPENLCA_DATASET_FILE}?download=true"
    echo "Downloading openLCA database from Hugging Face dataset ${HF_OPENLCA_DATASET_REPO}."
    if [ -n "${HF_TOKEN:-}" ]; then
      curl -fL --retry 3 -H "Authorization: Bearer ${HF_TOKEN}" "${hf_url}" -o "${target_zip}"
    else
      curl -fL --retry 3 "${hf_url}" -o "${target_zip}"
    fi
  else
    return 1
  fi

  echo "Extracting openLCA database into ${OPENLCA_DATA_DIR}."
  unzip -q -o "${target_zip}" -d "${OPENLCA_DATA_DIR}"

  if [ ! -d "${OPENLCA_DATA_DIR}/databases/${OPENLCA_DB_NAME}" ] && [ -d "${OPENLCA_DATA_DIR}/${OPENLCA_DB_NAME}" ]; then
    mkdir -p "${OPENLCA_DATA_DIR}/databases"
    mv "${OPENLCA_DATA_DIR}/${OPENLCA_DB_NAME}" "${OPENLCA_DATA_DIR}/databases/${OPENLCA_DB_NAME}"
  fi
}

start_openlca() {
  echo "Starting openLCA IPC server for database ${OPENLCA_DB_NAME}."
  java -Xmx4096M -cp "/srv/olca-ipc/lib/*" org.openlca.ipc.Server \
    -timeout 30 \
    -native /app/native \
    -data "${OPENLCA_DATA_DIR}" \
    -db "${OPENLCA_DB_NAME}" \
    --readonly \
    -port 8080 > /tmp/pfas-runtime/openlca.log 2>&1 &
  OPENLCA_PID="$!"

  local waited=0
  while [ "${waited}" -lt "${OPENLCA_WAIT_SECONDS}" ]; do
    if curl -fsS \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc":"2.0","id":1,"method":"data/get/descriptors","params":{"@type":"ProductSystem"}}' \
      "${OPENLCA_IPC_URL}" >/dev/null 2>&1; then
      echo "openLCA IPC server is ready at ${OPENLCA_IPC_URL}."
      return 0
    fi
    if ! kill -0 "${OPENLCA_PID}" >/dev/null 2>&1; then
      echo "openLCA IPC server exited during startup. Last log lines:"
      tail -80 /tmp/pfas-runtime/openlca.log || true
      return 1
    fi
    sleep 3
    waited=$((waited + 3))
  done

  echo "openLCA IPC server did not become ready within ${OPENLCA_WAIT_SECONDS}s. Last log lines:"
  tail -80 /tmp/pfas-runtime/openlca.log || true
  return 1
}

if ! database_ready; then
  if ! download_database_zip; then
    echo "No openLCA database ZIP configured. Starting PFAS app in proxy LCA/LCC mode."
    export PFAS_ENVIRONMENTAL_MODE=proxy
  fi
fi

if database_ready; then
  if ! start_openlca; then
    if [ "${OPENLCA_FALLBACK_TO_PROXY:-true}" = "true" ]; then
      echo "Continuing with proxy fallback enabled."
    else
      exit 1
    fi
  fi
else
  export PFAS_ENVIRONMENTAL_MODE=proxy
fi

echo "Starting PFAS platform on ${HOST}:${PORT}."
cd /srv/pfas
exec python generated_outputs/predictor_app/app.py --host "${HOST}" --port "${PORT}"
