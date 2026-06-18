# PFAS Platform Deployment Guide

This project is prepared for a two-layer production architecture:

```text
Browser
  -> PFAS platform web service
  -> private LCA evaluator API
  -> private openLCA/gdt-server engine
  -> openLCA Biochar database
```

The user-facing platform does not need a local openLCA installation or an IPC
port. It calls the evaluator by URL.

## Services

### 1. `pfas-platform`

Public web service. Runs:

```bash
python generated_outputs/predictor_app/app.py
```

Important environment variables:

```text
HOST=0.0.0.0
PORT=<set by host>
PFAS_ENVIRONMENTAL_MODE=openlca_cloud
OPENLCA_EVALUATOR_URL=http://pfas-openlca-evaluator:8000
OPENLCA_EVALUATOR_API_KEY=<same value as evaluator, optional on private network>
```

### 2. `pfas-openlca-evaluator`

Private FastAPI service. It receives candidate payloads from the PFAS platform
and returns LCC, EBI, impact vectors, and breakdowns.

Important environment variables:

```text
OPENLCA_EVALUATOR_MODE=openlca_rest
OPENLCA_REST_URL=http://pfas-openlca-engine:8080
OPENLCA_PRODUCT_SYSTEM=PS_Biochar_PFAS_Incineration
OPENLCA_IMPACT_METHOD=ReCiPe Midpoint (H)
OPENLCA_FALLBACK_TO_PROXY=true
OPENLCA_EVALUATOR_API_KEY=<optional shared secret>
```

If you want a demo without the real openLCA engine, set:

```text
OPENLCA_EVALUATOR_MODE=proxy
```

### 3. `pfas-openlca-engine`

Private gdt-server/openLCA REST engine. It owns the full openLCA database.

Expected data layout in the container:

```text
/app/data/
  databases/
    Biochar/
  libraries/
    ...
```

The service runs in read-only mode:

```bash
/app/run.sh -data /app/data -db Biochar --readonly -port 8080
```

## Render Deployment

1. Put this project in a GitHub or GitLab repository.
2. Confirm you are allowed to deploy the openLCA database and any background
   datasets to cloud infrastructure.
3. In Render, create a new Blueprint from `render.yaml`.
4. Enter the same `OPENLCA_EVALUATOR_API_KEY` for both:
   - `pfas-platform`
   - `pfas-openlca-evaluator`
5. Deploy the services.
6. Upload or mount the full openLCA workspace to the `pfas-openlca-engine`
   persistent disk at `/app/data`.
   A practical path is:
   - Run `deploy/package_openlca_data.ps1 -Zip` locally.
   - Upload `deploy/openlca-data-Biochar.zip` to a private temporary URL
     such as S3, Google Cloud Storage, or Google Drive direct download.
   - Open the Render Shell for `pfas-openlca-engine` and run:

```bash
cd /app/data
curl -L "<SIGNED_OR_PRIVATE_DOWNLOAD_URL>" -o openlca-data-Biochar.zip
unzip -o openlca-data-Biochar.zip -d /app/data
ls /app/data/databases/Biochar
```

7. Confirm the engine responds at:

```text
http://pfas-openlca-engine:8080/api/version
```

8. Confirm the evaluator responds at:

```text
http://pfas-openlca-evaluator:8000/health
```

9. Open the public `pfas-platform` URL and click **Test cloud evaluator**.

## Preparing the openLCA Database

The local database detected on this machine is:

```text
C:\Users\aisci\openLCA-data-1.4\databases\Biochar
```

To stage it locally:

```powershell
.\deploy\package_openlca_data.ps1 `
  -OpenLcaDataDir "C:\Users\aisci\openLCA-data-1.4" `
  -DatabaseName "Biochar" `
  -Zip
```

This produces:

```text
deploy/openlca-data/
deploy/openlca-data-Biochar.zip
```

Do not commit the database folder unless licensing permits it.

## Local Docker Test

Proxy evaluator only:

```bash
docker build -f deploy/openlca-evaluator.Dockerfile -t pfas-openlca-evaluator .
docker run --rm -p 8000:8000 -e OPENLCA_EVALUATOR_MODE=proxy pfas-openlca-evaluator
```

PFAS app calling the evaluator:

```bash
docker build -f deploy/pfas-app.Dockerfile -t pfas-platform .
docker run --rm -p 8057:8057 ^
  -e PORT=8057 ^
  -e HOST=0.0.0.0 ^
  -e PFAS_ENVIRONMENTAL_MODE=openlca_cloud ^
  -e OPENLCA_EVALUATOR_URL=http://host.docker.internal:8000 ^
  pfas-platform
```

Then open:

```text
http://127.0.0.1:8057
```

## Deployment Notes

- Keep the openLCA engine private. Do not expose raw gdt-server/openLCA IPC
  publicly.
- Use `--readonly` for the openLCA database service.
- Use a paid instance with enough memory for the engine. Start with at least
  Render `standard` for the gdt-server service.
- Keep `OPENLCA_FALLBACK_TO_PROXY=true` during demos so inverse design still
  runs if the engine is temporarily unavailable.
- The PFAS platform can also be deployed without the openLCA engine by setting
  evaluator mode to `proxy`.
