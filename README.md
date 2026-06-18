# PFAS Optimizer

AISCIA PFAS removal decision engine for:

- Biochar and resin removal prediction from saved PKL models.
- PSO optimization for top removal candidates.
- BoTorch inverse design for target removal value/range matching.
- Biochar LCA/LCC screening through proxy, local IPC, or cloud OpenLCA evaluator.

## Local Run

```powershell
py -3.11 generated_outputs\predictor_app\app.py --port 8057
```

Open:

```text
http://127.0.0.1:8057
```

## Deployment

Render deployment files are included:

- `render.yaml`
- `deploy/pfas-app.Dockerfile`
- `deploy/openlca-evaluator.Dockerfile`
- `deploy/openlca-engine.Dockerfile`

Read the full guide:

```text
deploy/DEPLOYMENT_GUIDE.md
```

## openLCA Database

The full openLCA database is not committed because it is large and may contain
licensed data. Package it only when you are ready to deploy:

```powershell
.\deploy\package_openlca_data.ps1 `
  -OpenLcaDataDir "C:\Users\aisci\openLCA-data-1.4" `
  -DatabaseName "Biochar" `
  -Zip
```

Upload the resulting ZIP to the Render persistent disk for the private
`pfas-openlca-engine` service.
