---
title: PFAS Removal Decision Engine
sdk: docker
app_port: 7860
pinned: false
license: other
---

# PFAS Removal Decision Engine

Docker Space for the AISCIA PFAS prediction, optimization, inverse design, and LCA/LCC demo.

The container starts:

- PFAS web platform on port `7860`
- openLCA IPC server internally on port `8080` when a Biochar database ZIP is configured
- proxy LCA/LCC fallback when no database is configured or the engine is unavailable

Required Space secrets or variables for real openLCA mode:

```text
HF_OPENLCA_DATASET_REPO=<username-or-org>/<private-dataset>
HF_OPENLCA_DATASET_FILE=openlca-data-Biochar.zip
HF_TOKEN=<token with read access to the private dataset>
```

Alternative direct-download mode:

```text
OPENLCA_DB_ZIP_URL=<private-or-public-zip-url>
OPENLCA_DB_BEARER_TOKEN=<optional-token>
```

Default database name:

```text
OPENLCA_DB_NAME=Biochar
OPENLCA_PRODUCT_SYSTEM=PFAS Adsorption Treatment with Biochar
OPENLCA_IMPACT_METHOD=ReCiPe 2016 Midpoint (H)
```
