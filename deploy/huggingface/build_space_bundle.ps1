param(
    [string]$OutputDir = ".\generated_outputs\huggingface_space",
    [switch]$IncludeOpenLcaZip
)

$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path -LiteralPath ".").Path
$target = Join-Path $workspace $OutputDir
$targetFull = [System.IO.Path]::GetFullPath($target)

if (-not $targetFull.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside workspace: $targetFull"
}

if (Test-Path -LiteralPath $targetFull) {
    Remove-Item -LiteralPath $targetFull -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $targetFull | Out-Null

Copy-Item -LiteralPath ".\deploy\huggingface\Dockerfile" -Destination (Join-Path $targetFull "Dockerfile")
Copy-Item -LiteralPath ".\deploy\huggingface\olca-ipc-pom.xml" -Destination (Join-Path $targetFull "olca-ipc-pom.xml")
Copy-Item -LiteralPath ".\deploy\huggingface\README.md" -Destination (Join-Path $targetFull "README.md")
Copy-Item -LiteralPath ".\deploy\huggingface\start.sh" -Destination (Join-Path $targetFull "start.sh")
Copy-Item -LiteralPath ".\requirements-pfas-app.txt" -Destination (Join-Path $targetFull "requirements-pfas-app.txt")

New-Item -ItemType Directory -Force -Path (Join-Path $targetFull "generated_outputs") | Out-Null
Copy-Item -LiteralPath ".\generated_outputs\predictor_app" -Destination (Join-Path $targetFull "generated_outputs\predictor_app") -Recurse
New-Item -ItemType Directory -Force -Path (Join-Path $targetFull "generated_outputs\pkl_models") | Out-Null
Copy-Item -LiteralPath ".\generated_outputs\pkl_models\dataset1_biochar_removal_model.pkl.gz" -Destination (Join-Path $targetFull "generated_outputs\pkl_models")
Copy-Item -LiteralPath ".\generated_outputs\pkl_models\dataset2_resin_removal_known_catalog_model.pkl.gz" -Destination (Join-Path $targetFull "generated_outputs\pkl_models")
Copy-Item -LiteralPath ".\generated_outputs\lca_lcc_evaluator.py" -Destination (Join-Path $targetFull "generated_outputs")
Copy-Item -LiteralPath ".\generated_outputs\inverse_design_engine.py" -Destination (Join-Path $targetFull "generated_outputs")
Copy-Item -LiteralPath ".\generated_outputs\run_pso_optimization.py" -Destination (Join-Path $targetFull "generated_outputs")
Copy-Item -LiteralPath ".\generated_outputs\pfas_smiles_lookup.csv" -Destination (Join-Path $targetFull "generated_outputs")

Copy-Item -LiteralPath ".\deploy_data" -Destination (Join-Path $targetFull "deploy_data") -Recurse

if ($IncludeOpenLcaZip) {
    $zip = ".\deploy\openlca-data-Biochar.zip"
    if (!(Test-Path -LiteralPath $zip)) {
        throw "openLCA ZIP not found: $zip"
    }
    Copy-Item -LiteralPath $zip -Destination (Join-Path $targetFull "openlca-data-Biochar.zip")
}

Write-Host "Prepared Hugging Face Space bundle at $targetFull"
