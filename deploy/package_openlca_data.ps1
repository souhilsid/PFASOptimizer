param(
    [string]$OpenLcaDataDir = "C:\Users\aisci\openLCA-data-1.4",
    [string]$DatabaseName = "Biochar",
    [string]$OutputDir = ".\deploy\openlca-data",
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

$sourceRoot = Resolve-Path -LiteralPath $OpenLcaDataDir
$sourceDb = Join-Path $sourceRoot "databases\$DatabaseName"
$sourceLibraries = Join-Path $sourceRoot "libraries"

if (!(Test-Path -LiteralPath $sourceDb)) {
    throw "Database folder not found: $sourceDb"
}

$targetRoot = Resolve-Path -LiteralPath (New-Item -ItemType Directory -Force -Path $OutputDir)
$targetDbRoot = Join-Path $targetRoot "databases"
$targetDb = Join-Path $targetDbRoot $DatabaseName
$targetLibraries = Join-Path $targetRoot "libraries"

New-Item -ItemType Directory -Force -Path $targetDbRoot | Out-Null

if (Test-Path -LiteralPath $targetDb) {
    Remove-Item -LiteralPath $targetDb -Recurse -Force
}
Copy-Item -LiteralPath $sourceDb -Destination $targetDbRoot -Recurse -Force

if (Test-Path -LiteralPath $sourceLibraries) {
    if (Test-Path -LiteralPath $targetLibraries) {
        Remove-Item -LiteralPath $targetLibraries -Recurse -Force
    }
    Copy-Item -LiteralPath $sourceLibraries -Destination $targetRoot -Recurse -Force
}

$bytes = (Get-ChildItem -LiteralPath $targetRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host "Prepared openLCA workspace at $targetRoot"
Write-Host ("Size: {0:N2} MB" -f ($bytes / 1MB))

if ($Zip) {
    $zipPath = Join-Path (Split-Path $targetRoot -Parent) "openlca-data-$DatabaseName.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -LiteralPath (Join-Path $targetRoot "*") -DestinationPath $zipPath -Force
    Write-Host "Created $zipPath"
}
