param(
  [string]$In="data\silver_explicit.parquet",
  [string]$Runs="artifacts\runs",
  [int]$K=10,
  [float]$Frac=1.0,
  [float]$PopAlpha=0.05
)

$ErrorActionPreference="Stop"
$env:PYTHONPATH="$PWD"

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
python .\jobs\train_als_local.py `
  --in $In `
  --out artifacts\tmp `
  --run-dir $Runs `
  --split-mode peruser `
  --k $K `
  --exclude-seen `
  --pop-alpha $PopAlpha `
  --metrics-out "$Runs\$ts\metrics.json" `
  --metrics-sample-frac $Frac `
  --disable-batch `
  --skip-recs-json

Write-Host "Done. See latest run under $Runs"
