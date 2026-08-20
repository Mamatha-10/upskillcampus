<#
Run the full training + report pipeline using the Anaconda Python used earlier.
Adjust the Python executable path below if you want to use a different interpreter.
#>
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Path $MyInvocation.MyCommand.Path -Parent)

# Change this path if your Python is installed elsewhere
$PY = 'C:\Users\sures\anaconda3\python.exe'

& $PY train_models.py
& $PY scripts\generate_report.py

Write-Host "`nDone. Generated: models/traffic_prediction_model.pkl, data/submission.csv, reports/"
Pop-Location
