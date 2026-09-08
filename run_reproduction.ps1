param(
    [string]$PythonExe = "python",
    [string]$SegmentationData,
    [string]$ClassificationData
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

& $PythonExe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $PythonExe .\verify_streaming.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($SegmentationData) {
    & $PythonExe .\train_segmentation.py --data-dir $SegmentationData
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($ClassificationData) {
    & $PythonExe .\train_classification.py --data-dir $ClassificationData
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
