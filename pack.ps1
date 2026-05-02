# Delete dist if it already exists
if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}

$PythonCommand = $env:PYTHON_BIN
$PythonArgs = @()
if (-not $PythonCommand) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonCommand = "python"
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $PythonCommand = "py"
        $PythonArgs = @("-3.12")
    }
    else {
        throw "Could not find a Python interpreter for packaging. Set PYTHON_BIN to continue."
    }
}

# Create dist
New-Item "dist" -ItemType Directory

# Reinstall dependencies cleanly so packaged native wheels match the selected Python runtime
if (Test-Path "deps") {
    Remove-Item -Recurse -Force "deps"
}

# Install dependencies
if (Test-Path "requirements.txt") {
    & $PythonCommand @PythonArgs -m pip install --target ./deps -r requirements.txt
}

# Ensure PocketTTS ONNX bundle files and bundled reference audio are present
if ((-not (Test-Path "model\english_2026-04\bundle.json")) -or ((-not (Test-Path "assets\voices\nova.wav")) -and (-not (Test-Path "assets\voices\selfie.wav")))) {
    if (Test-Path "scripts\download_pocket_tts_assets.ps1") {
        Write-Host "PocketTTS ONNX assets not found; downloading into .\model and .\assets\voices ..."
        .\scripts\download_pocket_tts_assets.ps1
    }
    else {
        throw "Missing PocketTTS assets and scripts\download_pocket_tts_assets.ps1 not found."
    }
}

# Remember to add any additional files, and change the name of the plugin
$artifacts = "cn-plugin-pocket-tts.py", "requirements.txt", "manifest.json", "__init__.py", "THIRD_PARTY_NOTICES.md"

if (Test-Path "deps") {
    $artifacts += "deps"
}

if (Test-Path "model") {
    $artifacts += "model"
}

if (Test-Path "assets") {
    $artifacts += "assets"
}

if (Test-Path "vendor") {
    $artifacts += "vendor"
}

$compress = @{
LiteralPath = $artifacts
CompressionLevel = "Fastest"
DestinationPath = "dist\cn-plugin-pocket-tts.zip"
}
Compress-Archive @compress
