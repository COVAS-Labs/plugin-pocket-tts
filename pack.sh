#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    for candidate in "$HOME/.pyenv/versions/3.12.13/bin/python" "python3.12" "python3" "python"; do
        if [ -x "$candidate" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "Could not find a Python interpreter for packaging. Set PYTHON_BIN to continue." >&2
    exit 1
fi

# Delete dist if it already exists
if [ -d "dist" ]; then
    rm -rf dist
fi

# Create dist
mkdir dist

# Reinstall dependencies cleanly so packaged native wheels match the selected Python runtime
if [ -d "deps" ]; then
    rm -rf deps
fi

# Install dependencies
if [ -f "requirements.txt" ]; then
    "$PYTHON_BIN" -m pip install --target ./deps -r requirements.txt
fi

# Ensure PocketTTS model files and bundled reference audio are present
if [ ! -f "model/lm_flow.int8.onnx" ] || [ ! -f "assets/voices/selfie.wav" ]; then
    if [ -f "scripts/download_pocket_tts_assets.sh" ]; then
        echo "PocketTTS assets not found; downloading into ./model and ./assets/voices ..."
        chmod +x scripts/download_pocket_tts_assets.sh
        ./scripts/download_pocket_tts_assets.sh
    else
        echo "Missing PocketTTS assets and scripts/download_pocket_tts_assets.sh not found." >&2
        exit 1
    fi
fi

# Remember to add any additional files, and change the name of the plugin
artifacts=(
    "cn-plugin-pocket-tts.py"
    "requirements.txt"
    "manifest.json" "__init__.py" "THIRD_PARTY_NOTICES.md"
)

if [ -d "deps" ]; then
    artifacts+=("deps")
fi

if [ -d "model" ]; then
    artifacts+=("model")
fi

if [ -d "assets" ]; then
    artifacts+=("assets")
fi

# Create the zip archive
zip -r -9 "dist/cn-plugin-pocket-tts.zip" "${artifacts[@]}"
