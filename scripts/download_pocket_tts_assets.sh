#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PLUGIN_DIR/model"
MODEL_BUNDLE="${MODEL_BUNDLE:-english_2026-04}"
MODEL_BUNDLE_DIR="$MODEL_DIR/$MODEL_BUNDLE"
VOICE_DIR="$PLUGIN_DIR/assets/voices"
TMP_DIR="$PLUGIN_DIR/.tmp/pocket-tts"
MODEL_BASE_URL="https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/main/onnx/$MODEL_BUNDLE"
DEFAULT_VOICE_URL="https://huggingface.co/kyutai/tts-voices/resolve/main/voice-donations/Selfie.wav"

case "$MODEL_BUNDLE" in
    ""|.*|*/*|*\\*|*[!A-Za-z0-9_-]*)
        echo "Invalid model bundle name: $MODEL_BUNDLE" >&2
        exit 1
        ;;
esac

mkdir -p "$MODEL_DIR" "$MODEL_BUNDLE_DIR" "$VOICE_DIR" "$TMP_DIR"
rm -rf "$TMP_DIR"/*

echo "Downloading PocketTTS ONNX bundle '$MODEL_BUNDLE'..."

rm -f \
    "$MODEL_DIR/lm_flow.int8.onnx" \
    "$MODEL_DIR/lm_main.int8.onnx" \
    "$MODEL_DIR/encoder.onnx" \
    "$MODEL_DIR/decoder.int8.onnx" \
    "$MODEL_DIR/text_conditioner.onnx" \
    "$MODEL_DIR/vocab.json" \
    "$MODEL_DIR/token_scores.json"
rm -rf "$MODEL_BUNDLE_DIR"
mkdir -p "$MODEL_BUNDLE_DIR"

bundle_files=(
    "bundle.json"
    "tokenizer.model"
    "bos_before_voice.npy"
    "flow_lm_main_int8.onnx"
    "flow_lm_flow_int8.onnx"
    "mimi_decoder_int8.onnx"
    "mimi_encoder.onnx"
    "text_conditioner.onnx"
)

for file_name in "${bundle_files[@]}"; do
    curl -L --fail --output "$MODEL_BUNDLE_DIR/$file_name" "$MODEL_BASE_URL/$file_name?download=true"
done

curl -L --fail --output "$VOICE_DIR/selfie.wav" "$DEFAULT_VOICE_URL"

echo "PocketTTS ONNX assets downloaded into $MODEL_BUNDLE_DIR and $VOICE_DIR"
