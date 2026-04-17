#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PLUGIN_DIR/model"
VOICE_DIR="$PLUGIN_DIR/assets/voices"
TMP_DIR="$PLUGIN_DIR/.tmp/pocket-tts"
ARCHIVE_NAME="sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2"
ARCHIVE_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$ARCHIVE_NAME"
EXTRACTED_DIR="$TMP_DIR/sherpa-onnx-pocket-tts-int8-2026-01-26"

mkdir -p "$MODEL_DIR" "$VOICE_DIR" "$TMP_DIR"
rm -rf "$TMP_DIR"/*

echo "Downloading PocketTTS assets..."
curl -L --fail --output "$TMP_DIR/$ARCHIVE_NAME" "$ARCHIVE_URL"

echo "Extracting PocketTTS assets..."
tar -xjf "$TMP_DIR/$ARCHIVE_NAME" -C "$TMP_DIR"

cp "$EXTRACTED_DIR/lm_flow.int8.onnx" "$MODEL_DIR/"
cp "$EXTRACTED_DIR/lm_main.int8.onnx" "$MODEL_DIR/"
cp "$EXTRACTED_DIR/encoder.onnx" "$MODEL_DIR/"
cp "$EXTRACTED_DIR/decoder.int8.onnx" "$MODEL_DIR/"
cp "$EXTRACTED_DIR/text_conditioner.onnx" "$MODEL_DIR/"
cp "$EXTRACTED_DIR/vocab.json" "$MODEL_DIR/"
cp "$EXTRACTED_DIR/token_scores.json" "$MODEL_DIR/"
cp "$EXTRACTED_DIR/test_wavs/bria.wav" "$VOICE_DIR/"

echo "PocketTTS assets downloaded into $MODEL_DIR and $VOICE_DIR"
