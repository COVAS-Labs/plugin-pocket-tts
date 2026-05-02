# COVAS:NEXT Plugin Pocket TTS

Run zero-shot TTS locally using PocketTTS via a bundled `onnxruntime` backend.

## About

This plugin provides offline Text-to-Speech (TTS) for COVAS:NEXT using **PocketTTS** through a bundled **ONNX Runtime** backend. PocketTTS clones a voice from a short reference clip, so you can keep synthesis local while swapping voices just by changing the reference audio.

## Features

- **Offline synthesis**: No internet connection required.
- **Streaming output**: Streams audio directly from the bundled PocketTTS ONNX runtime.
- **Voice cloning**: Uses a short reference clip instead of fixed speaker IDs.
- **Bundled default voice**: The plugin ships with a bundled reference clip and points to `assets/voices/selfie.wav` by default.

## Installation

Download the latest release under the *Releases* section on the right. Follow the instructions on [COVAS:NEXT Plugins](https://ratherrude.github.io/Elite-Dangerous-AI-Integration/plugins/) to install the plugin.

Unpack the plugin into the `plugins` folder in the COVAS:NEXT AppData folder, leading to the following folder structure:
* `plugins`
    * `cn-plugin-pocket-tts`
        * `cn-plugin-pocket-tts.py`
        * `requirements.txt`
        * `deps`
        * `model`
        * `assets`
        * `__init__.py`
        * etc.
    * `OtherPlugin`

## Configuration

Select **Pocket TTS (Offline)** as your TTS provider in COVAS:NEXT.

The plugin exposes:
- **Fallback voice file**: Sets the default reference clip. Runtime voice names are first tried as absolute paths or as paths relative to the fallback file's directory. If a voice name has no extension, the plugin also tries `.wav` automatically.
- **Generation steps**: Higher values improve quality but add latency.
- **Max tokens per inference pass**: Uses the official Pocket TTS chunking logic. The plugin first splits on sentence punctuation, then falls back to commas, semicolons, and colons when a segment is too long, while packing each pass up to the configured token limit. The default is `50`.
- **Gap between passes (ms)**: Adds a short silence between multi-pass chunks so stitched output flows more naturally. Set it to `0` to disable the gap.

## Development

During development, clone the COVAS:NEXT repository and place your plugin project in the plugins folder.
Install the dependencies to your local `.venv` virtual environment using `pip`, by running this command in the `cn-plugin-pocket-tts` folder:

```bash
pip install -r requirements.txt
```

Follow the [COVAS:NEXT Plugin Development Guide](https://ratherrude.github.io/Elite-Dangerous-AI-Integration/plugins/Development/) for more information on developing plugins.

## Packaging

Use the `./pack.ps1` or `./pack.sh` scripts to package the plugin and any Python dependencies in the `deps` folder.

If the PocketTTS model assets are not present, the pack scripts automatically download the PocketTTS ONNX bundle into this layout:
- `model/english_2026-04/bundle.json`
- `model/english_2026-04/tokenizer.model`
- `model/english_2026-04/bos_before_voice.npy`
- `model/english_2026-04/flow_lm_main_int8.onnx`
- `model/english_2026-04/flow_lm_flow_int8.onnx`
- `model/english_2026-04/mimi_decoder_int8.onnx`
- `model/english_2026-04/mimi_encoder.onnx`
- `model/english_2026-04/text_conditioner.onnx`
- `assets/voices/selfie.wav`

## Licensing

- The bundled PocketTTS model weights are derived from Kyutai PocketTTS and require attribution under `CC-BY-4.0`.
- The bundled default reference voice `selfie.wav` comes from `kyutai/tts-voices/voice-donations` and is released as `CC0`.
- Third-party attribution details are included in `THIRD_PARTY_NOTICES.md` and are packaged into the release zip.

You can also download the assets explicitly:
- Linux/macOS: `./scripts/download_pocket_tts_assets.sh`
- Windows: `./scripts/download_pocket_tts_assets.ps1`

## Releasing

This project includes a GitHub Actions workflow that automatically creates releases. To create a new release:

1. Tag your commit with a version number:
   ```bash
   git tag v1.0.0
   ```
2. Push the tag to GitHub:
   ```bash
   git push origin v1.0.0
   ```

The workflow automatically downloads the PocketTTS assets, builds the plugin, and creates a GitHub Release with the zip file attached.

## Acknowledgements

- [COVAS:NEXT](https://github.com/RatherRude/Elite-Dangerous-AI-Integration)
- [onnxruntime](https://github.com/microsoft/onnxruntime) - Local inference runtime.
- [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) - The underlying TTS model.
