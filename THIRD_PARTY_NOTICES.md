# Third-Party Notices

This plugin redistributes third-party software and model assets.

## PocketTTS ONNX runtime helper

- Project: `KevinAHM/pocket-tts-onnx`
- Role: Vendored ONNX Runtime-based helper used by the plugin for PocketTTS inference
- License: Apache-2.0
- Source: https://huggingface.co/KevinAHM/pocket-tts-onnx

## ONNX Runtime

- Project: `microsoft/onnxruntime`
- Role: ONNX execution backend redistributed in the packaged Python dependencies
- License: MIT
- Source: https://github.com/microsoft/onnxruntime

## samplerate

- Project: `fakufaku/samplerate`
- Role: Reference audio resampling backend redistributed in the packaged Python dependencies
- License: MIT
- Source: https://github.com/tuxu/python-samplerate

## PocketTTS model assets

- Project: `kyutai/pocket-tts`
- Role: Upstream model whose weights are redistributed in converted ONNX form
- License: CC-BY-4.0
- Source model card: https://huggingface.co/kyutai/pocket-tts
- Project page: https://github.com/kyutai-labs/pocket-tts

Attribution:

- Authors: Manu Orsini, Simon Rouard, Gabriel De Marmiesse, Vaclav Volhejn, Neil Zeghidour, Alexandre Defossez
- The packaged ONNX files are redistributed from the PocketTTS ONNX export maintained at `KevinAHM/pocket-tts-onnx` and are based on the Kyutai PocketTTS model.

CC-BY-4.0 requires attribution and a link to the license:

- https://creativecommons.org/licenses/by/4.0/

## Bundled default reference voice

- File: `assets/voices/selfie.wav`
- Source: https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Selfie.wav
- License: CC0-1.0

The Kyutai `tts-voices` repository documents `voice-donations/` as CC0:

- https://huggingface.co/kyutai/tts-voices
