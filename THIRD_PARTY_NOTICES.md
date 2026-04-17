# Third-Party Notices

This plugin redistributes third-party software and model assets.

## sherpa-onnx

- Project: `k2-fsa/sherpa-onnx`
- Role: Runtime used by the plugin for PocketTTS inference
- License: Apache-2.0
- Source: https://github.com/k2-fsa/sherpa-onnx

The packaged `sherpa-onnx` wheel also includes its own license files under `deps/sherpa_onnx-*.dist-info/licenses/`.

## PocketTTS model assets

- Project: `kyutai/pocket-tts`
- Role: Upstream model whose weights are redistributed in converted ONNX form
- License: CC-BY-4.0
- Source model card: https://huggingface.co/kyutai/pocket-tts
- Project page: https://github.com/kyutai-labs/pocket-tts

Attribution:

- Authors: Manu Orsini, Simon Rouard, Gabriel De Marmiesse, Vaclav Volhejn, Neil Zeghidour, Alexandre Defossez
- The packaged ONNX files are redistributed unchanged from the sherpa-onnx PocketTTS release archive and are based on the Kyutai PocketTTS model.

CC-BY-4.0 requires attribution and a link to the license:

- https://creativecommons.org/licenses/by/4.0/

## Bundled default reference voice

- File: `assets/voices/selfie.wav`
- Source: https://huggingface.co/kyutai/tts-voices/blob/main/voice-donations/Selfie.wav
- License: CC0-1.0

The Kyutai `tts-voices` repository documents `voice-donations/` as CC0:

- https://huggingface.co/kyutai/tts-voices
