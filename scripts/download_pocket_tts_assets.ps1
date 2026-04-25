$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PluginDir = Split-Path -Parent $ScriptDir
$ModelDir = Join-Path $PluginDir "model"
$ModelBundle = "english_2026-04"
$ModelBundleDir = Join-Path $ModelDir $ModelBundle
$VoiceDir = Join-Path $PluginDir "assets\voices"
$TmpDir = Join-Path $PluginDir ".tmp\pocket-tts"
$ModelBaseUrl = "https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/main/onnx/$ModelBundle"
$DefaultVoiceUrl = "https://huggingface.co/kyutai/tts-voices/resolve/main/voice-donations/Selfie.wav"

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $ModelBundleDir | Out-Null
New-Item -ItemType Directory -Force -Path $VoiceDir | Out-Null
if (Test-Path $TmpDir) {
    Remove-Item -Recurse -Force $TmpDir
}
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

Write-Host "Downloading PocketTTS ONNX bundle assets..."

@(
    "lm_flow.int8.onnx",
    "lm_main.int8.onnx",
    "encoder.onnx",
    "decoder.int8.onnx",
    "text_conditioner.onnx",
    "vocab.json",
    "token_scores.json"
) | ForEach-Object {
    $LegacyFile = Join-Path $ModelDir $_
    if (Test-Path $LegacyFile) {
        Remove-Item -Force $LegacyFile
    }
}

if (Test-Path $ModelBundleDir) {
    Remove-Item -Recurse -Force $ModelBundleDir
}
New-Item -ItemType Directory -Force -Path $ModelBundleDir | Out-Null

@(
    "bundle.json",
    "tokenizer.model",
    "bos_before_voice.npy",
    "flow_lm_main_int8.onnx",
    "flow_lm_flow_int8.onnx",
    "mimi_decoder_int8.onnx",
    "mimi_encoder.onnx",
    "text_conditioner.onnx"
) | ForEach-Object {
    $TargetPath = Join-Path $ModelBundleDir $_
    Invoke-WebRequest -Uri "$ModelBaseUrl/$_?download=true" -OutFile $TargetPath
}

Invoke-WebRequest -Uri $DefaultVoiceUrl -OutFile (Join-Path $VoiceDir "selfie.wav")

Write-Host "PocketTTS ONNX assets downloaded into $ModelBundleDir and $VoiceDir"
