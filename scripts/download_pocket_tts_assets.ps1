$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PluginDir = Split-Path -Parent $ScriptDir
$ModelDir = Join-Path $PluginDir "model"
$VoiceDir = Join-Path $PluginDir "assets\voices"
$TmpDir = Join-Path $PluginDir ".tmp\pocket-tts"
$ArchiveName = "sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2"
$ArchiveUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$ArchiveName"
$ExtractedDir = Join-Path $TmpDir "sherpa-onnx-pocket-tts-int8-2026-01-26"
$ArchivePath = Join-Path $TmpDir $ArchiveName
$DefaultVoiceUrl = "https://huggingface.co/kyutai/tts-voices/resolve/main/voice-donations/Selfie.wav"

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $VoiceDir | Out-Null
if (Test-Path $TmpDir) {
    Remove-Item -Recurse -Force $TmpDir
}
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

Write-Host "Downloading PocketTTS assets..."
Invoke-WebRequest -Uri $ArchiveUrl -OutFile $ArchivePath

Write-Host "Extracting PocketTTS assets..."
tar -xf $ArchivePath -C $TmpDir

Copy-Item (Join-Path $ExtractedDir "lm_flow.int8.onnx") $ModelDir -Force
Copy-Item (Join-Path $ExtractedDir "lm_main.int8.onnx") $ModelDir -Force
Copy-Item (Join-Path $ExtractedDir "encoder.onnx") $ModelDir -Force
Copy-Item (Join-Path $ExtractedDir "decoder.int8.onnx") $ModelDir -Force
Copy-Item (Join-Path $ExtractedDir "text_conditioner.onnx") $ModelDir -Force
Copy-Item (Join-Path $ExtractedDir "vocab.json") $ModelDir -Force
Copy-Item (Join-Path $ExtractedDir "token_scores.json") $ModelDir -Force
Invoke-WebRequest -Uri $DefaultVoiceUrl -OutFile (Join-Path $VoiceDir "selfie.wav")

Write-Host "PocketTTS assets downloaded into $ModelDir and $VoiceDir"
