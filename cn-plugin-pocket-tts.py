"""
Pocket TTS Plugin for COVAS:NEXT.
Provides offline zero-shot Text-to-Speech via sherpa-onnx.
"""

from typing import override, Iterable, Any, Optional
import os
import queue
import threading

import numpy as np
import samplerate
import sherpa_onnx
import soundfile as sf

from lib.PluginHelper import TTSModel
from lib.PluginSettingDefinitions import (
    PluginSettings,
    ModelProviderDefinition,
    SettingsGrid,
    ParagraphSetting,
    TextSetting,
    NumericalSetting,
)
from lib.PluginBase import PluginBase, PluginManifest
from lib.Logger import log


class PocketTTSModel(TTSModel):
    """PocketTTS text-to-speech model implementation."""

    TARGET_SAMPLE_RATE = 24000
    STREAM_CHUNK_SAMPLES = 2400

    def __init__(
        self,
        plugin_dir: str,
        model_dir: str,
        reference_audio_path: str,
        num_steps: int = 2,
    ):
        super().__init__("pocket-tts")
        self.plugin_dir = plugin_dir
        self.model_dir = model_dir
        self.reference_audio_path = reference_audio_path
        self.num_steps = max(int(num_steps), 1)

        self._tts: Optional[sherpa_onnx.OfflineTts] = None
        self._load_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._cached_reference_audio_path: Optional[str] = None
        self._cached_reference_audio: Optional[np.ndarray] = None

    def _resolve_required_file(self, *relative_candidates: str) -> str:
        for candidate in relative_candidates:
            path = os.path.join(self.model_dir, candidate)
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            f"Missing PocketTTS model asset. Tried: {', '.join(os.path.join(self.model_dir, c) for c in relative_candidates)}"
        )

    def _load_model(self) -> sherpa_onnx.OfflineTts:
        if self._tts is not None:
            return self._tts

        with self._load_lock:
            if self._tts is not None:
                return self._tts

            log("info", f"Loading PocketTTS models from {self.model_dir}")

            config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    pocket=sherpa_onnx.OfflineTtsPocketModelConfig(
                        lm_flow=self._resolve_required_file(
                            "lm_flow.int8.onnx",
                            "lm_flow.onnx",
                            "flow_lm_flow_int8.onnx",
                            "flow_lm_flow.onnx",
                        ),
                        lm_main=self._resolve_required_file(
                            "lm_main.int8.onnx",
                            "lm_main.onnx",
                            "flow_lm_main_int8.onnx",
                            "flow_lm_main.onnx",
                        ),
                        encoder=self._resolve_required_file(
                            "encoder.onnx",
                            "mimi_encoder.onnx",
                        ),
                        decoder=self._resolve_required_file(
                            "decoder.int8.onnx",
                            "decoder.onnx",
                            "mimi_decoder_int8.onnx",
                            "mimi_decoder.onnx",
                        ),
                        text_conditioner=self._resolve_required_file("text_conditioner.onnx"),
                        vocab_json=self._resolve_required_file("vocab.json"),
                        token_scores_json=self._resolve_required_file("token_scores.json"),
                    ),
                    debug=False,
                    num_threads=max(1, (os.cpu_count() or 2) // 2),
                    provider="cpu",
                ),
                max_num_sentences=1,
            )

            if not config.validate():
                raise ValueError("PocketTTS configuration is invalid. Check model assets in the plugin's model directory.")

            self._tts = sherpa_onnx.OfflineTts(config)
            log("info", f"PocketTTS sample rate: {self._tts.sample_rate} Hz")
            return self._tts

    def _normalize_user_path(self, path: str) -> Optional[str]:
        value = (path or "").strip()
        if not value:
            return None

        value = os.path.expanduser(os.path.expandvars(value))
        if not os.path.isabs(value):
            value = os.path.abspath(os.path.join(self.plugin_dir, value))
        return value

    def _resolve_reference_audio_candidate(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None

        candidate = self._normalize_user_path(path)
        if candidate is None:
            return None

        if os.path.isfile(candidate):
            return candidate

        # Older plugin versions defaulted to bria.wav. If a persisted config still
        # points there, transparently fall back to the new CC0 bundled voice.
        if os.path.basename(candidate).lower() == "bria.wav":
            selfie_candidate = os.path.join(os.path.dirname(candidate), "selfie.wav")
            if os.path.isfile(selfie_candidate):
                return selfie_candidate

        if os.path.isdir(candidate):
            for preferred_name in ("selfie.wav", "bria.wav"):
                preferred = os.path.join(candidate, preferred_name)
                if os.path.isfile(preferred):
                    return preferred

            supported_exts = {".wav", ".flac", ".ogg", ".mp3"}
            audio_files = []
            for entry in os.listdir(candidate):
                full_path = os.path.join(candidate, entry)
                if not os.path.isfile(full_path):
                    continue
                if os.path.splitext(entry)[1].lower() not in supported_exts:
                    continue
                audio_files.append(full_path)

            if audio_files:
                return sorted(audio_files)[0]

        return None

    def _resolve_reference_audio_path(self, configured_path: str, requested_voice: str) -> str:
        default_voice_dir = os.path.join(self.plugin_dir, "assets", "voices")
        candidates = [requested_voice, configured_path, default_voice_dir]

        for candidate in candidates:
            resolved = self._resolve_reference_audio_candidate(candidate)
            if resolved is not None:
                if candidate and resolved != self._normalize_user_path(candidate):
                    log("info", f"Using reference audio '{resolved}'")
                return resolved

        raise FileNotFoundError(
            "Could not resolve a PocketTTS reference audio file. "
            f"Checked requested voice '{requested_voice}', configured path '{configured_path}', and '{default_voice_dir}'."
        )

    def _load_reference_audio(self, reference_audio_path: str) -> np.ndarray:
        tts = self._load_model()
        if self._cached_reference_audio_path == reference_audio_path and self._cached_reference_audio is not None:
            return self._cached_reference_audio

        log("info", f"Loading PocketTTS reference audio from {reference_audio_path}")
        samples, sample_rate = sf.read(reference_audio_path, dtype="float32")
        samples = np.asarray(samples, dtype=np.float32)

        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = samples.reshape(-1)

        if sample_rate != tts.sample_rate:
            samples = samplerate.resample(samples, tts.sample_rate / sample_rate, "sinc_best")

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        self._cached_reference_audio_path = reference_audio_path
        self._cached_reference_audio = samples
        return samples

    def _pcm16_chunks(self, samples: np.ndarray, source_sample_rate: int) -> Iterable[bytes]:
        if source_sample_rate != self.TARGET_SAMPLE_RATE:
            samples = samplerate.resample(samples, self.TARGET_SAMPLE_RATE / source_sample_rate, "sinc_best")

        pcm = (np.asarray(samples, dtype=np.float32) * 32767.0).clip(-32768, 32767).astype(np.int16)
        for start in range(0, len(pcm), self.STREAM_CHUNK_SAMPLES):
            yield pcm[start : start + self.STREAM_CHUNK_SAMPLES].tobytes()

    @override
    def synthesize(self, text: str, voice: str) -> Iterable[bytes]:
        if not text.strip():
            return

        tts = self._load_model()
        reference_audio_path = self._resolve_reference_audio_path(self.reference_audio_path, voice)
        reference_audio = self._load_reference_audio(reference_audio_path)

        generation_config = sherpa_onnx.GenerationConfig()
        generation_config.reference_audio = reference_audio
        generation_config.reference_sample_rate = tts.sample_rate
        generation_config.num_steps = self.num_steps

        with self._synthesis_lock:
            audio_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
            errors: list[Exception] = []
            streamed_samples = 0

            def generated_audio_callback(samples: np.ndarray, progress: float) -> int:
                del progress
                nonlocal streamed_samples
                chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
                streamed_samples += len(chunk)
                audio_queue.put(chunk)
                return 1

            def run_generation() -> None:
                try:
                    audio = tts.generate(text, generation_config, callback=generated_audio_callback)
                    final_samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
                    if final_samples.size == 0:
                        raise RuntimeError("PocketTTS returned no audio")
                    if streamed_samples < final_samples.size:
                        audio_queue.put(final_samples[streamed_samples:])
                except Exception as exc:
                    errors.append(exc)
                finally:
                    audio_queue.put(None)

            thread = threading.Thread(target=run_generation, name="pocket-tts-stream", daemon=True)
            thread.start()

            try:
                while True:
                    chunk = audio_queue.get()
                    if chunk is None:
                        break
                    for pcm_chunk in self._pcm16_chunks(chunk, tts.sample_rate):
                        yield pcm_chunk
            finally:
                thread.join()

            if errors:
                raise errors[0]


class PocketTTSPlugin(PluginBase):
    """Plugin providing PocketTTS services."""

    def __init__(self, plugin_manifest: PluginManifest):
        super().__init__(plugin_manifest)

        self.plugin_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(self.plugin_dir, "model")
        self.default_reference_audio_dir = os.path.join(self.plugin_dir, "assets", "voices")
        self.default_reference_audio_path = os.path.join(self.default_reference_audio_dir, "selfie.wav")

        self.settings_config = PluginSettings(
            key="Pocket TTS",
            label="Pocket TTS",
            icon="record_voice_over",
            grids=[
                SettingsGrid(
                    key="general",
                    label="General",
                    fields=[
                        ParagraphSetting(
                            key="info_text",
                            label=None,
                            type="paragraph",
                            readonly=False,
                            placeholder=None,
                            content='To use Pocket TTS, select it as your "TTS provider" in "Advanced" → "TTS Settings".',
                        ),
                    ],
                ),
            ],
        )

        self.model_providers = [
            ModelProviderDefinition(
                kind="tts",
                id="pocket-tts",
                label="Pocket TTS (Offline)",
                settings_config=[
                    SettingsGrid(
                        key="settings",
                        label="Settings",
                        fields=[
                            ParagraphSetting(
                                key="reference_audio_help",
                                label=None,
                                type="paragraph",
                                readonly=False,
                                placeholder=None,
                                content=(
                                    "Pocket TTS clones a voice from a reference clip. "
                                    "You can point this setting at a file or a directory. "
                                    "If you point it at the bundled assets directory, the plugin prefers `selfie.wav`."
                                ),
                            ),
                            TextSetting(
                                key="reference_audio_path",
                                label="Reference audio path",
                                type="text",
                                readonly=False,
                                placeholder=self.default_reference_audio_path,
                                default_value=self.default_reference_audio_path,
                            ),
                            NumericalSetting(
                                key="num_steps",
                                label="Generation steps",
                                type="number",
                                readonly=False,
                                placeholder="2",
                                default_value=2,
                                min_value=1,
                                max_value=8,
                                step=1,
                            ),
                        ],
                    )
                ],
            )
        ]

    @override
    def create_model(self, provider_id: str, settings: dict[str, Any]) -> TTSModel:
        if provider_id == "pocket-tts":
            reference_audio_path = settings.get("reference_audio_path", self.default_reference_audio_path)
            num_steps = int(settings.get("num_steps", 2))
            return PocketTTSModel(
                plugin_dir=self.plugin_dir,
                model_dir=self.model_dir,
                reference_audio_path=reference_audio_path,
                num_steps=num_steps,
            )

        raise ValueError(f"Unknown PocketTTS provider: {provider_id}")


if __name__ == "__main__":
    plugin_manifest = PluginManifest(
        name="Pocket TTS Plugin",
        version="0.0.1",
        author="COVAS:NEXT",
        description="Pocket TTS Plugin for COVAS:NEXT",
    )
    plugin = PocketTTSPlugin(plugin_manifest)
    try:
        plugin.create_model("pocket-tts", {})
        log("info", "Pocket TTS Plugin initialized successfully.")
    except Exception as exc:
        log("error", f"Failed to initialize Pocket TTS Plugin: {exc}")
