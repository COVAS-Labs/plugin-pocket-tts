"""Pocket TTS plugin for COVAS:NEXT using a vendored ONNX runtime."""

from typing import override, Iterable, Any, Optional
import os
import re
import sys
import threading

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
DEPS_DIR = os.path.join(PLUGIN_DIR, "deps")

if os.path.isdir(DEPS_DIR) and DEPS_DIR not in sys.path:
    sys.path.insert(0, DEPS_DIR)

import numpy as np

try:
    from .vendor.pocket_tts_onnx import PocketTTSOnnx
except ImportError:
    from vendor.pocket_tts_onnx import PocketTTSOnnx

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

    DEFAULT_LANGUAGE_BUNDLE = "english_2026-04"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_INTER_PASS_GAP_MS = 150
    DEFAULT_SENTENCES_PER_PASS = 2
    SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
    TARGET_SAMPLE_RATE = 24000
    STREAM_CHUNK_SAMPLES = 2400

    def __init__(
        self,
        plugin_dir: str,
        model_dir: str,
        reference_audio_path: str,
        num_steps: int = 2,
        inter_pass_gap_ms: int = DEFAULT_INTER_PASS_GAP_MS,
        sentences_per_pass: int = DEFAULT_SENTENCES_PER_PASS,
    ):
        super().__init__("pocket-tts")
        self.plugin_dir = plugin_dir
        self.model_dir = model_dir
        self.reference_audio_path = reference_audio_path
        self.num_steps = max(int(num_steps), 1)
        self.inter_pass_gap_ms = max(int(inter_pass_gap_ms), 0)
        self.sentences_per_pass = max(int(sentences_per_pass), 1)

        self._tts: Optional[PocketTTSOnnx] = None
        self._load_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()

    def _load_model(self) -> PocketTTSOnnx:
        if self._tts is not None:
            return self._tts

        with self._load_lock:
            if self._tts is not None:
                return self._tts

            bundle_dir = os.path.join(self.model_dir, self.DEFAULT_LANGUAGE_BUNDLE)
            bundle_manifest = os.path.join(bundle_dir, "bundle.json")
            if not os.path.isfile(bundle_manifest):
                raise FileNotFoundError(
                    "Missing PocketTTS ONNX bundle manifest. "
                    f"Expected {bundle_manifest}."
                )

            log(
                "info",
                f"Loading PocketTTS ONNX bundle from {bundle_dir} with {self.num_steps} generation steps",
            )
            self._tts = PocketTTSOnnx(
                models_dir=self.model_dir,
                language=self.DEFAULT_LANGUAGE_BUNDLE,
                precision="int8",
                temperature=self.DEFAULT_TEMPERATURE,
                lsd_steps=self.num_steps,
            )
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

    def _pcm16_chunks(self, samples: np.ndarray, source_sample_rate: int) -> Iterable[bytes]:
        if source_sample_rate != self.TARGET_SAMPLE_RATE:
            raise RuntimeError(
                f"PocketTTS ONNX returned {source_sample_rate} Hz audio, expected {self.TARGET_SAMPLE_RATE} Hz"
            )

        pcm = (np.asarray(samples, dtype=np.float32) * 32767.0).clip(-32768, 32767).astype(np.int16)
        for start in range(0, len(pcm), self.STREAM_CHUNK_SAMPLES):
            yield pcm[start : start + self.STREAM_CHUNK_SAMPLES].tobytes()

    def _inter_pass_silence_chunks(self) -> Iterable[bytes]:
        silence_samples = int(round(self.TARGET_SAMPLE_RATE * (self.inter_pass_gap_ms / 1000.0)))
        if silence_samples <= 0:
            return

        silence = np.zeros((silence_samples,), dtype=np.float32)
        yield from self._pcm16_chunks(silence, self.TARGET_SAMPLE_RATE)

    def _group_text_for_inference(self, text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []

        sentences = [
            sentence.strip()
            for sentence in self.SENTENCE_SPLIT_PATTERN.split(normalized)
            if sentence.strip()
        ]
        if not sentences:
            return [normalized]

        return [
            " ".join(sentences[index : index + self.sentences_per_pass])
            for index in range(0, len(sentences), self.sentences_per_pass)
        ]

    @override
    def synthesize(self, text: str, voice: str) -> Iterable[bytes]:
        if not text.strip():
            return

        tts = self._load_model()
        reference_audio_path = self._resolve_reference_audio_path(self.reference_audio_path, voice)
        text_passes = self._group_text_for_inference(text)

        with self._synthesis_lock:
            yielded_audio = False
            log(
                "info",
                "Streaming PocketTTS audio for "
                f"{len(text)} characters across {len(text_passes)} inference pass(es) "
                f"with up to {self.sentences_per_pass} sentence(s) each",
            )
            for pass_index, text_pass in enumerate(text_passes):
                pass_yielded_audio = False
                for audio_chunk in tts.stream(text_pass, voice=reference_audio_path):
                    chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
                    if chunk.size == 0:
                        continue
                    yielded_audio = True
                    pass_yielded_audio = True
                    for pcm_chunk in self._pcm16_chunks(chunk, tts.sample_rate):
                        yield pcm_chunk

                if pass_yielded_audio and pass_index < len(text_passes) - 1:
                    yield from self._inter_pass_silence_chunks()

            if not yielded_audio:
                raise RuntimeError("PocketTTS returned no audio")


class PocketTTSPlugin(PluginBase):
    """Plugin providing PocketTTS services."""

    def __init__(self, plugin_manifest: PluginManifest):
        super().__init__(plugin_manifest)

        self.plugin_dir = PLUGIN_DIR
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
                            ParagraphSetting(
                                key="sentences_per_pass_help",
                                label=None,
                                type="paragraph",
                                readonly=False,
                                placeholder=None,
                                content=(
                                    "Higher values let Pocket TTS keep more sentence context together, "
                                    "which can sound more natural. If you pass too much text in one "
                                    "inference pass, the model can break down, so keep this value modest."
                                ),
                            ),
                            NumericalSetting(
                                key="sentences_per_pass",
                                label="Sentences per inference pass",
                                type="number",
                                readonly=False,
                                placeholder="2",
                                default_value=2,
                                min_value=1,
                                max_value=20,
                                step=1,
                            ),
                            NumericalSetting(
                                key="inter_pass_gap_ms",
                                label="Gap between passes (ms)",
                                type="number",
                                readonly=False,
                                placeholder="150",
                                default_value=150,
                                min_value=0,
                                max_value=2000,
                                step=25,
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
            inter_pass_gap_ms = int(
                settings.get("inter_pass_gap_ms", PocketTTSModel.DEFAULT_INTER_PASS_GAP_MS)
            )
            sentences_per_pass = int(
                settings.get("sentences_per_pass", PocketTTSModel.DEFAULT_SENTENCES_PER_PASS)
            )
            return PocketTTSModel(
                plugin_dir=self.plugin_dir,
                model_dir=self.model_dir,
                reference_audio_path=reference_audio_path,
                num_steps=num_steps,
                inter_pass_gap_ms=inter_pass_gap_ms,
                sentences_per_pass=sentences_per_pass,
            )

        raise ValueError(f"Unknown PocketTTS provider: {provider_id}")


if __name__ == "__main__":
    plugin_manifest = PluginManifest(
        name="Pocket TTS Plugin",
        version="0.0.7",
        author="COVAS:NEXT",
        description="Pocket TTS Plugin for COVAS:NEXT",
    )
    plugin = PocketTTSPlugin(plugin_manifest)
    try:
        plugin.create_model("pocket-tts", {})
        log("info", "Pocket TTS Plugin initialized successfully.")
    except Exception as exc:
        log("error", f"Failed to initialize Pocket TTS Plugin: {exc}")
