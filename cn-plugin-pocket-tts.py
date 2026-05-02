"""Pocket TTS plugin for COVAS:NEXT using a vendored ONNX runtime."""

from typing import override, Iterable, Any, Optional
import os
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
    DEFAULT_MAX_TOKENS = 50
    TARGET_SAMPLE_RATE = 24000
    STREAM_CHUNK_SAMPLES = 2400

    def __init__(
        self,
        plugin_dir: str,
        model_dir: str,
        reference_audio_path: str,
        num_steps: int = 2,
        inter_pass_gap_ms: int = DEFAULT_INTER_PASS_GAP_MS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__("pocket-tts")
        self.plugin_dir = plugin_dir
        self.model_dir = model_dir
        self.reference_audio_path = reference_audio_path
        self.num_steps = max(int(num_steps), 1)
        self.inter_pass_gap_ms = max(int(inter_pass_gap_ms), 0)
        self.max_tokens = max(int(max_tokens), 1)

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

    def _normalize_user_path(self, path: str, relative_base_dir: Optional[str] = None) -> Optional[str]:
        value = (path or "").strip()
        if not value:
            return None

        value = os.path.expanduser(os.path.expandvars(value))
        if not os.path.isabs(value):
            base_dir = relative_base_dir or self.plugin_dir
            value = os.path.abspath(os.path.join(base_dir, value))
        return value

    def _reference_audio_path_candidates(
        self,
        path: Optional[str],
        relative_base_dir: Optional[str] = None,
    ) -> list[str]:
        if not path:
            return []

        candidate = self._normalize_user_path(path, relative_base_dir=relative_base_dir)
        if candidate is None:
            return []

        candidates = [candidate]
        if not os.path.splitext(candidate)[1]:
            candidates.append(f"{candidate}.wav")
        return candidates

    def _resolve_reference_audio_candidate(
        self,
        path: Optional[str],
        relative_base_dir: Optional[str] = None,
    ) -> Optional[str]:
        for candidate in self._reference_audio_path_candidates(
            path,
            relative_base_dir=relative_base_dir,
        ):
            if os.path.isfile(candidate):
                return candidate

            # Older plugin versions defaulted to bria.wav. If a persisted config still
            # points there, transparently fall back to the new CC0 bundled voice.
            if os.path.basename(candidate).lower() == "bria.wav":
                selfie_candidate = os.path.join(os.path.dirname(candidate), "selfie.wav")
                if os.path.isfile(selfie_candidate):
                    return selfie_candidate

            if os.path.isdir(candidate):
                for preferred_name in ("nova.wav", "selfie.wav", "bria.wav"):
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

    def _reference_audio_base_dir(self, configured_path: str, fallback_path: str) -> str:
        configured_candidate = self._normalize_user_path(configured_path)
        if configured_candidate:
            if os.path.isdir(configured_candidate):
                return configured_candidate
            return os.path.dirname(configured_candidate)

        return os.path.dirname(fallback_path)

    def _resolve_reference_audio_path(self, configured_path: str, requested_voice: str) -> str:
        default_voice_dir = os.path.join(self.plugin_dir, "assets", "voices")
        fallback_path = self._resolve_reference_audio_candidate(configured_path)
        if fallback_path is None:
            fallback_path = self._resolve_reference_audio_candidate(default_voice_dir)

        if fallback_path is None:
            raise FileNotFoundError(
                "Could not resolve a PocketTTS fallback voice file. "
                f"Checked configured path '{configured_path}' and '{default_voice_dir}'."
            )

        requested_voice_base_dir = self._reference_audio_base_dir(configured_path, fallback_path)
        resolved_voice = self._resolve_reference_audio_candidate(
            requested_voice,
            relative_base_dir=requested_voice_base_dir,
        )
        if resolved_voice is not None:
            normalized_requested = self._normalize_user_path(
                requested_voice,
                relative_base_dir=requested_voice_base_dir,
            )
            if requested_voice and resolved_voice != normalized_requested:
                log("info", f"Using reference audio '{resolved_voice}'")
            return resolved_voice

        return fallback_path

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

    def _find_boundary_indices(self, list_of_tokens: list[int], boundary_tokens: list[int]) -> list[int]:
        indices = [0]
        previous_was_boundary = False
        for index, token in enumerate(list_of_tokens):
            if token in boundary_tokens:
                previous_was_boundary = True
            else:
                if previous_was_boundary:
                    indices.append(index)
                previous_was_boundary = False
        indices.append(len(list_of_tokens))
        return indices

    def _segments_from_boundaries(
        self,
        tokenizer: spm.SentencePieceProcessor,
        list_of_tokens: list[int],
        boundary_indices: list[int],
    ) -> list[tuple[int, str]]:
        segments = []
        for index in range(len(boundary_indices) - 1):
            start = boundary_indices[index]
            end = boundary_indices[index + 1]
            text = tokenizer.Decode(list_of_tokens[start:end])
            segments.append((end - start, text))
        return segments

    def _group_text_for_inference(self, text: str) -> list[str]:
        tts = self._load_model()
        prepared_text, _ = tts._prepare_text_prompt(text)
        prepared_text = prepared_text.strip()
        if not prepared_text:
            return []

        tokenizer = tts.tokenizer
        list_of_tokens = tokenizer.Encode(prepared_text)

        end_of_sentence_tokens = tokenizer.Encode(".!...?")[1:]
        sentence_boundaries = self._find_boundary_indices(list_of_tokens, end_of_sentence_tokens)
        token_count_and_sentences = self._segments_from_boundaries(
            tokenizer,
            list_of_tokens,
            sentence_boundaries,
        )

        fallback_tokens = tokenizer.Encode(",;:")[1:]
        refined_segments: list[tuple[int, str]] = []
        for token_count, sentence in token_count_and_sentences:
            if token_count <= self.max_tokens:
                refined_segments.append((token_count, sentence))
                continue

            sub_tokens = tokenizer.Encode(sentence.strip())
            sub_boundaries = self._find_boundary_indices(sub_tokens, fallback_tokens)
            sub_segments = self._segments_from_boundaries(tokenizer, sub_tokens, sub_boundaries)
            if len(sub_segments) > 1:
                refined_segments.extend(sub_segments)
            else:
                refined_segments.append((token_count, sentence))

        chunks: list[str] = []
        current_chunk = ""
        current_chunk_token_count = 0
        for token_count, sentence in refined_segments:
            if current_chunk == "":
                current_chunk = sentence
                current_chunk_token_count = token_count
                continue

            if current_chunk_token_count + token_count > self.max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_chunk_token_count = token_count
            else:
                current_chunk += " " + sentence
                current_chunk_token_count += token_count

        if current_chunk != "":
            chunks.append(current_chunk.strip())

        for chunk in chunks:
            chunk_tokens = tokenizer.Encode(chunk.strip())
            if len(chunk_tokens) > self.max_tokens:
                log(
                    "warning",
                    f"PocketTTS chunk has {len(chunk_tokens)} tokens (max {self.max_tokens}), generation may skip words: '{chunk[:50]}...'",
                )

        return chunks

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
                f"with max {self.max_tokens} token(s) per pass",
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
        self.default_reference_audio_path = os.path.join(self.default_reference_audio_dir, "nova.wav")

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
                                    "Pocket TTS clones a voice from a reference clip. Configure a fallback voice "
                                    "file here. When a runtime voice name is provided, the plugin first tries it "
                                    "as an absolute path or as a path relative to the fallback file's directory. "
                                    "If the voice name has no extension, `.wav` is tried automatically."
                                ),
                            ),
                            TextSetting(
                                key="reference_audio_path",
                                label="Fallback voice file",
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
                                key="max_tokens_help",
                                label=None,
                                type="paragraph",
                                readonly=False,
                                placeholder=None,
                                content=(
                                    "Pocket TTS follows the upstream chunking logic here: it first splits on "
                                    "sentence punctuation, then falls back to commas, semicolons, and colons if a "
                                    "segment is too long. Each inference pass is packed up to this token limit."
                                ),
                            ),
                            NumericalSetting(
                                key="max_tokens",
                                label="Max tokens per inference pass",
                                type="number",
                                readonly=False,
                                placeholder="50",
                                default_value=50,
                                min_value=1,
                                max_value=400,
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
            max_tokens = int(
                settings.get("max_tokens", PocketTTSModel.DEFAULT_MAX_TOKENS)
            )
            return PocketTTSModel(
                plugin_dir=self.plugin_dir,
                model_dir=self.model_dir,
                reference_audio_path=reference_audio_path,
                num_steps=num_steps,
                inter_pass_gap_ms=inter_pass_gap_ms,
                max_tokens=max_tokens,
            )

        raise ValueError(f"Unknown PocketTTS provider: {provider_id}")


if __name__ == "__main__":
    plugin_manifest = PluginManifest(
        name="Pocket TTS Plugin",
        version="0.0.10",
        author="COVAS:NEXT",
        description="Pocket TTS Plugin for COVAS:NEXT",
    )
    plugin = PocketTTSPlugin(plugin_manifest)
    try:
        plugin.create_model("pocket-tts", {})
        log("info", "Pocket TTS Plugin initialized successfully.")
    except Exception as exc:
        log("error", f"Failed to initialize Pocket TTS Plugin: {exc}")
