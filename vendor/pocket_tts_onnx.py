"""PocketTTS ONNX runtime with plugin-controlled chunking."""

from __future__ import annotations

import json
import os
import threading
import time
import wave
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

try:
    import soundfile as sf

    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import samplerate

    HAS_SAMPLERATE = True
except ImportError:
    HAS_SAMPLERATE = False


class PocketTTSOnnx:
    DEFAULT_LANGUAGE = "english_2026-04"
    VALID_PRECISIONS = ("int8", "fp32")
    TOKENS_PER_SECOND_ESTIMATE = 3.0
    GEN_SECONDS_PADDING = 2.0

    def __init__(
        self,
        models_dir: str,
        language: str = DEFAULT_LANGUAGE,
        tokenizer_path: Optional[str] = None,
        precision: str = "int8",
        device: str = "auto",
        temperature: float = 0.7,
        lsd_steps: int = 2,
        num_threads: int = max(1, (os.cpu_count() or 1) // 2),
    ):
        if precision not in self.VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {self.VALID_PRECISIONS}, got '{precision}'")
        if lsd_steps < 1:
            raise ValueError("lsd_steps must be at least 1")

        self.models_root = Path(models_dir)
        self.language = self._normalize_language(language)
        self.bundle_dir = self._resolve_bundle_dir(self.models_root, self.language)
        self.metadata = self._load_metadata(self.bundle_dir)

        self.precision = precision
        self.temperature = float(temperature)
        self.lsd_steps = int(lsd_steps)
        self.num_threads = max(1, int(num_threads))
        self.providers = self._get_providers(device)

        self.sample_rate = int(self.metadata["sample_rate"])
        self.frame_rate = float(self.metadata["frame_rate"])
        self.samples_per_frame = int(self.metadata["samples_per_frame"])
        self.frame_duration = self.samples_per_frame / self.sample_rate
        self.latent_dim = int(self.metadata["latent_dim"])
        self.conditioning_dim = int(self.metadata["conditioning_dim"])
        self.pad_with_spaces_for_short_inputs = bool(
            self.metadata.get("pad_with_spaces_for_short_inputs", False)
        )
        self.remove_semicolons = bool(self.metadata.get("remove_semicolons", False))
        self.model_recommended_frames_after_eos = self.metadata.get(
            "model_recommended_frames_after_eos"
        )
        self.insert_bos_before_voice = bool(self.metadata.get("insert_bos_before_voice", False))

        tokenizer_file = tokenizer_path or str(self.bundle_dir / self.metadata["tokenizer_file"])
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_file)

        self.bos_before_voice = None
        bos_file = self.metadata.get("bos_before_voice_file")
        if bos_file:
            self.bos_before_voice = np.load(self.bundle_dir / bos_file).astype(np.float32)

        self.flow_state_manifest = self.metadata["flow_lm_state_manifest"]
        self.mimi_state_manifest = self.metadata["mimi_state_manifest"]

        self._load_models()
        self._precompute_flow_buffers()
        self._voice_cache: dict[str, np.ndarray] = {}
        self._voice_state_cache: dict[str, dict[str, np.ndarray]] = {}
        self._voice_cache_lock = threading.Lock()

    @staticmethod
    def _normalize_language(language: str) -> str:
        if language == "english":
            return "english_2026-04"
        return language.replace("_2026_", "_2026-")

    @staticmethod
    def _resolve_bundle_dir(models_root: Path, language: str) -> Path:
        candidate = models_root / language
        if candidate.is_dir():
            return candidate
        if (models_root / "bundle.json").exists():
            return models_root
        raise FileNotFoundError(
            f"Could not find ONNX bundle for '{language}' under {models_root}."
        )

    @staticmethod
    def _load_metadata(bundle_dir: Path) -> dict:
        metadata_path = bundle_dir / "bundle.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing bundle metadata: {metadata_path}")
        return json.loads(metadata_path.read_text())

    def _get_providers(self, device: str) -> list[str]:
        if device == "cpu":
            return ["CPUExecutionProvider"]
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _make_session_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        return opts

    def _model_file(self, stem: str) -> str:
        if self.precision == "int8":
            quantized = self.bundle_dir / f"{stem}_int8.onnx"
            if quantized.exists():
                return quantized.name
        fp32 = self.bundle_dir / f"{stem}.onnx"
        if fp32.exists():
            return fp32.name
        raise FileNotFoundError(f"Missing ONNX file for {stem} in {self.bundle_dir}")

    def _load_models(self) -> None:
        opts = self._make_session_options()
        self.mimi_encoder = ort.InferenceSession(
            str(self.bundle_dir / "mimi_encoder.onnx"),
            sess_options=opts,
            providers=self.providers,
        )
        self.text_conditioner = ort.InferenceSession(
            str(self.bundle_dir / "text_conditioner.onnx"),
            sess_options=opts,
            providers=self.providers,
        )
        self.flow_lm_main = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("flow_lm_main")),
            sess_options=opts,
            providers=self.providers,
        )
        self.flow_lm_flow = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("flow_lm_flow")),
            sess_options=opts,
            providers=self.providers,
        )
        self.mimi_decoder = ort.InferenceSession(
            str(self.bundle_dir / self._model_file("mimi_decoder")),
            sess_options=opts,
            providers=self.providers,
        )

    def _precompute_flow_buffers(self) -> None:
        dt = 1.0 / self.lsd_steps
        self._st_buffers = []
        for index in range(self.lsd_steps):
            start = index / self.lsd_steps
            end = start + dt
            self._st_buffers.append(
                (
                    np.array([[start]], dtype=np.float32),
                    np.array([[end]], dtype=np.float32),
                )
            )

    @staticmethod
    def _numpy_dtype(dtype: str):
        return {
            "float32": np.float32,
            "float16": np.float16,
            "int64": np.int64,
            "bool": np.bool_,
        }[dtype]

    def _make_filled_array(self, shape: list[int], dtype, fill: str) -> np.ndarray:
        if fill == "nan":
            return np.full(shape, np.nan, dtype=dtype)
        if fill == "ones":
            return np.ones(shape, dtype=dtype)
        if fill == "empty":
            return np.empty(shape, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    def _init_state(self, manifest: list[dict]) -> dict[str, np.ndarray]:
        state = {}
        for entry in manifest:
            dtype = self._numpy_dtype(entry["dtype"])
            state[entry["input_name"]] = self._make_filled_array(
                entry["shape"], dtype=dtype, fill=entry["fill"]
            )
        return state

    @staticmethod
    def _clone_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in state.items()}

    def _update_state_from_outputs(
        self,
        state: dict[str, np.ndarray],
        result: list[np.ndarray],
        manifest: list[dict],
        output_offset: int,
    ) -> None:
        for entry in manifest:
            state[entry["input_name"]] = result[output_offset + entry["index"]]

    def _load_audio(self, path: Union[str, Path]) -> np.ndarray:
        audio_path = Path(path)
        if HAS_SOUNDFILE:
            audio, sample_rate = sf.read(str(audio_path), dtype="float32")
            if np.asarray(audio).ndim == 2:
                audio = np.asarray(audio, dtype=np.float32).mean(axis=1)
            else:
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        elif audio_path.suffix.lower() == ".wav":
            with wave.open(str(audio_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                raw_data = wav_file.readframes(-1)
                audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            raise ImportError("soundfile is required for non-WAV voice cloning inputs")

        if sample_rate != self.sample_rate:
            if not HAS_SAMPLERATE:
                raise ImportError("samplerate is required when voice inputs need resampling")
            ratio = float(self.sample_rate) / float(sample_rate)
            audio = samplerate.resample(audio, ratio, "sinc_best").astype(np.float32)

        return audio.reshape(1, 1, -1)

    def encode_voice(self, audio_path: Union[str, Path]) -> np.ndarray:
        audio = self._load_audio(audio_path)
        embeddings = self.mimi_encoder.run(None, {"audio": audio})[0]
        while embeddings.ndim > 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim < 3:
            embeddings = embeddings[None]
        return embeddings.astype(np.float32, copy=False)

    def _prepare_voice_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        prepared = np.asarray(embeddings, dtype=np.float32)
        while prepared.ndim > 3:
            prepared = prepared.squeeze(0)
        if prepared.ndim < 3:
            prepared = prepared.reshape(1, -1, prepared.shape[-1])
        if self.insert_bos_before_voice and self.bos_before_voice is not None:
            prepared = np.concatenate([self.bos_before_voice, prepared], axis=1)
        return prepared

    def _condition_with_voice_embeddings(self, embeddings: np.ndarray) -> dict[str, np.ndarray]:
        voice_embeddings = self._prepare_voice_embeddings(embeddings)
        state = self._init_state(self.flow_state_manifest)
        empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        result = self.flow_lm_main.run(
            None,
            {"sequence": empty_seq, "text_embeddings": voice_embeddings, **state},
        )
        self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)
        return state

    def prepare_voice_state(self, voice: Union[str, Path, np.ndarray]) -> dict[str, np.ndarray]:
        if isinstance(voice, np.ndarray):
            return self._condition_with_voice_embeddings(voice)

        voice_key = str(voice)
        with self._voice_cache_lock:
            cached_state = self._voice_state_cache.get(voice_key)
            if cached_state is not None:
                return self._clone_state(cached_state)
            cached_embeddings = self._voice_cache.get(voice_key)

        if cached_embeddings is None:
            voice_path = Path(voice_key)
            if not voice_path.exists():
                raise ValueError(f"Voice '{voice}' not found")
            cached_embeddings = self.encode_voice(voice_path)
            with self._voice_cache_lock:
                self._voice_cache[voice_key] = cached_embeddings

        state = self._condition_with_voice_embeddings(cached_embeddings)
        with self._voice_cache_lock:
            self._voice_state_cache[voice_key] = self._clone_state(state)
        return state

    def _prepare_text_prompt(self, text: str) -> tuple[str, int]:
        prepared = " ".join(text.strip().split())
        if not prepared:
            raise ValueError("Text cannot be empty")
        if self.remove_semicolons:
            prepared = prepared.replace(";", ",")

        word_count = len(prepared.split())
        frames_after_eos_guess = 3 if word_count <= 4 else 1

        if not prepared[0].isupper():
            prepared = prepared[0].upper() + prepared[1:]
        if prepared[-1].isalnum():
            prepared = prepared + "."
        if self.pad_with_spaces_for_short_inputs and word_count < 5:
            prepared = " " * 8 + prepared
        return prepared, frames_after_eos_guess

    def _tokenize(self, text: str) -> np.ndarray:
        prepared, _ = self._prepare_text_prompt(text)
        token_ids = self.tokenizer.Encode(prepared)
        return np.asarray(token_ids, dtype=np.int64).reshape(1, -1)

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self.TOKENS_PER_SECOND_ESTIMATE + self.GEN_SECONDS_PADDING
        return int(np.ceil(gen_len_sec * self.frame_rate))

    def _run_flow_lm(
        self,
        initial_state: dict[str, np.ndarray],
        text_ids: np.ndarray,
        max_frames: Optional[int],
        frames_after_eos: int,
    ) -> Generator[np.ndarray, None, None]:
        state = self._clone_state(initial_state)
        text_embeddings = self.text_conditioner.run(None, {"token_ids": text_ids})[0]
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings[None]

        empty_seq = np.zeros((1, 0, self.latent_dim), dtype=np.float32)
        empty_text = np.zeros((1, 0, self.conditioning_dim), dtype=np.float32)

        result = self.flow_lm_main.run(
            None,
            {"sequence": empty_seq, "text_embeddings": text_embeddings, **state},
        )
        self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)

        current = np.full((1, 1, self.latent_dim), np.nan, dtype=np.float32)
        eos_step = None
        frame_limit = max_frames or self._estimate_max_gen_len(text_ids.shape[1])
        dt = 1.0 / self.lsd_steps

        for step in range(frame_limit):
            result = self.flow_lm_main.run(
                None,
                {"sequence": current, "text_embeddings": empty_text, **state},
            )
            conditioning = result[0]
            eos_logit = result[1]
            self._update_state_from_outputs(state, result, self.flow_state_manifest, output_offset=2)

            if eos_logit[0][0] > -4.0 and eos_step is None:
                eos_step = step
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            if self.temperature > 0:
                std = np.sqrt(self.temperature)
                sample = np.random.normal(0.0, std, (1, self.latent_dim)).astype(np.float32)
            else:
                sample = np.zeros((1, self.latent_dim), dtype=np.float32)

            for start, end in self._st_buffers:
                flow = self.flow_lm_flow.run(
                    None,
                    {"c": conditioning, "s": start, "t": end, "x": sample},
                )[0]
                sample = sample + flow * dt

            latent = sample.reshape(1, 1, self.latent_dim)
            yield latent
            current = latent

    def decode_latents(self, latents: np.ndarray, chunk_size: int = 15) -> np.ndarray:
        state = self._init_state(self.mimi_state_manifest)
        audio_chunks = []
        for index in range(0, latents.shape[1], chunk_size):
            chunk = latents[:, index : index + chunk_size, :]
            result = self.mimi_decoder.run(None, {"latent": chunk, **state})
            audio_chunks.append(result[0].reshape(-1))
            self._update_state_from_outputs(state, result, self.mimi_state_manifest, output_offset=1)
        if not audio_chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(audio_chunks)

    def _decode_worker(
        self,
        latent_queue,
        audio_chunks: list[np.ndarray],
        decode_chunk_size: int = 12,
    ) -> None:
        mimi_state = self._init_state(self.mimi_state_manifest)
        buffered = []
        decoded = 0

        while True:
            item = latent_queue.get()
            if item is None:
                break
            buffered.append(item)

            if len(buffered) - decoded >= decode_chunk_size:
                chunk = np.concatenate(buffered[decoded : decoded + decode_chunk_size], axis=1)
                result = self.mimi_decoder.run(None, {"latent": chunk, **mimi_state})
                audio_chunks.append(result[0].reshape(-1))
                self._update_state_from_outputs(
                    mimi_state,
                    result,
                    self.mimi_state_manifest,
                    output_offset=1,
                )
                decoded += decode_chunk_size

        if decoded < len(buffered):
            chunk = np.concatenate(buffered[decoded:], axis=1)
            result = self.mimi_decoder.run(None, {"latent": chunk, **mimi_state})
            audio_chunks.append(result[0].reshape(-1))

    def generate(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
    ) -> np.ndarray:
        base_state = self.prepare_voice_state(voice)
        _, guess = self._prepare_text_prompt(text)
        effective_frames = (
            frames_after_eos
            if frames_after_eos is not None
            else (self.model_recommended_frames_after_eos or (guess + 2))
        )
        text_ids = self._tokenize(text)

        import queue

        latent_queue = queue.Queue()
        audio_chunks: list[np.ndarray] = []
        decoder = threading.Thread(
            target=self._decode_worker,
            args=(latent_queue, audio_chunks),
            daemon=True,
        )
        decoder.start()

        for latent in self._run_flow_lm(base_state, text_ids, max_frames, effective_frames):
            latent_queue.put(latent)
        latent_queue.put(None)
        decoder.join()

        if not audio_chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(audio_chunks)

    def stream(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: Optional[int] = None,
        frames_after_eos: Optional[int] = None,
        first_chunk_frames: int = 2,
        target_buffer_sec: float = 0.2,
        max_chunk_frames: int = 15,
    ) -> Generator[np.ndarray, None, None]:
        base_state = self.prepare_voice_state(voice)
        _, guess = self._prepare_text_prompt(text)
        effective_frames = (
            frames_after_eos
            if frames_after_eos is not None
            else (self.model_recommended_frames_after_eos or (guess + 2))
        )
        text_ids = self._tokenize(text)

        mimi_state = self._init_state(self.mimi_state_manifest)
        generated_latents = []
        decoded_frames = 0
        playback_start_time = None
        start_time = time.time()

        for latent in self._run_flow_lm(base_state, text_ids, max_frames, effective_frames):
            generated_latents.append(latent)
            pending = len(generated_latents) - decoded_frames
            chunk_size = 0

            if playback_start_time is None:
                if pending >= first_chunk_frames:
                    chunk_size = first_chunk_frames
            else:
                elapsed = time.time() - start_time
                audio_decoded_sec = decoded_frames * self.frame_duration
                playback_elapsed = elapsed - playback_start_time
                buffer_sec = audio_decoded_sec - playback_elapsed

                if buffer_sec < target_buffer_sec and pending >= 1:
                    chunk_size = min(pending, 3)
                elif pending >= max_chunk_frames:
                    chunk_size = max_chunk_frames

            if chunk_size > 0:
                latents_chunk = np.concatenate(
                    generated_latents[decoded_frames : decoded_frames + chunk_size],
                    axis=1,
                )
                result = self.mimi_decoder.run(None, {"latent": latents_chunk, **mimi_state})
                self._update_state_from_outputs(
                    mimi_state,
                    result,
                    self.mimi_state_manifest,
                    output_offset=1,
                )
                decoded_frames += chunk_size
                if playback_start_time is None:
                    playback_start_time = time.time() - start_time
                yield result[0].reshape(-1)

        if decoded_frames < len(generated_latents):
            latents_chunk = np.concatenate(generated_latents[decoded_frames:], axis=1)
            result = self.mimi_decoder.run(None, {"latent": latents_chunk, **mimi_state})
            yield result[0].reshape(-1)

    def __repr__(self) -> str:
        return (
            f"PocketTTSOnnx(language={self.language!r}, device={self.device!r}, "
            f"precision={self.precision!r}, temperature={self.temperature}, "
            f"lsd_steps={self.lsd_steps}, sample_rate={self.sample_rate})"
        )

    @property
    def device(self) -> str:
        if "CUDAExecutionProvider" in self.providers:
            return "cuda"
        return "cpu"
