from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def get_audio_info(path: str | Path) -> tuple[int, int, int, float]:
    info = sf.info(str(path))
    duration_sec = info.frames / info.samplerate if info.samplerate else 0.0
    return info.samplerate, info.frames, info.channels, float(duration_sec)


def read_audio_mono(
    path: str | Path,
    target_sample_rate: int | None = None,
) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        mono = resample_audio(mono, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return mono.astype(np.float32, copy=False), sample_rate


def write_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="PCM_16")


def resample_audio(
    audio: np.ndarray,
    original_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if original_sample_rate == target_sample_rate:
        return audio.astype(np.float32, copy=False)
    gcd = np.gcd(original_sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = original_sample_rate // gcd
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))


def slice_audio(
    audio: np.ndarray,
    sample_rate: int,
    offset_sec: float,
    duration_sec: float,
) -> np.ndarray:
    start = max(0, int(round(offset_sec * sample_rate)))
    length = max(1, int(round(duration_sec * sample_rate)))
    end = min(audio.shape[0], start + length)
    chunk = audio[start:end]
    if chunk.shape[0] >= length:
        return chunk.astype(np.float32, copy=False)

    padded = np.zeros(length, dtype=np.float32)
    padded[: chunk.shape[0]] = chunk
    return padded


def tile_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    if audio.shape[0] == target_length:
        return audio.astype(np.float32, copy=False)
    if audio.shape[0] > target_length:
        return audio[:target_length].astype(np.float32, copy=False)
    repeats = int(np.ceil(target_length / max(1, audio.shape[0])))
    tiled = np.tile(audio, repeats)
    return tiled[:target_length].astype(np.float32, copy=False)


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    clean_rms = max(rms(clean), eps)
    noise_rms = max(rms(noise), eps)
    desired_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (desired_noise_rms / noise_rms)
    noisy = clean + scaled_noise
    return noisy.astype(np.float32, copy=False), scaled_noise.astype(np.float32, copy=False)


def normalize_triplet(
    clean: np.ndarray,
    noise: np.ndarray,
    noisy: np.ndarray,
    target_peak: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    peak = float(max(np.max(np.abs(clean)), np.max(np.abs(noise)), np.max(np.abs(noisy)), 1e-8))
    scale = target_peak / peak
    return (
        (clean * scale).astype(np.float32, copy=False),
        (noise * scale).astype(np.float32, copy=False),
        (noisy * scale).astype(np.float32, copy=False),
    )


def apply_rir(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if rir.size == 0:
        return clean.astype(np.float32, copy=False)
    rir = rir.astype(np.float32, copy=False)
    rir = rir / max(np.sqrt(np.sum(rir**2, dtype=np.float64)), 1e-8)
    convolved = fftconvolve(clean, rir, mode="full")
    return convolved[: clean.shape[0]].astype(np.float32, copy=False)
