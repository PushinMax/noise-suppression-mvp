from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Подготовка маленького набора данных для первого Colab-run."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-clean", type=int, default=120)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-duration-sec", type=float, default=1.5)
    parser.add_argument("--max-duration-sec", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    current_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current_peak <= 1e-8:
        return audio
    return (audio * (peak / current_peak)).astype(np.float32, copy=False)


def save_clean_subset(
    clean_dir: Path,
    num_clean: int,
    sample_rate: int,
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[Path]:
    clean_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset = load_dataset(
            "google/fleurs",
            "ru_ru",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    except Exception as exc:
        raise RuntimeError(
            "Не удалось загрузить google/fleurs ru_ru. Для этого MVP нужен "
            "datasets>=3.6.0,<4.0.0: datasets 4.x больше не поддерживает "
            "старые loading scripts, а FLEURS пока использует fleurs.py. "
            "В Colab заново выполните установку зависимостей через "
            "`uv sync --extra train --extra data --extra dev`."
        ) from exc

    saved_paths: list[Path] = []
    for idx, example in enumerate(dataset):
        audio = np.asarray(example["audio"]["array"], dtype=np.float32)
        duration_sec = audio.shape[0] / sample_rate
        if duration_sec < min_duration_sec or duration_sec > max_duration_sec:
            continue

        transcription = str(example.get("transcription") or "").strip()
        if not transcription:
            continue

        speaker_id = str(example.get("speaker_id", f"speaker_{idx:05d}"))
        speaker_dir = clean_dir / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        stem = f"fleurs_ru_{len(saved_paths):04d}"
        wav_path = speaker_dir / f"{stem}.wav"
        txt_path = speaker_dir / f"{stem}.txt"

        sf.write(wav_path, normalize(audio), sample_rate)
        txt_path.write_text(transcription, encoding="utf-8")
        saved_paths.append(wav_path)

        if len(saved_paths) >= num_clean:
            break

    if len(saved_paths) < num_clean:
        raise RuntimeError(
            "Удалось подготовить только "
            f"{len(saved_paths)} clean examples из запрошенных {num_clean}."
        )

    return saved_paths


def make_colored_noise(rng: np.random.Generator, num_samples: int, kernel_size: int) -> np.ndarray:
    noise = rng.standard_normal(num_samples).astype(np.float32)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(noise, kernel, mode="same").astype(np.float32)


def create_noise_pool(
    noise_dir: Path,
    clean_paths: list[Path],
    sample_rate: int,
    seed: int,
) -> list[Path]:
    noise_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    duration_sec = 20
    num_samples = duration_sec * sample_rate
    noise_paths: list[Path] = []

    fan = make_colored_noise(rng, num_samples, kernel_size=1024)
    fan = normalize(fan * 0.08)
    fan_path = noise_dir / "fan_like.wav"
    sf.write(fan_path, fan, sample_rate)
    noise_paths.append(fan_path)

    keyboard = 0.01 * rng.standard_normal(num_samples).astype(np.float32)
    positions = rng.integers(low=0, high=num_samples - 300, size=64)
    for pos in positions:
        end = min(pos + 220, num_samples)
        keyboard[pos:end] += 0.15 * np.hanning(end - pos).astype(np.float32)
    keyboard = normalize(keyboard)
    keyboard_path = noise_dir / "keyboard_like.wav"
    sf.write(keyboard_path, keyboard, sample_rate)
    noise_paths.append(keyboard_path)

    traffic_low = make_colored_noise(rng, num_samples, kernel_size=1600)
    traffic_mid = make_colored_noise(rng, num_samples, kernel_size=64)
    traffic = normalize(0.7 * traffic_low + 0.3 * traffic_mid)
    traffic_path = noise_dir / "traffic_like.wav"
    sf.write(traffic_path, 0.08 * traffic, sample_rate)
    noise_paths.append(traffic_path)

    cafe = 0.008 * rng.standard_normal(num_samples).astype(np.float32)
    offsets = rng.integers(
        low=0,
        high=max(1, num_samples - sample_rate),
        size=min(12, len(clean_paths)),
    )
    for clean_path, offset in zip(clean_paths[:12], offsets, strict=False):
        speech, sr = sf.read(clean_path, dtype="float32")
        if sr != sample_rate:
            raise RuntimeError("Для babble-noise ожидалась единая частота дискретизации.")
        speech = speech[: min(speech.shape[0], num_samples - offset)]
        cafe[offset : offset + speech.shape[0]] += 0.08 * speech
    cafe = normalize(cafe)
    cafe_path = noise_dir / "cafe_babble_like.wav"
    sf.write(cafe_path, cafe, sample_rate)
    noise_paths.append(cafe_path)

    return noise_paths


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    clean_dir = output_root / "clean"
    noise_dir = output_root / "noise"

    clean_paths = save_clean_subset(
        clean_dir=clean_dir,
        num_clean=args.num_clean,
        sample_rate=args.sample_rate,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )
    noise_paths = create_noise_pool(
        noise_dir=noise_dir,
        clean_paths=clean_paths,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )

    print(f"Готово. Clean files: {len(clean_paths)}")
    print(f"Готово. Noise files: {len(noise_paths)}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
