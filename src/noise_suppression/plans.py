from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from .audio import (
    apply_rir,
    mix_at_snr,
    normalize_triplet,
    read_audio_mono,
    slice_audio,
    tile_or_trim,
    write_audio,
)
from .manifests import load_manifest, write_jsonl


@dataclass
class MixRecipe:
    clean_manifest: Path
    noise_manifest: Path
    rir_manifest: Path | None
    sample_rate: int
    num_examples: int
    min_duration_sec: float
    max_duration_sec: float
    snr_min_db: float
    snr_max_db: float
    focus_snr_min_db: float
    focus_snr_max_db: float
    focus_probability: float
    reverb_probability: float
    target_peak: float
    seed: int


def load_mix_recipe(path: Path) -> MixRecipe:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mixing = payload["mixing"]
    base_dir = path.parent
    return MixRecipe(
        clean_manifest=(base_dir / mixing["clean_manifest"]).resolve(),
        noise_manifest=(base_dir / mixing["noise_manifest"]).resolve(),
        rir_manifest=(
            (base_dir / mixing["rir_manifest"]).resolve()
            if mixing.get("rir_manifest")
            else None
        ),
        sample_rate=int(mixing["sample_rate"]),
        num_examples=int(mixing["num_examples"]),
        min_duration_sec=float(mixing["min_duration_sec"]),
        max_duration_sec=float(mixing["max_duration_sec"]),
        snr_min_db=float(mixing["snr_min_db"]),
        snr_max_db=float(mixing["snr_max_db"]),
        focus_snr_min_db=float(mixing["focus_snr_min_db"]),
        focus_snr_max_db=float(mixing["focus_snr_max_db"]),
        focus_probability=float(mixing["focus_probability"]),
        reverb_probability=float(mixing["reverb_probability"]),
        target_peak=float(mixing["target_peak"]),
        seed=int(payload.get("seed", 42)),
    )


def choose_offset(duration_sec: float, clip_duration_sec: float, rng: random.Random) -> float:
    if duration_sec <= clip_duration_sec:
        return 0.0
    return rng.uniform(0.0, duration_sec - clip_duration_sec)


def draw_snr(recipe: MixRecipe, rng: random.Random) -> float:
    if rng.random() < recipe.focus_probability:
        return rng.uniform(recipe.focus_snr_min_db, recipe.focus_snr_max_db)
    return rng.uniform(recipe.snr_min_db, recipe.snr_max_db)


def generate_mix_plan(recipe: MixRecipe, output_path: Path) -> int:
    rng = random.Random(recipe.seed)
    clean_rows = load_manifest(recipe.clean_manifest)
    noise_rows = load_manifest(recipe.noise_manifest)
    rir_rows = load_manifest(recipe.rir_manifest) if recipe.rir_manifest else []

    if not clean_rows:
        raise ValueError("Clean manifest пуст.")
    if not noise_rows:
        raise ValueError("Noise manifest пуст.")

    plan_rows: list[dict] = []
    for index in range(recipe.num_examples):
        clean = rng.choice(clean_rows)
        noise = rng.choice(noise_rows)
        duration_sec = rng.uniform(recipe.min_duration_sec, recipe.max_duration_sec)
        rir = None
        if rir_rows and rng.random() < recipe.reverb_probability:
            rir = rng.choice(rir_rows)

        plan_rows.append(
            {
                "id": f"mix-{index:06d}",
                "sample_rate": recipe.sample_rate,
                "duration_sec": round(duration_sec, 6),
                "snr_db": round(draw_snr(recipe, rng), 4),
                "target_peak": recipe.target_peak,
                "clean_id": clean["id"],
                "clean_path": clean["path"],
                "clean_offset_sec": round(
                    choose_offset(float(clean["duration_sec"]), duration_sec, rng), 6
                ),
                "noise_id": noise["id"],
                "noise_path": noise["path"],
                "noise_offset_sec": round(
                    choose_offset(float(noise["duration_sec"]), duration_sec, rng), 6
                ),
                "rir_id": None if rir is None else rir["id"],
                "rir_path": None if rir is None else rir["path"],
            }
        )

    write_jsonl(output_path, plan_rows)
    return len(plan_rows)


def render_mix_plan(
    plan_path: Path,
    output_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
) -> tuple[Path, int]:
    with plan_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit is not None:
        rows = rows[:limit]

    clean_dir = output_dir / "clean"
    noise_dir = output_dir / "noise"
    noisy_dir = output_dir / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    rendered_rows: list[dict] = []
    for row in rows:
        clean_path = clean_dir / f"{row['id']}.wav"
        noise_path = noise_dir / f"{row['id']}.wav"
        noisy_path = noisy_dir / f"{row['id']}.wav"
        if not overwrite and clean_path.exists() and noise_path.exists() and noisy_path.exists():
            rendered_rows.append(
                {
                    "id": row["id"],
                    "sample_rate": row["sample_rate"],
                    "duration_sec": row["duration_sec"],
                    "snr_db": row["snr_db"],
                    "clean_path": str(clean_path.resolve()),
                    "noise_path": str(noise_path.resolve()),
                    "noisy_path": str(noisy_path.resolve()),
                }
            )
            continue

        sample_rate = int(row["sample_rate"])
        clean_audio, _ = read_audio_mono(row["clean_path"], target_sample_rate=sample_rate)
        noise_audio, _ = read_audio_mono(row["noise_path"], target_sample_rate=sample_rate)

        clean_chunk = slice_audio(
            clean_audio,
            sample_rate=sample_rate,
            offset_sec=float(row["clean_offset_sec"]),
            duration_sec=float(row["duration_sec"]),
        )
        noise_chunk = slice_audio(
            noise_audio,
            sample_rate=sample_rate,
            offset_sec=float(row["noise_offset_sec"]),
            duration_sec=float(row["duration_sec"]),
        )
        noise_chunk = tile_or_trim(noise_chunk, clean_chunk.shape[0])

        if row.get("rir_path"):
            rir_audio, _ = read_audio_mono(row["rir_path"], target_sample_rate=sample_rate)
            clean_chunk = apply_rir(clean_chunk, rir_audio)

        noisy_chunk, scaled_noise = mix_at_snr(clean_chunk, noise_chunk, float(row["snr_db"]))
        clean_chunk, scaled_noise, noisy_chunk = normalize_triplet(
            clean_chunk,
            scaled_noise,
            noisy_chunk,
            float(row["target_peak"]),
        )

        write_audio(clean_path, clean_chunk, sample_rate)
        write_audio(noise_path, scaled_noise, sample_rate)
        write_audio(noisy_path, noisy_chunk, sample_rate)
        rendered_rows.append(
            {
                "id": row["id"],
                "sample_rate": sample_rate,
                "duration_sec": row["duration_sec"],
                "snr_db": row["snr_db"],
                "clean_path": str(clean_path.resolve()),
                "noise_path": str(noise_path.resolve()),
                "noisy_path": str(noisy_path.resolve()),
            }
        )

    rendered_manifest_path = output_dir / f"rendered_{plan_path.stem}.jsonl"
    write_jsonl(rendered_manifest_path, rendered_rows)
    return rendered_manifest_path, len(rendered_rows)
