from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from jiwer import wer

from .audio import read_audio_mono
from .manifests import load_manifest


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64, copy=False)
    estimate = estimate.astype(np.float64, copy=False)
    min_length = min(reference.shape[0], estimate.shape[0])
    if min_length == 0:
        raise ValueError("Пустой сигнал нельзя использовать для SI-SDR.")
    reference = reference[:min_length]
    estimate = estimate[:min_length]

    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    reference_energy = np.sum(reference**2)
    if reference_energy <= 1e-12:
        raise ValueError("Reference signal имеет почти нулевую энергию.")

    projection = np.dot(estimate, reference) * reference / reference_energy
    noise = estimate - projection
    noise_energy = np.sum(noise**2)
    if noise_energy <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(np.sum(projection**2) / noise_energy))


def score_si_sdr_manifest(rendered_manifest_path: Path, estimate_dir: Path) -> dict:
    rows = load_manifest(rendered_manifest_path)
    values: list[float] = []
    missing = 0

    for row in rows:
        estimate_path = estimate_dir / f"{row['id']}.wav"
        if not estimate_path.exists():
            missing += 1
            continue
        reference, sample_rate = read_audio_mono(row["clean_path"])
        estimate, _ = read_audio_mono(estimate_path, target_sample_rate=sample_rate)
        values.append(si_sdr(reference, estimate))

    mean_value = float(np.mean(values)) if values else float("nan")
    median_value = float(np.median(values)) if values else float("nan")
    return {
        "count": len(values),
        "missing": missing,
        "si_sdr_mean": mean_value,
        "si_sdr_median": median_value,
    }


def load_id_text_map(path: Path) -> dict[str, str]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row["id"]): str(row["text"]) for row in rows}


def score_wer(references_path: Path, hypotheses_path: Path) -> dict:
    references = load_id_text_map(references_path)
    hypotheses = load_id_text_map(hypotheses_path)
    common_ids = sorted(set(references) & set(hypotheses))
    if not common_ids:
        raise ValueError("Нет общих id между references и hypotheses.")

    ref_texts = [references[item_id] for item_id in common_ids]
    hyp_texts = [hypotheses[item_id] for item_id in common_ids]
    return {
        "count": len(common_ids),
        "missing_references": len(set(hypotheses) - set(references)),
        "missing_hypotheses": len(set(references) - set(hypotheses)),
        "wer": float(wer(ref_texts, hyp_texts)),
    }
