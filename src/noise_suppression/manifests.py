from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .audio import get_audio_info

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_audio_paths(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def infer_tags(relative_path: Path) -> list[str]:
    tokens: list[str] = []
    for part in relative_path.parts:
        tokens.extend(re.split(r"[^0-9A-Za-zА-Яа-я_]+", part.lower()))
    suffix = relative_path.suffix.lower().lstrip(".")
    return sorted({token for token in tokens if token and token != suffix})


def build_manifest(
    source_dir: Path,
    output_path: Path,
    kind: str,
    speaker_depth: int | None = None,
    transcript_suffix: str = ".txt",
) -> int:
    rows: list[dict] = []
    for audio_path in iter_audio_paths(source_dir):
        relative_path = audio_path.relative_to(source_dir)
        sample_rate, frames, channels, duration_sec = get_audio_info(audio_path)
        digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]
        transcript_path = audio_path.with_suffix(transcript_suffix)
        transcript = None
        if transcript_path.exists():
            transcript = transcript_path.read_text(encoding="utf-8").strip()

        speaker_id = None
        if speaker_depth is not None and speaker_depth < len(relative_path.parts):
            speaker_id = relative_path.parts[speaker_depth]

        rows.append(
            {
                "id": f"{kind}-{digest}",
                "kind": kind,
                "path": str(audio_path.resolve()),
                "relative_path": relative_path.as_posix(),
                "sample_rate": sample_rate,
                "duration_sec": round(duration_sec, 6),
                "num_samples": int(frames),
                "channels": int(channels),
                "speaker_id": speaker_id,
                "transcript": transcript,
                "tags": infer_tags(relative_path),
            }
        )

    write_jsonl(output_path, rows)
    return len(rows)


def summarize_manifest(rows: list[dict]) -> dict:
    durations = [float(row["duration_sec"]) for row in rows]
    sample_rates = Counter(int(row["sample_rate"]) for row in rows)
    speakers = {row["speaker_id"] for row in rows if row.get("speaker_id")}
    kinds = Counter(str(row["kind"]) for row in rows)

    return {
        "items": len(rows),
        "hours": round(sum(durations) / 3600.0, 3),
        "mean_duration_sec": round((sum(durations) / len(durations)) if durations else 0.0, 3),
        "sample_rates": dict(sample_rates),
        "kinds": dict(kinds),
        "speakers": len(speakers),
    }
