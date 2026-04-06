from __future__ import annotations

import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .manifests import load_manifest, write_jsonl


@dataclass
class BaselineResult:
    id: str
    noisy_path: str
    enhanced_path: str
    clean_path: str | None


def apply_identity(noisy_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(noisy_path, output_path)


def apply_shell_command(command_template: str, noisy_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = command_template.format(
        input=shlex.quote(str(noisy_path)),
        output=shlex.quote(str(output_path)),
    )
    subprocess.run(command, shell=True, check=True)


def run_baseline(
    rendered_manifest_path: Path,
    output_dir: Path,
    mode: str,
    command_template: str | None = None,
) -> Path:
    entries = load_manifest(rendered_manifest_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, str | None]] = []

    for entry in entries:
        noisy_path = Path(entry["noisy_path"])
        clean_path = entry.get("clean_path")
        enhanced_path = output_dir / f"{entry['id']}.wav"

        if mode == "identity":
            apply_identity(noisy_path, enhanced_path)
        elif mode == "command":
            if not command_template:
                raise ValueError("Для режима 'command' требуется --command-template")
            apply_shell_command(command_template, noisy_path, enhanced_path)
        else:
            raise ValueError(f"Неизвестный baseline mode: {mode}")

        results.append(
            {
                "id": entry["id"],
                "noisy_path": str(noisy_path),
                "clean_path": clean_path,
                "enhanced_path": str(enhanced_path.resolve()),
            }
        )

    result_manifest = output_dir / "enhanced_manifest.jsonl"
    write_jsonl(result_manifest, results)
    return result_manifest
