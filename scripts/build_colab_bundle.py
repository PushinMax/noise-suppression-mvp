from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REQUIRED_PATHS = [
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "configs/colab_first_result.example.yaml",
    "docs/first_colab_run_plan.md",
    "notebooks/first_colab_run.ipynb",
    "scripts/prepare_first_colab_dataset.py",
    "src/noise_suppression/__init__.py",
    "src/noise_suppression/audio.py",
    "src/noise_suppression/baselines.py",
    "src/noise_suppression/cli.py",
    "src/noise_suppression/manifests.py",
    "src/noise_suppression/metrics.py",
    "src/noise_suppression/modeling.py",
    "src/noise_suppression/plans.py",
    "src/noise_suppression/training.py",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Собрать минимальный zip-бандл для запуска Colab notebook."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/colab_bundle.zip"),
        help="Путь к итоговому zip-файлу.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing = [rel for rel in REQUIRED_PATHS if not (project_root / rel).exists()]
    if missing:
        raise FileNotFoundError(
            "Не найдены обязательные файлы для Colab bundle:\n- " + "\n- ".join(missing)
        )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel_path in REQUIRED_PATHS:
            abs_path = project_root / rel_path
            archive.write(abs_path, arcname=rel_path)

    print(f"Colab bundle создан: {output_path}")
    print("Внутри архива:")
    for rel_path in REQUIRED_PATHS:
        print(f"  - {rel_path}")


if __name__ == "__main__":
    main()
