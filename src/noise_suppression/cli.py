from __future__ import annotations

import importlib
import json
import platform
import random
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .baselines import run_baseline
from .manifests import (
    build_manifest,
    load_manifest,
    summarize_manifest,
    write_jsonl,
)
from .metrics import score_si_sdr_manifest, score_wer
from .plans import generate_mix_plan, load_mix_recipe, render_mix_plan

app = typer.Typer(help="Инструменты для MVP по шумоподавлению русской речи.")
env_app = typer.Typer(help="Проверка локального окружения.")
manifest_app = typer.Typer(help="Построение и сводка manifest-файлов.")
mix_app = typer.Typer(help="Планирование и рендер synthetic mixtures.")
baseline_app = typer.Typer(help="Применение baseline-обработки.")
metrics_app = typer.Typer(help="Подсчет стартовых метрик.")
train_app = typer.Typer(help="Tiny training pipeline для smoke test и Colab MVP.")

app.add_typer(env_app, name="env")
app.add_typer(manifest_app, name="manifest")
app.add_typer(mix_app, name="mix")
app.add_typer(baseline_app, name="baseline")
app.add_typer(metrics_app, name="metrics")
app.add_typer(train_app, name="train")

console = Console()


@env_app.command("check")
def env_check() -> None:
    table = Table(title="Environment Check")
    table.add_column("Параметр")
    table.add_column("Значение")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("Platform", platform.platform())
    table.add_row("Machine", platform.machine())
    table.add_row("uv", shutil.which("uv") or "not found")
    table.add_row("ffmpeg", shutil.which("ffmpeg") or "not found")
    table.add_row("sox", shutil.which("sox") or "not found")

    for package_name in ["numpy", "soundfile", "scipy", "jiwer", "typer"]:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", None) or metadata.version(package_name)
        except ImportError:
            version = "not installed"
        table.add_row(package_name, str(version))

    console.print(table)


@manifest_app.command("build")
def manifest_build(
    source_dir: Path,
    output_path: Path,
    kind: str = typer.Option(..., help="clean | noise | rir | real"),
    speaker_depth: int | None = typer.Option(
        None,
        help="Индекс части относительного пути, из которой брать speaker_id.",
    ),
    transcript_suffix: str = typer.Option(".txt", help="Суффикс sidecar transcript-файла."),
) -> None:
    count = build_manifest(
        source_dir=source_dir,
        output_path=output_path,
        kind=kind,
        speaker_depth=speaker_depth,
        transcript_suffix=transcript_suffix,
    )
    console.print(f"Сохранен manifest: [bold]{output_path}[/bold] ({count} файлов)")


@manifest_app.command("summarize")
def manifest_summarize(manifest_path: Path) -> None:
    summary = summarize_manifest(load_manifest(manifest_path))
    table = Table(title=f"Manifest Summary: {manifest_path.name}")
    table.add_column("Поле")
    table.add_column("Значение")
    for key, value in summary.items():
        if isinstance(value, dict):
            table.add_row(key, json.dumps(value, ensure_ascii=False, sort_keys=True))
        else:
            table.add_row(key, str(value))
    console.print(table)


@manifest_app.command("split")
def manifest_split(
    manifest_path: Path,
    train_output_path: Path,
    val_output_path: Path,
    val_ratio: float = typer.Option(0.2, min=0.01, max=0.99, help="Доля валидации."),
    seed: int = typer.Option(42, help="Seed для детерминированного split."),
) -> None:
    rows = load_manifest(manifest_path)
    rows = sorted(rows, key=lambda item: item["id"])
    randomizer = random.Random(f"{seed}:{len(rows)}")
    randomizer.shuffle(rows)
    val_count = max(1, int(round(len(rows) * val_ratio)))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    if not train_rows:
        raise typer.BadParameter("После split train-подмножество оказалось пустым.")
    write_jsonl(train_output_path, train_rows)
    write_jsonl(val_output_path, val_rows)
    console.print(
        f"Split готов: train={len(train_rows)} -> [bold]{train_output_path}[/bold], "
        f"val={len(val_rows)} -> [bold]{val_output_path}[/bold]"
    )


@mix_app.command("plan")
def mix_plan(config_path: Path, output_path: Path) -> None:
    recipe = load_mix_recipe(config_path)
    count = generate_mix_plan(recipe=recipe, output_path=output_path)
    console.print(f"Сохранен mix plan: [bold]{output_path}[/bold] ({count} смесей)")


@mix_app.command("render")
def mix_render(
    plan_path: Path,
    output_dir: Path,
    limit: int | None = typer.Option(None, help="Ограничить число рендеримых примеров."),
    overwrite: bool = typer.Option(False, help="Перезаписывать существующие файлы."),
) -> None:
    rendered_manifest_path, count = render_mix_plan(
        plan_path=plan_path,
        output_dir=output_dir,
        limit=limit,
        overwrite=overwrite,
    )
    console.print(
        f"Срендерен набор: [bold]{output_dir}[/bold] ({count} файлов), manifest: "
        f"[bold]{rendered_manifest_path}[/bold]"
    )


@baseline_app.command("apply")
def baseline_apply(
    rendered_manifest_path: Path,
    output_dir: Path,
    mode: str = typer.Option("identity", help="identity | command"),
    command_template: str | None = typer.Option(
        None,
        help="Шаблон shell-команды с плейсхолдерами {input} и {output}.",
    ),
) -> None:
    manifest_path = run_baseline(
        rendered_manifest_path=rendered_manifest_path,
        output_dir=output_dir,
        mode=mode,
        command_template=command_template,
    )
    console.print(f"Сохранен enhanced manifest: [bold]{manifest_path}[/bold]")


@metrics_app.command("si-sdr")
def metrics_si_sdr(
    rendered_manifest_path: Path,
    estimate_dir: Path,
    output_path: Annotated[
        Path | None,
        typer.Option(help="Необязательный JSON output."),
    ] = None,
) -> None:
    summary = score_si_sdr_manifest(rendered_manifest_path, estimate_dir)
    table = Table(title="SI-SDR Summary")
    table.add_column("Поле")
    table.add_column("Значение")
    for key, value in summary.items():
        table.add_row(key, f"{value:.6f}" if isinstance(value, float) else str(value))
    console.print(table)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


@metrics_app.command("wer")
def metrics_wer(
    references_path: Path,
    hypotheses_path: Path,
    output_path: Annotated[
        Path | None,
        typer.Option(help="Необязательный JSON output."),
    ] = None,
) -> None:
    summary = score_wer(references_path, hypotheses_path)
    table = Table(title="WER Summary")
    table.add_column("Поле")
    table.add_column("Значение")
    for key, value in summary.items():
        table.add_row(key, f"{value:.6f}" if isinstance(value, float) else str(value))
    console.print(table)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


@train_app.command("fit")
def train_fit(config_path: Path) -> None:
    from .training import train_from_config

    summary = train_from_config(config_path)
    table = Table(title="Training Summary")
    table.add_column("Поле")
    table.add_column("Значение")
    for key, value in summary.items():
        table.add_row(key, f"{value:.6f}" if isinstance(value, float) else str(value))
    console.print(table)


@train_app.command("infer")
def train_infer(
    checkpoint_path: Path,
    rendered_manifest_path: Path,
    output_dir: Path,
    device: str = typer.Option("auto", help="auto | cpu | mps | cuda"),
    limit: int | None = typer.Option(None, help="Ограничить число файлов для инференса."),
) -> None:
    from .training import enhance_manifest

    manifest_path = enhance_manifest(
        checkpoint_path=checkpoint_path,
        rendered_manifest_path=rendered_manifest_path,
        output_dir=output_dir,
        device=device,
        limit=limit,
    )
    console.print(f"Сохранен enhanced manifest: [bold]{manifest_path}[/bold]")


if __name__ == "__main__":
    app()
