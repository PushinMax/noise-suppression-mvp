# Noise Suppression MVP

Локальный каркас проекта для MVP по шумоподавлению русской речи.

## Что уже есть

- обзор задачи: `docs/noise_suppression_overview.md`
- MVP-спецификация: `docs/noise_suppression_mvp.md`
- стек данных: `docs/noise_suppression_dataset_stack.md`
- план первого запуска в Colab: `docs/first_colab_run_plan.md`
- инструкция по запуску в Colab: `docs/colab_launch_instructions.md`
- первый Colab notebook: `notebooks/first_colab_run.ipynb`
- CLI для:
  - проверки окружения
  - построения manifest-файлов
  - split train/val
  - генерации плана synthetic mixtures
  - рендера noisy-clean пар
  - запуска baseline-обработки
  - tiny training / inference
  - расчета стартовых метрик

## Быстрый старт

```bash
uv sync
uv run noise-suppression env check
```

## Полезные команды

Быстрый локальный smoke test на маленьком русскоязычном примере:

```bash
./scripts/run_small_local_demo.sh
```

Маленький smoke test уже с обучением tiny модели:

```bash
./scripts/run_tiny_training_demo.sh
```

Первый notebook для Google Colab:

```text
notebooks/first_colab_run.ipynb
```

Собрать минимальный bundle для Colab в один zip:

```bash
python3 scripts/build_colab_bundle.py
```

Построить manifest для clean speech:

```bash
uv run noise-suppression manifest build /path/to/golos manifests/clean_golos.jsonl --kind clean --speaker-depth 0
```

Построить manifest для noise:

```bash
uv run noise-suppression manifest build /path/to/fsd50k manifests/noise_fsd50k.jsonl --kind noise
```

Посмотреть сводку по manifest:

```bash
uv run noise-suppression manifest summarize manifests/clean_golos.jsonl
```

Разделить manifest на train / val:

```bash
uv run noise-suppression manifest split data/rendered/train/rendered_mix_plan_train.jsonl manifests/train.jsonl manifests/val.jsonl --val-ratio 0.2
```

Собрать план synthetic mixtures:

```bash
uv run noise-suppression mix plan configs/local_macbook_m3.example.yaml manifests/mix_plan_train.jsonl
```

Срендерить небольшой synthetic eval набор:

```bash
uv run noise-suppression mix render manifests/mix_plan_train.jsonl data/rendered/train --limit 128
```

Прогнать identity baseline:

```bash
uv run noise-suppression baseline apply data/rendered/train/rendered_mix_plan_train.jsonl outputs/identity --mode identity
```

Посчитать SI-SDR для enhanced файлов:

```bash
uv run noise-suppression metrics si-sdr manifests/rendered_train.jsonl outputs/identity
```

Запустить tiny training:

```bash
uv sync --extra train
uv run noise-suppression train fit configs/smoke_train.yaml
```

Запустить инференс из checkpoint:

```bash
uv run noise-suppression train infer outputs/train_smoke/best.pt manifests/smoke_val.jsonl outputs/train_smoke_val
```

## Замечания по локальной машине

- окружение ориентировано на `macOS arm64`
- стартовый sample rate для MVP: `16 kHz`
- heavy training зависимости пока не ставятся по умолчанию
- для tiny training нужно отдельно выполнить `uv sync --extra train`
- `ffmpeg` в системе пока не найден, но для базового пайплайна manifest/mix это не блокер
