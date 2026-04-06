#!/usr/bin/env zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "0/7 Проверяю, что установлен train stack..."
uv sync --extra train >/dev/null

echo "1/7 Готовлю маленький synthetic набор..."
"$ROOT_DIR/scripts/run_small_local_demo.sh" >/dev/null

echo "2/7 Делю rendered manifest на train / val..."
uv run noise-suppression manifest split \
  "$ROOT_DIR/data/smoke_demo/rendered/rendered_smoke_mix_plan.jsonl" \
  "$ROOT_DIR/manifests/smoke_train.jsonl" \
  "$ROOT_DIR/manifests/smoke_val.jsonl" \
  --val-ratio 0.25 \
  --seed 11

echo "3/7 Считаю identity baseline на val..."
uv run noise-suppression baseline apply \
  "$ROOT_DIR/manifests/smoke_val.jsonl" \
  "$ROOT_DIR/outputs/identity_val" \
  --mode identity >/dev/null

echo "4/7 Обучаю tiny модель..."
uv run noise-suppression train fit "$ROOT_DIR/configs/smoke_train.yaml"

for checkpoint in "$ROOT_DIR/outputs/train_smoke/epoch_001.pt" "$ROOT_DIR/outputs/train_smoke/best.pt" "$ROOT_DIR/outputs/train_smoke/last.pt"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Не найден checkpoint: $checkpoint" >&2
    exit 1
  fi
done

for mirrored in "$ROOT_DIR/outputs/train_smoke_mirror/epoch_001.pt" "$ROOT_DIR/outputs/train_smoke_mirror/best.pt" "$ROOT_DIR/outputs/train_smoke_mirror/last.pt" "$ROOT_DIR/outputs/train_smoke_mirror/history.json"; do
  if [[ ! -f "$mirrored" ]]; then
    echo "Не найден mirrored checkpoint: $mirrored" >&2
    exit 1
  fi
done

echo "5/7 Прогоняю val через best checkpoint..."
uv run noise-suppression train infer \
  "$ROOT_DIR/outputs/train_smoke/best.pt" \
  "$ROOT_DIR/manifests/smoke_val.jsonl" \
  "$ROOT_DIR/outputs/train_smoke_val"

echo "6/7 Считаю SI-SDR для identity baseline..."
uv run noise-suppression metrics si-sdr \
  "$ROOT_DIR/manifests/smoke_val.jsonl" \
  "$ROOT_DIR/outputs/identity_val"

echo "7/7 Считаю SI-SDR для tiny модели..."
uv run noise-suppression metrics si-sdr \
  "$ROOT_DIR/manifests/smoke_val.jsonl" \
  "$ROOT_DIR/outputs/train_smoke_val"

echo
echo "Если нужен перенос в Google Drive из Colab:"
echo "  укажите training.checkpoint_mirror_dir=/content/drive/MyDrive/noise_suppression/<run_name>"
echo "Тогда после каждой эпохи туда будут копироваться epoch_XXX.pt, last.pt, best.pt и history.json."
