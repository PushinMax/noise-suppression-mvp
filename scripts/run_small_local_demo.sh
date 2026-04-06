#!/usr/bin/env zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE_DIR="$ROOT_DIR/data/smoke_demo"
CLEAN_DIR="$SMOKE_DIR/clean/spk_milena"
NOISE_DIR="$SMOKE_DIR/noise"
MANIFEST_DIR="$ROOT_DIR/manifests"
OUTPUT_DIR="$ROOT_DIR/outputs/smoke_identity"

mkdir -p "$CLEAN_DIR" "$NOISE_DIR" "$MANIFEST_DIR" "$OUTPUT_DIR"

echo "1/6 Генерирую маленький набор русской clean speech..."

TEXTS=(
  "Сегодня хороший день для проверки пайплайна шумоподавления."
  "Мы хотим убедиться, что локальный MVP корректно работает на маленьких данных."
  "Этот пример нужен только для технической проверки и не отражает реальное качество модели."
)

for idx in 1 2 3; do
  aiff_path="$CLEAN_DIR/utt${idx}.aiff"
  wav_path="$CLEAN_DIR/utt${idx}.wav"
  txt_path="$CLEAN_DIR/utt${idx}.txt"
  say -v Milena -r 175 -o "$aiff_path" "${TEXTS[$idx]}"
  afconvert -f WAVE -d LEI16@16000 "$aiff_path" "$wav_path" >/dev/null 2>&1
  rm -f "$aiff_path"
  printf "%s\n" "${TEXTS[$idx]}" >"$txt_path"
done

echo "2/6 Генерирую маленький пул шумов..."

uv run python - <<'PY'
from pathlib import Path
import numpy as np
import soundfile as sf

sr = 16000
root = Path("data/smoke_demo/noise")
root.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)
t = np.linspace(0, 5, sr * 5, endpoint=False, dtype=np.float32)

# Fan-like low-frequency colored noise.
fan = rng.standard_normal(sr * 5).astype(np.float32)
fan = np.convolve(fan, np.ones(512, dtype=np.float32) / 512.0, mode="same")
fan = 0.08 * fan / np.max(np.abs(fan))
sf.write(root / "fan_like.wav", fan, sr)

# Keyboard-like impulses over quiet background.
keyboard = 0.01 * rng.standard_normal(sr * 5).astype(np.float32)
click_positions = rng.integers(low=0, high=sr * 5, size=18)
for pos in click_positions:
    end = min(pos + 120, keyboard.shape[0])
    keyboard[pos:end] += np.hanning(end - pos).astype(np.float32) * 0.2
keyboard = 0.08 * keyboard / np.max(np.abs(keyboard))
sf.write(root / "keyboard_like.wav", keyboard, sr)
PY

echo "3/6 Строю manifest-файлы..."

uv run noise-suppression manifest build \
  "$CLEAN_DIR/.." \
  "$MANIFEST_DIR/smoke_clean.jsonl" \
  --kind clean \
  --speaker-depth 0

uv run noise-suppression manifest build \
  "$NOISE_DIR" \
  "$MANIFEST_DIR/smoke_noise.jsonl" \
  --kind noise

echo "4/6 Планирую и рендерю synthetic mixtures..."

uv run noise-suppression mix plan \
  "$ROOT_DIR/configs/smoke_demo.yaml" \
  "$MANIFEST_DIR/smoke_mix_plan.jsonl"

uv run noise-suppression mix render \
  "$MANIFEST_DIR/smoke_mix_plan.jsonl" \
  "$SMOKE_DIR/rendered" \
  --limit 8 \
  --overwrite

echo "5/6 Прогоняю baseline identity..."

uv run noise-suppression baseline apply \
  "$SMOKE_DIR/rendered/rendered_smoke_mix_plan.jsonl" \
  "$OUTPUT_DIR" \
  --mode identity

echo "6/6 Считаю стартовую метрику SI-SDR..."

uv run noise-suppression metrics si-sdr \
  "$SMOKE_DIR/rendered/rendered_smoke_mix_plan.jsonl" \
  "$OUTPUT_DIR"

echo
echo "Готово."
echo "Clean speech:   $CLEAN_DIR"
echo "Noise pool:     $NOISE_DIR"
echo "Rendered set:   $SMOKE_DIR/rendered"
echo "Baseline out:   $OUTPUT_DIR"
