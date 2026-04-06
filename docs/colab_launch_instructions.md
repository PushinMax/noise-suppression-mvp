# Инструкция по запуску в Google Colab

## Вариант A. Самый удобный: проект лежит целиком на Google Drive

Это лучший вариант.

### Что сделать локально

Просто положите папку проекта целиком на Google Drive, например:

- `MyDrive/dl/project`

### Что сделать в notebook

В первой ячейке notebook проверьте путь:

```python
PROJECT_DIR_ON_DRIVE = Path('/content/drive/MyDrive/dl/project')
```

Если ваша папка лежит в другом месте, исправьте только эту строку.

### Когда этот вариант подходит лучше всего

- если вы работаете с проектом регулярно
- если хотите обновлять код без ручной загрузки файлов
- если хотите, чтобы notebook всегда видел актуальные `src/`, `scripts/`, `configs/`

## Вариант B. Минимальный bundle в один zip

Если не хотите класть весь проект на Drive, можно отправлять в Colab только минимальный набор нужных файлов.

### Шаг 1. Соберите bundle локально

В корне проекта выполните:

```bash
python3 scripts/build_colab_bundle.py
```

По умолчанию появится файл:

- `dist/colab_bundle.zip`

В нем уже будут нужные файлы:

- `pyproject.toml`
- `uv.lock`
- `src/noise_suppression/...`
- `scripts/prepare_first_colab_dataset.py`
- `configs/colab_first_result.example.yaml`
- `notebooks/first_colab_run.ipynb`

### Шаг 2. Загрузите zip на Google Drive

Например сюда:

- `MyDrive/noise_suppression/colab_bundle.zip`

### Шаг 3. Вставьте в первую ячейку Colab этот bootstrap-код

```python
from google.colab import drive
from pathlib import Path
import shutil
import zipfile
import os

drive.mount('/content/drive')

BUNDLE_ZIP = Path('/content/drive/MyDrive/noise_suppression/colab_bundle.zip')
RUNTIME_DIR = Path('/content/project')

assert BUNDLE_ZIP.exists(), f'Не найден архив: {BUNDLE_ZIP}'

if RUNTIME_DIR.exists():
    shutil.rmtree(RUNTIME_DIR)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(BUNDLE_ZIP, 'r') as zf:
    zf.extractall(RUNTIME_DIR)

os.chdir(RUNTIME_DIR)
print('Готово:', RUNTIME_DIR)
print('pyproject exists =', Path('pyproject.toml').exists())
print('cli exists =', Path('src/noise_suppression/cli.py').exists())
print('prep script exists =', Path('scripts/prepare_first_colab_dataset.py').exists())
```

После этого уже можно запускать остальные ячейки notebook.

## Что выбрать

Я рекомендую так:

- если проект живет долго: **Вариант A**
- если нужен просто перенос минимального набора в Colab: **Вариант B**

## Самая короткая рабочая схема

### Если проект уже на Google Drive

1. Откройте Colab
2. Смонтируйте Google Drive
3. Убедитесь, что `PROJECT_DIR_ON_DRIVE` указывает на правильную папку
4. Запустите первую ячейку notebook
5. Потом запускайте ячейки сверху вниз

### Если у вас только notebook

1. Выполните локально:

```bash
python3 scripts/build_colab_bundle.py
```

2. Загрузите `dist/colab_bundle.zip` на Google Drive
3. В Colab выполните bootstrap-ячейку из раздела выше
4. Только потом запускайте notebook

## Что обязательно сохранить на Google Drive во время обучения

В train config должно быть заполнено поле:

- `training.checkpoint_mirror_dir`

Например:

```python
'/content/drive/MyDrive/noise_suppression/first_colab_result'
```

Тогда после **каждой эпохи** сохраняются:

- `epoch_001.pt`, `epoch_002.pt`, ...
- `last.pt`
- `best.pt`
- `history.json`
- `resolved_config.json`

Это нужно на случай:

- падения runtime
- отключения браузера
- разрыва сессии
- ручной остановки

## Если хотите запускать без ручного редактирования notebook

Самая практичная автоматизация такая:

1. локально собрать zip-бандл через `scripts/build_colab_bundle.py`
2. загрузить его на Google Drive
3. в Colab всегда запускать одну и ту же bootstrap-ячейку распаковки

Тогда не нужно каждый раз вручную переносить `src/`, `scripts/` и `configs/`.
