# Инструкция по запуску через VSCode Colab extension

## Принцип запуска

Теперь notebook **не использует Google Drive**.

Ожидаемый сценарий:

1. проект ведется в git;
2. в Colab runtime проект появляется через `git clone`;
3. notebook запускается из папки проекта;
4. модель и checkpoint-ы сохраняются локально в `outputs/colab_first_result`.

## Последовательность действий

### 1. Запушьте проект в git

В корне проекта:

```bash
cd /Users/m.pushin/dl/project
git status --short
git add .gitignore README.md pyproject.toml uv.lock configs docs notebooks scripts src tests
git commit -m "Add Colab noise suppression MVP"
git push
```

### 2. В Colab runtime клонируйте проект

В первой ячейке notebook укажите URL вашего репозитория:

```python
REPO_URL = 'https://github.com/<your-username>/<your-repo>.git'
```

Если `/content/project` еще нет, notebook сам выполнит `git clone`.

Если репозиторий приватный, используйте персональный токен или настройте доступ в VSCode/Colab extension.

### 3. Установите зависимости

```python
run('pip -q install uv')
run('uv sync --extra train --extra data --extra dev')
run('uv run noise-suppression env check')
```

Важно: для первого Colab-run мы закрепляем `datasets<4`, потому что `google/fleurs`
пока загружается через старый `fleurs.py`, а `datasets 4.x` такие loading scripts
уже не поддерживает. Не удаляйте `uv.lock` перед запуском.

### 4. Откройте notebook из проекта

Открывайте:

```text
/content/project/notebooks/first_colab_run.ipynb
```

Если VSCode extension открывает локальный notebook, убедитесь, что первая ячейка видит корень проекта:

```python
PROJECT_ROOT = Path('/content/project')
```

В актуальной версии notebook первая ячейка сама ищет корень проекта по `pyproject.toml`.

### 5. Запускайте ячейки сверху вниз

Модель будет сохраняться локально в:

```text
/content/project/outputs/colab_first_result
```

После каждой эпохи там должны появляться:

```text
epoch_001.pt
epoch_002.pt
...
last.pt
best.pt
history.json
resolved_config.json
```

## Важное ограничение

Локальное сохранение в Colab runtime **не переживает пересоздание runtime**.

Это нормально для текущего сценария, потому что вы попросили сохранять модель локально, а не на Google Drive.

Если нужно забрать модель после запуска, скачайте ее до завершения runtime:

```python
from google.colab import files

files.download('/content/project/outputs/colab_first_result/best.pt')
files.download('/content/project/outputs/colab_first_result/history.json')
```

Или скопируйте весь каталог:

```python
!zip -r /content/colab_first_result.zip /content/project/outputs/colab_first_result
from google.colab import files
files.download('/content/colab_first_result.zip')
```

## Если notebook говорит, что не найден `pyproject.toml`

Значит runtime не находится в папке проекта.

Исправление:

```python
%cd /content/project
!ls
```

В выводе должны быть:

```text
pyproject.toml
src
configs
scripts
notebooks
```

## Если появляется `Clean manifest пуст`

Это значит, что ячейка подготовки данных не скачала/не создала clean WAV-файлы.
В актуальной версии notebook выполнение остановится сразу на этой ошибке, а не
пойдет дальше к `mix plan` и `colab_val.jsonl`.

Что сделать:

```python
%cd /content/project
!git pull
!uv sync --extra train --extra data --extra dev
```

Затем перезапустите notebook сверху или хотя бы начиная с ячейки установки
зависимостей. В выводе `uv sync` должно быть видно, что используется
`datasets==3.6.0`, а не `datasets==4.x`.
