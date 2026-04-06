# Первый запуск в Google Colab

## Цель

Получить **первый осмысленный результат** за один короткий прогон в Colab:

- убедиться, что обучение идет без падений
- увидеть, что модель может обыгрывать `identity` baseline
- получить первые checkpoint-ы, которые не потеряются при аварии среды

## Рекомендуемый размер данных для первого прогона

Для первого Colab-run не нужен большой датасет.

Рекомендуемая конфигурация:

- **clean speech**: `30-60` минут русской речи
- **noise pool**: `10-20` минут шумов
- **synthetic train mixtures**: `1500-2500`
- **synthetic val mixtures**: `200-300`
- **длина сегмента**: `3` секунды
- **sample rate**: `16 kHz`

Этого уже достаточно, чтобы:

- получить первый train/val curve
- сравнить с `identity`
- понять, сходится ли tiny STFT-model вообще на выбранном recipe

## Какие данные брать в этот первый run

Минимальный практичный вариант:

- clean speech:
  - маленький поднабор из `Golos`
  - или небольшой кусок `Common Voice RU`, если с ним быстрее стартовать
- noise:
  - `MUSAN` как базовый noise pool
  - плюс маленький curated subset из `FSD50K`

Для very-first run достаточно 4 шумовых категорий:

- fan / AC
- keyboard / typing
- cafe / crowd ambience
- traffic / street

## Что именно тренируем

Для первого Colab-run берём:

- `TinyMaskNet`
- `16 kHz`
- `n_fft=512`
- `hop_length=128`
- `segment_seconds=3.0`
- `epochs=6`
- `batch_size=8`

Это специально не “лучшая возможная” архитектура, а **быстрая диагностическая модель**, которая нужна, чтобы:

- проверить данные
- проверить training loop
- проверить checkpointing
- понять, есть ли хоть какой-то learning signal

## Что считаем первым полезным результатом

Первый прогон можно считать удачным, если выполняются все три условия:

1. обучение проходит все эпохи без технических падений
2. checkpoint-ы сохраняются после каждой эпохи
3. на synthetic validation модель хотя бы немного лучше `identity` по `SI-SDR` или `val_loss`

На этом этапе нам не нужен SOTA-quality.

Нам нужен ответ на вопрос:

> “Пайплайн учится и дает хоть какой-то улучшенный сигнал, или фундаментально что-то не так?”

## Обязательное сохранение модели

Для текущего сценария через VSCode Colab extension используем локальное сохранение:

- локальный output:
  - `/content/project/outputs/colab_first_result`

В нашем конфиге:

- `training.output_dir` указывает на локальную папку с checkpoint-ами
- `training.checkpoint_mirror_dir` выключен и равен `None`

После **каждой эпохи** локально сохраняются:

- `epoch_XXX.pt`
- `last.pt`
- `best.pt`
- `history.json`
- `resolved_config.json`

Важно: локальные файлы Colab runtime не переживают пересоздание runtime. Если модель нужно забрать после запуска, скачайте `best.pt` или zip-архив с `outputs/colab_first_result` до остановки runtime.


## Какой конфиг брать за основу

Стартовый шаблон для Colab:

- [colab_first_result.example.yaml](../configs/colab_first_result.example.yaml)

Что в нем важно:

- маленький `limit_train`
- маленький `limit_val`
- `save_every_epoch: true`
- `checkpoint_mirror_dir` выключен для локального сохранения

## Следующий шаг после первого удачного Colab-run

Если первый run отработал и tiny model показывает хоть небольшой gain, тогда следующий шаг:

1. увеличить число synthetic mixtures
2. расширить noise library
3. перейти от `TinyMaskNet` к `FullSubNet-like` backbone
4. добавить сравнение с `DeepFilterNet`

То есть первый Colab-run нужен не для финального качества, а для быстрой проверки направления.
