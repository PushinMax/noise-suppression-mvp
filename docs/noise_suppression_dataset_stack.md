# MVP dataset stack: шумоподавление русской речи

## 1. Цель этого документа

Зафиксировать минимальный, но достаточно сильный стек данных для первой версии MVP:

- откуда берем чистую речь
- откуда берем шумы
- как синтезируем noisy-clean пары
- как строим synthetic и real evaluation

Этот документ намеренно ориентирован на **быстрый запуск**, а не на идеальное покрытие всех возможных доменов.

## 2. Выбранный домен MVP

Для первой итерации фиксируем такой домен:

> **русская речь + бытовые и полуофисные фоновые шумы + микрофон ноутбука / телефона**

Это означает, что мы хотим хорошо работать на:

- записи с ноутбука
- записи с телефона
- записи в комнате, офисе, кафе, транспорте

И пока **не** оптимизируемся отдельно под:

- сильное многоголосье
- колл-центр с телефонным каналом 8 kHz
- far-field meeting rooms с несколькими микрофонами
- музыкальные фоны как основной сценарий

## 3. Clean speech

## 3.1 Основной источник

### Golos / OpenSLR 114

- ссылка: <https://www.openslr.org/114/>
- что берем: русский read speech
- роль: основной clean speech source для supervised synthetic training

Почему именно он:

- большой объем
- открытый доступ
- русский язык
- подходит под нашу ключевую гипотезу про domain specialization

## 3.2 Дополнительный источник

### DNS Challenge clean speech pool

- ссылка: <https://github.com/microsoft/DNS-Challenge>
- что берем: clean speech subsets, включая русскую речь при наличии в релизе
- роль: добавить разнообразие дикторов и акустических условий

Почему берем как дополнительный, а не основной:

- наш приоритет — русский speech prior
- DNS полезен как совместимость с baseline-рецептами и как источник дополнительной clean speech diversity

## 4. Noise sources

## 4.1 Базовый noise pool

### MUSAN

- ссылка: <https://www.openslr.org/17/>
- что берем: только noise category для первой итерации
- роль: базовый широкий пул реальных шумов

Почему:

- стандартный открытый корпус
- хорошо подходит для генерации synthetic mixtures
- быстро интегрируется в baseline training pipeline

## 4.2 Целевой noise pool

### FSD50K

- ссылка: <https://zenodo.org/records/4060432>
- роль: curated target-domain noise library

Для MVP нас в первую очередь интересуют клипы, связанные с:

- keyboard / typing
- fan / ventilation / air conditioner
- office ambience
- cafe / restaurant ambience
- traffic / road / street
- vehicle interior
- crowd / chatter background
- household appliance sounds
- clicks / impact / door / dish-like noises

Почему FSD50K критичен:

- позволяет явно сместить training distribution в нашу целевую область
- именно это один из наиболее вероятных источников выигрыша над generic baseline-ами

## 4.3 Шумы и RIR для стандартного recipe

### DNS Challenge noise + RIR

- ссылка: <https://github.com/microsoft/DNS-Challenge>
- роль:
  - добавить стандартный enhancement-compatible noise pool
  - добавить room impulse responses
  - улучшить robustness к reverberation и microphone coloration

## 5. Real-world evaluation

Synthetic test нам нужен, но его недостаточно.

Поэтому в MVP фиксируем обязательный **real noisy Russian eval set**.

## 5.1 Минимальная цель по сбору

- `30-60` минут реальных записей
- `8-12` говорящих
- минимум `4` сценария:
  - комната / офис
  - ноутбук с клавиатурой
  - улица / транспорт
  - кафе / фоновая толпа

## 5.2 Что сохраняем для каждой записи

- wav файл
- текстовая расшифровка
- сценарий записи
- устройство записи
- примерный тип шума

## 5.3 Зачем это нужно

- именно здесь мы будем мерить реальный `WER`
- именно здесь generic baseline-ы чаще всего теряют устойчивость
- именно здесь доменная адаптация должна проявиться лучше всего

## 6. Сплиты

## 6.1 Synthetic train

Состав:

- clean speech: Golos + optional DNS clean additions
- noise: MUSAN + curated FSD50K + DNS noise

Правила:

- train speakers должны быть disjoint от val/test speakers
- train noise clips должны быть disjoint от val/test noise clips

## 6.2 Synthetic validation

Назначение:

- model selection
- early stopping
- контроль overfitting на synthetic domain

Состав:

- held-out speakers
- held-out noise clips
- отдельный SNR mix

## 6.3 Synthetic test

Назначение:

- intrusive metrics:
  - SI-SDR
  - STOI
  - PESQ

Состав:

- полностью disjoint speakers
- полностью disjoint noise clips
- фиксированные SNR buckets

## 6.4 Real test

Назначение:

- реальная финальная проверка

Метрики:

- WER
- DNSMOS
- ручное прослушивание

## 7. Рецепт смешивания

Для первого MVP берем простой, но сильный recipe.

## 7.1 Длина клипов

- train chunks: `2-6` секунд
- val/test chunks: фиксированные `4-8` секунд

## 7.2 SNR

Распределение для train:

- диапазон: `-5 dB` до `20 dB`
- oversampling диапазона `0-10 dB`

Почему:

- именно здесь обычно возникает практический trade-off между агрессивным suppression и сохранением речи

## 7.3 Reverberation

- применять RIR не ко всем примерам
- стартовая вероятность: `30%`

## 7.4 Дополнительные искажения

Для realism augmentation:

- random gain
- clipping simulation с малой вероятностью
- simple EQ / coloration
- device-like filtering

## 7.5 Состав смеси

Первая итерация:

- один спикер
- один доминирующий шум
- optional second low-level background noise с меньшей вероятностью

Такой recipe проще отлаживать и лучше подходит под первый честный baseline.

## 8. Что пойдет в baseline comparison

На первом круге сравнения используем:

1. noisy input
2. DeepFilterNet3
3. FullSubNet+ или максимально близкий публичный checkpoint
4. наша модель

Если времени хватит:

5. MetricGAN+
6. UNIVERSE++

## 9. Что не стоит делать в первой итерации

Не стоит сразу:

- смешивать несколько говорящих
- строить слишком широкий noise zoo
- оптимизировать под музыку
- делать fullband `48 kHz` обучение
- собирать гигантский real dataset до первого baseline comparison

Все это полезно позже, но замедлит путь до первого честного вывода.

## 10. Практический вывод

Для MVP достаточно следующего стека:

- **clean speech**: Golos как база, DNS clean как добавка
- **noise**: MUSAN как база, FSD50K как target-domain усиление, DNS noise/RIR как стандартное дополнение
- **evaluation**:
  - synthetic paired test для intrusive metrics
  - real Russian noisy test для WER и DNSMOS

Это уже достаточно сильная конфигурация, чтобы:

- обучить первую domain-adapted модель
- сравнить ее с сильными open-source baseline-ами
- понять, есть ли у гипотезы практический сигнал
