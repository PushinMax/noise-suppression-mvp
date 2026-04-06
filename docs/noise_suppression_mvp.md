# MVP: шумоподавление русской речи

См. также:

- [noise_suppression_overview.md](./noise_suppression_overview.md)
- [noise_suppression_dataset_stack.md](./noise_suppression_dataset_stack.md)

## 1. Цель MVP

Построить **single-channel speech enhancement** модель для русской речи, которая на узком целевом домене:

- снижает фоновый шум в бытовых и полуофисных условиях
- сохраняет разборчивость речи
- дает выигрыш относительно сильных open-source baseline-ов

Ключевой момент: мы целимся **не** в “лучший denoiser вообще”, а в **лучший результат на конкретном домене русской шумной речи**.

## 2. Что именно считаем задачей

Рабочая постановка MVP:

- вход: зашумленная одноканальная запись русской речи
- выход: улучшенная запись той же речи
- начальный sample rate: `16 kHz`
- начальный сценарий: ноутбук / телефон / бытовой микрофон

Шумы, на которые ориентируемся в MVP:

- вентилятор, кондиционер, гул
- клавиатура, клики, бытовые щелчки
- кафе, офис, комнатный шум
- улица, транспорт, фон толпы

На первом этапе **не** решаем:

- разделение нескольких говорящих
- сложную far-field beamforming-задачу
- студийное восстановление качества
- универсальное enhancement для всех искажений во всех доменах

## 3. Критерий успеха MVP

MVP считаем успешным, если на нашем целевом домене модель:

- лучше оффлайн или сопоставимо лучше generic baseline-а по **WER русского ASR**
- не проседает заметно по perceptual quality
- остается достаточно легкой для дальнейших итераций

Практический целевой сценарий сравнения:

> наша domain-adapted модель против `DeepFilterNet3` и хотя бы одного research baseline-а на русскоязычном noisy test set.

## 4. Главные метрики

## 4.1 Основные

- **WER** русского ASR до и после enhancement
  - главная прикладная метрика
  - именно она лучше всего отвечает на вопрос, улучшили ли мы речь для реального downstream-сценария

- **DNSMOS**
  - no-reference метрика качества
  - важна для real-world eval, где нет clean ground truth

- **STOI**
  - метрика разборчивости речи

- **SI-SDR**
  - удобная objective signal-level метрика для synthetic noisy-clean eval

## 4.2 Дополнительные

- **PESQ**
  - полезна как стандартная benchmark-метрика
  - но не должна быть единственным критерием отбора

- **RTF**
  - real-time factor

- **latency**
  - особенно важно, если позже пойдем в live-mic сценарий

- **число параметров**
- **память**

## 4.3 Почему WER ставим выше PESQ

Это важная фиксация для всего проекта.

В speech enhancement часто есть конфликт между:

- “звук субъективно приятнее”
- “ASR распознает лучше”

Это видно и в современной литературе:

- в [FINALLY](https://arxiv.org/abs/2410.05920) авторы прямо показывают, что модель может выигрывать по MOS и no-reference метрикам, но проигрывать по `PESQ/STOI/SI-SDR`
- в [Mel-FullSubNet](https://arxiv.org/abs/2402.13511) цель уже формулируется как улучшение **и** speech quality, **и** ASR

Поэтому для нас:

- `WER` — главная продуктовая метрика
- `DNSMOS/STOI` — guardrail, чтобы модель не портила звук
- `PESQ/SI-SDR` — вспомогательные intrusive benchmark-метрики

## 5. Ландшафт существующих моделей

Ниже перечислены модели и семейства, которые важно зафиксировать как baseline-ландшафт.

## 5.1 DeepFilterNet / DeepFilterNet2 / DeepFilterNet3

Источники:

- [репозиторий](https://github.com/Rikorose/DeepFilterNet)
- [DeepFilterNet](https://arxiv.org/abs/2110.05588)
- [DeepFilterNet3 demo paper](https://arxiv.org/abs/2305.08227)

Что это за подход:

- STFT-based enhancement
- двухэтапная схема
- сначала улучшается spectral envelope в ERB-шкале
- затем применяется deep filtering для periodic components речи
- сильный упор на low complexity и real-time

Почему модель сильная:

- очень практичная
- открытая
- пригодна для live noise suppression
- сильная точка отсчета для реального использования

Публичные числа, которые удобно держать как reference:

- из Interspeech 2023 demo paper:
  - DeepFilterNet: PESQ `2.81`, STOI `0.942`
  - DeepFilterNet2: PESQ `3.08`, STOI `0.943`
  - DeepFilterNet3: PESQ `3.17`, STOI `0.944`
  - real-time factor: `0.19` на single-thread CPU `i5-8250U`
  - latency: `40 ms`

Что для нас важно:

- это главный **практический baseline**
- если мы не можем обыграть off-the-shelf DeepFilterNet3 на русскоязычном целевом домене, проекту будет трудно доказать ценность

Ограничения для нашей задачи:

- модель generic, а не доменная
- публичные reference-результаты в основном не про русский ASR
- целевая оптимизация в первую очередь не под `WER`

## 5.2 FullSubNet / FullSubNet+

Источники:

- [FullSubNet paper](https://arxiv.org/abs/2010.15508)
- [FullSubNet repo](https://github.com/Audio-WestlakeU/FullSubNet)
- [FullSubNet+ paper](https://arxiv.org/abs/2203.12188)
- [FullSubNet+ repo](https://github.com/RookieJunChen/FullSubNet-plus)

Что это за подход:

- spectrogram-based fusion of full-band and sub-band processing
- full-band часть ловит глобальный спектральный контекст
- sub-band часть лучше работает с локальной stationarity и узкополосными паттернами
- в FullSubNet+ добавлены complex spectrogram input, channel attention и более эффективный full-band extractor

Почему модель сильная:

- один из самых влиятельных и воспроизводимых research baseline-ов
- удобна для быстрых итераций в PyTorch
- архитектурно хорошо подходит для нашего домена

Почему она важна для MVP:

- это лучший кандидат на **исследовательский backbone**
- DeepFilterNet сильнее как “готовый продукт”, но FullSubNet-подобные модели проще и быстрее адаптировать под новые loss-ы, данные и ablation-эксперименты

Ограничения для нашей задачи:

- публичные claims модели в основном опираются на DNS/benchmark-режим
- базовые версии не оптимизируются специально под русский ASR
- generic synthetic training не решает domain gap автоматически

## 5.3 MetricGAN+

Источники:

- [paper](https://arxiv.org/abs/2104.03538)
- [SpeechBrain toolkit](https://github.com/speechbrain/speechbrain)

Что это за подход:

- GAN-like enhancement, напрямую ориентированный на perceptual metric
- discriminator приближает quality metric и направляет generator

Публичный ориентир:

- в статье заявлен PESQ `3.15` на VoiceBank-DEMAND

Почему модель важна:

- это хороший пример моделей, которые очень сильны по benchmark-like quality metric
- полезна как напоминание, что оптимизация под одну метрику может не совпадать с downstream ASR-задачей

Ограничения для нашей задачи:

- качество по одному metric proxy не гарантирует лучший `WER`
- модель не заточена под узкий русский бытовой домен

## 5.4 MP-SENet

Источники:

- [paper](https://arxiv.org/abs/2305.13686)
- [repo](https://github.com/yxlu-0102/MP-SENet)

Что это за подход:

- параллельная оценка magnitude и phase spectra
- multi-level losses на magnitude, phase, complex spectra и waveform

Публичный ориентир:

- в abstract статьи заявлен PESQ `3.50` на VoiceBank+DEMAND

Почему модель важна:

- это сильный quality-oriented baseline
- показывает, насколько далеко можно уйти в benchmark quality при более тяжелой модели

Ограничения для нашей задачи:

- это не лучший путь для быстрого прикладного MVP
- выигрыш по `PESQ` сам по себе не означает выигрыш по `WER`
- риск выше по вычислительной стоимости и сложности экспериментов

## 5.5 ZipEnhancer

Источник:

- [paper](https://arxiv.org/abs/2501.05183)

Что это за подход:

- dual-path down-up sampling architecture на базе Zipformer
- упор на сильные benchmark-результаты при умеренном числе параметров

Публичный ориентир:

- PESQ `3.69` на DNS 2020 Challenge
- PESQ `3.63` на VoiceBank+DEMAND
- `2.04M` параметров
- `62.41G` FLOPS

Почему модель важна:

- это уже ориентир на recent high-end benchmark quality
- показывает верхнюю планку того, что generic models умеют на популярных benchmark-ах

Ограничения для нашей задачи:

- обыгрывать такую модель “в среднем по всем benchmark-ам” для нас сейчас нецелесообразно
- рациональная цель — обыграть generic baseline-ы на **узком целевом домене**

## 5.6 UNIVERSE / UNIVERSE++

Источники:

- [UNIVERSE paper](https://arxiv.org/abs/2206.03065)
- [open-universe repo](https://github.com/line/open-universe)

Что это за подход:

- universal speech enhancement через diffusion/generative family
- модель покрывает более широкий класс искажений, чем обычный denoiser

Публичные метрики из open-universe repo на VoiceBank-DEMAND 16 kHz:

- UNIVERSE++:
  - SI-SDR `18.624`
  - PESQ-WB `3.017`
  - STOI-ext `0.864`
  - DNSMOS OVRL `3.200`
- UNIVERSE:
  - SI-SDR `17.600`
  - PESQ-WB `2.830`
  - STOI-ext `0.844`
  - DNSMOS OVRL `3.157`

Почему модель важна:

- сильный представитель universal/generative линии
- полезен как reference для “широкой” модели, не заточенной под один домен

Ограничения для нашей задачи:

- inference сложнее и тяжелее, чем у легких STFT-моделей
- универсальность не гарантирует лучший результат на узком русском домене
- для MVP риск/стоимость выше, чем у более простой STFT-based модели

## 5.7 FINALLY

Источники:

- [paper](https://arxiv.org/abs/2410.05920)
- [demo page](https://mmacosha.github.io/finally-demo/)

Что это за подход:

- GAN-based enhancement
- backbone на базе HiFi++
- WavLM-based perceptual loss
- сильный упор на high-quality universal enhancement

Публичные ориентиры из статьи:

- на VoxCeleb real data:
  - MOS `4.63`
  - UTMOS `4.05`
  - DNSMOS `3.31`
  - RTF `0.03`
- на VCTK-DEMAND:
  - MOS `4.66`
  - UTMOS `4.32`
  - DNSMOS `3.22`
  - PESQ `2.94`
  - STOI `0.92`
  - SI-SDR `4.6`
  - WER `0.07`

Что особенно важно:

- сами авторы пишут, что модель выигрывает по subjective и no-reference метрикам, но **проигрывает по `PESQ/STOI/SI-SDR`**

Почему модель важна:

- это отличный пример того, что speech enhancement нельзя оценивать только одной intrusive метрикой
- она помогает нам обосновать мульти-метрический протокол оценки

Ограничения для нашей задачи:

- модель сложнее и более research-heavy
- воспроизводимость и адаптация под наш MVP сложнее, чем у легких STFT baseline-ов
- опять же, модель не оптимизирована конкретно под русский бытовой noisy domain

## 5.8 RemixIT / UDASE-семейство

Источники:

- [RemixIT](https://arxiv.org/abs/2110.10103)
- [CHiME-7 UDASE task](https://arxiv.org/abs/2307.03533)
- [repo](https://github.com/etzinis/unsup_speech_enh_adaptation)

Что это за подход:

- не отдельный denoiser-класс, а **подход к domain adaptation**
- идея: использовать real noisy target-domain audio без clean target reference

Почему это критично для нас:

- в CHiME-7 UDASE authors прямо фиксируют проблему: supervised speech enhancement обучается на synthetic noisy-clean смесях, и **mismatch между synthetic train и real test** может сильно ухудшать качество
- это почти дословно совпадает с нашим будущим риском

Вывод:

- даже если мы берем lightweight STFT backbone, domain adaptation — не optional nice-to-have, а один из самых реалистичных источников будущего выигрыша

## 6. Что именно мы выбираем как основной подход

Для MVP выбираем следующий основной путь:

> **domain-adapted lightweight STFT-based speech enhancement model для русской речи**

В рабочем виде это означает:

- backbone: **FullSubNet+-lite / FullSubNet-подобная модель**
- вход: noisy complex STFT features
- выход: complex mask или complex residual refinement
- sample rate: `16 kHz`
- обучение: сначала synthetic supervised, затем optional unsupervised domain adaptation

## 6.1 Почему не DeepFilterNet как основную обучаемую модель

DeepFilterNet очень важен как baseline, но не лучший основной research-backbone для первого MVP.

Причины:

- FullSubNet-подобные модели проще модифицировать в чистом PyTorch
- удобнее добавлять собственные loss-ы
- удобнее делать ablation
- быстрее перестраивать data recipe и эксперименты

При этом:

- **DeepFilterNet3** остается нашим главным практическим baseline-ом для сравнения

## 6.2 Почему не diffusion / FINALLY / UNIVERSE как основной путь

Причины простые:

- слишком высокий инженерный риск
- сложнее обучение и inference
- выше стоимость итераций
- хуже подходят для быстрого узкодоменного MVP

Это хорошие reference-модели, но не лучший первый путь, если наша цель:

- быстро дойти до честного сравнения
- понять, есть ли доменный выигрыш
- не утонуть в инфраструктуре до первого результата

## 7. Почему наш подход имеет шанс обойти существующие open-source модели

Важно: ниже не доказанный факт, а **рабочая гипотеза проекта**.

Мы не утверждаем заранее, что модель уже лучше. Мы утверждаем, что у этого подхода есть понятные и правдоподобные источники выигрыша.

## 7.1 Источник выигрыша 1: целевой домен вместо universal optimization

Большинство сильных open-source моделей:

- обучаются на широких synthetic mixtures
- проверяются на VoiceBank+DEMAND, DNS или универсальных distortion benchmark-ах

Мы же хотим:

- русский язык
- бытовые шумы
- микрофоны ноутбука/телефона
- downstream русский ASR

Именно здесь specialization может выиграть у general-purpose модели.

## 7.2 Источник выигрыша 2: оптимизация под русский ASR WER, а не только под quality proxy

Многие baseline-ы в явном или неявном виде оптимизируют:

- PESQ
- MOS-like proxy
- spectral reconstruction

Но нам нужен выигрыш в сценарии:

- “распознавать русскую речь после denoising”

Поэтому наше model selection должно идти по:

- `WER`
- `DNSMOS`
- `STOI`

а не по одному `PESQ`.

Это и есть один из самых правдоподобных способов обойти generic baseline:

- не на их любимой метрике
- а на нашей целевой прикладной задаче

## 7.3 Источник выигрыша 3: правильная noise library

Если generic модель обучалась на broad noise pool, а мы:

- переусилим клавиатуру
- переусилим офис и комнатный шум
- добавим бытовые импульсные события
- подмешаем device coloration и reverb

то наш train distribution будет ближе к test distribution.

Это особенно важно, потому что [CHiME-7 UDASE](https://arxiv.org/abs/2307.03533) прямо указывает synthetic-real mismatch как центральную проблему.

## 7.4 Источник выигрыша 4: русская clean speech как prior

Это более слабый, но тоже правдоподобный источник выигрыша.

Если backbone учится на русской clean speech, то:

- фонетика
- темп
- просодика
- спектральные паттерны

будут ближе к реальному целевому домену, чем у англоязычно-центричного рецепта.

Сам по себе этот фактор не гарантирует прорыв, но в связке с domain adaptation он вполне может дать плюс.

## 7.5 Источник выигрыша 5: optional unsupervised adaptation на реальном шумном аудио

Это, вероятно, самый сильный потенциальный differentiator после выбора данных.

Если мы добавим этап adaptation в стиле:

- RemixIT-like self-training
- consistency regularization
- teacher-student на real noisy speech

то сможем подстроить модель под реальный target domain без clean reference.

Именно этот шаг generic baseline-ы “из коробки” обычно не делают под ваш конкретный сценарий.

## 8. Базовая архитектура MVP

## 8.1 Backbone

Первый выбор:

- **FullSubNet+-lite**

Упрощенная реализация с акцентом на:

- complex STFT input
- full-band + sub-band fusion
- умеренный размер модели
- возможность causal или low-lookahead режима

## 8.2 Выход модели

Два допустимых варианта:

- **complex ratio mask**
- **complex residual mapping**

Для первого MVP более консервативный выбор:

- complex mask

Потому что:

- проще стабилизировать обучение
- проще сравнивать с классическими baseline-ами

## 8.3 Loss functions

Базовый loss stack для MVP:

- L1/L2 loss на magnitude
- complex spectral loss
- waveform loss
- SI-SDR loss

Вторая итерация:

- добавить ASR-aware regularization или feature-level loss от frozen ASR encoder

Это не обязательно для самой первой тренировки, но очень желательно для второй волны экспериментов.

## 9. План сравнения

Минимальный baseline stack:

1. noisy input
2. `DeepFilterNet3`
3. `FullSubNet+` или близкий public checkpoint
4. наша модель

Расширенный stack:

5. `MetricGAN+`
6. `UNIVERSE++`

Сравниваем на двух уровнях:

- **synthetic holdout**
  - SI-SDR
  - STOI
  - PESQ
- **real noisy Russian eval**
  - WER
  - DNSMOS
  - ручное прослушивание

## 10. Что считаем честным выигрышем

Честный выигрыш для первой версии проекта:

- относительное снижение `WER` на целевом real noisy Russian set
- при этом `DNSMOS` не хуже baseline-а более чем на небольшую величину
- и без сильного деградационного артефакта на слух

Нечестный выигрыш:

- улучшили только `PESQ` на synthetic benchmark
- ухудшили распознавание
- переобучились на один тип шума

## 11. Основные риски

Риск 1:

- модель станет “чище” по ощущениям, но ухудшит `WER`

Риск 2:

- synthetic train recipe окажется слишком далек от real test domain

Риск 3:

- gains будут только на одном шумовом сценарии

Риск 4:

- тяжело воспроизвести некоторые сильные generative baseline-ы

## 12. Рабочий вывод

На текущем этапе самый рациональный MVP выглядит так:

- **не** пытаться строить universal speech enhancer
- **не** пытаться обыгрывать FINALLY и ZipEnhancer “вообще”
- взять **FullSubNet+-lite как research backbone**
- взять **DeepFilterNet3 как practical baseline**
- сделать **русский целевой train/eval pipeline**
- мерить успех в первую очередь через **WER русского ASR**
- при необходимости добавить **RemixIT-like domain adaptation**

Именно так у нас появляется реальный шанс получить не просто “еще одну модель шумоподавления”, а специфический measurable win на выбранном домене.

## 13. Следующий шаг

Следующий документ должен зафиксировать:

- точный dataset stack
- recipe синтетического смешивания
- baseline inference plan
- train / val / test split
- структуру экспериментов
