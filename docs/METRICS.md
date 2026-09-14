# NeuroMatrix — все измеренные метрики

Собрано из файлов отчётов `docs/locomo-local-*.md` автоматически (`scripts/metrics_summary.py`); «—» означает, что отчёт этой метрики не содержит.

## Сводка по прогонам

| отчёт | n | модель | попадание факта@8 | ответы F1 | точность даты |
|---|---|---|---|---|---|
| locomo-local-baseline.md | 154 | qwen3.5:9b | — | 7.9% | — |
| locomo-local-memory-dated.md | 154 | qwen3.5:9b | 31.8% | 15.7% | — |
| locomo-local-memory.md | 154 | qwen3.5:9b | 31.8% | 15.7% | — |
| locomo-local-v010.md | 154 | qwen3.5:9b | 53.2% | 24.6% | — |
| locomo-local-v011.md | 154 | qwen3.5:9b | 53.2% | 28.2% | — |
| locomo-local-v012-lemmas.md | 154 | qwen3.5:9b | 55.2% | 25.3% | — |
| locomo-local-v012-typeboost.md | 154 | qwen3.5:9b | 53.9% | 25.1% | — |
| locomo-local-v013-edgeorder.md | 154 | qwen3.5:9b | 49.4% | 22.2% | — |
| locomo-local-v013-lemmas-beside.md | 154 | qwen3.5:9b | 62.3% | 28.9% | — |
| locomo-local-v014-temporal-typeboost.md | ? | qwen3.5:9b | — | — | — |
| locomo-local-v015-reader-qwen2.5-1.5b.md | 203 | qwen2.5:1.5b | 61.1% | 28.2% | — |
| locomo-local-v016-event-date.md | 203 | qwen2.5:1.5b | 64.5% | 29.8% | — |
| locomo-local-v016a-record.md | 607 | qwen2.5:1.5b | 62.3% | 30.4% | — |
| locomo-local-v016b-event.md | 607 | qwen2.5:1.5b | 62.1% | 32.3% | — |
| locomo-local-v017-event-human.md | 607 | qwen2.5:1.5b | 62.6% | 32.1% | 62.7% |

## Ответы F1 по категориям

| отчёт | одиночный факт | время | связь фактов | открытые | ловушки |
|---|---|---|---|---|---|
| locomo-local-baseline.md | 4.0% | 5.2% | 17.9% | 9.3% | 8.0% |
| locomo-local-memory-dated.md | 11.0% | 2.7% | 8.3% | 21.7% | 19.2% |
| locomo-local-memory.md | 11.0% | 4.7% | 16.7% | 20.2% | 18.7% |
| locomo-local-v010.md | 19.7% | 7.4% | 16.7% | 35.3% | 23.4% |
| locomo-local-v011.md | 21.0% | 7.4% | 16.7% | 38.1% | 32.7% |
| locomo-local-v012-lemmas.md | 19.6% | 7.4% | 8.3% | 35.8% | 26.9% |
| locomo-local-v012-typeboost.md | 20.7% | 8.5% | 16.7% | 33.8% | 26.6% |
| locomo-local-v013-edgeorder.md | 23.3% | 11.8% | 11.2% | 29.3% | 19.2% |
| locomo-local-v013-lemmas-beside.md | 27.3% | 8.5% | 11.1% | 39.9% | 29.4% |
| locomo-local-v015-reader-qwen2.5-1.5b.md | 20.6% | 5.6% | 14.7% | 39.0% | 35.7% |
| locomo-local-v016-event-date.md | 30.2% | 3.0% | 2.2% | 39.7% | 39.8% |
| locomo-local-v016a-record.md | 24.3% | 3.8% | 7.6% | 40.6% | 38.4% |
| locomo-local-v016b-event.md | 23.1% | 2.8% | 16.7% | 43.6% | 41.2% |
| locomo-local-v017-event-human.md | 24.6% | 5.5% | 18.9% | 42.5% | 39.1% |

## Точность даты по категориям

| отчёт | одиночный факт | время | связь фактов | открытые | ловушки |
|---|---|---|---|---|---|
| locomo-local-v017-event-human.md | 0.0% | 63.4% | — | — | — |
