# LoCoMo evidence-hit@8 report

- date: 2026-09-14 18:57
- conversations: 10, questions scored: 607
- overall evidence-hit@8: **362/607 (59.6%)**

| Category | n | hit@8 |
|---|---|---|
| 1 single-hop | 85 | 43/85 (50.6%) |
| 2 temporal | 102 | 68/102 (66.7%) |
| 3 multi-hop | 22 | 7/22 (31.8%) |
| 4 open-domain | 252 | 157/252 (62.3%) |
| 5 adversarial | 146 | 87/146 (59.6%) |

## QA-accuracy (F1), n=607 sampled

| Category | n | avg F1 |
|---|---|---|
| 1 single-hop | 85 | 23.8% |
| 2 temporal | 102 | 22.7% |
| 3 multi-hop | 22 | 12.9% |
| 4 open-domain | 252 | 40.7% |
| 5 adversarial | 146 | 34.5% |

**Overall F1: 32.8%** (model=qwen2.5:1.5b)

## Date accuracy (calendar point), n=83

| Category | n | date-hit |
|---|---|---|
| 1 single-hop | 1 | 0/1 (0.0%) |
| 2 temporal | 82 | 55/82 (67.1%) |

**Overall date accuracy: 66.3%** (55/83)

## Judge accuracy (meaning), n=607

| Category | n | judge score |
|---|---|---|
| 1 single-hop | 85 | 35.3% |
| 2 temporal | 102 | 20.6% |
| 3 multi-hop | 22 | 9.1% |
| 4 open-domain | 252 | 46.0% |
| 5 adversarial | 146 | 35.6% |

**Overall judge accuracy: 36.4%** (n=607)
