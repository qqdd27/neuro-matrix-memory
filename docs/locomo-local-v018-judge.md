# LoCoMo evidence-hit@8 report

- date: 2026-09-14 14:23
- conversations: 10, questions scored: 607
- overall evidence-hit@8: **378/607 (62.3%)**

| Category | n | hit@8 |
|---|---|---|
| 1 single-hop | 85 | 44/85 (51.8%) |
| 2 temporal | 102 | 70/102 (68.6%) |
| 3 multi-hop | 22 | 6/22 (27.3%) |
| 4 open-domain | 252 | 165/252 (65.5%) |
| 5 adversarial | 146 | 93/146 (63.7%) |

## QA-accuracy (F1), n=607 sampled

| Category | n | avg F1 |
|---|---|---|
| 1 single-hop | 85 | 21.6% |
| 2 temporal | 102 | 3.8% |
| 3 multi-hop | 22 | 18.9% |
| 4 open-domain | 252 | 43.5% |
| 5 adversarial | 146 | 36.3% |

**Overall F1: 31.1%** (model=qwen2.5:1.5b)

## Date accuracy (calendar point), n=83

| Category | n | date-hit |
|---|---|---|
| 1 single-hop | 1 | 0/1 (0.0%) |
| 2 temporal | 82 | 55/82 (67.1%) |

**Overall date accuracy: 66.3%** (55/83)

## Judge accuracy (meaning), n=607

| Category | n | judge score |
|---|---|---|
| 1 single-hop | 85 | 34.1% |
| 2 temporal | 102 | 27.5% |
| 3 multi-hop | 22 | 18.2% |
| 4 open-domain | 252 | 50.8% |
| 5 adversarial | 146 | 37.0% |

**Overall judge accuracy: 40.0%** (n=607)
