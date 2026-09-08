# NeuroMatrix recall report — 2026-09-08 (baseline)

- date: 2026-09-08
- db: live Hermes profile db (91+ facts / 286 entities / 22 dossiers at the time)
- harness: `scripts/eval_recall.py` (search path identical to prefetch: `store.search` top-k)

## Results

| Suite | recall@5 |
|---|---|
| Semantic scenarios (7 real past-work questions incl. typos `cloudflre` / `cloudfare` → Cloudflare) | **7/7 = 100%** |
| Dossier self-test (top-10 dossier entities; query = entity key; ground truth = the dossier itself must resurface in top-5) | **10/10 = 100%** |

## What the run taught us (already fixed)

- Typo tolerance added: `cloudflre`/`cloudfare` resolve to the `cloudflare`
  dossier via difflib close-match (retrieval suggestion only, never a
  persisted merge) — semantic suite 7/7.
- Lowercase queries (`cloudflare`, `url`, `api`) are not anchors, so the entity
  path never fired and the entity's dossier stayed hidden under weak FTS rows →
  added a case-insensitive entity-key/alias fallback for dossier retrieval
  (recall on the self-test went 6/10 → 10/10).
- Dossiers with a flat 5.0 score flooded recall earlier (60+ rows for limit=3) →
  capped at 2 slots with token-relevance filtering (fixed in v0.5.4).

## Notes / honesty

- The dossier self-test is a strict floor: it proves the memory resurfaces its
  own consolidated knowledge for a bare entity key. Semantic recall on natural
  multi-word questions is the upper layer and currently also 100%.
- Baseline to beat with every future feature: **any change must keep or raise
  these numbers on the same scenarios** (regression gate), then improve on new
  harder scenarios (e.g., cross-session "why did we switch X→Y" questions).
