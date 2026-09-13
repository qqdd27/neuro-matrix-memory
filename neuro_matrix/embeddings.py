"""Optional local dense embeddings (stdlib only, over local HTTP).

Why this exists: retrieval in this project is entity/lexical (alias expansion
plus FTS5).  That is exactly what makes it fast, offline and dependency-free —
and exactly why a paraphrase that shares neither an anchor nor a content word
with the stored fact is missed.  Sparse and dense retrieval fail in
*orthogonal* ways, and fusing their ranked lists with Reciprocal Rank Fusion
beats either alone: EACL 2026 (T2-RAGBench, 23k queries) finds hybrid RRF
leading on every metric; on WANDS, RRF reaches 0.7068 NDCG against BM25 0.6983
and dense 0.6953, with a tuned hybrid at 0.7497.

Design constraints kept from the rest of this project:

* **No runtime dependency.**  Vectors come over local HTTP from Ollama — the
  same server the optional LLM already runs on.  No numpy, no model files
  shipped here, nothing leaves the machine.
* **Strictly optional.**  No embedding model installed means ``available()``
  is False and every caller silently keeps the existing entity/lexical path.
* **Never raises.**  Any failure (no server, no model, timeout, bad JSON)
  degrades to ``None``; a missing embedding must never break recall.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"

# bge-m3 is used instead of an English-only model because the facts this
# memory stores are frequently Russian while the queries may be English (or
# the reverse) — a monolingual encoder would put them in unrelated regions of
# the space.  Any Ollama-available embedding model works: set it in config.
_PREFERRED_MODELS = ("bge-m3", "qwen3-embedding", "mxbai-embed-large",
                     "nomic-embed-text", "snowflake-arctic-embed")


def _l2_normalize(vec: list[float]) -> list[float]:
    """Normalise once at write time so cosine similarity is a plain dot
    product later — the inner loop runs over every candidate fact."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return vec
    return [v / norm for v in vec]


class OllamaEmbedder:
    """Minimal client for Ollama's embedding endpoints.

    Uses ``/api/embed`` (batched, current) and falls back to the older
    one-text-per-request ``/api/embeddings`` when the server predates it.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 30.0,
    ) -> None:
        self.model = (model or "").strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        if self.base_url.endswith("/v1"):  # accept a pasted OpenAI-style URL
            self.base_url = self.base_url[:-3].rstrip("/")
        self.timeout_s = timeout_s
        self._dim: Optional[int] = None
        self._probe: Optional[bool] = None

    # ------------------------------------------------------------------ probe
    def _installed_models(self, timeout: float = 3.0) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return [str((m or {}).get("name") or "") for m in (body.get("models") or [])]
        except Exception:
            return []

    def available(self) -> bool:
        """True when an embedding model is usable.  Probed once and cached:
        this is consulted on every search call."""
        if self._probe is not None:
            return self._probe
        if not self.model:
            names = self._installed_models()
            for pref in _PREFERRED_MODELS:
                for n in names:
                    if n.split(":")[0] == pref:
                        self.model = n
                        break
                if self.model:
                    break
            if not self.model:
                self._probe = False
                return False
        names = self._installed_models()
        if not names:
            self._probe = False
            return False
        base = self.model.split(":")[0]
        self._probe = any(n == self.model or n.split(":")[0] == base for n in names)
        return self._probe

    @property
    def dim(self) -> Optional[int]:
        if self._dim is None and self.available():
            vec = self.embed_one("dimension probe")
            if vec:
                self._dim = len(vec)
        return self._dim

    # ------------------------------------------------------------------ embed
    def _post(self, path: str, payload: dict[str, Any], timeout: Optional[float] = None) -> Optional[dict[str, Any]]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body if isinstance(body, dict) else None
        except Exception:
            return None

    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Batch-embed `texts`.  Returns a list of L2-normalised vectors, or
        None if the batch could not be produced at all."""
        if not texts:
            return []
        if not self.available():
            return None
        body = self._post("/api/embed", {"model": self.model, "input": list(texts)})
        vecs: Optional[list[list[float]]] = None
        if body and isinstance(body.get("embeddings"), list):
            try:
                vecs = [[float(x) for x in v] for v in body["embeddings"]]
            except (TypeError, ValueError):
                vecs = None
        if vecs is None:  # older server: one request per text
            vecs = []
            for t in texts:
                one = self.embed_one(t)
                if one is None:
                    return None
                vecs.append(one)
            return vecs
        if len(vecs) != len(texts):
            return None
        if self._dim is None and vecs:
            self._dim = len(vecs[0])
        return [_l2_normalize(v) for v in vecs]

    def embed_one(self, text: str) -> Optional[list[float]]:
        if not text:
            return None
        if not self.available():
            return None
        body = self._post("/api/embeddings", {"model": self.model, "prompt": text})
        if body and isinstance(body.get("embedding"), list):
            try:
                vec = [float(x) for x in body["embedding"]]
            except (TypeError, ValueError):
                return None
            if self._dim is None:
                self._dim = len(vec)
            return _l2_normalize(vec)
        return None


def cosine(a: list[float], b: list[float]) -> float:
    """Similarity of two vectors.  Both sides are L2-normalised at write time,
    so this is a dot product; the fallback keeps it correct for any vector
    that arrived un-normalised (e.g. a store written by an older version)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    if len(a) > 4096:  # noqa: SIM108 - clarity over cleverness
        dot = 0.0
        for x, y in zip(a, b):
            dot += x * y
    else:
        dot = sum(x * y for x, y in zip(a, b))
    return dot


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    k: int = 20,
    weights: Optional[list[float]] = None,
) -> dict[int, float]:
    """Fuse ranked id lists by rank only (Cormack et al., SIGIR 2009).

    RRF was chosen over score blending deliberately: BM25/entity scores are
    unbounded while cosine sits in [-1, 1], so any weighted average silently
    hands the ranking to whichever scale happens to be larger.  Fusing ranks
    needs no normalisation and no tuning.

    ``k`` default is 20 rather than the usual 60: the TREC-scale default was
    set for corpora of thousands of documents, and this memory is small
    (hundreds to a few thousand facts) where rank differences are more
    meaningful — published ablations put the optimum for smaller corpora at
    k=10..20.
    """
    fused: dict[int, float] = {}
    for i, ids in enumerate(ranked_lists):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, doc_id in enumerate(ids, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w * (1.0 / (k + rank))
    return fused


def pack_vector(vec: list[float]) -> bytes:
    """float32 little-endian — 4 bytes per dimension.  At 1024 dims that is
    4 KB per fact: a 5000-fact store stays around 20 MB, and decoding in pure
    Python costs nothing at these sizes."""
    import struct
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    import struct
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob[: n * 4]))
