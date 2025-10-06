# -*- coding: utf-8 -*-
# All code comments are in English.

from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Optional BM25
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

def _tok(t: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_#+\-]+", (t or "").lower())

class MedQuADRetriever:
    """
    Dense (FAISS) retriever for medquad_corpus, with optional BM25 hybrid.
    """
    def __init__(self, corpus_dir: str, model_name: str,
                 use_hybrid: bool = False, alpha_dense: float = 0.6):
        p = Path(corpus_dir)
        self.index = faiss.read_index(str(p / "faiss.index"))
        # load texts & metas aligned to build order
        self.texts = [json.loads(l) for l in (p / "texts.jsonl").open("r", encoding="utf-8")]
        self.metas = [json.loads(l) for l in (p / "meta.jsonl").open("r", encoding="utf-8")]
        self.encoder = SentenceTransformer(model_name)
        self.use_hybrid = use_hybrid
        self.alpha_dense = alpha_dense
        if self.use_hybrid and BM25Okapi is not None:
            self.bm25 = BM25Okapi([_tok(t) for t in self.texts])
        else:
            self.bm25 = None

    def _dense(self, query: str, k: int) -> Tuple[List[float], List[int]]:
        qv = self.encoder.encode([query], convert_to_numpy=True,
                                 normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, k)
        return D[0].tolist(), I[0].tolist()

    @staticmethod
    def _minmax(x: List[float]) -> np.ndarray:
        arr = np.array(x, dtype=np.float32)
        if arr.size == 0:
            return arr
        mn, mx = float(arr.min()), float(arr.max())
        return np.zeros_like(arr) if mx - mn < 1e-8 else (arr - mn) / (mx - mn)

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        if not self.use_hybrid or self.bm25 is None:
            D, I = self._dense(query, top_k)
            hits = []
            for s, i in zip(D, I):
                meta = self.metas[i]; txt = self.texts[i]
                hits.append({
                    "doc_id": f"{meta.get('document_id')}#{meta.get('chunk_index',0)}",
                    "text": txt, "score": float(s),
                    "title": None,
                    "source": meta.get("document_url")
                })
            return hits

        # Hybrid: min-max normalize then linear mix
        D, I = self._dense(query, max(top_k, 200))
        d_norm = self._minmax(D)
        fused: Dict[int, float] = {i: self.alpha_dense * float(s) for i, s in zip(I, d_norm)}

        qtok = _tok(query); bm = self.bm25
        if bm is not None:
            scores = bm.get_scores(qtok)
            top = np.argpartition(scores, -max(top_k, 500))[-max(top_k, 500):]
            b_pairs = [(int(i), float(scores[i])) for i in top]
            b_norm = self._minmax([s for _, s in b_pairs])
            for (i, _), s in zip(b_pairs, b_norm):
                fused[i] = fused.get(i, 0.0) + (1 - self.alpha_dense) * float(s)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        hits = []
        for i, sc in ranked:
            meta = self.metas[i]; txt = self.texts[i]
            hits.append({
                "doc_id": f"{meta.get('document_id')}#{meta.get('chunk_index',0)}",
                "text": txt, "score": float(sc),
                "title": None,
                "source": meta.get("document_url")
            })
        return hits
