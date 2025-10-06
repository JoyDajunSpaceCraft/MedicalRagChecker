# -*- coding: utf-8 -*-
# All code comments are in English.

import os, re, json, time, hashlib, argparse, mimetypes
from pathlib import Path
from typing import List, Dict, Any

import requests
from datasets import load_dataset
from bs4 import BeautifulSoup
import trafilatura
from trafilatura.settings import use_config

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ------------------ Config ------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
REQ_TIMEOUT = 20
SLEEP_BETWEEN = 0.6
UA = "medquAD-crawler/1.0 (contact: you@example.com)"
EMBED_MODEL = os.environ.get("EMBED_MODEL_NAME", "pritamdeka/S-PubMedBert-MS-MARCO")

# ------------------ Utils -------------------
def _hash16(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def _is_pdf_url(url: str) -> bool:
    if url.lower().endswith(".pdf"):
        return True
    try:
        r = requests.head(url, timeout=REQ_TIMEOUT, allow_redirects=True, headers={"User-Agent": UA})
        ctype = r.headers.get("Content-Type", "")
        return "application/pdf" in ctype.lower()
    except Exception:
        return False

def _fetch_html(url: str) -> str:
    r = requests.get(url, timeout=REQ_TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    return r.text

def _extract_from_html(url: str, html: str) -> str:
    # Trafilatura focuses on main content extraction (precise mode)
    cfg = use_config(); cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
    text = trafilatura.extract(html, url=url, config=cfg, include_comments=False, favor_precision=True)
    if text and text.strip():
        return text
    # Fallback: simple readability-lite with BS4
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n").strip())

def _extract_from_pdf(url: str) -> str:
    import io
    from pdfminer.high_level import extract_text_to_fp
    r = requests.get(url, timeout=REQ_TIMEOUT, headers={"User-Agent": UA})
    r.raise_for_status()
    out = io.StringIO()
    extract_text_to_fp(io.BytesIO(r.content), out)
    return out.getvalue()

def _normalize_ws(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _chunk(text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    out = []
    for i, c in enumerate(chunks):
        out.append({
            "text": c,
            "meta": {**meta, "chunk_index": i, "num_chunks": len(chunks)}
        })
    return out

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------ Build -------------------
def build(out_dir: str = "medquad_corpus", split: str = "train", limit: int | None = None) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    corpus_path = out / "corpus.jsonl"

    ds = load_dataset("lavita/MedQuAD", split=split)  # HF dataset loader
    seen = set()
    kept = 0

    for i, row in enumerate(ds):
        if limit is not None and kept >= limit:
            break
        url = row.get("document_url") or row.get("source") or row.get("link")
        if not url:
            continue
        key = (row.get("document_id"), url)
        if key in seen:
            continue
        seen.add(key)

        try:
            if _is_pdf_url(url):
                txt = _extract_from_pdf(url)
            else:
                html = _fetch_html(url)
                txt = _extract_from_html(url, html)
            txt = _normalize_ws(txt)
            if not txt or len(txt) < 300:
                time.sleep(SLEEP_BETWEEN); continue

            base_meta = {
                "document_id": row.get("document_id"),
                "document_url": url,
                "document_source": row.get("document_source"),
                "question_id": row.get("question_id"),
                "question_type": row.get("question_type"),
                "question_focus": row.get("question_focus"),
                "umls_cui": row.get("umls_cui"),
                "umls_semantic_types": row.get("umls_semantic_types"),
            }
            chunks = _chunk(txt, base_meta)
            _write_jsonl(corpus_path, chunks)
            kept += 1
        except Exception as e:
            print(f"[WARN] {url} -> {e}")
        finally:
            time.sleep(SLEEP_BETWEEN)

    print(f"[build] wrote corpus to {corpus_path.resolve()}")

    # ---- Embedding + FAISS ----
    texts, metas = [], []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"]); metas.append(r["meta"])
    print(f"[embed] total chunks: {len(texts)}")

    model = SentenceTransformer(EMBED_MODEL)
    # normalize_embeddings=True -> unit vectors for cosine by inner product
    X = model.encode(texts, batch_size=64, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)          # cosine via normalized vectors + inner product
    index.add(X)

    faiss.write_index(index, str(out / "faiss.index"))
    with (out / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with (out / "texts.jsonl").open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[done] index saved to {out/'faiss.index'}; meta/texts saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="medquad_corpus")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    build(args.out_dir, args.split, args.limit)
