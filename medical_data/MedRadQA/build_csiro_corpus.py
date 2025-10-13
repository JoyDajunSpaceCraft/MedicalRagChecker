# -*- coding: utf-8 -*-
# All code comments are in English.
# 构建新的 CSIRO 索引
# python build_csiro_corpus.py --train medredqa+pubmed_train.json  --val   medredqa+pubmed_val.json  --test  medredqa+pubmed_test.json  --out_dir csiro_corpus

from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import List, Dict, Any, Iterable

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------- Config --------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBED_MODEL = os.environ.get("EMBED_MODEL_NAME", "pritamdeka/S-PubMedBert-MS-MARCO")

# -------------- I/O helpers ----------
def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("CSIRO files should have a top-level list.")
        return data

def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -------------- Chunking ------------
def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)

# -------------- Main build ----------
def build_corpus_and_index(
    train_path: str,
    val_path: str,
    test_path: str,
    out_dir: str = "csiro_corpus",
) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) Load all three splits
    files = [
        ("train", train_path),
        ("val",   val_path),
        ("test",  test_path),
    ]
    all_chunks: List[Dict[str, Any]] = []
    for split, jpath in files:
        if not jpath:
            continue
        rows = read_json(jpath)
        for i, ex in enumerate(rows):
            q = (ex.get("question") or "").strip()
            doc = (ex.get("document") or "").strip()
            # skip empty docs
            if not doc:
                continue
            chunks = chunk_text(doc)
            for ci, ch in enumerate(chunks):
                all_chunks.append({
                    "text": ch,
                    "meta": {
                        "dataset": "csiro/medredqa_pubmed",
                        "split": split,
                        "local_id": i,
                        "chunk_index": ci,
                        "num_chunks": len(chunks),
                        # keep question and (optionally) response for backref/debug
                        "question": q[:5000],
                        "has_response": bool(ex.get("response")),
                    }
                })

    # 2) Persist corpus.jsonl (aligned order is critical for FAISS)
    corpus_path = out / "corpus.jsonl"
    write_jsonl(all_chunks, str(corpus_path))

    # 3) Build embeddings
    texts = [r["text"] for r in all_chunks]
    metas = [r["meta"] for r in all_chunks]

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine via IP
    ).astype("float32")

    # 4) Build FAISS (cosine = normalized + inner product)
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    # 5) Save index + meta (and optionally texts)
    faiss.write_index(index, str(out / "faiss.index"))
    write_jsonl(metas, str(out / "meta.jsonl"))
    # Optional: if you want a standalone texts file
    with (out / "texts.jsonl").open("w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[done] corpus={corpus_path}  index={out/'faiss.index'}  meta={out/'meta.jsonl'}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build FAISS over CSIRO MedRedQA+PubMed documents.")
    ap.add_argument("--train", required=True, help="Path to medredqa+pubmed_train.json")
    ap.add_argument("--val",   required=True, help="Path to medredqa+pubmed_val.json")
    ap.add_argument("--test",  required=True, help="Path to medredqa+pubmed_test.json")
    ap.add_argument("--out_dir", default="csiro_corpus")
    args = ap.parse_args()
    build_corpus_and_index(args.train, args.val, args.test, args.out_dir)
