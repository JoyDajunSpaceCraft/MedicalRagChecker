# -*- coding: utf-8 -*-
# All code comments are in English per your preference.

import os, re, time, mimetypes, hashlib, json
from pathlib import Path
import requests
from datasets import load_dataset
import trafilatura
from trafilatura.settings import use_config
from bs4 import BeautifulSoup  # fallback cleaner
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -----------------------
# Configs
# -----------------------
OUT_DIR = Path("medquad_corpus")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
REQUEST_TIMEOUT = 20
SLEEP_BETWEEN = 0.8  # be nice to NIH sites; adjust if rate-limited
USER_AGENT = "research-bot/1.0 (contact: your_email@example.com)"

# Choose embedding model (biomedical domain)
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "pritamdeka/S-PubMedBert-MS-MARCO")
# Alternative: "NeuML/pubmedbert-base-embeddings"

# -----------------------
# Helpers
# -----------------------
def hash_id(text: str) -> str:
    # Stable short hash for dedup keys and filenames
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

def is_pdf_url(url: str) -> bool:
    # Quick heuristics + MIME head check
    if url.lower().endswith(".pdf"):
        return True
    try:
        resp = requests.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True,
                             headers={"User-Agent": USER_AGENT})
        ctype = resp.headers.get("Content-Type", "")
        return "application/pdf" in ctype.lower()
    except Exception:
        return False

def fetch_html(url: str) -> str:
    # Fetch raw HTML with proper headers
    r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.text

def extract_text_html(url: str, raw_html: str) -> str:
    # Use trafilatura for robust main-content extraction, fallback to simple BS4
    cfg = use_config()
    cfg.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")  # no per-page timeout inside trafilatura
    extracted = trafilatura.extract(raw_html, url=url, config=cfg,
                                    include_comments=False,
                                    favor_precision=True)  # prefer precision on medical pages
    if extracted and extracted.strip():
        return extracted
    # Fallback: strip scripts/styles and get text
    soup = BeautifulSoup(raw_html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.extract()
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n").strip())
    return text

def extract_text_pdf(url: str) -> str:
    # Lightweight PDF extraction via pdfminer.six (no images/OCR)
    # pip install pdfminer.six
    import io
    from pdfminer.high_level import extract_text_to_fp
    r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    output = io.StringIO()
    extract_text_to_fp(io.BytesIO(r.content), output)
    return output.getvalue()

def normalize_whitespace(text: str) -> str:
    # Normalize whitespace while keeping paragraph boundaries
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, doc_meta: dict):
    # Recursive splitter preserves paragraph boundaries where possible
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    results = []
    for i, c in enumerate(chunks):
        results.append({
            "chunk_id": f"{doc_meta['document_id']}#{i}",
            "text": c,
            "meta": {
                **doc_meta,
                "chunk_index": i,
                "num_chunks": len(chunks)
            }
        })
    return results

def safe_write_jsonl(path: Path, records: list):
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------
# Step 1: Load dataset rows
# -----------------------
ds = load_dataset("lavita/MedQuAD", split="train")  # 47.4k rows

# -----------------------
# Step 2: Crawl & extract corpus
# -----------------------
seen_urls = set()
doc_records = []
corpus_path = OUT_DIR / "corpus.jsonl"

for row in ds:
    doc_id = row.get("document_id")
    url = row.get("document_url")
    if not url:
        continue
    key = (doc_id, url)
    if key in seen_urls:
        continue
    seen_urls.add(key)

    try:
        if is_pdf_url(url):
            text = extract_text_pdf(url)
        else:
            html = fetch_html(url)
            text = extract_text_html(url, html)

        text = normalize_whitespace(text)
        if not text or len(text) < 300:
            # Skip too-short pages; often navigation pages
            time.sleep(SLEEP_BETWEEN)
            continue

        # Build base metadata that keeps dataset linkage
        meta = {
            "document_id": doc_id,
            "document_source": row.get("document_source"),
            "document_url": url,
            "question_id": row.get("question_id"),
            "question_type": row.get("question_type"),
            "question_focus": row.get("question_focus"),
            "umls_cui": row.get("umls_cui"),
            "umls_semantic_types": row.get("umls_semantic_types"),
        }
        # Chunking
        chunks = chunk_text(text, meta)
        safe_write_jsonl(corpus_path, chunks)

    except Exception as e:
        # Log and continue
        print(f"[WARN] Failed {url}: {e}")
    finally:
        time.sleep(SLEEP_BETWEEN)

print(f"Corpus written to {corpus_path.resolve()}")

# -----------------------
# Step 3: Build embeddings + FAISS index
# -----------------------
# Load all chunks from JSONL (stream to avoid RAM spikes)
texts, metadatas = [], []
with corpus_path.open("r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        texts.append(r["text"])
        metadatas.append(r["meta"])

print(f"Total chunks: {len(texts)}")

model = SentenceTransformer(EMBED_MODEL_NAME)
emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

# Build FAISS index (cosine via Inner Product if normalized)
d = emb.shape[1]
index = faiss.IndexFlatIP(d)
index.add(emb)

# Persist index + metadata
faiss.write_index(index, str(OUT_DIR / "faiss.index"))
with (OUT_DIR / "meta.jsonl").open("w", encoding="utf-8") as f:
    for m in metadatas:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print("FAISS index and metadata saved.")

# -----------------------
# Step 4: Simple retriever demo
# -----------------------
def search(query: str, top_k=5):
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv, top_k)
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        results.append({
            "score": float(score),
            "text": texts[idx],
            "meta": metadatas[idx]
        })
    return results

# Example:
# hits = search("What are the symptoms of keratoderma with woolly hair?")
# for h in hits:
#     print(h["score"], h["meta"]["document_url"], h["meta"]["chunk_id"])
