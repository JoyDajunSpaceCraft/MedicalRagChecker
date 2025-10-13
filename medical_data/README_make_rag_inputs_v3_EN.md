
# Make RAG Inputs v3 — End‑to‑End Guide (English)

This README is tailored to your current repository layout and the script **`medical_data/make_rag_inputs_v3.py`**.  
It explains data preparation, per‑dataset runs, **full‑dataset runs**, and common debugging steps.

---

## 0) Folder Assumptions

- You run commands from the repo root (which contains `medical_data/`).
- You have local, *prebuilt* corpora / indices for:
  - **CSIRO**: `.../medical_data/MedRadQA/csiro_corpus/` (contains `faiss.index`, `texts.jsonl`, `meta.jsonl`)
  - **MedQuAD**: `.../medical_data/MedQuAD/medquad_corpus/` (same three files as above)
- Your Python environment has `faiss`, `sentence-transformers`, `datasets`, etc.

> If you later use **OpenAI** for evaluation (e.g., `smokeopenai.py`), export your key:
>
> ```bash
> export OPENAI_API_KEY=YOUR_KEY
> ```

---

## 1) Output Format (Target JSONL)

The script writes JSONL files into `--out-dir`, e.g.:

- `rag_generation_outputs_csiro.jsonl`
- `rag_generation_outputs_medquad.jsonl`
- `rag_generation_outputs_pubmedqa.jsonl`

Each JSON line follows a unified RAG schema, for example:
```json
{
  "query_id": "xxx",
  "query": "Question text",
  "gt_answer": "Reference answer",
  "rag_response": "",
  "retrieved_context": [
    {"doc_id": "doc-1", "text": "passage text", "title": null, "score": 0.83, "source": "csiro_faiss"}
  ],
  "retrieved_images": []
}
```

`retrieved_context` is populated by your selected retriever(s): MedRAG, CSIRO FAISS, MedQuAD, or their fusion.

---

## 2) Datasets, Splits, and Subsets

- **PubMedQA (`qiaojin/PubMedQA`)**:
  - Commonly used split: **train** only.
  - If your loader supports subsets, you can pass the subset (e.g., `pqa_labeled`, `pqa_artificial`).

- **MedQuAD (`lavita/MedQuAD`)**:
  - Has **train** only (no val/test).

- **LiveQA (`hyesunyun/liveqa_medical_trec2017`)**:
  - HF usually exposes **test** only. If you use LiveQA, set `--split test`.

- **CSIRO MedRedQA+PubMed**:
  - Uses local JSON files, typically named `medredqa+pubmed_{train,val,test}.json`.
  - Provide a folder via `--csiro-dir` or point to files via `--csiro-<split>`.

> **Note:** `--split` is a *uniform* CLI flag, but each dataset has its own reality:
> - PubMedQA / MedQuAD: *train* only.
> - LiveQA: *test* only.
> - CSIRO: whatever JSON files you have (train/val/test).

---

## 3) Minimal Examples (Single Dataset)

### 3.1 CSIRO Only (FAISS‐based retrieval)
```bash
python medical_data/make_rag_inputs_v3.py \
  --datasets csiro \
  --split train \
  --limit 50 \
  --out-dir tests/_min_input \
  --csiro-dir /ABS/PATH/medical_data/MedRadQA \
  --csiro-mode csiro_faiss \
  --csiro-index-dir /ABS/PATH/medical_data/MedRadQA/csiro_corpus \
  --csiro-model "pritamdeka/S-PubMedBert-MS-MARCO" \
  --k 8
```

### 3.2 PubMedQA Only (MedRAG retrieval)
```bash
python medical_data/make_rag_inputs_v3.py \
  --datasets pubmedqa \
  --split train \
  --limit 100 \
  --out-dir tests/_min_input \
  --retriever MedCPT \
  --corpus PubMed \
  --k 8
```

### 3.3 MedQuAD Only (Fuse MedQuAD + MedRAG)
```bash
python medical_data/make_rag_inputs_v3.py \
  --datasets medquad \
  --split train \
  --limit 100 \
  --out-dir tests/_min_input \
  --use-medquad \
  --medquad-dir /ABS/PATH/medical_data/MedQuAD/medquad_corpus \
  --medquad-hybrid \
  --medquad-alpha 0.65 \
  --retriever MedCPT \
  --corpus PubMed \
  --k 8
```

---

## 4) **Full‐Dataset Run** (recommended)

**Goal:** Produce RAG inputs for **CSIRO + MedQuAD + PubMedQA**, with **no item cap** (remove `--limit` or pass `--limit -1`).  
CSIRO uses *gold + FAISS* fusion; MedQuAD uses *MedQuAD + MedRAG* fusion; PubMedQA uses *MedRAG*.

```bash
python medical_data/make_rag_inputs_v3.py \
  --datasets csiro medquad pubmedqa \
  --split train \
  --out-dir runs/rag_inputs_full \
  --csiro-dir /ABS/PATH/medical_data/MedRadQA \
  --csiro-mode both \
  --csiro-index-dir /ABS/PATH/medical_data/MedRadQA/csiro_corpus \
  --csiro-model "pritamdeka/S-PubMedBert-MS-MARCO" \
  --csiro-rrf-k 60 \
  --csiro-weights "gold=1.0,csiro_faiss=1.2" \
  --use-medquad \
  --medquad-dir /ABS/PATH/medical_data/MedQuAD/medquad_corpus \
  --medquad-hybrid \
  --medquad-alpha 0.65 \
  --retriever MedCPT \
  --corpus PubMed \
  --k 8
```

- **No limit**: do **not** pass `--limit`, or pass `--limit -1` (the script treats negative as *no limit*).
- Adjust `--out-dir` to any writable location.

---

## 5) Expected Outputs

After success, `runs/rag_inputs_full/` should contain at least:

- `rag_generation_outputs_csiro.jsonl`
- `rag_generation_outputs_medquad.jsonl`
- `rag_generation_outputs_pubmedqa.jsonl`

If you only select some datasets, only those files will be created.

---

## 6) Common Issues & Debugging

1. **CSIRO file not found**  
   Use `--csiro-dir` or pass explicit `--csiro-train/--csiro-val/--csiro-test` paths that contain files like `medredqa+pubmed_train.json`.

2. **FAISS index not found**  
   - `--csiro-index-dir` must contain `faiss.index`, `texts.jsonl`, `meta.jsonl`.  
   - `--medquad-dir` must contain the same set of files for MedQuAD.  
   - If paths contain spaces, wrap them in quotes or switch to absolute paths.

3. **Empty retrieval for MedQuAD**  
   - Ensure the three files exist and are **non‑empty**.  
   - You can add a `--debug` mode in your script (print `len(hits)` per query, dump sample hits).

4. **Hugging Face access errors**  
   - Pre‑cache the datasets or configure a mirror if needed.

5. **Speed**  
   - First use of any retriever may build/load caches and be slower.  
   - Lowering `--k` reduces I/O and JSON size.

---

## 7) Optional: Quick Evaluation (Smoke OpenAI)

Once the JSONLs are created, you can do a quick sanity check with your `smokeopenai.py`:

```bash
export OPENAI_API_KEY=YOUR_KEY
python smokeopenai.py \
  --eval-backend gpt-4o-mini \
  --input-dir runs/rag_inputs_full \
  --output-dir runs/_eval_out \
  --limit 20 \
  --claims-jsonl runs/_eval_out/claims_dump.jsonl
```

> For a very fast check, keep `--limit` small.

---

## 8) LiveQA Only (if you need it)

```bash
python medical_data/make_rag_inputs_v3.py \
  --datasets liveqa \
  --split test \
  --out-dir runs/rag_inputs_liveqa \
  --retriever MedCPT \
  --corpus PubMed \
  --k 8
```

> LiveQA generally exposes **test** only on HF.

---

## 9) Tips

- Keep the embedding model (`--csiro-model`, `--medquad-model`) consistent with what you used to build the FAISS indices.
- Print key paths and parameter values (your script already prints several). If you need deeper insight, print `len(hits)` for each retriever and optionally save a few sample hits into a temporary JSONL for manual inspection.

Good luck!
