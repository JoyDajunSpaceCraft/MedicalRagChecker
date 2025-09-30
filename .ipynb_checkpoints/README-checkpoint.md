# MedRAG-Checker  
**Claim-level faithfulness evaluation for medical RAG, with distillation-ready pipeline**

> This repo builds a biomedical corpus and hybrid retriever, runs RAG with **claim-level** checking (faithfulness, hallucination, context precision/utilization, claim recall), and **distills** the checker (claim extractor + biomedical NLI) into efficient students for scalable evaluation.

---

## 0) TL;DR Quickstart

```bash
# 0) Create env (Python 3.9/3.10 recommended)
conda create -n medrag python=3.10 -y
conda activate medrag

# 1) Core deps
pip install -U pip wheel setuptools

# IR + Java bits
pip install pyserini==0.23.0  # brings Anserini bindings
# If you don't have Java in this env, install a conda OpenJDK:
conda install -c conda-forge openjdk=17 -y

# Vector search + encoders
pip install faiss-cpu==1.8.0.post1 sentence-transformers==2.7.0 transformers==4.43.4
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or your CUDA build

# Biomedical NLP helpers
pip install spacy scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

# Optional entity linking models
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_craft_md-0.5.4.tar.gz

# 2) Repo layout bootstrap (if missing)
mkdir -p data/{bioasq,medquad,pubmedqa,trec_liveqa,meddialog,healthsearchqa,ms2}
mkdir -p corpus/{pubmed,statpearls,nih_medlineplus,textbooks,wikipedia}/chunk
mkdir -p indexes/{bm25,faiss}
mkdir -p experiments/{runs,distill,reports,_demo_out}
```

**Run a smoke test** (replace with your driver):  
```bash
python run_rag_pipeline.py
```

> If Pyserini complains about Java/JNI, see **Appendix A: Java/JNI notes** below.

---

## 1) Datasets we use

We start with complementary, public medical/biomedical sets:

- **BioASQ Task B (2019–2023)** – expert biomedical questions (Y/N, Factoid, List, Summary). Has **PMIDs** and **ideal answers** (great for claim-level alignment).
- **MedQuAD** – ~47k Q/A pairs from 12 NIH sites (authoritative; linkable to the source page). Answers are long and well-structured.
- **PubMedQA** – research questions with **Yes/No/Maybe** and a **long conclusion paragraph** (use as reference explanation).
- **TREC LiveQA Medical (NLM)** – real consumer health Qs; reference long answers; helpful for long-form checking.
- **MedDialog (EN)** – doctor–patient dialogues; physician replies are long-form; we retrieve evidence from PubMed/MedlinePlus/StatPearls.
- **HealthSearchQA** – free-form consumer health questions (used by Med-PaLM); good for realistic long answers.
- **MS² (medical multi-doc summarization)** – evidence synthesis across multiple studies (stress-tests multi-source consistency).

> Place raw files in `data/<dataset>` or keep HF loader code in your dataset adapters.

---

## 2) Corpus construction & chunking

**Sources**  
- PubMed titles/abstracts (especially PMIDs referenced by BioASQ).  
- StatPearls (Bookshelf export).  
- NIH/MedlinePlus disease/condition pages.  
- Textbooks (public domain or licensed).  
- Wikipedia medical pages.

**Chunking rules**  
- **Section-aware split** by headings (e.g., *Causes*, *Diagnosis*, *Treatment*).  
- **Sliding windows** inside sections: **400–700 tokens**, stride **100–150**, sentence-aligned.  
- **Metadata** per chunk: `{id, url, title, section_path, text, source}` for precise attribution.  
- **De-dup** near-duplicates (cosine > 0.95 / MinHash).  
- **Normalization**: unicode/whitespace cleanup; **do not** stem biomedical terms.

Save chunks as **JSONL** in `corpus/<source>/chunk/*.jsonl` with fields:
```json
{"id": "srcA_000001", "url": "...", "title": "PCOS - Causes", "section_path": ["PCOS", "Causes"], "text": "…", "source": "nih_medlineplus"}
```

---

## 3) Retrieval indexes

We build **hybrid** retrieval:

- **BM25** with **Pyserini** (Lucene).  
- **Dense** with **FAISS** over Sentence-Transformer encoders (SPECTER / Contriever / MedCPT).

**Encoders you can enable**
- `"allenai/specter"` (paper-level, citation-aware)  
- `"facebook/contriever"` (unsupervised dense IR)  
- `"ncbi/MedCPT-Query-Encoder"` & its article encoder (biomedical domain)  
- (Optional) `"michiyasunaga/BioLinkBERT-base"` as an additional dense model

**Index build (examples)**

BM25 (per source):
```bash
pyserini.index.lucene \
  --collection JsonCollection \
  --input corpus/nih_medlineplus/chunk \
  --index indexes/bm25/nih_medlineplus \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16
```

Dense:
```bash
python tools/embed_corpus.py \
  --input corpus/nih_medlineplus/chunk \
  --encoder allenai/specter \
  --output indexes/faiss/nih_medlineplus/specter

python tools/build_faiss.py \
  --emb indexes/faiss/nih_medlineplus/specter/embedding.npy \
  --index indexes/faiss/nih_medlineplus/specter/faiss.index \
  --metric ip  # or l2
```

**Fusion at query time**: **RRF** across BM25 + (one or more) dense retrievers, then keep Top-K (e.g., 20–50) for generation **and** for checking.

---

## 4) Model components

### 4.1 Claim extractor (Teacher → Student)
- **Teacher**: a strong LLM converts a long answer into **atomic, verifiable claims** (1 claim/line).  
- **Student**: a compact biomedical model (e.g., **PubMedBERT-base**) fine-tuned either  
  - as **seq2seq** (claims as target text), or  
  - as **BIO tagger** to cut sentences/clauses into claims.

### 4.2 Biomedical NLI (Teacher → Student)
- **Teacher**: labels (claim, evidence-chunk) pairs as **entails / neutral / contradict**.  
- **Student**: **BioLinkBERT**/**PubMedBERT** fine-tuned for 3-way NLI; optional warm-start from MedNLI; then adapt with your teacher’s silver labels.

### 4.3 Entity linking (optional but useful)
- **scispaCy** UMLS linker (or SapBERT) to normalize entities → helps guideline adherence, safety detection, and de-duplication in evidence matching.

---

## 5) End-to-end pipeline

```
Question  → hybrid retrieval (Top-K chunks) → generator (optional) → long answer
            ↓
   Claim extractor (teacher/student) → list of atomic claims C = {c1..cN}
            ↓
Biomedical NLI (teacher/student) over (ci, chunk_j) → label ∈ {ENTAIL, NEUTRAL, CONTRADICT}
            ↓
Aggregate to metrics (Sec. 6) and per-claim attributions (supporting chunks)
```

**Driver example**: `run_rag_pipeline.py`  
- Builds the query (dataset adapter)  
- Calls `MedRAG.medrag_answer()` (RRF Top-K → prompt)  
- Saves `snippets.json` (Top-K) and `response.json` (answers)  
- Runs the checker to produce per-claim labels and final metrics

---

## 6) Metrics we report

All metrics are **claim-level** unless noted.

- **Claim-Faithfulness** = `#SUPPORTED / #AllClaims`  
- **Hallucination Rate** = `#REFUTED / #AllClaims`  
- **Claim-Recall** = `#GoldKeyClaimsCovered / #GoldKeyClaims` (keys from BioASQ ideal answers or MedQuAD bullets)  
- **Context-Precision (retrieval precision)** = `#r_chunks / TopK`  
  - *r_chunk*: any retrieved chunk that **entails at least one gold claim**  
- **Context-Utilization** = `#claims_with_support / #AllClaims`  
- **Safety-Critical Error Rate** (optional) = `#safety_violation_claims / #AllClaims` (dose/contraindication/triage, etc.)  
- **Guideline Adherence (weighted)** (optional) – weight supports by source authority (SR/RCT > cohort > review/site).

### Worked example (compact)
**Query**: “What causes PCOS?”  
**Gold claims (5)**: hormones↑; anovulation/follicle retention; not all have cystic ovaries; dx often teens/20–30s; cause unclear/multifactorial.  
**Top-K=10** chunks; r-chunks = 5 (chunks #0,#1,#2,#3,#6 each entail ≥1 gold claim).  
- Context-Precision = 5/10 = **0.50**  
- Suppose supported claims = 3/5 → Claim-Faithfulness = **0.60**  
- Refuted claims = 2/5 → Hallucination = **0.40**  
- Gold key claims covered = 3/5 → Claim-Recall = **0.60**  
- Claims with at least one supporting chunk = 3/5 → Context-Utilization = **0.60**

---

## 7) Distillation (how to make it fast)

**Teacher → Student** two heads:

1) **Claim extractor student**
   - Inputs: long answers (gold or generated)
   - Targets: teacher claims  
   - Loss: token-CE (seq2seq) or BIO tagging CE; add length/coverage constraints

2) **Biomedical NLI student**
   - Inputs: (claim, retrieved chunk) pairs  
   - Targets: teacher labels (E/N/C)  
   - Loss: 3-way CE; curriculum from high-overlap to paraphrastic pairs  
   - Add regularizers: entity overlap (CUI-Jaccard) to encourage medically-grounded decisions

**Export**: a **small checker** (extractor + NLI) you can batch over thousands of items without LLM calls.

---

## 8) Reporting & ablations

For each dataset slice (e.g., MedQuAD, BioASQ-B Summary):

- **Retrieval**: Recall@K on r-chunks; nDCG@K (gain = #claims entailed); latency (ms), index size (GB).  
- **Claim-level**: Faithfulness, Hallucination, Claim-Recall, Context-Precision, Context-Utilization (+ safety/guideline if enabled).  
- **Ablations**: BM25 vs MedCPT vs RRF; with/without entity linking; Teacher vs Student checker; chunk size/stride.

Provide a **Result Card** like:

| Slice | Retriever | K | Rec@K (r-chunk) | nDCG@K | Faithful | Halluc | C-Recall | C-Prec | C-Util |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MedQuAD-NIH | RRF-4 | 20 | 0.79 | 0.61 | 0.72 | 0.11 | 0.68 | 0.55 | 0.74 |

*(Numbers above are placeholders to illustrate the table.)*

---

## 9) Repo structure

```
.
├── adapters/                   # dataset adapters (build queries, read gold answers)
├── chunker/                    # heading-aware & sliding-window chunking
├── corpus/                     # {pubmed,statpearls,nih_medlineplus,textbooks,wikipedia}/chunk/*.jsonl
├── indexes/
│   ├── bm25/...                # Lucene indexes by source
│   └── faiss/...               # FAISS indexes by source & encoder
├── retriever/
│   ├── utils.py                # RRF hybrid retrieval (BM25 + dense)
│   └── medrag.py               # MedRAG class (answer, interactive loops)
├── checker/
│   ├── claim_extractor/        # teacher & student (seq2seq or tagger)
│   ├── biomed_nli/             # teacher & student (E/N/C)
│   └── metrics.py              # claim-level metrics
├── distill/
│   ├── make_silver.py          # run teacher to create silver labels
│   ├── train_extractor.py      # train student claim extractor
│   └── train_nli.py            # train student NLI
├── tools/
│   ├── embed_corpus.py         # encode chunks & dump embeddings
│   └── build_faiss.py          # build FAISS index
├── experiments/
│   ├── runs/                   # saved runs
│   ├── distill/                # checkpoints
│   └── reports/                # result cards
├── run_rag_pipeline.py         # demo driver (end-to-end)
└── README.md
```

---

## 10) Configuration knobs

- **Chunking**: `max_tokens=512`, `stride=128`, `min_sent=2`, `max_sent=12`  
- **Retrieval**: `K=20/32/50`, `rrf_k=100`, `bm25_k1=0.9`, `b=0.4`  
- **Dense encoders**: SPECTER (cosine/IP), Contriever (cosine/IP), MedCPT (pair encoders)  
- **Checker batching**: `batch_size=64` for student models; mixed-precision for speed  
- **Safety/guideline**: enable rulebook + NLI + source weighting

---

## 11) Reproducibility tips

- Fix random seeds (`pythonhashseed`, `torch`, `numpy`).  
- Persist `id→(url, section_path)` maps for every chunk.  
- Log Top-K doc IDs, final claims, per-pair NLI logits, and chosen supports.  
- Keep **exact** versions of encoders and Pyserini/FAISS.

---

## Appendix A: Java/JNI notes (Pyserini)

If you see errors like *“no segments* file found”, “cannot open libjvm.so”, or “Unknown split”*:

1. Make sure **OpenJDK** is installed inside this conda env:
   ```bash
   conda install -c conda-forge openjdk=17 -y
   ```
2. Ensure `JAVA_HOME` and `LD_LIBRARY_PATH` point to this JDK (before importing Pyserini):
   ```bash
   export JAVA_HOME="$CONDA_PREFIX/lib/jvm"
   export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
   ```
3. Rebuild the BM25 index **after** chunking exists:
   ```bash
   ls corpus/statpearls/chunk | head   # should list .jsonl files
   pyserini.index.lucene ...           # as in Section 3
   ```
4. For **PyJNIUS** custom setups, set (only if needed, before imports):
   ```bash
   export PYJNIUS_JAVA_HOME="$JAVA_HOME"
   export PYJNIUS_JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
   ```

---

## Appendix B: Minimal evaluation recipe

1) **Build corpus & indexes** as above.  
2) **Pick a dataset slice** (e.g., 1k MedQuAD QAs).  
3) **Run end-to-end**:
   ```bash
   python run_rag_pipeline.py \
     --dataset medquad \
     --k 20 \
     --retrievers RRF-4 \
     --out experiments/runs/medquad_rrf20
   ```
4) **Compute metrics** (claim extractor + NLI teacher or students):
   ```bash
   python checker/metrics.py \
     --run_dir experiments/runs/medquad_rrf20 \
     --gold data/medquad/gold.jsonl \
     --out experiments/reports/medquad_rrf20.json
   ```
5) **Distill**:
   ```bash
   # create silver labels first
   python distill/make_silver.py --in experiments/runs/... --out experiments/distill/silver/

   # train extractor
   python distill/train_extractor.py --data experiments/distill/silver/extractor.jsonl --out experiments/distill/extractor_student/

   # train NLI
   python distill/train_nli.py --data experiments/distill/silver/nli.jsonl --out experiments/distill/nli_student/
   ```

---

## License & Attribution

- This repo integrates public resources (NIH/MedlinePlus, PubMed/Bookshelf, Wikipedia) and public datasets (BioASQ, MedQuAD, PubMedQA, TREC LiveQA Medical, MedDialog, HealthSearchQA, MS²).  
- Please follow each dataset’s usage policy and cite as appropriate.

---

**Contact / Issues**  
Open a GitHub issue with: env details, exact command, and full stderr/stdout. Include your `JAVA_HOME`, `LD_LIBRARY_PATH`, and `pip freeze | grep -E '(pyserini|faiss|sentence-transformers|transformers)'`.
