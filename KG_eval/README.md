# KG_eval: Knowledge-Graph Fusion Evaluation Pipeline

This folder contains the scripts used to (1) generate **KG support scores** (from DRKG),
(2) **fuse** KG support into claim-level NLI probabilities produced by your checker,
and (3) **summarize**/export the results into a single CSV and LaTeX-ready numbers.

> Recommended usage: keep all scripts under `KG_eval/` and run them from your project root, e.g.
> `python KG_eval/kg_fusion_end2end.py ...`

---

## Directory layout (recommended)

```
project_root/
  KG_eval/
    kg_allinone_drkg.py
    kg_fusion_end2end.py
    collect_kgfuse_summary.py
    cleanup_kgfused.py
    legacy/
      kg_fusion_eval_allinone.py
    example_outputs/
      summary_kgfuse.csv
    README.md
  medical_data/                      # your `--root_dir` (data/results)
    ...                              # eval run folders with text_eval/ and soft_transe_scores*.jsonl
```

`KG_eval/` only hosts **code** and light artifacts. Your experiment outputs remain under your data root (`--root_dir`).

---

## What each script does

### 1) `kg_allinone_drkg.py` (optional / upstream)
**Goal:** produce a per-query KG score file (JSONL) such as:
- `soft_transe_scores.jsonl`
- `soft_transe_scores-*.jsonl`

**Input:**
- `--root_dir`: the experiment root that contains many subfolders (it will scan recursively)
- `--entity_emb`: DRKG entity embedding file (TransE/RotatE-style embeddings)
- `--entities_tsv`: DRKG entity list/TSV (entity id ↔ entity name)
- `--name_map`: optional name mapping file (normalization / alias mapping)

**Output (in each run folder):**
- `soft_transe_scores*.jsonl` (one JSON per line, keyed by `query_id`, includes a score like `p_final`,
  and optional hit flags like `node_hit` / `pair_hit`)
- an overall summary CSV/TEX if you set `--out_csv` / `--out_tex`

> Note: Your fusion script (`kg_fusion_end2end.py`) only needs the **JSONL** score file(s).  
> If you already have `soft_transe_scores*.jsonl`, you can skip this step.

---

### 2) `kg_fusion_end2end.py` (main fusion step)
**Goal:** fuse KG support scores into each claim’s NLI probabilities and write a new JSON file per checker output.

**Input (batch mode):**
- `--root_dir`: root directory to scan
- `--student_glob`: glob pattern to find checker outputs (default searches for
  `**/text_eval/student_checker_claim_probs__checker_*.json`)
- `--prefer_kge`: preference list to pick the KG score JSONL in the same folder (default includes
  `soft_transe_scores.jsonl,soft_transe_scores-*.jsonl,...`)

**Output:**
- For each student file:
  - `student_checker_claim_probs__checker_*.json`  →  `student_checker_claim_probs__checker_*__kgfused.json`
- The fused file contains:
  - `claim_outputs[*].kg_score` (raw or calibrated KG score)
  - `claim_outputs[*].kg_hit` / `kg_hit_type`
  - `claim_outputs[*].kg_fused` (fused probabilities + prediction)
  - `stu["kg_fusion_config"]` and `stu["kg_fusion_diagnostics"]` for reproducibility

**Key knobs:**
- `--beta`: weight on the NLI model in the logit-mixture fusion
  - `beta → 1`: trust NLI more
  - `beta → 0`: trust KG score more (only when KG hit exists)
- `--kg_calib`: calibration of raw KG score before fusion
  - `none`: clamp raw score into `[0,1]`
  - `minmax`: per-file min–max scaling to `[0,1]` using the JSONL’s `score_min/score_max`
  - `sigmoid`: monotonic squash (useful if raw scores are too peaky)
- `--hit_only`: only keep KG rows with `node_hit` or `pair_hit` when building the “top edges”
- `--attach_top_edges N`: store top-N KG edges under each claim (debug/analysis)

**Suffix safety (recommended defaults):**
- By default we **skip** inputs that already contain `__kgfused` and **clean up** redundant outputs that contain repeated tags.
- To disable these defaults:
  - `--no_skip_fused_inputs`
  - `--no_cleanup_redundant_outputs`

---

### 3) `collect_kgfuse_summary.py` (aggregate results)
**Goal:** aggregate all `*__kgfused.json` files under a root directory into one CSV summary.

**Input:**
- `--root_dir`
- `--in_glob` (default: `**/*__kgfused.json`)

**Output:**
- `--out_csv` (default: `summary_kgfuse.csv`)

This CSV is what you can use to generate tables/plots (ablation, backbone comparisons, etc.).

Example columns from `example_outputs/summary_kgfuse.csv` include:
- file, path, model, method, fusion_type, alpha, beta, n_queries, n_claims, fused_available_claims, fused_available_rate, kg_cov_claim_rate, kg_hit_claim_rate, kg_cov_fused_rate, kg_hit_fused_rate, kg_score_min, kg_score_p50, kg_score_p90, kg_score_max, kg_score_cov_min
- ... (total 44 columns)

---

### 4) `cleanup_kgfused.py` (utility)
**Goal:** remove redundant outputs (e.g., `__kgfused__kgfused.json`) created by accidental re-running.

Use it when you already generated many duplicates and want a quick cleanup before re-running summaries.

---

### 5) `legacy/kg_fusion_eval_allinone.py` (legacy / early prototype)
This was an earlier “all-in-one” evaluator that mixed multiple steps (and wrote its own summaries/tex).
It is kept for reference, but the recommended pipeline now is:

**DRKG score JSONL** → **`kg_fusion_end2end.py`** → **`collect_kgfuse_summary.py`**

---

## Typical workflow

### Step A. (Optional) Generate KG score JSONLs (DRKG)
If you do not yet have `soft_transe_scores*.jsonl` in each run folder:

```bash
python KG_eval/kg_allinone_drkg.py \
  --root_dir /path/to/medical_data \
  --entity_emb /path/to/drkg_entity_embeddings.npy \
  --entities_tsv /path/to/entities.tsv \
  --name_map /path/to/name_map.json \
  --out_csv drkg_scores_summary.csv
```

### Step B. Fuse KG into checker outputs (batch)
```bash
python KG_eval/kg_fusion_end2end.py \
  --root_dir /path/to/medical_data \
  --student_glob "**/text_eval/student_checker_claim_probs__checker_*.json" \
  --prefer_kge "soft_transe_scores.jsonl,soft_transe_scores-*.jsonl" \
  --hit_only --agg max --topk 10 \
  --beta 0.8 --kg_calib minmax \
  --verbose
```

Disable default skip/cleanup behavior if you need:
```bash
python KG_eval/kg_fusion_end2end.py \
  --root_dir /path/to/medical_data \
  --no_skip_fused_inputs \
  --no_cleanup_redundant_outputs \
  ...
```

### Step C. Summarize into a single CSV
```bash
python KG_eval/collect_kgfuse_summary.py \
  --root_dir /path/to/medical_data \
  --in_glob "**/*__kgfused.json" \
  --out_csv summary_kgfuse.csv \
  --verbose
```

### Step D. (Optional) Cleanup duplicated fused outputs
```bash
python KG_eval/cleanup_kgfused.py \
  --root_dir /path/to/medical_data \
  --glob "**/*__kgfused__kgfused*.json" \
  --dry_run   # remove this flag to actually delete
```

---

## Notes / common pitfalls

- **Why does my `store_true default=True` not work?**  
  `store_true` can only flip False → True. If you set `default=True`, the flag becomes “always on”.
  The fixed pattern used in `kg_fusion_end2end.py` is a pair of “disable” flags:
  `--no_skip_fused_inputs` and `--no_cleanup_redundant_outputs`.

- **What does “KG cov. (claim)” mean?**  
  Coverage is computed as the fraction of claims whose query has a KG **hit** and a non-zero/usable score,
  depending on your summarizer settings.

- **Single-file debugging:**  
  Use `--student_json`, `--kge_jsonl`, and `--out_json` to run fusion on one file.

---

## Quick checklist

- [ ] Each run folder contains `text_eval/student_checker_claim_probs__checker_*.json`
- [ ] Each run folder contains `soft_transe_scores*.jsonl`
- [ ] Run fusion → `*__kgfused.json` appears
- [ ] Run summary → `summary_kgfuse.csv` updated
