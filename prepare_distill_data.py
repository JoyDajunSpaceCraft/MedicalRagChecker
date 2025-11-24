#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build SFT/GRPO datasets for extractor and checker from your results files.
Inputs:
  - results_text.json (aggregated) OR claims.jsonl (line-by-line) OR multi-root directory scan
Outputs (under --outdir):
  - extractor_sft.jsonl
  - checker_sft.jsonl
  - checker_grpo.jsonl
  - extractor_infer_inputs.jsonl (if --emit_infer_inputs)
  - teacher_claims.jsonl (if --emit_teacher_claims)
  - gt_claims.jsonl (if --emit_gt_claims)

# Full data use
python prepare_distill_data.py \
  --results_roots \
 /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_liveqa-4o/liveqa_test.Meditron3-8B.gen100__gpt-4o-6d9372 \
 /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_csiro_val_gpt-4.o/csiro_val.Meditron3-8B.gen100__gpt-4o-6d9372 \
 /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_medquad-4o/medquad_train.Meditron3-8B.gen100__gpt-4o-6d9372 \
 /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_pubmedqa-4.1/pubmedqa_train.Meditron3-8B.gen100__gpt-4.1-df63c0 \
 /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_pubmedqa-4o/pubmedqa_train.Meditron3-8B.gen100__gpt-4o-6d9372 \
 --results_glob "results_text*.json" \
 --recurse \
 --dedup_key query_id \
 --outdir ./data \
 --emit_infer_inputs \
 --emit_teacher_claims \
 --emit_gt_claims
"""
import os
import json
import argparse
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

# --------- Label canon & constants ---------
LBL_CANON = {"Entailment": "entailed", "Contradiction": "contradicted", "Neutral": "neutral"}
LABELS = ["entailed", "contradicted", "neutral"]
MIN_RESP_LEN = 10            # minimal non-whitespace length for response
MIN_EVID_LEN = 40            # minimal evidence length for checker
ALLOW_FALLBACK_TO_GT = True  # enable using gt claims when response/claims are empty

# --------- Robust evidence/text extraction helpers ---------
TEXT_KEYS = ["text", "content", "passage", "chunk", "body", "abstract", "snippet", "document"]
ID_KEYS   = ["doc_id", "pmid", "id", "uuid", "hash"]
NAME_KEYS = ["title", "name", "doc_name", "filename"]

def norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    return rows

def maybe_triples_list(obj: Any) -> Optional[List[List[str]]]:
    """Return list[[s,r,o], ...] if obj looks like triples; else None."""
    if not isinstance(obj, list):
        return None
    out: List[List[str]] = []
    for it in obj:
        if isinstance(it, list) and len(it) == 3:
            s, r, o = [norm(str(x)) for x in it]
            if s and r and o:
                out.append([s, r, o])
    return out if out else None

def row_id(r: Dict[str, Any]) -> Optional[str]:
    return r.get("id") or r.get("query_id")

def is_nonempty(s: str, min_len: int = 1) -> bool:
    return bool((s or "").strip()) and len((s or "").strip()) >= min_len

def triples_to_sentence(x: Any) -> str:
    if isinstance(x, list) and len(x) == 3:
        return f"{x[0]} {x[1]} {x[2]}."
    return str(x)

def flatten_claims(claims: Any) -> List[str]:
    out: List[str] = []
    for c in (claims or []):
        if isinstance(c, list) and len(c) == 3:
            out.append(triples_to_sentence(c))
        else:
            out.append(str(c))
    # de-dup and normalize
    seen = set()
    dedup: List[str] = []
    for t in out:
        nt = norm(t)
        if nt and nt.lower() not in seen:
            dedup.append(nt)
            seen.add(nt.lower())
    return dedup

def coerce_triples(claims: Any) -> List[List[str]]:
    out: List[List[str]] = []
    for c in (claims or []):
        if isinstance(c, list) and len(c) == 3:
            s, r, o = [norm(str(x)) for x in c]
            if s and r and o:
                out.append([s, r, o])
    return out

def get_doc_text(d: Dict[str, Any]) -> str:
    """Return best-effort evidence text from a context dict."""
    # direct fields
    for k in TEXT_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return norm(v)
    # nested metadata
    meta = d.get("metadata") or d.get("meta") or {}
    if isinstance(meta, dict):
        for k in TEXT_KEYS + NAME_KEYS:
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return norm(v)
    # last resort: join medium strings
    pieces = []
    for _, v in d.items():
        if isinstance(v, str) and 20 <= len(v) <= 2000:
            pieces.append(v.strip())
    return norm(" ".join(pieces)) if pieces else ""

def doc_key_for_match(d: Dict[str, Any]) -> Optional[str]:
    """Return a stable key to match labels dict, if available."""
    for k in ID_KEYS + NAME_KEYS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def align_labels_for_claim(labels_row: Any, ctx_docs: List[Dict[str, Any]]) -> List[Optional[str]]:
    """
    Align one row of labels (list or dict) to ctx_docs.
    Returns list of labels length == len(ctx_docs); unknowns -> None.
    """
    out: List[Optional[str]] = [None] * len(ctx_docs)
    if isinstance(labels_row, list):
        m = min(len(labels_row), len(ctx_docs))
        for i in range(m):
            out[i] = labels_row[i]
        return out
    if isinstance(labels_row, dict):
        key2idx = {}
        for i, d in enumerate(ctx_docs):
            k = doc_key_for_match(d) or ""
            if k:
                key2idx[k] = i
        for k, v in labels_row.items():
            if k in key2idx:
                out[key2idx[k]] = v
            else:
                kl = str(k).lower()
                for kk, ii in key2idx.items():
                    if kk.lower() == kl:
                        out[ii] = v
                        break
        return out
    return out

# --------- Multi-root loader & de-duplicator ---------
def _iter_result_files(roots: Iterable[str], pattern: str, recurse: bool) -> Iterable[Path]:
    """Yield result JSON files from multiple roots matching pattern."""
    seen: set[Path] = set()
    for r in roots:
        base = Path(r)
        it = base.rglob(pattern) if recurse else base.glob(pattern)
        for p in it:
            if p.is_file():
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield rp

def collect_from_multi_roots(
    roots: Iterable[str],
    pattern: str = "results_text*.json",
    recurse: bool = True,
    max_files: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, int]]]:
    """Load many results_text*.json files across directories. Returns (rows, per_file_counts)."""
    all_rows: List[Dict[str, Any]] = []
    per_file_counts: List[Tuple[str, int]] = []
    for i, fp in enumerate(_iter_result_files(roots, pattern, recurse)):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            rows = obj.get("results", [])
            if isinstance(rows, list) and rows:
                all_rows.extend(rows)
                per_file_counts.append((str(fp), len(rows)))
        except Exception as e:
            print(f"[WARN] failed to load {fp}: {e}")
        if max_files is not None and (i + 1) >= max_files:
            break
    return all_rows, per_file_counts

def _row_uid(row: Dict[str, Any], key_pref: List[str]) -> str:
    """Stable uid for cross-file de-duplication."""
    for k in key_pref:
        v = (row.get(k) or "").strip()
        if v:
            return f"{k}:{v}"
    # fallback to content hash of (query + response)
    q = (row.get("query") or row.get("question") or "").strip()
    a = (row.get("response") or row.get("answer") or "").strip()
    h = hashlib.md5((q + "\n" + a).encode("utf-8")).hexdigest()
    return f"md5:{h}"

def dedup_rows(rows: List[Dict[str, Any]], key_pref: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        u = _row_uid(r, key_pref)
        if u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out

# --------- Readers for legacy single-file inputs ---------
def collect_from_results_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])

def collect_from_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

# --------- Builders ---------
def to_extractor_example(ex: Dict[str, Any]) -> Dict[str, str]:
    # SFT target = triples, not plain strings
    q = ex.get("query", "") or ""
    resp = ex.get("response", "") or ""

    tgt = coerce_triples(ex.get("gt_answer_claims"))
    if not tgt:
        tgt = coerce_triples(ex.get("response_claims"))
    if not tgt:
        raise ValueError("no triple targets")

    if not is_nonempty(resp, MIN_RESP_LEN):
        resp = ex.get("gt_answer", "") or ""
    if not is_nonempty(resp, MIN_RESP_LEN) and not is_nonempty(q, 1):
        raise ValueError("no answer/question content")

    prompt = (
        "You are an information extraction assistant. "
        "Extract factual triples from the given answer.\n"
        "Return a pure JSON array of triples as [[subject, relation, object], ...].\n"
        "Do not include explanations."
        "\n\nQuestion:\n"
        f"{q}\n\nAnswer:\n{resp}\n"
    )
    target = json.dumps(tgt, ensure_ascii=False)
    return {"instruction": prompt, "output": target}

def to_checker_examples(ex: Dict[str, Any], min_overlap: float = 0.0) -> List[Dict[str, str]]:
    # Build checker pairs robust to schema drift
    ctx = ex.get("retrieved_context") or []
    r2r = ex.get("retrieved2response") or []
    claims = flatten_claims(ex.get("response_claims"))
    if not claims and ALLOW_FALLBACK_TO_GT:
        claims = flatten_claims(ex.get("gt_answer_claims"))

    out: List[Dict[str, str]] = []
    if not claims or not ctx or not r2r:
        return out

    # Build cleaned evidence docs (text + raw)
    ev_docs = []
    for d in ctx:
        t = get_doc_text(d or {})
        if is_nonempty(t, MIN_EVID_LEN):
            ev_docs.append({"text": t, "_raw": d})
    if not ev_docs:
        return out

    for ci, claim in enumerate(claims):
        if ci >= len(r2r):
            break
        labels_row = align_labels_for_claim(r2r[ci], [e["_raw"] for e in ev_docs])
        for di, lab_raw in enumerate(labels_row):
            if not lab_raw:
                continue
            lbl = LBL_CANON.get(lab_raw) or LBL_CANON.get(str(lab_raw).title())
            if lbl not in LABELS:
                continue
            prompt = (
                "Decide whether the EVIDENCE entails, contradicts, or is neutral to the CLAIM.\n"
                "Respond with one of: entailed | contradicted | neutral\n\n"
                f"CLAIM:\n{claim}\n\nEVIDENCE:\n{ev_docs[di]['text']}\n\nLabel:"
            )
            out.append({"prompt": prompt, "label": lbl})
    return out

# --------- Optional rebalancing (downsample/upsample/none) ---------
def rebalance_pairs(pairs: List[Dict[str, str]], mode: str = "none", seed: int = 42) -> List[Dict[str, str]]:
    if mode not in {"downsample", "upsample"}:
        return pairs
    rng = random.Random(seed)
    buckets = {"entailed": [], "contradicted": [], "neutral": []}
    for x in pairs:
        y = (x.get("label") or "").strip()
        if y in buckets:
            buckets[y].append(x)
    sizes = {k: len(v) for k, v in buckets.items()}
    if not sizes or min(sizes.values()) == 0:
        return pairs  # avoid degenerate cases
    if mode == "downsample":
        m = min(sizes.values())
        out: List[Dict[str, str]] = []
        for k, v in buckets.items():
            out.extend(rng.sample(v, m) if len(v) > m else list(v))
        rng.shuffle(out)
        return out
    else:  # upsample
        M = max(sizes.values())
        out: List[Dict[str, str]] = []
        for _, v in buckets.items():
            if not v:
                continue
            cur = list(v)
            while len(cur) < M:
                cur.append(rng.choice(v))
            out.extend(cur[:M])
        rng.shuffle(out)
        return out

# --------- Claims jsonl helpers ---------
def build_infer_input_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ex in rows:
        qid = ex.get("query_id") or ex.get("id")
        q = ex.get("query") or ex.get("question") or ""
        a = ex.get("response") or ""
        if not is_nonempty(a, MIN_RESP_LEN):
            a = ex.get("gt_answer") or ""
        if not (is_nonempty(a, MIN_RESP_LEN) or is_nonempty(q, 1)):
            continue
        out.append({"query_id": qid, "question": norm(q), "answer": norm(a)})
    return out

def rows_to_claims_jsonl(rows: List[Dict[str, Any]], key: str = "response_claims") -> List[Dict[str, Any]]:
    out = []
    for ex in rows:
        qid = ex.get("query_id") or ex.get("id")
        claims = flatten_claims(ex.get(key))
        if claims:
            out.append({"query_id": qid, "pred_claims": claims})
    return out

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser()
    # inputs
    ap.add_argument("--results_json", default=None, help="Single aggregated results_text.json")
    ap.add_argument("--claims_jsonl", default=None, help="Alternative: claims.jsonl (line-by-line)")
    ap.add_argument("--results_roots", nargs="+", default=None, help="One or more roots with results_text*.json")
    ap.add_argument("--results_glob", default="results_text*.json", help="Glob pattern inside roots")
    ap.add_argument("--recurse", action="store_true", help="Recurse into subdirectories when scanning roots")
    ap.add_argument("--max_files", type=int, default=None, help="Optional cap on matched files (debug)")
    ap.add_argument("--dedup_key", choices=["query_id", "id", "query", "none"], default="query_id",
                    help="Primary key for cross-file de-duplication (fallback to content hash)")
    # outputs
    ap.add_argument("--outdir", default="data", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    # extras
    ap.add_argument("--emit_infer_inputs", action="store_true",
                    help="Also write extractor_infer_inputs.jsonl for student inference.")
    ap.add_argument("--infer_inputs_name", default="extractor_infer_inputs.jsonl",
                    help="Filename for inference-only inputs (question/answer/query_id).")
    ap.add_argument("--emit_teacher_claims", action="store_true",
                    help="Also write teacher_claims.jsonl from response_claims (if available).")
    ap.add_argument("--teacher_claims_name", default="teacher_claims.jsonl",
                    help="Filename for teacher claims jsonl.")
    ap.add_argument("--emit_gt_claims", action="store_true",
                    help="Also write gt_claims.jsonl from gt_answer_claims (if available).")
    ap.add_argument("--gt_claims_name", default="gt_claims.jsonl",
                    help="Filename for GT claims jsonl.")
    # unified-eval branch
    ap.add_argument("--infer_inputs_jsonl", default=None,
                    help="id/query/answer rows used for unified eval input.")
    ap.add_argument("--teacher_claims_jsonl", default=None,
                    help="Optional teacher references (pred_triples or pred_claims).")
    ap.add_argument("--gt_claims_jsonl", default=None,
                    help="Optional GT references (pred_triples or pred_claims).")
    ap.add_argument("--emit_eval_unified", default=None,
                    help="Path to write unified eval jsonl (id, query, answer, teacher_triples/gt_triples).")
    ap.add_argument("--emit_extractor_sft", default=None,
                    help="Path to write extractor SFT jsonl built from unified eval.")
    # rebalancing option
    ap.add_argument("--rebalance_checker", choices=["none", "downsample", "upsample"], default="none",
                    help="Rebalance checker pairs before saving (affects both SFT/GRPO)")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # unified-eval short-circuit
    if args.infer_inputs_jsonl:
        infer_rows = load_jsonl(args.infer_inputs_jsonl)
        teacher_rows = load_jsonl(args.teacher_claims_jsonl) if args.teacher_claims_jsonl else []
        gt_rows = load_jsonl(args.gt_claims_jsonl) if args.gt_claims_jsonl else []

        t_map = {row_id(r): r for r in teacher_rows if row_id(r)}
        g_map = {row_id(r): r for r in gt_rows if row_id(r)}

        eval_unified = []
        sft_rows = []
        dropped_sft = 0

        for r in infer_rows:
            qid = row_id(r)
            q = r.get("query") or r.get("question") or ""
            a = r.get("answer") or r.get("response") or ""
            u = {"id": qid, "query": q, "answer": a}

            # teacher
            tr = t_map.get(qid)
            if tr:
                t_tri = maybe_triples_list(tr.get("pred_triples"))
                if not t_tri:
                    tri2 = maybe_triples_list(tr.get("pred_claims"))
                    if tri2:
                        t_tri = tri2
                    else:
                        t_claims = tr.get("pred_claims") or []
                        if t_claims:
                            u["teacher_claims"] = [norm(str(x)) for x in t_claims if str(x).strip()]
                if t_tri:
                    u["teacher_triples"] = t_tri

            # gt
            gr = g_map.get(qid)
            if gr:
                g_tri = maybe_triples_list(gr.get("pred_triples"))
                if not g_tri:
                    tri2 = maybe_triples_list(gr.get("pred_claims"))
                    if tri2:
                        g_tri = tri2
                    else:
                        g_claims = gr.get("pred_claims") or []
                        if g_claims:
                            u["gt_claims"] = [norm(str(x)) for x in g_claims if str(x).strip()]
                if g_tri:
                    u["gt_triples"] = g_tri

            eval_unified.append(u)

            # SFT row (prefer triples; else claims)
            instr = (
                "You are an information extraction assistant.\n"
                "Extract factual triples strictly from the given content.\n"
                "Return ONLY a valid JSON array of triples: [[\"subject\",\"relation\",\"object\"], ...]."
            )
            content = []
            if q: content.append(f"Question:\n{q}")
            if a: content.append(f"Answer:\n{a}")
            prompt = instr + "\n\n" + "\n\n".join(content)

            target_triples = u.get("gt_triples") or u.get("teacher_triples")
            if target_triples:
                sft_rows.append({"instruction": prompt, "output": json.dumps(target_triples, ensure_ascii=False)})
            else:
                target_claims = u.get("gt_claims") or u.get("teacher_claims")
                if target_claims:
                    sft_rows.append({"instruction": prompt, "output": json.dumps(target_claims, ensure_ascii=False)})
                else:
                    dropped_sft += 1

        eval_unified_path = args.emit_eval_unified or os.path.join(args.outdir, "eval_unified.jsonl")
        sft_out_path      = args.emit_extractor_sft or os.path.join(args.outdir, "extractor_sft.jsonl")

        with open(eval_unified_path, "w", encoding="utf-8") as f:
            for r in eval_unified:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(sft_out_path, "w", encoding="utf-8") as f:
            for r in sft_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"✅ eval_unified: {len(eval_unified)} | extractor_sft: {len(sft_rows)} (dropped {dropped_sft})")
        return

    # load rows: single file / multi-roots / claims-jsonl
    rows: List[Dict[str, Any]] = []
    per_file_stats: List[Tuple[str, int]] = []

    if args.results_json and os.path.isfile(args.results_json):
        rows = collect_from_results_json(args.results_json)

    elif args.results_roots:
        rows, per_file_stats = collect_from_multi_roots(
            roots=args.results_roots,
            pattern=args.results_glob,
            recurse=args.recurse,
            max_files=args.max_files,
        )
        pref = [] if args.dedup_key == "none" else [args.dedup_key]
        if pref:
            rows = dedup_rows(rows, key_pref=pref)

    elif args.claims_jsonl and os.path.isfile(args.claims_jsonl):
        rows = collect_from_jsonl(args.claims_jsonl)

    else:
        raise FileNotFoundError("Provide one of: --results_json OR --results_roots ... OR --claims_jsonl")

    if per_file_stats:
        print("[info] loaded files:")
        for path_s, n in per_file_stats:
            print(f"  - {path_s}: {n}")
        print(f"[info] total raw rows before shuffle/drops: {len(rows)}")

    random.Random(args.seed).shuffle(rows)

    # Build extractor SFT
    extractor_sft: List[Dict[str, str]] = []
    drop_extractor = 0
    for ex in rows:
        try:
            sample = to_extractor_example(ex)
            extractor_sft.append(sample)
        except Exception:
            drop_extractor += 1

    # Build checker SFT/GRPO
    checker_sft: List[Dict[str, str]] = []
    checker_grpo: List[Dict[str, str]] = []
    drop_checker = 0
    for ex in rows:
        exs = to_checker_examples(ex)
        if exs:
            checker_sft.extend(exs)
            checker_grpo.extend(exs)
        else:
            drop_checker += 1

    # Optional rebalancing
    if args.rebalance_checker != "none":
        checker_sft = rebalance_pairs(checker_sft, mode=args.rebalance_checker, seed=args.seed)
        checker_grpo = list(checker_sft)

    # Save
    with open(os.path.join(args.outdir, "extractor_sft.jsonl"), "w", encoding="utf-8") as f:
        for r in extractor_sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.outdir, "checker_sft.jsonl"), "w", encoding="utf-8") as f:
        for r in checker_sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.outdir, "checker_grpo.jsonl"), "w", encoding="utf-8") as f:
        for r in checker_grpo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Extras
    infer_stats = {}
    if args.emit_infer_inputs:
        infer_rows = build_infer_input_rows(rows)
        infer_path = os.path.join(args.outdir, args.infer_inputs_name)
        with open(infer_path, "w", encoding="utf-8") as f:
            for r in infer_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        infer_stats["infer_inputs"] = {"path": infer_path, "n": len(infer_rows)}

    if args.emit_teacher_claims:
        tea_rows = rows_to_claims_jsonl(rows, key="response_claims")
        tea_path = os.path.join(args.outdir, args.teacher_claims_name)
        with open(tea_path, "w", encoding="utf-8") as f:
            for r in tea_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        infer_stats["teacher_claims"] = {"path": tea_path, "n": len(tea_rows)}

    if args.emit_gt_claims:
        gt_rows = rows_to_claims_jsonl(rows, key="gt_answer_claims")
        gt_path = os.path.join(args.outdir, args.gt_claims_name)
        with open(gt_path, "w", encoding="utf-8") as f:
            for r in gt_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        infer_stats["gt_claims"] = {"path": gt_path, "n": len(gt_rows)}

    msg = (
        f"✅ extractor_sft: {len(extractor_sft)} (dropped {drop_extractor}) | "
        f"checker_sft: {len(checker_sft)} (dropped {drop_checker}) | "
        f"checker_grpo: {len(checker_grpo)})"
    )
    if infer_stats:
        msg += " | extras=" + json.dumps(infer_stats, ensure_ascii=False)
    print(msg)

if __name__ == "__main__":
    main()

# # prepare_distill_data.py

# # # 任选其一作为数据源（推荐 results_text.json，因为包含上下文）
# # INPUT_JSON=/ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_pubmedqa-4.1/pubmedqa_train.Meditron3-8B.gen100__gpt-4.1-df63c0/results_text__gpt-4.1-df63c0.json
# # python prepare_distill_data.py   --results_json "$INPUT_JSON"   --outdir ./data   --emit_infer_inputs --emit_teacher_claims  --emit_gt_claims
# # ✅ extractor_sft: 200 (dropped 0) | checker_sft: 3112 (dropped 54) | checker_grpo: 3112) | extras={"infer_inputs": {"path": "./data/extractor_infer_inputs.jsonl", "n": 200}, "teacher_claims": {"path": "./data/teacher_claims.jsonl", "n": 146}, "gt_claims": {"path": "./data/gt_claims.jsonl", "n": 200}}

# # only for the extractor
# # python prepare_distill_data.py \
# #   --infer_inputs_jsonl ./data/extractor_infer_inputs.jsonl \
# #   --teacher_claims_jsonl ./data/teacher_claims.jsonl \
# #   --gt_claims_jsonl ./data/gt_claims.jsonl \
# #   --outdir ./data
# # # 输出：
# # #   ./data/eval_unified.jsonl
# # #   ./data/extractor_sft.jsonl
# """
# python prepare_distill_data.py \
#   --results_roots \
#     /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_csiro_val_gpt-4.o/csiro_val.Meditron3-8B.gen100__gpt-4o-6d9372 \
#     /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_csiro_val_gpt-4.o \
#     /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_otherset \
#   --results_glob "results_text*.json" \
#   --recurse \
#   --outdir ./data \
#   --emit_infer_inputs \
#   --emit_teacher_claims \
#   --emit_gt_claims
# """
# # -*- coding: utf-8 -*-
# """
# Build SFT/GRPO datasets for extractor and checker from your results files.
# Inputs:
#   - results_text.json (aggregated) OR claims.jsonl (line-by-line)
# Outputs:
#   - data/extractor_sft.jsonl
#   - data/checker_sft.jsonl
#   - data/checker_grpo.jsonl
#   - data/extractor_infer_inputs.jsonl
# """

# import os, json, argparse, random
# from typing import List, Dict, Any

# LBL_CANON = {"Entailment":"entailed","Contradiction":"contradicted","Neutral":"neutral"}
# LABELS = ["entailed","contradicted","neutral"]  # keep fixed order
# # English-only comments: thresholds and helpers to avoid empty/garbage samples
# MIN_RESP_LEN = 10           # minimal non-whitespace length for response
# MIN_EVID_LEN = 40           # minimal evidence length for checker
# ALLOW_FALLBACK_TO_GT = True # enable using gt claims when response/claims are empty
# # --- additions: helpers to unify eval & training (keep comments in English only) ---
# # ==== NEW: multi-root loader & deduper (English-only comments) ====
# from pathlib import Path
# import hashlib
# from typing import Iterable, Tuple

# def _iter_result_files(roots: Iterable[str], pattern: str, recurse: bool) -> Iterable[Path]:
#     """Yield result JSON files from multiple roots matching pattern."""
#     seen = set()
#     for r in roots:
#         base = Path(r)
#         it = base.rglob(pattern) if recurse else base.glob(pattern)  # '**' matches recursively
#         for p in it:
#             if p.is_file():
#                 rp = p.resolve()
#                 if rp not in seen:
#                     seen.add(rp)
#                     yield rp

# def collect_from_multi_roots(
#     roots: Iterable[str],
#     pattern: str = "results_text*.json",
#     recurse: bool = True,
#     max_files: int | None = None,
# ) -> tuple[list[dict], list[tuple[str,int]]]:
#     """Load many results_text*.json files across directories. Returns (rows, per_file_counts)."""
#     all_rows: list[dict] = []
#     per_file_counts: list[tuple[str,int]] = []
#     for i, fp in enumerate(_iter_result_files(roots, pattern, recurse)):
#         try:
#             obj = json.load(open(fp, "r", encoding="utf-8"))
#             rows = obj.get("results", [])
#             if isinstance(rows, list) and rows:
#                 all_rows.extend(rows)
#                 per_file_counts.append((str(fp), len(rows)))
#         except Exception as e:
#             print(f"[WARN] failed to load {fp}: {e}")
#         if max_files is not None and (i+1) >= max_files:
#             break
#     return all_rows, per_file_counts

# def _row_uid(row: dict, key_pref: list[str]) -> str:
#     """Stable uid for cross-file de-duplication."""
#     for k in key_pref:
#         v = (row.get(k) or "").strip()
#         if v:
#             return f"{k}:{v}"
#     # fallback to content hash of (query + response)
#     q = (row.get("query") or row.get("question") or "").strip()
#     a = (row.get("response") or row.get("answer") or "").strip()
#     h = hashlib.md5((q + "\n" + a).encode("utf-8")).hexdigest()
#     return f"md5:{h}"

# def dedup_rows(rows: list[dict], key_pref: list[str]) -> list[dict]:
#     """De-duplicate rows by preferred keys, then content hash fallback."""
#     seen = set()
#     out = []
#     for r in rows:
#         u = _row_uid(r, key_pref)
#         if u in seen:
#             continue
#         seen.add(u)
#         out.append(r)
#     return out
# # ==== /NEW ====

# def load_jsonl(path: str) -> List[Dict[str, Any]]:
#     rows = []
#     with open(path, "r", encoding="utf-8") as f:
#         for ln in f:
#             if ln.strip():
#                 rows.append(json.loads(ln))
#     return rows

# def maybe_triples_list(obj):
#     """Return list[[s,r,o], ...] if obj looks like triples; else None."""
#     if not isinstance(obj, list):
#         return None
#     out = []
#     for it in obj:
#         if isinstance(it, list) and len(it) == 3:
#             s, r, o = [norm(str(x)) for x in it]
#             if s and r and o:
#                 out.append([s, r, o])
#     return out if out else None

# def row_id(r):
#     return r.get("id") or r.get("query_id")

# def is_nonempty(s: str, min_len: int = 1) -> bool:
#     return bool((s or "").strip()) and len((s or "").strip()) >= min_len

# def flatten_claims(claims):
#     out = []
#     for c in (claims or []):
#         if isinstance(c, list) and len(c) == 3:
#             out.append(triples_to_sentence(c))
#         else:
#             out.append(str(c))
#     # de-dup and normalize
#     seen = set()
#     dedup = []
#     for t in out:
#         nt = norm(t)
#         if nt and nt.lower() not in seen:
#             dedup.append(nt)
#             seen.add(nt.lower())
#     return dedup
# def coerce_triples(claims):
#     """
#     Return a list of triples [[subj, rel, obj], ...].
#     Keep only items that are exactly 3-length lists after string-norm.
#     """
#     out = []
#     for c in (claims or []):
#         if isinstance(c, list) and len(c) == 3:
#             s, r, o = [norm(str(x)) for x in c]
#             if s and r and o:
#                 out.append([s, r, o])
#     return out

# def norm(s:str)->str:
#     return " ".join((s or "").strip().split())

# def triples_to_sentence(x)->str:
#     if isinstance(x,list) and len(x)==3:
#         return f"{x[0]} {x[1]} {x[2]}."
#     return str(x)
# def build_infer_input_rows(rows):
#     """
#     Build inference-only inputs for extractor:
#       [{"query_id","question","answer"}...]
#     Prefer 'response' as the answer; fallback to 'gt_answer' if response is empty.
#     """
#     out = []
#     for ex in rows:
#         qid = ex.get("query_id") or ex.get("id")
#         q = ex.get("query") or ex.get("question") or ""
#         a = ex.get("response") or ""
#         if not is_nonempty(a, MIN_RESP_LEN):
#             a = ex.get("gt_answer") or ""
#         if not (is_nonempty(a, MIN_RESP_LEN) or is_nonempty(q, 1)):
#             continue
#         out.append({"query_id": qid, "question": norm(q), "answer": norm(a)})
#     return out

# def rows_to_claims_jsonl(rows, key="response_claims"):
#     """
#     Convert results rows to {"query_id","pred_claims":[...]} jsonl using given field:
#       - key="response_claims": teacher claims (if teacher extracted)
#       - key="gt_answer_claims": ground-truth claims
#     """
#     out = []
#     for ex in rows:
#         qid = ex.get("query_id") or ex.get("id")
#         claims = flatten_claims(ex.get(key))
#         if claims:
#             out.append({"query_id": qid, "pred_claims": claims})
#     return out

# def collect_from_results_json(path:str)->List[Dict[str,Any]]:
#     data = json.load(open(path,"r",encoding="utf-8"))
#     # expect {"mode": "...", "metrics": {...}, "results":[...]}
#     return data.get("results", [])

# def collect_from_jsonl(path:str)->List[Dict[str,Any]]:
#     rows=[]
#     with open(path,"r",encoding="utf-8") as f:
#         for line in f:
#             rows.append(json.loads(line))
#     return rows

# def to_extractor_example(ex):
#     # English-only comments: SFT target = triples, not plain strings
#     q = ex.get("query", "") or ""
#     resp = ex.get("response", "") or ""
#     # prefer GT triples
#     tgt = coerce_triples(ex.get("gt_answer_claims"))
#     # fallback to teacher triples
#     if not tgt:
#         tgt = coerce_triples(ex.get("response_claims"))

#     # If no triples at all, drop this sample
#     if not tgt:
#         raise ValueError("no triple targets")

#     # If response is empty but gt_answer exists, use it as answer text for extraction supervision
#     if not is_nonempty(resp, MIN_RESP_LEN):
#         resp = ex.get("gt_answer", "") or ""

#     if not is_nonempty(resp, MIN_RESP_LEN) and not is_nonempty(q, 1):
#         raise ValueError("no answer/question content")

#     prompt = (
#         "You are an information extraction assistant. "
#         "Extract factual triples from the given answer.\n"
#         "Return a pure JSON array of triples as [[subject, relation, object], ...].\n"
#         "Do not include explanations."
#         "\n\nQuestion:\n"
#         f"{q}\n\nAnswer:\n{resp}\n"
#     )
#     target = json.dumps(tgt, ensure_ascii=False)
#     # NOTE: keep the same schema keys the trainer expects ("instruction","output")
#     return {"instruction": prompt, "output": target}

# # --- OPTIONAL: class rebalance for checker_sft / checker_grpo ---
# # English-only comments: lightweight rebalancing without external deps
# def _rebalance_pairs(pairs: list[dict], target: str = "downsample", seed: int = 42) -> list[dict]:
#     import random
#     rng = random.Random(seed)
#     by = {"entailed": [], "contradicted": [], "neutral": []}
#     for x in pairs:
#         y = (x.get("label") or "").strip()
#         if y in by: by[y].append(x)
#     sizes = {k: len(v) for k,v in by.items()}
#     if target == "downsample":
#         m = min(sizes.values()) if sizes else 0
#         out = []
#         for k,v in by.items():
#             if len(v) > m:
#                 out.extend(rng.sample(v, m))
#             else:
#                 out.extend(v)
#         rng.shuffle(out)
#         return out
#     elif target == "upsample":
#         M = max(sizes.values()) if sizes else 0
#         out = []
#         for k,v in by.items():
#             if not v: 
#                 continue
#             while len(v) < M:
#                 v.append(rng.choice(v))
#             out.extend(v[:M])
#         rng.shuffle(out)
#         return out
#     return pairs

# # Example: enable by env flag to avoid surprising changes
# if os.getenv("REBALANCE_CHECKER", "0") == "1":
#     checker_sft = _rebalance_pairs(checker_sft, target=os.getenv("REBALANCE_MODE", "downsample"))
#     checker_grpo = list(checker_sft)  # keep same pool for GRPO bootstrap



# def to_checker_examples(ex, min_overlap: float = 0.0):
#     # English-only comments: build checker pairs only when all fields are usable
#     ctx = ex.get("retrieved_context") or []
#     r2r = ex.get("retrieved2response") or []
#     claims = flatten_claims(ex.get("response_claims"))

#     # Optional fallback: if claims empty but gt claims exist, use them
#     if not claims and ALLOW_FALLBACK_TO_GT:
#         claims = flatten_claims(ex.get("gt_answer_claims"))

#     out = []
#     if not claims or not ctx or not r2r:
#         return out

#     # strip evidence and filter by minimal length
#     ev_texts = [norm((d.get("text") or "")) for d in ctx]
#     ev_texts = [t for t in ev_texts if is_nonempty(t, MIN_EVID_LEN)]
#     if not ev_texts:
#         return out

#     # align r2r with filtered evidence indices (conservative: keep indices up to original len)
#     # We will map by original order; if lengths mismatch, we truncate to the shortest to be safe
#     max_docs = min(len(ev_texts), min(len(x) for x in r2r)) if r2r else 0
#     if max_docs <= 0:
#         return out

#     for ci, claim in enumerate(claims):
#         if ci >= len(r2r):
#             break
#         labels_row = r2r[ci][:max_docs]
#         for di in range(max_docs):
#             ev = ev_texts[di]
#             lbl_raw = labels_row[di]
#             lbl = LBL_CANON.get(lbl_raw)
#             if lbl not in LABELS:
#                 continue
#             prompt = (
#                 "Decide whether the EVIDENCE entails, contradicts, or is neutral to the CLAIM.\n"
#                 "Respond with one of: entailed | contradicted | neutral\n\n"
#                 f"CLAIM:\n{claim}\n\nEVIDENCE:\n{ev}\n\nLabel:"
#             )
#             out.append({"prompt": prompt, "label": lbl})
#     return out

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--results_json", default=None, help="Path to results_text.json")
#     ap.add_argument("--claims_jsonl", default=None, help="Path to claims.jsonl (optional)")
#     ap.add_argument("--outdir", default="data", help="Output dir")
#     ap.add_argument("--seed", type=int, default=42)
#     # Extra outputs for inference and evaluation
#     ap.add_argument("--emit_infer_inputs", action="store_true",
#                     help="Also write extractor_infer_inputs.jsonl for student inference.")
#     ap.add_argument("--infer_inputs_name", default="extractor_infer_inputs.jsonl",
#                     help="Filename for inference-only inputs (question/answer/query_id).")
#     ap.add_argument("--emit_teacher_claims", action="store_true",
#                     help="Also write teacher_claims.jsonl from response_claims (if available).")
#     ap.add_argument("--teacher_claims_name", default="teacher_claims.jsonl",
#                     help="Filename for teacher claims jsonl.")
#     ap.add_argument("--emit_gt_claims", action="store_true",
#                     help="Also write gt_claims.jsonl from gt_answer_claims (if available).")
#     ap.add_argument("--gt_claims_name", default="gt_claims.jsonl",
#                     help="Filename for GT claims jsonl.")
#     # Optional unified-eval mode (no new file needed elsewhere)
#     ap.add_argument("--infer_inputs_jsonl", default=None,
#                     help="id/query/answer rows used for unified eval input.")
#     ap.add_argument("--teacher_claims_jsonl", default=None,
#                     help="Optional teacher references (pred_triples or pred_claims).")
#     ap.add_argument("--gt_claims_jsonl", default=None,
#                     help="Optional GT references (pred_triples or pred_claims).")
#     ap.add_argument("--emit_eval_unified", default=None,
#                     help="Path to write unified eval jsonl (id, query, answer, teacher_triples/gt_triples).")
#     ap.add_argument("--emit_extractor_sft", default=None,
#                     help="Path to write extractor SFT jsonl built from unified eval.")

#     args = ap.parse_args()
#     os.makedirs(args.outdir, exist_ok=True)
    
#     # 先处理统一评测分支；命中就直接 return，不再走老逻辑
#     if args.infer_inputs_jsonl:
#         infer_rows = load_jsonl(args.infer_inputs_jsonl)
#         teacher_rows = load_jsonl(args.teacher_claims_jsonl) if args.teacher_claims_jsonl else []
#         gt_rows = load_jsonl(args.gt_claims_jsonl) if args.gt_claims_jsonl else []
    
#         t_map = {row_id(r): r for r in teacher_rows if row_id(r)}
#         g_map = {row_id(r): r for r in gt_rows if row_id(r)}
    
#         eval_unified = []
#         sft_rows = []
#         dropped_sft = 0
    
#         for r in infer_rows:
#             qid = row_id(r)
#             q   = r.get("query") or r.get("question") or ""
#             a   = r.get("answer") or r.get("response") or ""
#             u = {"id": qid, "query": q, "answer": a}
    
#             # teacher
#             tr = t_map.get(qid)
#             if tr:
#                 t_tri = maybe_triples_list(tr.get("pred_triples"))
#                 if not t_tri:
#                     tri2 = maybe_triples_list(tr.get("pred_claims"))
#                     if tri2:
#                         t_tri = tri2
#                     else:
#                         t_claims = tr.get("pred_claims") or []
#                         if t_claims:
#                             u["teacher_claims"] = [norm(str(x)) for x in t_claims if str(x).strip()]
#                 if t_tri:
#                     u["teacher_triples"] = t_tri
    
#             # gt
#             gr = g_map.get(qid)
#             if gr:
#                 g_tri = maybe_triples_list(gr.get("pred_triples"))
#                 if not g_tri:
#                     tri2 = maybe_triples_list(gr.get("pred_claims"))
#                     if tri2:
#                         g_tri = tri2
#                     else:
#                         g_claims = gr.get("pred_claims") or []
#                         if g_claims:
#                             u["gt_claims"] = [norm(str(x)) for x in g_claims if str(x).strip()]
#                 if g_tri:
#                     u["gt_triples"] = g_tri
    
#             eval_unified.append(u)
    
#             # SFT 行（优先三元组、否则 claims）
#             instr = (
#                 "You are an information extraction assistant.\n"
#                 "Extract factual triples strictly from the given content.\n"
#                 "Return ONLY a valid JSON array of triples: [[\"subject\",\"relation\",\"object\"], ...]."
#             )
#             content = []
#             if q: content.append(f"Question:\n{q}")
#             if a: content.append(f"Answer:\n{a}")
#             prompt = instr + "\n\n" + "\n\n".join(content)
    
#             target_triples = u.get("gt_triples") or u.get("teacher_triples")
#             if target_triples:
#                 sft_rows.append({"instruction": prompt, "output": json.dumps(target_triples, ensure_ascii=False)})
#             else:
#                 target_claims = u.get("gt_claims") or u.get("teacher_claims")
#                 if target_claims:
#                     sft_rows.append({"instruction": prompt, "output": json.dumps(target_claims, ensure_ascii=False)})
#                 else:
#                     dropped_sft += 1
    
#         eval_unified_path = args.emit_eval_unified or os.path.join(args.outdir, "eval_unified.jsonl")
#         sft_out_path      = args.emit_extractor_sft or os.path.join(args.outdir, "extractor_sft.jsonl")
    
#         with open(eval_unified_path, "w", encoding="utf-8") as f:
#             for r in eval_unified:
#                 f.write(json.dumps(r, ensure_ascii=False) + "\n")
#         with open(sft_out_path, "w", encoding="utf-8") as f:
#             for r in sft_rows:
#                 f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
#         print(f"✅ eval_unified: {len(eval_unified)} | extractor_sft: {len(sft_rows)} (dropped {dropped_sft})")
#         return  # 命中统一评测分支后直接结束
    
#     # 走到这里说明没走统一评测；按旧逻辑加载 results_json / claims_jsonl
#     # rows=[]
#     # if args.results_json and os.path.isfile(args.results_json):
#     #     rows = collect_from_results_json(args.results_json)
#     # elif args.claims_jsonl and os.path.isfile(args.claims_jsonl):
#     #     rows = collect_from_jsonl(args.claims_jsonl)
#     # else:
#     #     raise FileNotFoundError("Provide --results_json or --claims_jsonl")
#         rows = []
#         per_file_stats = []
    
#         if args.results_json and os.path.isfile(args.results_json):
#             # single aggregated results JSON (old path)
#             rows = collect_from_results_json(args.results_json)
    
#         elif args.results_roots:
#             # multi-root mode: scan many result files across dirs
#             rows, per_file_stats = collect_from_multi_roots(
#                 roots=args.results_roots,
#                 pattern=args.results_glob,
#                 recurse=args.recurse,
#                 max_files=args.max_files,
#             )
#             # de-dup by preferred key(s)
#             pref = [args.dedup_key] if args.dedup_key != "none" else []
#             if pref:
#                 rows = dedup_rows(rows, key_pref=pref)
    
#         elif args.claims_jsonl and os.path.isfile(args.claims_jsonl):
#             # claims-jsonl path (old path)
#             rows = collect_from_jsonl(args.claims_jsonl)
    
#         else:
#             raise FileNotFoundError(
#                 "Provide one of: --results_json OR --results_roots ... OR --claims_jsonl"
#             )
    
#         if per_file_stats:
#             # light report: which files contributed how many rows
#             print("[info] loaded files:")
#             for path_s, n in per_file_stats:
#                 print(f"  - {path_s}: {n}")
#             print(f"[info] total raw rows: {len(rows)}")

   

#     random.Random(args.seed).shuffle(rows)

   
#     extractor_sft = []
#     drop_extractor = 0
#     for ex in rows:
#         try:
#             sample = to_extractor_example(ex)
#             extractor_sft.append(sample)
#         except Exception:
#             drop_extractor += 1
    
#     # Build Checker SFT/GRPO
#     checker_sft, checker_grpo = [], []
#     drop_checker = 0
#     for ex in rows:
#         exs = to_checker_examples(ex)
#         if exs:
#             checker_sft.extend(exs)
#             checker_grpo.extend(exs)
#         else:
#             drop_checker += 1

#     # Save
#     with open(os.path.join(args.outdir,"extractor_sft.jsonl"),"w",encoding="utf-8") as f:
#         for r in extractor_sft:
#             f.write(json.dumps(r, ensure_ascii=False)+"\n")
#     with open(os.path.join(args.outdir,"checker_sft.jsonl"),"w",encoding="utf-8") as f:
#         for r in checker_sft:
#             f.write(json.dumps(r, ensure_ascii=False)+"\n")
#     with open(os.path.join(args.outdir,"checker_grpo.jsonl"),"w",encoding="utf-8") as f:
#         for r in checker_grpo:
#             f.write(json.dumps(r, ensure_ascii=False)+"\n")
#         # ---- Extra outputs for inference / evaluation ----
#     infer_stats = {}
#     if args.emit_infer_inputs:
#         infer_rows = build_infer_input_rows(rows)
#         infer_path = os.path.join(args.outdir, args.infer_inputs_name)
#         with open(infer_path, "w", encoding="utf-8") as f:
#             for r in infer_rows:
#                 f.write(json.dumps(r, ensure_ascii=False) + "\n")
#         infer_stats["infer_inputs"] = {"path": infer_path, "n": len(infer_rows)}

#     if args.emit_teacher_claims:
#         tea_rows = rows_to_claims_jsonl(rows, key="response_claims")
#         tea_path = os.path.join(args.outdir, args.teacher_claims_name)
#         with open(tea_path, "w", encoding="utf-8") as f:
#             for r in tea_rows:
#                 f.write(json.dumps(r, ensure_ascii=False) + "\n")
#         infer_stats["teacher_claims"] = {"path": tea_path, "n": len(tea_rows)}

#     if args.emit_gt_claims:
#         gt_rows = rows_to_claims_jsonl(rows, key="gt_answer_claims")
#         gt_path = os.path.join(args.outdir, args.gt_claims_name)
#         with open(gt_path, "w", encoding="utf-8") as f:
#             for r in gt_rows:
#                 f.write(json.dumps(r, ensure_ascii=False) + "\n")
#         infer_stats["gt_claims"] = {"path": gt_path, "n": len(gt_rows)}

    
#     msg = (
#         f"✅ extractor_sft: {len(extractor_sft)} (dropped {drop_extractor}) | "
#         f"checker_sft: {len(checker_sft)} (dropped {drop_checker}) | "
#         f"checker_grpo: {len(checker_grpo)})"
#     )
#     if infer_stats:
#         msg += " | extras=" + json.dumps(infer_stats, ensure_ascii=False)
#     print(msg)


# if __name__ == "__main__":
#     main()
