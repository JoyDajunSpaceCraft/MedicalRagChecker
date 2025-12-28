#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:
python kg_fusion_end2end.py \
  --root_dir /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data \
  --student_glob "**/text_eval/student_checker_claim_probs__checker_*.json" \
  --prefer_kge "soft_transe_scores.jsonl,soft_transe_scores-*.jsonl" \
  --hit_only --agg max --beta 0.8 --kg_calib minmax --verbose
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _clamp(p: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, p))


def _logit(p: float, eps: float = 1e-6) -> float:
    # English-only comment: numerically stable logit
    p = _clamp(p, eps, 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    # English-only comment: numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _minmax(x: float, lo: Optional[float], hi: Optional[float], eps: float = 1e-12) -> float:
    # English-only comment: safe min-max scaling, returns 0 if range is degenerate
    if lo is None or hi is None or (hi - lo) <= eps:
        return 0.0
    return (x - lo) / (hi - lo)


def _calibrate_score(x: float, method: str, meta: Dict[str, Any]) -> float:
    """
    Calibrate raw KG score into [0,1].
    method:
      - none: clamp raw to [0,1]
      - minmax: file-level minmax based on meta['score_min'/'score_max']
      - sigmoid: a simple monotonic squash (not learned Platt scaling)
    """
    if method == "none":
        return _clamp(float(x), 0.0, 1.0)

    if method == "minmax":
        y = _minmax(float(x), meta.get("score_min"), meta.get("score_max"))
        return _clamp(y, 0.0, 1.0)

    if method == "sigmoid":
        # English-only comment: center raw score around 0.5 and squash; useful when raw is already [0,1]
        y = _sigmoid(6.0 * (float(x) - 0.5))
        return _clamp(y, 0.0, 1.0)

    raise ValueError(f"Unknown kg_calib method: {method}")


def build_kge_index(
    kge_jsonl: str,
    hit_only: bool = False,
    agg: str = "max",
    topk: int = 10,
    score_key: str = "p_final",
    alt_score_keys: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Build an index: query_id -> aggregated KG score and stats.
    Returns:
      - idx: dict(query_id -> record)
      - meta: diagnostics about the jsonl fields / distributions
    """
    if alt_score_keys is None:
        alt_score_keys = ["score", "prob", "final_score", "p", "p_final_score"]

    idx = defaultdict(
        lambda: {
            "max_p_final": 0.0,
            "n_rows": 0,
            "n_node_hit": 0,
            "n_pair_hit": 0,
            "top": [],  # list of (p_final, edge_obj)
            "score": 0.0,
            "hit": False,
            "hit_type": "none",
        }
    )

    meta: Dict[str, Any] = {
        "path": kge_jsonl,
        "n_lines": 0,
        "n_parse_err": 0,
        "n_missing_qid": 0,
        "n_missing_score": 0,
        "n_has_node_hit": 0,
        "n_has_pair_hit": 0,
        "n_hit_true": 0,
        "score_key_used": defaultdict(int),
        "score_min": None,
        "score_max": None,
        "score_gt0": 0,
    }

    def _get_score(d: Dict[str, Any]) -> Tuple[float, str]:
        if score_key in d:
            return _safe_float(d.get(score_key), 0.0), score_key
        for k in alt_score_keys:
            if k in d:
                return _safe_float(d.get(k), 0.0), k
        return 0.0, ""

    with open(kge_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            meta["n_lines"] += 1
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                meta["n_parse_err"] += 1
                continue

            qid = d.get("query_id", None)
            if qid is None:
                meta["n_missing_qid"] += 1
                continue
            qid = str(qid)

            p_final, used_key = _get_score(d)
            if not used_key:
                meta["n_missing_score"] += 1
            else:
                meta["score_key_used"][used_key] += 1

            # Track score stats
            if meta["score_min"] is None or p_final < meta["score_min"]:
                meta["score_min"] = p_final
            if meta["score_max"] is None or p_final > meta["score_max"]:
                meta["score_max"] = p_final
            if p_final > 0:
                meta["score_gt0"] += 1

            node_hit = bool(d.get("node_hit", False))
            pair_hit = bool(d.get("pair_hit", False))
            if "node_hit" in d:
                meta["n_has_node_hit"] += 1
            if "pair_hit" in d:
                meta["n_has_pair_hit"] += 1
            if node_hit or pair_hit:
                meta["n_hit_true"] += 1

            rec = idx[qid]
            rec["n_rows"] += 1
            rec["max_p_final"] = max(rec["max_p_final"], p_final)

            if node_hit:
                rec["n_node_hit"] += 1
            if pair_hit:
                rec["n_pair_hit"] += 1

            keep = True
            if hit_only and (not node_hit) and (not pair_hit):
                keep = False

            if keep:
                rec["top"].append((p_final, d))

    # Finalize each query record
    for _qid, rec in idx.items():
        rec["top"].sort(key=lambda x: x[0], reverse=True)
        rec["top"] = rec["top"][: max(0, topk)]

        if agg == "max":
            # If hit_only and we kept top edges, use top[0]; otherwise fallback to max_p_final
            rec["score"] = rec["top"][0][0] if (hit_only and rec["top"]) else rec["max_p_final"]
        elif agg == "mean_topk":
            vals = [p for p, _ in rec["top"]]
            rec["score"] = float(sum(vals) / len(vals)) if vals else 0.0
        else:
            raise ValueError(f"Unknown agg='{agg}' (use 'max' or 'mean_topk').")

        rec["hit"] = (rec["n_node_hit"] > 0) or (rec["n_pair_hit"] > 0)
        rec["hit_type"] = "pair" if rec["n_pair_hit"] > 0 else ("node" if rec["n_node_hit"] > 0 else "none")

    return dict(idx), meta


def fuse_probs(
    p_e: float,
    p_n: float,
    p_c: float,
    kg_score: float,
    beta: float = 0.8,
    eps: float = 1e-6,
):
    """
    Paper-style fusion (logit-mixture on entailment odds):
        logit(P*_E) = beta * logit(P_E) + (1-beta) * logit(s_KG)
        P*_E = sigmoid(...)
    Then redistribute the remaining mass to Neutral/Contradicted by original ratio.

    beta: weight on NLI (beta->1 means trust NLI more).
    kg_score must be in [0,1] and should only be used when KG "hit" is true.
    """
    # Safety: renormalize NLI probs
    s = p_e + p_n + p_c
    if s <= 1e-12:
        p_e, p_n, p_c = 1 / 3, 1 / 3, 1 / 3
    else:
        p_e, p_n, p_c = p_e / s, p_n / s, p_c / s

    sKG = _clamp(kg_score, eps, 1.0 - eps)
    fused_p_e = _sigmoid(beta * _logit(p_e, eps) + (1.0 - beta) * _logit(sKG, eps))
    fused_p_e = _clamp(fused_p_e, 0.0, 1.0)

    rest = 1.0 - fused_p_e
    other = p_n + p_c
    if other <= 1e-12:
        fused_p_n = rest * 0.5
        fused_p_c = rest * 0.5
    else:
        fused_p_n = rest * (p_n / other)
        fused_p_c = rest * (p_c / other)

    ss = fused_p_e + fused_p_n + fused_p_c
    fused_p_e, fused_p_n, fused_p_c = fused_p_e / ss, fused_p_n / ss, fused_p_c / ss

    pred = (
        "entailed"
        if fused_p_e >= fused_p_n and fused_p_e >= fused_p_c
        else ("neutral" if fused_p_n >= fused_p_c else "contradicted")
    )
    return fused_p_e, fused_p_n, fused_p_c, pred


def _pick_kge_file(folder: str, prefer: List[str]) -> Optional[str]:
    """
    Pick a KGE jsonl inside the folder by preference list.
    Each prefer item can be an exact filename or a glob pattern.
    """
    for pat in prefer:
        gpat = os.path.join(folder, pat)
        cands = sorted(glob.glob(gpat))
        if cands:
            return cands[0]
    return None


def _load_student(student_json: str) -> Dict[str, Any]:
    with open(student_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_student(obj: Dict[str, Any], out_json: str) -> None:
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def process_one(
    student_json: str,
    kge_jsonl: str,
    out_json: str,
    alpha: float,
    beta: float,
    hit_only: bool,
    agg: str,
    topk: int,
    attach_top_edges: int,
    score_key: str,
    kg_calib: str,
    kg_calib_scope: str,
    verbose: bool,
) -> Dict[str, Any]:
    stu = _load_student(student_json)
    results = stu.get("results", [])

    kge_idx, meta = build_kge_index(
        kge_jsonl,
        hit_only=hit_only,
        agg=agg,
        topk=topk,
        score_key=score_key,
    )

    if kg_calib_scope != "file":
        # English-only comment: global calibration requires a 2-pass scan; fallback to file-level for now
        if verbose:
            print(f"[WARN] kg_calib_scope='{kg_calib_scope}' not implemented; falling back to 'file'.")
        kg_calib_scope = "file"

    n_queries = len(results)
    n_queries_hit = 0
    n_claims = 0
    n_claims_with_kg = 0
    n_qid_overlap = 0

    # Quick overlap estimate
    kge_qids = set(kge_idx.keys())
    for r in results[: min(200, len(results))]:
        qid = str(r.get("query_id"))
        if qid in kge_qids:
            n_qid_overlap += 1
    overlap_ratio = (n_qid_overlap / float(min(200, len(results)))) if results else 0.0

    for r in results:
        qid = str(r.get("query_id"))
        kg = kge_idx.get(qid, None)

        use_kg = bool(kg is not None and kg.get("hit", False))
        raw_kg_score = float(kg.get("score", 0.0)) if (kg is not None) else 0.0
        calib_kg_score = _calibrate_score(raw_kg_score, kg_calib, meta) if use_kg else 0.0

        if use_kg:
            n_queries_hit += 1

        claim_outputs = r.get("claim_outputs", [])
        for c in claim_outputs:
            nli = c.get("nli", {})
            p_e = _safe_float(nli.get("p_entailed", 0.0), 0.0)
            p_n = _safe_float(nli.get("p_neutral", 0.0), 0.0)
            p_c = _safe_float(nli.get("p_contradicted", 0.0), 0.0)

            if use_kg:
                fused_e, fused_n, fused_c, fused_pred = fuse_probs(p_e, p_n, p_c, calib_kg_score, beta=beta)
            else:
                fused_e, fused_n, fused_c = p_e, p_n, p_c
                fused_pred = (
                    "entailed"
                    if fused_e >= fused_n and fused_e >= fused_c
                    else ("neutral" if fused_n >= fused_c else "contradicted")
                )

            # Store both raw and calibrated KG scores for debugging/repro
            c["kg_score_raw"] = raw_kg_score
            c["kg_score"] = calib_kg_score
            c["kg_hit"] = bool(use_kg)
            c["kg_hit_type"] = (kg.get("hit_type", "none") if kg else "none")
            c["kg_fused"] = {
                "alpha": float(alpha),
                "beta": float(beta),
                "agg": agg,
                "hit_only": bool(hit_only),
                "topk": int(topk),
                "used_kg": bool(use_kg),
                "kg_calib": kg_calib,
                "kg_calib_scope": kg_calib_scope,
                "p_entailed": fused_e,
                "p_neutral": fused_n,
                "p_contradicted": fused_c,
                "prediction": fused_pred,
            }

            if attach_top_edges > 0 and kg is not None:
                top_edges = [edge for _, edge in kg.get("top", [])[:attach_top_edges]]
                c["kg_top_edges"] = top_edges

            n_claims += 1
            if use_kg:
                n_claims_with_kg += 1

    stu["kg_fusion_config"] = {
        "alpha": float(alpha),
        "beta": float(beta),
        "agg": agg,
        "hit_only": bool(hit_only),
        "topk": int(topk),
        "attach_top_edges": int(attach_top_edges),
        "kge_jsonl": kge_jsonl,
        "score_key": score_key,
        "kg_calib": kg_calib,
        "kg_calib_scope": kg_calib_scope,
    }

    stu["kg_fusion_diagnostics"] = {
        "qid_overlap_ratio_first200": overlap_ratio,
        "kge_meta": {
            "n_lines": meta["n_lines"],
            "n_parse_err": meta["n_parse_err"],
            "n_missing_qid": meta["n_missing_qid"],
            "n_missing_score": meta["n_missing_score"],
            "score_key_used": dict(meta["score_key_used"]),
            "score_min": meta["score_min"],
            "score_max": meta["score_max"],
            "score_gt0": meta["score_gt0"],
            "n_hit_true": meta["n_hit_true"],
            "n_has_node_hit": meta["n_has_node_hit"],
            "n_has_pair_hit": meta["n_has_pair_hit"],
        },
    }

    _save_student(stu, out_json)

    q_cov = (n_queries_hit / n_queries * 100.0) if n_queries else 0.0
    c_cov = (n_claims_with_kg / n_claims * 100.0) if n_claims else 0.0

    if verbose:
        print(
            f"[OK] {os.path.basename(student_json)} -> {os.path.basename(out_json)} | "
            f"KGcov(query)={q_cov:.2f}% KGcov(claim)={c_cov:.2f}% | "
            f"qid_overlap(first200)={overlap_ratio*100:.1f}% | "
            f"kge_lines={meta['n_lines']} score_keys={dict(meta['score_key_used'])} "
            f"kg_calib={kg_calib}/{kg_calib_scope}"
        )

    return {
        "student_json": student_json,
        "kge_jsonl": kge_jsonl,
        "out_json": out_json,
        "n_queries": n_queries,
        "n_queries_hit": n_queries_hit,
        "n_claims": n_claims,
        "n_claims_with_kg": n_claims_with_kg,
        "qid_overlap_ratio_first200": overlap_ratio,
    }


def _strip_repeated_suffix(stem: str, suffix_stem: str) -> str:
    """
    Remove repeated trailing suffix_stem from stem.
    Example:
      stem="abc__kgfused__kgfused", suffix_stem="__kgfused" -> "abc"
    """
    while stem.endswith(suffix_stem):
        stem = stem[: -len(suffix_stem)]
    return stem


def _cleanup_extra_fused(folder: str, base_stem: str, fused_tag: str = "__kgfused") -> int:
    """
    Delete redundant fused files like:
      base_stem + "__kgfused__kgfused*.json"
    Keep the canonical single-tag file.
    """
    deleted = 0
    for fn in os.listdir(folder):
        if not fn.endswith(".json"):
            continue
        if not fn.startswith(base_stem):
            continue
        if fn.count(fused_tag) >= 2:
            try:
                os.remove(os.path.join(folder, fn))
                deleted += 1
            except Exception:
                pass
    return deleted


def main():
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--root_dir", default=None, help="Batch mode: scan this root dir recursively")
    mode.add_argument("--student_json", default=None, help="Single mode: path to one student_checker_claim_probs__*.json")

    # Single-mode explicit args
    ap.add_argument("--kge_jsonl", default=None, help="Single mode: path to one soft_transe_scores*.jsonl")
    ap.add_argument("--out_json", default=None, help="Single mode: output json path")

    # Batch-mode patterns
    ap.add_argument(
        "--student_glob",
        default="**/student_checker_claim_probs__checker_*.json",
        help="Batch mode: glob under root_dir to find student files",
    )
    ap.add_argument(
        "--prefer_kge",
        default="soft_transe_scores.jsonl,soft_transe_scores-*.jsonl,soft_transe_scores-DRKG.jsonl",
        help="Batch mode: comma-separated preference list to pick KGE jsonl in the same folder",
    )
    ap.add_argument(
        "--out_suffix",
        default="__kgfused.json",
        help="Batch mode: output suffix appended (default: __kgfused.json)",
    )

    # Fusion args
    ap.add_argument("--alpha", type=float, default=0.8, help="Fusion strength (kept for logging)")
    ap.add_argument("--beta", type=float, default=0.8, help="Fusion weight on NLI in logit-mixture")

    ap.add_argument("--hit_only", action="store_true", help="Use only node/pair hit rows to build top edges")
    ap.add_argument("--agg", choices=["max", "mean_topk"], default="max", help="Aggregation for kg_score")
    ap.add_argument("--topk", type=int, default=10, help="Top-k rows per query kept in index")
    ap.add_argument("--attach_top_edges", type=int, default=0, help="Attach top edges into each claim (0=off)")
    ap.add_argument("--score_key", type=str, default="p_final", help="Primary score key in KGE jsonl")

    ap.add_argument("--kg_calib", choices=["none", "minmax", "sigmoid"], default="none", help="KG score calibration")
    ap.add_argument(
        "--kg_calib_scope",
        choices=["file", "global"],
        default="file",
        help="Calibration scope (global requires 2-pass; currently falls back to file)",
    )

    ap.add_argument("--dry_run", action="store_true", help="Batch mode: only print what would be processed")
    ap.add_argument("--verbose", action="store_true", help="Print per-file summary")

    # Default ON behaviors implemented via "disable" flags
    ap.add_argument(
        "--no_skip_fused_inputs",
        action="store_true",
        help="Do not skip input json files that already contain '__kgfused' in filename",
    )
    ap.add_argument(
        "--no_cleanup_redundant_outputs",
        action="store_true",
        help="Do not delete redundant outputs with repeated '__kgfused' tags",
    )

    args = ap.parse_args()

    # Default behavior: skip fused inputs + cleanup redundant outputs
    skip_fused_inputs = (not args.no_skip_fused_inputs)
    cleanup_redundant_outputs = (not args.no_cleanup_redundant_outputs)

    if args.student_json is not None:
        # Single mode
        if not args.kge_jsonl or not args.out_json:
            raise ValueError("Single mode requires --kge_jsonl and --out_json")

        process_one(
            student_json=args.student_json,
            kge_jsonl=args.kge_jsonl,
            out_json=args.out_json,
            alpha=args.alpha,
            beta=args.beta,
            hit_only=args.hit_only,
            agg=args.agg,
            topk=args.topk,
            attach_top_edges=args.attach_top_edges,
            score_key=args.score_key,
            kg_calib=args.kg_calib,
            kg_calib_scope=args.kg_calib_scope,
            verbose=True,
        )
        return

    # Batch mode
    root_dir = args.root_dir
    student_pat = os.path.join(root_dir, args.student_glob)
    student_files = sorted(glob.glob(student_pat, recursive=True))

    if skip_fused_inputs:
        student_files = [p for p in student_files if "__kgfused" not in os.path.basename(p)]

    prefer = [x.strip() for x in args.prefer_kge.split(",") if x.strip()]
    total = 0
    fused = 0
    skipped_no_kge = 0
    suspicious_zero = 0
    deleted_redundant = 0

    if not student_files:
        print(f"[WARN] No student files found under: {root_dir} with pattern: {args.student_glob}")
        return

    out_suffix = args.out_suffix
    fused_tag = out_suffix[:-5] if out_suffix.endswith(".json") else out_suffix  # e.g. "__kgfused"

    for stu_path in student_files:
        total += 1
        folder = os.path.dirname(stu_path)
        kge_path = _pick_kge_file(folder, prefer)

        if not kge_path or (not os.path.exists(kge_path)):
            skipped_no_kge += 1
            if args.verbose:
                print(f"[SKIP] missing KGE in {folder}")
            continue

        base = os.path.basename(stu_path)
        base_stem = base[:-5] if base.endswith(".json") else base

        # Defensive: normalize stem by stripping any repeated fused tags
        base_stem = _strip_repeated_suffix(base_stem, fused_tag)

        out_name = base_stem + out_suffix
        out_path = os.path.join(folder, out_name)

        if cleanup_redundant_outputs:
            deleted_redundant += _cleanup_extra_fused(folder, base_stem, fused_tag=fused_tag)

        if args.dry_run:
            print(f"[DRY] {stu_path} + {kge_path} -> {out_path}")
            continue

        info = process_one(
            student_json=stu_path,
            kge_jsonl=kge_path,
            out_json=out_path,
            alpha=args.alpha,
            beta=args.beta,
            hit_only=args.hit_only,
            agg=args.agg,
            topk=args.topk,
            attach_top_edges=args.attach_top_edges,
            score_key=args.score_key,
            kg_calib=args.kg_calib,
            kg_calib_scope=args.kg_calib_scope,
            verbose=args.verbose,
        )

        fused += 1

        n_queries = info["n_queries"]
        q_cov = (info["n_queries_hit"] / n_queries * 100.0) if n_queries else 0.0
        c_cov = (info["n_claims_with_kg"] / info["n_claims"] * 100.0) if info["n_claims"] else 0.0
        if q_cov < 1e-6 and c_cov < 1e-6:
            suspicious_zero += 1

    print(
        "[Batch Summary]\n"
        f"  root_dir                : {root_dir}\n"
        f"  total_students          : {total}\n"
        f"  fused_students          : {fused}\n"
        f"  skipped_no_kge          : {skipped_no_kge}\n"
        f"  suspicious_zero         : {suspicious_zero}\n"
        f"  deleted_redundant_files : {deleted_redundant}\n"
        f"  skip_fused_inputs       : {skip_fused_inputs}\n"
        f"  cleanup_redundant       : {cleanup_redundant_outputs}\n"
        f"  kg_calib                : {args.kg_calib}/{args.kg_calib_scope}\n"
    )


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# python kg_fusion_end2end.py   --root_dir /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data   --student_glob "**/text_eval/student_checker_claim_probs__checker_*.json"   --prefer_kge "soft_transe_scores.jsonl,soft_transe_scores-*.jsonl"   --hit_only --agg max --alpha 0.8   --verbose
# """
# import argparse
# import glob
# import json
# import os
# from collections import defaultdict
# from typing import Dict, Any, List, Tuple, Optional


# def _safe_float(x: Any, default: float = 0.0) -> float:
#     try:
#         if x is None:
#             return default
#         return float(x)
#     except Exception:
#         return default


# def build_kge_index(
#     kge_jsonl: str,
#     hit_only: bool = False,
#     agg: str = "max",
#     topk: int = 10,
#     score_key: str = "p_final",
#     alt_score_keys: Optional[List[str]] = None,
# ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
#     """
#     Build an index: query_id -> aggregated KG score and stats.
#     Returns:
#       - idx: dict(query_id -> record)
#       - meta: diagnostics about the jsonl fields / distributions
#     """
#     if alt_score_keys is None:
#         alt_score_keys = ["score", "prob", "final_score", "p", "p_final_score"]

#     idx = defaultdict(lambda: {
#         "max_p_final": 0.0,
#         "n_rows": 0,
#         "n_node_hit": 0,
#         "n_pair_hit": 0,
#         "top": [],          # list of (p_final, edge_obj)
#         "score": 0.0,
#         "hit": False,
#         "hit_type": "none",
#     })

#     meta = {
#         "path": kge_jsonl,
#         "n_lines": 0,
#         "n_parse_err": 0,
#         "n_missing_qid": 0,
#         "n_missing_score": 0,
#         "n_has_node_hit": 0,
#         "n_has_pair_hit": 0,
#         "n_hit_true": 0,
#         "score_key_used": defaultdict(int),
#         "score_min": None,
#         "score_max": None,
#         "score_gt0": 0,
#     }

#     def _get_score(d: Dict[str, Any]) -> Tuple[float, str]:
#         if score_key in d:
#             return _safe_float(d.get(score_key), 0.0), score_key
#         for k in alt_score_keys:
#             if k in d:
#                 return _safe_float(d.get(k), 0.0), k
#         return 0.0, ""

#     with open(kge_jsonl, "r", encoding="utf-8") as f:
#         for line in f:
#             meta["n_lines"] += 1
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 d = json.loads(line)
#             except Exception:
#                 meta["n_parse_err"] += 1
#                 continue

#             qid = d.get("query_id", None)
#             if qid is None:
#                 meta["n_missing_qid"] += 1
#                 continue
#             qid = str(qid)

#             p_final, used_key = _get_score(d)
#             if not used_key:
#                 meta["n_missing_score"] += 1
#             else:
#                 meta["score_key_used"][used_key] += 1

#             # Track score stats
#             if meta["score_min"] is None or p_final < meta["score_min"]:
#                 meta["score_min"] = p_final
#             if meta["score_max"] is None or p_final > meta["score_max"]:
#                 meta["score_max"] = p_final
#             if p_final > 0:
#                 meta["score_gt0"] += 1

#             node_hit = bool(d.get("node_hit", False))
#             pair_hit = bool(d.get("pair_hit", False))
#             if "node_hit" in d:
#                 meta["n_has_node_hit"] += 1
#             if "pair_hit" in d:
#                 meta["n_has_pair_hit"] += 1
#             if node_hit or pair_hit:
#                 meta["n_hit_true"] += 1

#             rec = idx[qid]
#             rec["n_rows"] += 1
#             rec["max_p_final"] = max(rec["max_p_final"], p_final)

#             if node_hit:
#                 rec["n_node_hit"] += 1
#             if pair_hit:
#                 rec["n_pair_hit"] += 1

#             keep = True
#             if hit_only and (not node_hit) and (not pair_hit):
#                 keep = False

#             if keep:
#                 rec["top"].append((p_final, d))

#     # Finalize each query record
#     for qid, rec in idx.items():
#         rec["top"].sort(key=lambda x: x[0], reverse=True)
#         rec["top"] = rec["top"][:max(0, topk)]

#         if agg == "max":
#             # If hit_only and we kept top edges, use top[0]; otherwise fallback to max_p_final
#             rec["score"] = rec["top"][0][0] if (hit_only and rec["top"]) else rec["max_p_final"]
#         elif agg == "mean_topk":
#             vals = [p for p, _ in rec["top"]]
#             rec["score"] = float(sum(vals) / len(vals)) if vals else 0.0
#         else:
#             raise ValueError(f"Unknown agg='{agg}' (use 'max' or 'mean_topk').")

#         rec["hit"] = (rec["n_node_hit"] > 0) or (rec["n_pair_hit"] > 0)
#         rec["hit_type"] = "pair" if rec["n_pair_hit"] > 0 else ("node" if rec["n_node_hit"] > 0 else "none")

#     return dict(idx), meta


# # def fuse_probs(p_e: float, p_n: float, p_c: float, kg_score: float, alpha: float = 0.8):
# #     """
# #     Stable fusion:
# #       fused_p_e = 1 - (1 - p_e) * (1 - alpha * kg_score)
# #     Then redistribute remaining mass to neutral/contradicted by original ratios.
# #     """
# #     kg_term = max(0.0, min(1.0, alpha * kg_score))
# #     fused_p_e = 1.0 - (1.0 - p_e) * (1.0 - kg_term)
# #     fused_p_e = max(0.0, min(1.0, fused_p_e))
# #     rest = 1.0 - fused_p_e

# #     other = p_n + p_c
# #     if other <= 1e-12:
# #         fused_p_n = rest * 0.5
# #         fused_p_c = rest * 0.5
# #     else:
# #         fused_p_n = rest * (p_n / other)
# #         fused_p_c = rest * (p_c / other)

# #     s = fused_p_e + fused_p_n + fused_p_c
# #     if s <= 1e-12:
# #         return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, "neutral"
# #     fused_p_e, fused_p_n, fused_p_c = fused_p_e / s, fused_p_n / s, fused_p_c / s

# #     pred = "entailed" if fused_p_e >= fused_p_n and fused_p_e >= fused_p_c else ("neutral" if fused_p_n >= fused_p_c else "contradicted")
# #     return fused_p_e, fused_p_n, fused_p_c, pred

# import math

# def _clamp(p: float, lo: float, hi: float) -> float:
#     return max(lo, min(hi, p))

# def _logit(p: float, eps: float = 1e-6) -> float:
#     # English-only comment: numerically stable logit
#     p = _clamp(p, eps, 1.0 - eps)
#     return math.log(p / (1.0 - p))

# def _sigmoid(x: float) -> float:
#     # English-only comment: numerically stable sigmoid
#     if x >= 0:
#         z = math.exp(-x)
#         return 1.0 / (1.0 + z)
#     z = math.exp(x)
#     return z / (1.0 + z)

# def fuse_probs(p_e: float, p_n: float, p_c: float, kg_score: float, beta: float = 0.8, eps: float = 1e-6):
#     """
#     Paper-style fusion (logit-mixture on entailment odds):
#         logit(P*_E) = beta * logit(P_E) + (1-beta) * logit(s_KG)
#         P*_E = sigmoid(...)
#     Then redistribute the remaining mass to Neutral/Contradicted by original ratio (keep N:C shape).

#     beta: weight on NLI (beta->1 means trust NLI more).
#     kg_score must be in [0,1] and should only be used when KG "hit" is true (handled outside).
#     """
#     # Safety: renormalize NLI probs
#     s = p_e + p_n + p_c
#     if s <= 1e-12:
#         p_e, p_n, p_c = 1/3, 1/3, 1/3
#     else:
#         p_e, p_n, p_c = p_e / s, p_n / s, p_c / s

#     sKG = _clamp(kg_score, eps, 1.0 - eps)
#     fused_p_e = _sigmoid(beta * _logit(p_e, eps) + (1.0 - beta) * _logit(sKG, eps))
#     fused_p_e = _clamp(fused_p_e, 0.0, 1.0)

#     rest = 1.0 - fused_p_e
#     other = p_n + p_c
#     if other <= 1e-12:
#         fused_p_n = rest * 0.5
#         fused_p_c = rest * 0.5
#     else:
#         fused_p_n = rest * (p_n / other)
#         fused_p_c = rest * (p_c / other)

#     # Final renorm + pred
#     ss = fused_p_e + fused_p_n + fused_p_c
#     fused_p_e, fused_p_n, fused_p_c = fused_p_e/ss, fused_p_n/ss, fused_p_c/ss

#     pred = "entailed" if fused_p_e >= fused_p_n and fused_p_e >= fused_p_c else ("neutral" if fused_p_n >= fused_p_c else "contradicted")
#     return fused_p_e, fused_p_n, fused_p_c, pred

# def _pick_kge_file(
#     folder: str,
#     prefer: List[str],
# ) -> Optional[str]:
#     """
#     Pick a KGE jsonl inside the folder by preference list.
#     Each prefer item can be an exact filename or a glob pattern.
#     """
#     for pat in prefer:
#         # If pattern is relative, join with folder
#         gpat = os.path.join(folder, pat)
#         cands = sorted(glob.glob(gpat))
#         if cands:
#             return cands[0]
#     return None


# def _load_student(student_json: str) -> Dict[str, Any]:
#     with open(student_json, "r", encoding="utf-8") as f:
#         return json.load(f)


# def _save_student(obj: Dict[str, Any], out_json: str) -> None:
#     os.makedirs(os.path.dirname(out_json), exist_ok=True)
#     with open(out_json, "w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False, indent=2)



# def process_one(
#     student_json: str,
#     kge_jsonl: str,
#     out_json: str,
#     alpha: float,
#     beta: float,   # NEW
#     hit_only: bool,
#     agg: str,
#     topk: int,
#     attach_top_edges: int,
#     score_key: str,
#     verbose: bool,
# ) -> Dict[str, Any]:

#     stu = _load_student(student_json)
#     results = stu.get("results", [])
#     kge_idx, meta = build_kge_index(
#         kge_jsonl,
#         hit_only=hit_only,
#         agg=agg,
#         topk=topk,
#         score_key=score_key,
#     )

#     n_queries = len(results)
#     n_queries_hit = 0
#     n_claims = 0
#     n_claims_with_kg = 0
#     n_qid_overlap = 0

#     # Quick overlap estimate
#     kge_qids = set(kge_idx.keys())
#     for r in results[: min(200, len(results))]:
#         qid = str(r.get("query_id"))
#         if qid in kge_qids:
#             n_qid_overlap += 1
#     if results:
#         overlap_ratio = n_qid_overlap / float(min(200, len(results)))
#     else:
#         overlap_ratio = 0.0

#     for r in results:
#         qid = str(r.get("query_id"))
#         kg = kge_idx.get(qid, None)
#         raw_kg_score = float(kg.get("score", 0.0)) if kg is not None else 0.0
#         kg_score = _calibrate_score(raw_kg_score, args_kg_calib, meta)   
#         use_kg = bool(kg is not None and kg.get("hit", False))
#         kg_score = float(kg.get("score", 0.0)) if (kg is not None) else 0.0

#         # kg = kge_idx.get(qid, None)
#         if kg and kg.get("hit"):
#             n_queries_hit += 1
#         # kg_score = float(kg.get("score", 0.0)) if kg else 0.0

#         claim_outputs = r.get("claim_outputs", [])
#         for c in claim_outputs:
#             # probs = c.get("probs", {})
#             # p_e = _safe_float(probs.get("entailed", 0.0), 0.0)
#             # p_n = _safe_float(probs.get("neutral", 0.0), 0.0)
#             # p_c = _safe_float(probs.get("contradicted", 0.0), 0.0)
#             nli = c.get("nli", {})
#             p_e = _safe_float(nli.get("p_entailed", 0.0), 0.0)
#             p_n = _safe_float(nli.get("p_neutral", 0.0), 0.0)
#             p_c = _safe_float(nli.get("p_contradicted", 0.0), 0.0)

#             # fused_e, fused_n, fused_c, fused_pred = fuse_probs(p_e, p_n, p_c, kg_score, alpha=alpha)
#             if use_kg:
#                 fused_e, fused_n, fused_c, fused_pred = fuse_probs(p_e, p_n, p_c, kg_score, beta=beta)
#             else:
#                 # No KG evidence -> fallback to NLI-only
#                 fused_e, fused_n, fused_c = p_e, p_n, p_c
#                 # keep consistent prediction rule
#                 fused_pred = "entailed" if fused_e >= fused_n and fused_e >= fused_c else ("neutral" if fused_n >= fused_c else "contradicted")
#             # c["kg_score"] = kg_score
#             # c["kg_fused"] = {
#             #     "alpha": alpha,
#             #     "agg": agg,
#             #     "hit_only": bool(hit_only),
#             #     "topk": int(topk),
#             #     "p_entailed": fused_e,
#             #     "p_neutral": fused_n,
#             #     "p_contradicted": fused_c,
#             #     "prediction": fused_pred,
#             # }
#             c["kg_score"] = kg_score
#             c["kg_hit"] = bool(use_kg)
#             c["kg_hit_type"] = (kg.get("hit_type", "none") if kg else "none")
#             c["kg_fused"] = {
#                 "beta": float(beta),
#                 "agg": agg,
#                 "hit_only": bool(hit_only),
#                 "topk": int(topk),
#                 "used_kg": bool(use_kg),
#                 "p_entailed": fused_e,
#                 "p_neutral": fused_n,
#                 "p_contradicted": fused_c,
#                 "prediction": fused_pred,
#             }


#             if attach_top_edges > 0 and kg is not None:
#                 # Attach top edges (already filtered by hit_only during indexing)
#                 top_edges = [edge for _, edge in kg.get("top", [])[:attach_top_edges]]
#                 c["kg_top_edges"] = top_edges

#             n_claims += 1
#             if kg_score > 0:
#                 n_claims_with_kg += 1

#     stu["kg_fusion_config"] = {
#             "alpha": alpha,          # (可选保留，反正论文不用它)
#             "beta": float(beta),     # NEW
#             "agg": agg,
#             "hit_only": bool(hit_only),
#             "topk": int(topk),
#             "attach_top_edges": int(attach_top_edges),
#             "kge_jsonl": kge_jsonl,
#             "score_key": score_key,
#         }

#     stu["kg_fusion_diagnostics"] = {
#         "qid_overlap_ratio_first200": overlap_ratio,
#         "kge_meta": {
#             "n_lines": meta["n_lines"],
#             "n_parse_err": meta["n_parse_err"],
#             "n_missing_qid": meta["n_missing_qid"],
#             "n_missing_score": meta["n_missing_score"],
#             "score_key_used": dict(meta["score_key_used"]),
#             "score_min": meta["score_min"],
#             "score_max": meta["score_max"],
#             "score_gt0": meta["score_gt0"],
#             "n_hit_true": meta["n_hit_true"],
#             "n_has_node_hit": meta["n_has_node_hit"],
#             "n_has_pair_hit": meta["n_has_pair_hit"],
#         }
#     }

#     _save_student(stu, out_json)

#     q_cov = (n_queries_hit / n_queries * 100.0) if n_queries else 0.0
#     c_cov = (n_claims_with_kg / n_claims * 100.0) if n_claims else 0.0

#     if verbose:
#         print(
#             f"[OK] {os.path.basename(student_json)} -> {os.path.basename(out_json)} | "
#             f"KGcov(query)={q_cov:.2f}% KGcov(claim)={c_cov:.2f}% | "
#             f"qid_overlap(first200)={overlap_ratio*100:.1f}% | "
#             f"kge_lines={meta['n_lines']} score_keys={dict(meta['score_key_used'])}"
#         )

#     return {
#         "student_json": student_json,
#         "kge_jsonl": kge_jsonl,
#         "out_json": out_json,
#         "n_queries": n_queries,
#         "n_queries_hit": n_queries_hit,
#         "n_claims": n_claims,
#         "n_claims_with_kg": n_claims_with_kg,
#         "qid_overlap_ratio_first200": overlap_ratio,
#     }




# def _strip_repeated_suffix(stem: str, suffix_stem: str) -> str:
#     """
#     Remove repeated trailing suffix_stem from stem.
#     Example:
#       stem="abc__kgfused__kgfused", suffix_stem="__kgfused" -> "abc"
#     """
#     while stem.endswith(suffix_stem):
#         stem = stem[: -len(suffix_stem)]
#     return stem


# def _minmax(x: float, lo: float, hi: float, eps: float = 1e-12) -> float:
#     if lo is None or hi is None or hi - lo <= eps:
#         return 0.0
#     return (x - lo) / (hi - lo)

# def _calibrate_score(x: float, method: str, meta: Dict[str, Any]) -> float:
#     # x expected raw in [0,1] or arbitrary; we clamp after mapping
#     if method == "none":
#         return _clamp(float(x), 0.0, 1.0)
#     if method == "minmax":
#         y = _minmax(float(x), meta.get("score_min"), meta.get("score_max"))
#         return _clamp(y, 0.0, 1.0)
#     if method == "sigmoid":
#         # simple monotonic squash; not Platt, but gives you a knob
#         # You can later replace with learned temperature if needed.
#         y = _sigmoid(6.0 * (float(x) - 0.5))
#         return _clamp(y, 0.0, 1.0)
#     raise ValueError(f"Unknown kg_calib method: {method}")


# def _cleanup_extra_fused(folder: str, base_stem: str, fused_tag: str = "__kgfused") -> int:
#     """
#     Delete redundant fused files like:
#       base_stem + "__kgfused__kgfused*.json"
#     Keep the canonical single-tag file.
#     """
#     deleted = 0
#     for fn in os.listdir(folder):
#         if not fn.endswith(".json"):
#             continue
#         # Only touch files that belong to this base_stem
#         if not fn.startswith(base_stem):
#             continue
#         # If the fused tag appears 2+ times, it's redundant
#         if fn.count(fused_tag) >= 2:
#             try:
#                 os.remove(os.path.join(folder, fn))
#                 deleted += 1
#             except Exception:
#                 pass
#     return deleted


# def main():
#     ap = argparse.ArgumentParser()

#     mode = ap.add_mutually_exclusive_group(required=True)
#     mode.add_argument("--root_dir", default=None, help="Batch mode: scan this root dir recursively")
#     mode.add_argument("--student_json", default=None, help="Single mode: path to one student_checker_claim_probs__*.json")

#     # Single-mode explicit args
#     ap.add_argument("--kge_jsonl", default=None, help="Single mode: path to one soft_transe_scores*.jsonl")
#     ap.add_argument("--out_json", default=None, help="Single mode: output json path")

#     # Batch-mode patterns
#     ap.add_argument(
#         "--student_glob",
#         default="**/student_checker_claim_probs__checker_*.json",
#         help="Batch mode: glob under root_dir to find student files",
#     )
#     ap.add_argument(
#         "--prefer_kge",
#         default="soft_transe_scores.jsonl,soft_transe_scores-*.jsonl,soft_transe_scores-DRKG.jsonl",
#         help="Batch mode: comma-separated preference list to pick KGE jsonl in the same folder",
#     )
#     ap.add_argument(
#         "--out_suffix",
#         default="__kgfused.json",
#         help="Batch mode: output suffix appended (default: __kgfused.json)",
#     )

#     # Fusion args
#     ap.add_argument("--alpha", type=float, default=0.8, help="Fusion strength (default: 0.8)")
#     ap.add_argument("--beta", type=float, default=0.8, help="Paper fusion weight on NLI in logit-mixture (default: 0.8)")

#     ap.add_argument("--hit_only", action="store_true", help="Use only node/pair hit rows to build top edges (recommended)")
#     ap.add_argument("--agg", choices=["max", "mean_topk"], default="max", help="Aggregation for kg_score")
#     ap.add_argument("--topk", type=int, default=10, help="Top-k rows per query kept in index")
#     ap.add_argument("--attach_top_edges", type=int, default=0, help="Attach top edges into each claim (0=off)")
#     ap.add_argument("--score_key", type=str, default="p_final", help="Primary score key in KGE jsonl (default: p_final)")
#     ap.add_argument("--dry_run", action="store_true", help="Batch mode: only print what would be processed")
#     ap.add_argument("--verbose", action="store_true", help="Print per-file summary")

#     # NEW: skip already fused inputs and cleanup redundant outputs
#     ap.add_argument("--skip_fused_inputs", action="store_true", default=True,
#                     help="Skip input json files that already contain '__kgfused' in filename (default: true)")

#     ap.add_argument("--cleanup_redundant_outputs", action="store_true", default=True,
#                     help="Delete redundant outputs with repeated '__kgfused' tags (default: true)")
#     ap.add_argument("--kg_calib", choices=["none", "minmax", "sigmoid"], default="none")
#     ap.add_argument("--kg_calib_scope", choices=["file", "global"], default="file")

#     args = ap.parse_args()

#     if args.student_json is not None:
#         # Single mode
#         if not args.kge_jsonl or not args.out_json:
#             raise ValueError("Single mode requires --kge_jsonl and --out_json")
#         process_one(
#             student_json=args.student_json,
#             kge_jsonl=args.kge_jsonl,
#             out_json=args.out_json,
#             alpha=args.alpha,
#             beta=args.beta,   # NEW
#             hit_only=args.hit_only,
#             agg=args.agg,
#             topk=args.topk,
#             attach_top_edges=args.attach_top_edges,
#             score_key=args.score_key,
#             verbose=True,
            
#         )

#         return

#     # Batch mode
#     root_dir = args.root_dir
#     student_pat = os.path.join(root_dir, args.student_glob)
#     student_files = sorted(glob.glob(student_pat, recursive=True))

#     # NEW: skip fused inputs to avoid suffix accumulation
#     if args.skip_fused_inputs:
#         student_files = [
#             p for p in student_files
#             if "__kgfused" not in os.path.basename(p)
#         ]

#     prefer = [x.strip() for x in args.prefer_kge.split(",") if x.strip()]
#     total = 0
#     fused = 0
#     skipped_no_kge = 0
#     suspicious_zero = 0
#     deleted_redundant = 0

#     if not student_files:
#         print(f"[WARN] No student files found under: {root_dir} with pattern: {args.student_glob}")
#         return

#     # Prepare suffix stem like "__kgfused" from "__kgfused.json"
#     out_suffix = args.out_suffix
#     fused_tag = out_suffix[:-5] if out_suffix.endswith(".json") else out_suffix  # "__kgfused"

#     for stu_path in student_files:
#         total += 1
#         folder = os.path.dirname(stu_path)
#         kge_path = _pick_kge_file(folder, prefer)

#         if not kge_path or (not os.path.exists(kge_path)):
#             skipped_no_kge += 1
#             if args.verbose:
#                 print(f"[SKIP] missing KGE in {folder}")
#             continue

#         base = os.path.basename(stu_path)
#         base_stem = base[:-5] if base.endswith(".json") else base

#         # NEW: normalize stem by stripping any repeated fused tags (defensive)
#         base_stem = _strip_repeated_suffix(base_stem, fused_tag)

#         out_name = base_stem + out_suffix
#         out_path = os.path.join(folder, out_name)

#         # NEW: optionally cleanup redundant old outputs in the same folder
#         if args.cleanup_redundant_outputs:
#             deleted_redundant += _cleanup_extra_fused(folder, base_stem, fused_tag=fused_tag)

#         if args.dry_run:
#             print(f"[DRY] {stu_path} + {kge_path} -> {out_path}")
#             continue

#         info = process_one(
#             student_json=stu_path,
#             kge_jsonl=kge_path,
#             out_json=out_path,
#             alpha=args.alpha,
#             beta=args.beta,   # NEW
#             hit_only=args.hit_only,
#             agg=args.agg,
#             topk=args.topk,
#             attach_top_edges=args.attach_top_edges,
#             score_key=args.score_key,
#             verbose=args.verbose,
#         )


#         fused += 1

#         n_queries = info["n_queries"]
#         q_cov = (info["n_queries_hit"] / n_queries * 100.0) if n_queries else 0.0
#         c_cov = (info["n_claims_with_kg"] / info["n_claims"] * 100.0) if info["n_claims"] else 0.0
#         if q_cov < 1e-6 and c_cov < 1e-6:
#             suspicious_zero += 1

#     print(
#         "[Batch Summary]\n"
#         f"  root_dir                : {root_dir}\n"
#         f"  total_students          : {total}\n"
#         f"  fused_students          : {fused}\n"
#         f"  skipped_no_kge          : {skipped_no_kge}\n"
#         f"  suspicious_zero         : {suspicious_zero}\n"
#         f"  deleted_redundant_files : {deleted_redundant}\n"
#     )

# if __name__ == "__main__":
#     main()
