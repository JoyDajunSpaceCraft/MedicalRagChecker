# #!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python collect_kgfuse_summary.py \
  --root /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data \
  --pattern "**/text_eval/*kgfused*.json" \
  --out_csv summary_kgfuse.csv

"""
import os
import re
import json
import glob
import argparse
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd

LABELS = ["entailed", "neutral", "contradicted"]

def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def pred_from_probs(pe: float, pn: float, pc: float) -> str:
    # English-only comment: argmax with deterministic tie-breaking
    if pe >= pn and pe >= pc:
        return "entailed"
    if pn >= pc:
        return "neutral"
    return "contradicted"

def get_nli_probs(claim_obj: dict) -> Tuple[float, float, float]:
    # English-only comment: support either "probs" or "nli" schema
    if isinstance(claim_obj.get("probs"), dict):
        p = claim_obj["probs"]
        return _safe_float(p.get("entailed")), _safe_float(p.get("neutral")), _safe_float(p.get("contradicted"))
    if isinstance(claim_obj.get("nli"), dict):
        n = claim_obj["nli"]
        return _safe_float(n.get("p_entailed")), _safe_float(n.get("p_neutral")), _safe_float(n.get("p_contradicted"))
    return 0.0, 0.0, 0.0

def get_fused_probs(claim_obj: dict) -> Optional[Tuple[float, float, float]]:
    # English-only comment: support either "kg_fused" or "fused" schema
    fused = claim_obj.get("kg_fused")
    if not isinstance(fused, dict):
        fused = claim_obj.get("fused")
    if not isinstance(fused, dict):
        return None
    pe = fused.get("p_entailed")
    pn = fused.get("p_neutral")
    pc = fused.get("p_contradicted")
    if pe is None or pn is None or pc is None:
        return None
    return _safe_float(pe), _safe_float(pn), _safe_float(pc)

def get_kg_hit(claim_obj: dict) -> bool:
    # English-only comment: support multiple possible fields
    if "kg_hit" in claim_obj:
        return bool(claim_obj.get("kg_hit"))
    # fallback: infer from kg_score
    return _safe_float(claim_obj.get("kg_score")) > 0.0

def get_kg_score(claim_obj: dict) -> float:
    return _safe_float(claim_obj.get("kg_score"))

def safe_mean(arr) -> float:
    arr = list(arr)
    return float(np.mean(arr)) if len(arr) else 0.0

def safe_quantiles(x: np.ndarray) -> Dict[str, float]:
    # English-only comment: quantiles with empty handling
    if x.size == 0:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": float(np.min(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }

def parse_model_and_method_from_filename(fname: str) -> Tuple[str, str]:
    """
    Example:
      student_checker_claim_probs__checker_sft_med42-llama3-8b__kgfused_beta0.8.json
      -> method=sft, model=med42-llama3-8b
    """
    base = os.path.basename(fname)

    method = "unknown"
    m = re.search(r"__checker_(sft|grpo)_", base)
    if m:
        method = m.group(1)

    model = "unknown"
    # try common pattern ...__checker_{method}_{model}...
    m2 = re.search(r"__checker_(?:sft|grpo)_([^_]+(?:_[^_]+)*)", base)
    # the above can be too greedy; prefer stopping at "__kg"
    if m2:
        tmp = m2.group(1)
        tmp = re.split(r"__kg", tmp)[0]
        model = tmp

    return model, method

def extract_fusion_params(file_data: dict, claims: List[dict]) -> Tuple[Optional[float], Optional[float], str]:
    """
    Returns (alpha, beta, fusion_type)
    fusion_type: 'alpha', 'beta', 'unknown'
    """
    alpha = None
    beta = None
    fusion_type = "unknown"

    cfg = file_data.get("kg_fusion_config")
    if isinstance(cfg, dict):
        if "alpha" in cfg:
            alpha = _safe_float(cfg.get("alpha"))
            fusion_type = "alpha"
        if "beta" in cfg:
            beta = _safe_float(cfg.get("beta"))
            fusion_type = "beta"

    # fallback to claim-level kg_fused
    if (alpha is None and beta is None) and claims:
        for c in claims[:50]:
            fused = c.get("kg_fused")
            if isinstance(fused, dict):
                if "alpha" in fused:
                    alpha = _safe_float(fused.get("alpha"))
                    fusion_type = "alpha"
                    break
                if "beta" in fused:
                    beta = _safe_float(fused.get("beta"))
                    fusion_type = "beta"
                    break

    return alpha, beta, fusion_type

def dist(preds: List[str]) -> Dict[str, float]:
    return {lbl: safe_mean([p == lbl for p in preds]) for lbl in LABELS}

def summarize_one_file(fp: str) -> Dict[str, Any]:
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    claims = []
    for r in results:
        for c in r.get("claim_outputs", []):
            claims.append(c)

    if not claims:
        return {}

    model, method = parse_model_and_method_from_filename(fp)
    alpha, beta, fusion_type = extract_fusion_params(data, claims)

    # per-claim signals
    kg_scores = np.array([get_kg_score(c) for c in claims], dtype=float)
    kg_hits = np.array([get_kg_hit(c) for c in claims], dtype=bool)

    fused_exists = np.array([get_fused_probs(c) is not None for c in claims], dtype=bool)

    # subsets
    idx_all = np.arange(len(claims))
    idx_cov = np.where(kg_scores > 0)[0]
    idx_cov_hit = np.where(kg_hits)[0]
    idx_cov_fused = np.where((kg_scores > 0) & fused_exists)[0]
    idx_cov_hit_fused = np.where(kg_hits & fused_exists)[0]

    # distributions
    nli_preds = []
    fused_preds = []
    changed = 0
    fused_available = 0

    # also track delta_pe when fused exists
    delta_pe = []

    for c in claims:
        pe, pn, pc = get_nli_probs(c)
        p0 = pred_from_probs(pe, pn, pc)
        nli_preds.append(p0)

        fused_probs = get_fused_probs(c)
        if fused_probs is None:
            continue

        fe, fn, fc = fused_probs
        p1 = pred_from_probs(fe, fn, fc)
        fused_preds.append(p1)
        fused_available += 1
        if p1 != p0:
            changed += 1

        delta_pe.append(fe - pe)

    nli_dist = dist(nli_preds)
    fused_dist = dist(fused_preds) if fused_available else {lbl: 0.0 for lbl in LABELS}

    # coarse proxies
    faith_nli = nli_dist["entailed"]
    hall_nli = 1.0 - faith_nli
    faith_fused = fused_dist["entailed"] if fused_available else 0.0
    hall_fused = (1.0 - faith_fused) if fused_available else 0.0

    # subset entail rates
    def entail_rate_for_indices(pred_list: List[str], indices: np.ndarray) -> float:
        if len(pred_list) == 0 or indices.size == 0:
            return 0.0
        # pred_list aligns with claims for NLI, but for fused_preds it aligns only to claims with fused_exists.
        # So here we compute subset rates explicitly from probs instead (robust).
        return 0.0

    # English-only comment: compute subset rates robustly via per-claim probs
    def subset_entail_rates(indices: np.ndarray) -> Tuple[float, float]:
        if indices.size == 0:
            return 0.0, 0.0
        nli_e = 0
        fused_e = 0
        fused_cnt = 0
        for i in indices.tolist():
            pe, pn, pc = get_nli_probs(claims[i])
            if pred_from_probs(pe, pn, pc) == "entailed":
                nli_e += 1
            fused_probs = get_fused_probs(claims[i])
            if fused_probs is not None:
                fe, fn, fc = fused_probs
                fused_cnt += 1
                if pred_from_probs(fe, fn, fc) == "entailed":
                    fused_e += 1
        nli_rate = nli_e / float(indices.size)
        fused_rate = (fused_e / float(fused_cnt)) if fused_cnt else 0.0
        return float(nli_rate), float(fused_rate)

    cov_ent_nli, cov_ent_fused = subset_entail_rates(idx_cov)
    covhit_ent_nli, covhit_ent_fused = subset_entail_rates(idx_cov_hit)

    # kg score stats
    kgq_all = safe_quantiles(kg_scores)
    kg_cov_scores = kg_scores[idx_cov] if idx_cov.size else np.array([], dtype=float)
    kgq_cov = safe_quantiles(kg_cov_scores)

    # delta_pe stats
    dpe = np.array(delta_pe, dtype=float) if len(delta_pe) else np.array([], dtype=float)
    dpe_q = safe_quantiles(dpe)

    out = {
        "file": os.path.basename(fp),
        "path": fp,
        "model": model,
        "method": method,
        "fusion_type": fusion_type,
        "alpha": alpha,
        "beta": beta,

        "n_queries": len(results),
        "n_claims": len(claims),

        "fused_available_claims": int(fused_available),
        "fused_available_rate": fused_available / float(len(claims)),

        "kg_cov_claim_rate": safe_mean(kg_scores > 0),
        "kg_hit_claim_rate": safe_mean(kg_hits),
        "kg_cov_fused_rate": (idx_cov_fused.size / float(len(claims))),
        "kg_hit_fused_rate": (idx_cov_hit_fused.size / float(len(claims))),

        "kg_score_min": kgq_all["min"],
        "kg_score_p50": kgq_all["p50"],
        "kg_score_p90": kgq_all["p90"],
        "kg_score_max": kgq_all["max"],

        "kg_score_cov_min": kgq_cov["min"],
        "kg_score_cov_p50": kgq_cov["p50"],
        "kg_score_cov_p90": kgq_cov["p90"],
        "kg_score_cov_max": kgq_cov["max"],
        "kg_score_cov_mean": float(np.mean(kg_cov_scores)) if kg_cov_scores.size else 0.0,

        "nli_ent_rate": nli_dist["entailed"],
        "nli_neu_rate": nli_dist["neutral"],
        "nli_con_rate": nli_dist["contradicted"],

        "fused_ent_rate": fused_dist["entailed"],
        "fused_neu_rate": fused_dist["neutral"],
        "fused_con_rate": fused_dist["contradicted"],

        "faith_nli": faith_nli,
        "hall_nli": hall_nli,
        "faith_fused": faith_fused,
        "hall_fused": hall_fused,

        "pred_change_rate": (changed / float(fused_available)) if fused_available else 0.0,

        # covered subset
        "kg_cov_ent_rate_nli": cov_ent_nli,
        "kg_cov_ent_rate_fused": cov_ent_fused,
        "kg_hit_ent_rate_nli": covhit_ent_nli,
        "kg_hit_ent_rate_fused": covhit_ent_fused,

        # delta pe stats (only where fused exists)
        "delta_pe_min": dpe_q["min"],
        "delta_pe_p50": dpe_q["p50"],
        "delta_pe_p90": dpe_q["p90"],
        "delta_pe_max": dpe_q["max"],
        "delta_pe_mean": float(np.mean(dpe)) if dpe.size else 0.0,
    }

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root folder that contains *kgfused*.json")
    ap.add_argument("--out_csv", type=str, default="summary_kgfuse.csv")
    ap.add_argument("--pattern", type=str, default="**/*kgfused*.json", help="Glob pattern under root")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, args.pattern), recursive=True))
    if not files:
        print(f"[WARN] No files matched under root={args.root} pattern={args.pattern}")
        return

    rows = []
    for fp in files:
        try:
            row = summarize_one_file(fp)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"[ERR] Failed to parse: {fp} | {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Found {len(df)} kgfused files. Wrote: {args.out_csv}")

    # quick peek
    show_cols = ["file", "model", "method", "fusion_type", "alpha", "beta",
                 "n_claims", "kg_cov_claim_rate", "fused_ent_rate", "nli_ent_rate", "pred_change_rate"]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].head(30).to_string(index=False))

if __name__ == "__main__":
    main()
