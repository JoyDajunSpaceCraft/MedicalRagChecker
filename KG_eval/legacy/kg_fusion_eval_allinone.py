#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
End-to-end KG fusion evaluation for MedRAGChecker.

It:
1) Scans --root for .../text_eval/results_text*.json
2) For each run directory, finds student checker outputs:
   student_checker_claim_probs__*.json
3) Computes claim-level KG score from DRKG TransE entity embeddings
4) Runs NLI-only vs KG-fused predictions
5) Writes per-run CSV + overall summary CSV + LaTeX table

Expected student JSON item format (recommended):
{
  "query_id": "...",
  "claim_idx": 3,
  "claim_text": "...",              # optional but helpful
  "nli": {
    "p_entailed": 0.72,
    "p_neutral": 0.21,
    "p_contradicted": 0.07
  },
  "teacher_label": "entailed"       # optional (for F1)
}
"""

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple, Set

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[a-z0-9\-_\/]+)?", re.I)


# ---------------- I/O utils ----------------
def safe_mean(xs: List[float]) -> Optional[float]:
    return float(mean(xs)) if xs else None


def scan_results_text(root: Path) -> List[Path]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("results_text") and fn.endswith(".json"):
                out.append(Path(dirpath) / fn)
    return sorted(out)


def find_eval_root(p: Path) -> Path:
    for parent in p.parents:
        if parent.name.startswith("eval_"):
            return parent
    return p.parents[2]


def parse_eval_folder(eval_root: Path) -> Tuple[str, str, str]:
    # eval_{dataset}-{teacher}-{generator...}
    name = eval_root.name[len("eval_"):]
    parts = name.split("-")
    dataset = parts[0] if len(parts) >= 1 else eval_root.name
    teacher = parts[1] if len(parts) >= 2 else "unknown"
    generator = "-".join(parts[2:]) if len(parts) >= 3 else "unknown"
    return dataset, teacher, generator


def parse_inner_run_dir(results_path: Path) -> Tuple[str, str]:
    # .../<run_dir>/text_eval/results_text.json
    run_dir = results_path.parent.parent
    name = run_dir.name
    dataset_split = name.split(".")[0] if "." in name else name
    gen = "unknown"
    if "." in name:
        parts = name.split(".")
        if len(parts) >= 2:
            gen = parts[1]
    return dataset_split, gen


def load_results_text(path: Path) -> Tuple[dict, List[dict]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("metrics", {}), obj.get("results", [])


# ---------------- claim extraction from results_text ----------------
def claims_to_texts(rec: dict) -> List[str]:
    # Handles your common formats
    if "response_claims" in rec:
        c = rec["response_claims"]
        if isinstance(c, list) and c:
            if isinstance(c[0], list):
                return [" ".join(x).strip() for x in c if x]
            return [str(x).strip() for x in c]
        return []
    for k in ["claims", "atomic_claims", "extracted_claims"]:
        if k in rec and isinstance(rec[k], list):
            return [str(x).strip() for x in rec[k]]
    return []


# ---------------- DRKG mapping + KG score ----------------
def normalize_surface(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall((text or "").lower())]


def load_entities_tsv(entities_path: Path) -> Dict[str, int]:
    """
    entities.tsv: entity_name <tab> entity_id  (or reversed)
    Returns entity_name -> entity_id
    """
    name2id: Dict[str, int] = {}
    with entities_path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    first = lines[0].split("\t")
    has_header = any(x.lower() in {"entity_name", "name", "entity"} for x in first)

    start = 1 if has_header else 0

    def is_int(x: str) -> bool:
        return bool(re.fullmatch(r"\d+", x))

    for ln in lines[start:]:
        cols = ln.split("\t")
        if len(cols) < 2:
            continue
        a, b = cols[0].strip(), cols[1].strip()
        if is_int(a) and not is_int(b):
            eid, ename = int(a), b
        elif is_int(b) and not is_int(a):
            eid, ename = int(b), a
        else:
            # fallback
            eid = len(name2id)
            ename = a
        name2id[ename] = eid

    return name2id


def load_name_map(name_map_path: Optional[Path], name2id: Dict[str, int]) -> Dict[str, int]:
    """
    Tries to auto-detect columns. Works with your screenshot style: columns like [name, raw_id]
    If raw_id matches DRKG entity_name (e.g. "Gene::10581"), we can map it.
    """
    if name_map_path is None:
        return {}

    import pandas as pd  # local import

    df = pd.read_csv(name_map_path)

    # Heuristic column detection
    surface_candidates = ["surface_form", "surface", "mention", "alias", "name", "text"]
    rawid_candidates = ["raw_id", "drkg_entity", "entity_name", "entity", "drkg_id", "entity_id", "id"]

    surface_col = next((c for c in surface_candidates if c in df.columns), df.columns[0])
    raw_col = next((c for c in rawid_candidates if c in df.columns), df.columns[1] if len(df.columns) > 1 else df.columns[0])

    surface2id: Dict[str, int] = {}
    for _, row in df.iterrows():
        s = normalize_surface(str(row.get(surface_col, "")))
        rid = str(row.get(raw_col, "")).strip()
        if not s or s == "nan" or not rid or rid == "nan":
            continue
        # If the raw id equals DRKG entity_name
        if rid in name2id:
            surface2id[s] = name2id[rid]
            continue
        # Some maps may store integer ids
        if re.fullmatch(r"\d+", rid):
            surface2id[s] = int(rid)

    return surface2id


def build_fallback_surface_map_from_entities(name2id: Dict[str, int], max_alias_per_entity: int = 6) -> Dict[str, int]:
    surface2id: Dict[str, int] = {}
    for ename, eid in name2id.items():
        raw = ename.strip()
        candidates = [raw]
        for sep in ["::", ":", "/", "|"]:
            if sep in raw:
                candidates.append(raw.split(sep)[-1])
        candidates.append(raw.replace("_", " "))
        candidates.append(raw.replace("-", " "))

        seen = set()
        kept = []
        for c in candidates:
            c2 = normalize_surface(c)
            if len(c2) < 3:
                continue
            if c2 in seen:
                continue
            seen.add(c2)
            kept.append(c2)
            if len(kept) >= max_alias_per_entity:
                break

        for s in kept:
            surface2id.setdefault(s, eid)
    return surface2id


def match_surfaces_in_text(text: str, surface2id: Dict[str, int], max_ngram: int = 5) -> Set[int]:
    toks = tokenize(text)
    if not toks:
        return set()
    matched: Set[int] = set()
    L = len(toks)
    for n in range(min(max_ngram, L), 0, -1):
        for i in range(0, L - n + 1):
            phrase = " ".join(toks[i:i+n])
            eid = surface2id.get(phrase)
            if eid is not None:
                matched.add(eid)
    return matched


def kg_score_for_claim(emb: np.ndarray, ent_ids: List[int], topk: int = 10) -> float:
    """
    Relation-agnostic TransE distance between entity embeddings:
      dist = ||e1 - e2||_2
      p = 1 / (1 + dist)
    Claim KG score = mean of top-k p over entity pairs (or 0 if <2 entities)
    """
    if len(ent_ids) < 2:
        return 0.0

    ps = []
    for i in range(len(ent_ids)):
        for j in range(i + 1, len(ent_ids)):
            e1, e2 = ent_ids[i], ent_ids[j]
            if e1 < 0 or e2 < 0 or e1 >= emb.shape[0] or e2 >= emb.shape[0]:
                continue
            dist = float(np.linalg.norm(emb[e1] - emb[e2]))
            p = 1.0 / (1.0 + dist)
            ps.append(p)

    if not ps:
        return 0.0

    ps.sort(reverse=True)
    return float(mean(ps[: min(topk, len(ps))]))


# ---------------- Fusion + metrics ----------------
LABELS = ["entailed", "contradicted", "neutral"]


def argmax_label(p_e: float, p_c: float, p_n: float) -> str:
    trip = [("entailed", p_e), ("contradicted", p_c), ("neutral", p_n)]
    trip.sort(key=lambda x: x[1], reverse=True)
    return trip[0][0]


def fuse_probs(
    p_e: float,
    p_c: float,
    p_n: float,
    s_kg: float,
    alpha: float,
    tau: float,
) -> Tuple[float, float, float]:
    """
    Simple, explainable fusion:
      If KG score is "high" (s_kg > tau), we boost entail by adding alpha*(s_kg - tau).
      Neutral/contradicted stay unchanged except for renormalization.
    """
    eps = 1e-9
    boost = max(0.0, s_kg - tau) * alpha
    pe2 = min(1.0, p_e + boost)
    pc2 = max(0.0, p_c)
    pn2 = max(0.0, p_n)

    z = pe2 + pc2 + pn2 + eps
    return pe2 / z, pc2 / z, pn2 / z


def compute_claim_metrics(preds: List[str], golds: List[str]) -> Dict[str, Optional[float]]:
    # macro-F1 and per-class F1
    if not golds:
        return {"acc": None, "macro_f1": None, "f1_ent": None, "f1_con": None, "f1_neu": None}

    def f1_for(label: str) -> float:
        tp = sum(1 for p, g in zip(preds, golds) if p == label and g == label)
        fp = sum(1 for p, g in zip(preds, golds) if p == label and g != label)
        fn = sum(1 for p, g in zip(preds, golds) if p != label and g == label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    acc = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds)
    f1s = {lab: f1_for(lab) for lab in LABELS}
    macro_f1 = sum(f1s.values()) / 3.0
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "f1_ent": f1s["entailed"],
        "f1_con": f1s["contradicted"],
        "f1_neu": f1s["neutral"],
    }


def qlevel_from_claim_labels(claim_labels: List[str]) -> Tuple[float, float]:
    """
    Faithfulness = %entailed; Hallucination = %contradicted (in percent).
    """
    if not claim_labels:
        return 0.0, 0.0
    n = len(claim_labels)
    faith = 100.0 * sum(1 for x in claim_labels if x == "entailed") / n
    hall = 100.0 * sum(1 for x in claim_labels if x == "contradicted") / n
    return faith, hall


@dataclass
class Row:
    dataset: str
    generator: str
    teacher: str
    student: str
    n_claim: int
    kg_claim_cov: float

    acc_nli: Optional[float]
    macrof1_nli: Optional[float]
    acc_fused: Optional[float]
    macrof1_fused: Optional[float]

    faith_nli: float
    hall_nli: float
    faith_fused: float
    hall_fused: float


def load_student_claim_probs(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    # Accept list or dict{root:[...]}
    if isinstance(obj, dict) and "root" in obj and isinstance(obj["root"], list):
        return obj["root"]
    if isinstance(obj, list):
        return obj
    return []


def extract_probs(item: dict) -> Optional[Tuple[str, int, float, float, float, Optional[str]]]:
    """
    Returns (query_id, claim_idx, pE, pC, pN, gold_label/teacher_label)
    """
    qid = item.get("query_id")
    cidx = item.get("claim_idx")
    nli = item.get("nli", item)

    def getp(k1, k2=None):
        if k1 in nli:
            return nli.get(k1)
        if k2 and k2 in nli:
            return nli.get(k2)
        return None

    pE = getp("p_entailed", "p_entail")
    pN = getp("p_neutral")
    pC = getp("p_contradicted", "p_contra")

    gold = item.get("gold_label", item.get("teacher_label"))

    if qid is None or cidx is None or pE is None or pN is None or pC is None:
        return None
    return str(qid), int(cidx), float(pE), float(pC), float(pN), (str(gold) if gold is not None else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--entity_emb", type=Path, required=True, help="DRKG_TransE_l2_entity.npy")
    ap.add_argument("--entities_tsv", type=Path, required=True, help="entities.tsv")
    ap.add_argument("--name_map", type=Path, default=None, help="Optional name_map.csv")
    ap.add_argument("--alpha", type=float, default=0.8, help="Fusion strength")
    ap.add_argument("--tau", type=float, default=0.3, help="KG threshold for boosting entail")
    ap.add_argument("--out_csv", type=Path, default=Path("fusion_summary.csv"))
    ap.add_argument("--out_tex", type=Path, default=Path("fusion_summary.tex"))
    args = ap.parse_args()

    results_paths = scan_results_text(args.root)
    print(f"Found {len(results_paths)} results_text*.json under {args.root}")
    if not results_paths:
        return

    emb = np.load(args.entity_emb)
    name2id = load_entities_tsv(args.entities_tsv)

    surface2id = {}
    if args.name_map is not None:
        surface2id.update(load_name_map(args.name_map, name2id))
    fallback = build_fallback_surface_map_from_entities(name2id)
    for k, v in fallback.items():
        surface2id.setdefault(k, v)

    rows: List[Row] = []

    for rp in results_paths:
        eval_root = find_eval_root(rp)
        _, teacher, gen_from_eval = parse_eval_folder(eval_root)
        dataset_split, gen_from_inner = parse_inner_run_dir(rp)
        generator = gen_from_inner if gen_from_inner != "unknown" else gen_from_eval

        metrics, records = load_results_text(rp)
        run_dir = rp.parent.parent  # right above text_eval

        # Find student files
        student_files = sorted(run_dir.glob("student_checker_claim_probs__*.json"))
        if not student_files:
            continue

        # Precompute claim texts by (query_id, claim_idx)
        claim_text_map: Dict[Tuple[str, int], str] = {}
        for ex in records:
            qid = str(ex.get("query_id"))
            claims = claims_to_texts(ex)
            for i, c in enumerate(claims):
                claim_text_map[(qid, i)] = c

        for sf in student_files:
            student_name = sf.stem.replace("student_checker_claim_probs__", "")
            items = load_student_claim_probs(sf)

            preds_nli, preds_fused, golds = [], [], []
            faith_q_nli, hall_q_nli = [], []
            faith_q_fused, hall_q_fused = [], []

            kg_hit_claim = 0
            total_claim = 0

            # Group claim labels per question for q-level metrics
            per_q_labels_nli: Dict[str, List[str]] = {}
            per_q_labels_fused: Dict[str, List[str]] = {}

            for it in items:
                parsed = extract_probs(it)
                if parsed is None:
                    continue
                qid, cidx, pE, pC, pN, gold = parsed
                total_claim += 1

                claim_text = it.get("claim_text") or claim_text_map.get((qid, cidx), "")
                ent_ids = sorted(match_surfaces_in_text(claim_text, surface2id))
                sKG = kg_score_for_claim(emb, ent_ids)

                if sKG > 0.0:
                    kg_hit_claim += 1

                pred0 = argmax_label(pE, pC, pN)
                pE2, pC2, pN2 = fuse_probs(pE, pC, pN, sKG, alpha=args.alpha, tau=args.tau)
                pred1 = argmax_label(pE2, pC2, pN2)

                preds_nli.append(pred0)
                preds_fused.append(pred1)
                if gold is not None:
                    golds.append(gold)

                per_q_labels_nli.setdefault(qid, []).append(pred0)
                per_q_labels_fused.setdefault(qid, []).append(pred1)

            # q-level aggregation
            for qid, labs in per_q_labels_nli.items():
                f, h = qlevel_from_claim_labels(labs)
                faith_q_nli.append(f); hall_q_nli.append(h)
            for qid, labs in per_q_labels_fused.items():
                f, h = qlevel_from_claim_labels(labs)
                faith_q_fused.append(f); hall_q_fused.append(h)

            m0 = compute_claim_metrics(preds_nli[:len(golds)], golds) if golds else compute_claim_metrics([], [])
            m1 = compute_claim_metrics(preds_fused[:len(golds)], golds) if golds else compute_claim_metrics([], [])

            kg_cov = 100.0 * (kg_hit_claim / total_claim) if total_claim else 0.0

            rows.append(
                Row(
                    dataset=dataset_split,
                    generator=generator,
                    teacher=teacher,
                    student=student_name,
                    n_claim=total_claim,
                    kg_claim_cov=kg_cov,
                    acc_nli=m0["acc"],
                    macrof1_nli=m0["macro_f1"],
                    acc_fused=m1["acc"],
                    macrof1_fused=m1["macro_f1"],
                    faith_nli=float(safe_mean(faith_q_nli) or 0.0),
                    hall_nli=float(safe_mean(hall_q_nli) or 0.0),
                    faith_fused=float(safe_mean(faith_q_fused) or 0.0),
                    hall_fused=float(safe_mean(hall_q_fused) or 0.0),
                )
            )

    # Write summary CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "generator", "teacher", "student",
            "n_claim", "kg_claim_cov",
            "acc_nli", "macrof1_nli", "acc_fused", "macrof1_fused",
            "faith_nli", "hall_nli", "faith_fused", "hall_fused"
        ])
        for r in rows:
            w.writerow([
                r.dataset, r.generator, r.teacher, r.student,
                r.n_claim, round(r.kg_claim_cov, 2),
                r.acc_nli, r.macrof1_nli, r.acc_fused, r.macrof1_fused,
                round(r.faith_nli, 2), round(r.hall_nli, 2),
                round(r.faith_fused, 2), round(r.hall_fused, 2),
            ])

    # Write LaTeX table
    def fmt(x: Optional[float]) -> str:
        return "--" if x is None else f"{x:.3f}"

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r.dataset, r.generator, r.student))
    with args.out_tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l l l r r r r r r}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Generator & Student & KG-Cov(\\%) & Acc$_{NLI}$ & Acc$_{fused}$ & MacroF1$_{NLI}$ & MacroF1$_{fused}$ & $\\Delta$Hall(\\%) \\\\\n")
        f.write("\\midrule\n")
        for r in rows_sorted:
            delta_hall = r.hall_fused - r.hall_nli
            f.write(
                f"{r.dataset} & {r.generator} & {r.student} & {r.kg_claim_cov:.1f} & "
                f"{fmt(r.acc_nli)} & {fmt(r.acc_fused)} & {fmt(r.macrof1_nli)} & {fmt(r.macrof1_fused)} & "
                f"{delta_hall:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\caption{KG fusion ablation on student checkers. Accuracy/MacroF1 require gold/teacher labels in student files; otherwise they are --. $\\Delta$Hall is fused minus NLI-only (lower is better).}\n")
        f.write("\\label{tab:kg_fusion_ablation}\n")
        f.write("\\end{tabular}\n\\end{table*}\n")

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote LaTeX: {args.out_tex}")


if __name__ == "__main__":
    main()
