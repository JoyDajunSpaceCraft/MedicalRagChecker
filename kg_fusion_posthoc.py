#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[a-z0-9\-_\/]+)?", re.I)


def normalize_surface(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text.lower())]


def load_entities_tsv(entities_path: Path) -> Dict[str, int]:
    name2id: Dict[str, int] = {}
    with entities_path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    first = lines[0].split("\t")
    has_header = any(x.lower() in {"entity_name", "entity", "name"} for x in first) or any(
        x.lower() in {"entity_id", "id"} for x in first
    )
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
            continue
        name2id[ename] = eid
    return name2id


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
            if len(c2) < 3 or c2 in seen:
                continue
            seen.add(c2)
            kept.append(c2)
            if len(kept) >= max_alias_per_entity:
                break

        for s in kept:
            surface2id.setdefault(s, eid)
    return surface2id


def load_name_map(name_map_path: Optional[Path], name2id: Dict[str, int]) -> Dict[str, int]:
    if name_map_path is None:
        return {}
    import pandas as pd

    df = pd.read_csv(name_map_path)

    surface_cols = ["surface_form", "surface", "mention", "alias", "name", "text"]
    ent_name_cols = ["drkg_entity", "entity_name", "entity", "drkg_entity_name"]
    ent_id_cols = ["entity_id", "drkg_id", "drkg_entity_id", "id", "raw_id"]

    surface_col = next((c for c in surface_cols if c in df.columns), df.columns[0])
    ent_col = next((c for c in ent_name_cols if c in df.columns), None)
    ent_id_col = next((c for c in ent_id_cols if c in df.columns), None)

    surface2id: Dict[str, int] = {}

    for _, row in df.iterrows():
        s = normalize_surface(str(row[surface_col]))
        if not s or s == "nan":
            continue

        if ent_id_col is not None and ent_id_col in df.columns:
            v = str(row[ent_id_col])
            if v != "nan":
                # raw_id can be like "Gene::10581"
                if re.fullmatch(r"\d+", v):
                    surface2id[s] = int(v)
                    continue

        if ent_col is not None and ent_col in df.columns:
            en = str(row[ent_col]).strip()
            if en in name2id:
                surface2id[s] = name2id[en]
                continue

        # Common case: two columns (name, raw_id) where raw_id is DRKG entity string.
        if ent_col is None and len(df.columns) >= 2:
            en2 = str(row[df.columns[1]]).strip()
            if en2 in name2id:
                surface2id[s] = name2id[en2]

    return surface2id


def match_surfaces_in_text(text: str, surface2id: Dict[str, int], max_ngram: int = 5) -> Set[int]:
    toks = tokenize(text)
    if not toks:
        return set()
    matched: Set[int] = set()
    L = len(toks)
    for n in range(min(max_ngram, L), 0, -1):
        for i in range(0, L - n + 1):
            phrase = " ".join(toks[i : i + n])
            eid = surface2id.get(phrase)
            if eid is not None:
                matched.add(eid)
    return matched


def score_entity_pair(emb: np.ndarray, e1: int, e2: int) -> Optional[float]:
    if e1 < 0 or e2 < 0 or e1 >= emb.shape[0] or e2 >= emb.shape[0]:
        return None
    dist = float(np.linalg.norm(emb[e1] - emb[e2]))
    return 1.0 / (1.0 + dist)


def kg_score_for_claim(claim_text: str, emb: np.ndarray, surface2id: Dict[str, int], max_pairs: int = 60) -> Tuple[bool, Optional[float]]:
    ent_ids = sorted(match_surfaces_in_text(claim_text, surface2id))
    if len(ent_ids) < 2:
        return False, None

    scores: List[float] = []
    cnt = 0
    for i in range(len(ent_ids)):
        for j in range(i + 1, len(ent_ids)):
            s = score_entity_pair(emb, ent_ids[i], ent_ids[j])
            if s is not None:
                scores.append(s)
            cnt += 1
            if cnt >= max_pairs:
                break
        if cnt >= max_pairs:
            break

    if not scores:
        return False, None
    # Use max as a "strongest plausibility" signal; you can also try mean(top-k)
    return True, float(max(scores))


def fuse_label(label: str, pair_hit: bool, kg_score: Optional[float], tau_low: float, tau_high: float) -> str:
    if (not pair_hit) or (kg_score is None):
        return label
    if label == "Entailment" and kg_score < tau_low:
        return "Neutral"
    if label == "Neutral" and kg_score > tau_high:
        return "Entailment"
    return label


def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    f1s = []
    for c in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def collect_pairs_for_eval(student: dict, teacher: Optional[dict]) -> Tuple[List[str], List[str]]:
    """
    Align by (query_id, response_claim_index).
    Assumption: response_claims list order is stable between student/teacher for the same generator output.
    """
    if teacher is None:
        return [], []

    t_map = {str(ex["query_id"]): ex for ex in teacher.get("results", [])}
    y_true, y_pred = [], []

    for ex in student.get("results", []):
        qid = str(ex.get("query_id"))
        if qid not in t_map:
            continue
        tex = t_map[qid]
        s_labels = ex.get("response2answer", [])
        t_labels = tex.get("response2answer", [])

        if not isinstance(s_labels, list) or not isinstance(t_labels, list):
            continue
        m = min(len(s_labels), len(t_labels))
        for i in range(m):
            y_pred.append(str(s_labels[i]))
            y_true.append(str(t_labels[i]))
    return y_true, y_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_results", type=Path, required=True, help="Path to student results_text.json")
    ap.add_argument("--teacher_results", type=Path, default=None, help="Optional: teacher results_text.json as pseudo-gold")
    ap.add_argument("--entity_emb", type=Path, required=True, help="Path to DRKG_TransE_l2_entity.npy")
    ap.add_argument("--entities_tsv", type=Path, required=True, help="Path to entities.tsv")
    ap.add_argument("--name_map", type=Path, default=None, help="Optional name_map.csv")
    ap.add_argument("--tau_low", type=float, default=0.25)
    ap.add_argument("--tau_high", type=float, default=0.55)
    ap.add_argument("--out", type=Path, default=None, help="Output fused results_text.json (default: *.kgfused.json)")
    args = ap.parse_args()

    student = json.load(args.student_results.open("r"))
    teacher = json.load(args.teacher_results.open("r")) if args.teacher_results else None

    emb = np.load(args.entity_emb)
    name2id = load_entities_tsv(args.entities_tsv)

    surface2id = {}
    surface2id.update(load_name_map(args.name_map, name2id) if args.name_map else {})
    fallback = build_fallback_surface_map_from_entities(name2id)
    for k, v in fallback.items():
        surface2id.setdefault(k, v)

    # Baseline agreement (optional)
    if teacher is not None:
        y_true0, y_pred0 = collect_pairs_for_eval(student, teacher)
        base = macro_f1(y_true0, y_pred0, ["Entailment", "Neutral", "Contradiction"])
        print(f"[Baseline] macro-F1 vs teacher: {base:.4f}")

    # Fuse claim-level labels
    for ex in student.get("results", []):
        claims = ex.get("response_claims", [])
        labels = ex.get("response2answer", [])
        if not isinstance(claims, list) or not isinstance(labels, list):
            continue
        fused = []
        kg_meta = []
        m = min(len(claims), len(labels))
        for i in range(m):
            claim_text = " ".join(claims[i]) if isinstance(claims[i], list) else str(claims[i])
            pair_hit, s = kg_score_for_claim(claim_text, emb, surface2id)
            fused.append(fuse_label(str(labels[i]), pair_hit, s, args.tau_low, args.tau_high))
            kg_meta.append({"pair_hit": bool(pair_hit), "kg_score": s})
        ex["response2answer_fused"] = fused
        ex["kg_claim_meta"] = kg_meta

    # Fused agreement (optional)
    if teacher is not None:
        # temporarily evaluate using fused labels
        tmp = {"results": []}
        for ex in student.get("results", []):
            ex2 = dict(ex)
            ex2["response2answer"] = ex.get("response2answer_fused", ex.get("response2answer", []))
            tmp["results"].append(ex2)
        y_true1, y_pred1 = collect_pairs_for_eval(tmp, teacher)
        fused_f1 = macro_f1(y_true1, y_pred1, ["Entailment", "Neutral", "Contradiction"])
        print(f"[Fused]    macro-F1 vs teacher: {fused_f1:.4f}  (tau_low={args.tau_low}, tau_high={args.tau_high})")

    out = args.out
    if out is None:
        out = args.student_results.with_suffix(".kgfused.json")
    json.dump(student, out.open("w"), ensure_ascii=False, indent=2)
    print(f"Wrote fused results to: {out}")


if __name__ == "__main__":
    main()
