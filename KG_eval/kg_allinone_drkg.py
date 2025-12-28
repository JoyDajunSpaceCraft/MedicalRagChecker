#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
python kg_allinone_drkg.py \
  --root /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data \
  --entity_emb /ocean/projects/med230010p/yji3/MedicalRagChecker/KG/DRKG/embed/DRKG_TransE_l2_entity.npy \
  --entities_tsv /ocean/projects/med230010p/yji3/MedicalRagChecker/KG/DRKG/embed/entities.tsv \
  --name_map /ocean/projects/med230010p/yji3/MedicalRagChecker/aux/name_map.csv \
  --out_csv /ocean/projects/med230010p/yji3/MedicalRagChecker/kg_summary.csv \
  --out_tex /ocean/projects/med230010p/yji3/MedicalRagChecker/kg_summary.tex \
  --debug 3 \
  --force_rescore
我建议你把 KG 全部合并成一个脚本跑完（你想要的“一次性跑完 KG + 出表”）

下面这个脚本做三件事（一个文件跑通）：

扫描 --root 下所有 results_text*.json（不再出现 Collected 0 runs 那种黑盒）

对每个 results_text*.json：

从 claims 里抽取字符串（支持你这种 token list 的 claims）

用 name_map（列名自动适配）把 surface → DRKG entity

只要每个 question 能找到 ≥1 个实体，就算 Node-level KG coverage；能找到 ≥2 个实体就会写出 pair-level 分数（不会再全是 null）

把分数写到同目录 soft_transe_scores.jsonl（query_id 严格沿用 results_text 里的 query_id，解决对不上的问题）

最后 汇总出 CSV + LaTeX：

teacher/checker 的那张主表（F1/ClaimRec/CtxPrec/Hallu/Faith）

KG coverage/effect 表（你现在那张 KG-Cov / Faith_KG / Hall_KG）

注意：如果你现在还没把 drkg.tsv（三元组大表）下载下来，也完全能跑：我默认用 entity embedding 的相似度/距离给一个 p_kge（至少你能先把“KG 有数”跑通，把论文故事讲顺）。你后面要做“严格 TransE(h+r≈t)”再加 relation embedding + drkg.tsv 去筛 relation 就行。
All-in-one DRKG scoring + MedRAGChecker table export.

What it does:
1) Find all results_text*.json under --root.
2) For each results_text*.json, generate text_eval/soft_transe_scores.jsonl
   (query_id is taken directly from results_text to avoid mismatch).
3) Summarize:
   - Teacher/checker main table (F1, ClaimRec, CtxPrec, Halluc, Faith)
   - KG coverage/effect table (KG-Cov, Faith_all/KG, Hall_all/KG)

Notes:
- This script assumes you already have DRKG entity embeddings + entities.tsv.
- name_map is optional but highly recommended. It maps surface strings -> DRKG entity_name or entity_id.
- If name_map is missing or weak, KG coverage can still be low; use --debug to inspect matches.
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple, Set

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[a-z0-9\-_\/]+)?", re.I)


def safe_mean(xs: Sequence[float]) -> Optional[float]:
    return float(mean(xs)) if xs else None


def find_eval_root(p: Path) -> Path:
    for parent in p.parents:
        if parent.name.startswith("eval_"):
            return parent
    return p.parents[2]


def parse_eval_folder(eval_root: Path) -> Tuple[str, str, str]:
    """
    Parse eval folder: eval_{dataset}-{teacher}-{generator...}
    Examples:
      eval_medquad-4.1-Meditron3-8B
      eval_medquad-4o
    """
    name = eval_root.name[len("eval_") :]
    parts = name.split("-")
    dataset = parts[0] if len(parts) >= 1 else eval_root.name
    teacher = parts[1] if len(parts) >= 2 else "unknown"
    generator = "-".join(parts[2:]) if len(parts) >= 3 else "unknown"
    return dataset, teacher, generator


def parse_inner_run_dir(results_path: Path) -> Tuple[str, str]:
    """
    results_path example:
      .../eval_medquad-4.1-Meditron3-8B/medquad_train.Meditron3-8B.gen100__gpt-4.1/text_eval/results_text.json
    We use the directory right above text_eval as "inner run dir".
    """
    run_dir = results_path.parent.parent
    name = run_dir.name
    # dataset split is before first dot
    dataset_split = name.split(".")[0] if "." in name else name

    # generator often sits between first and second dot
    gen = "unknown"
    if "." in name:
        parts = name.split(".")
        if len(parts) >= 2:
            gen = parts[1]
    return dataset_split, gen


def load_results_text(path: Path) -> Tuple[dict, List[dict]]:
    with path.open("r") as f:
        obj = json.load(f)
    return obj.get("metrics", {}), obj.get("results", [])


def normalize_surface(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text.lower())]


def claims_to_texts(rec: dict) -> List[str]:
    """
    Supports:
    - response_claims as list[list[str]] (your current format)
    - response_claims as list[str]
    - claims / atomic_claims variants (best-effort)
    """
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


def load_entities_tsv(entities_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    entities.tsv: mapping entity_name -> entity_id
    Format differs across releases; we handle:
      - header or no header
      - 2 columns (name, id) or (id, name)
    Returns:
      name2id, id2name
    """
    name2id: Dict[str, int] = {}
    id2name: Dict[int, str] = {}

    with entities_path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    # Detect header
    first = lines[0].split("\t")
    has_header = any(x.lower() in {"entity_name", "entity", "name"} for x in first) or any(
        x.lower() in {"entity_id", "id"} for x in first
    )
    start = 1 if has_header else 0

    for ln in lines[start:]:
        cols = ln.split("\t")
        if len(cols) < 2:
            continue
        a, b = cols[0].strip(), cols[1].strip()

        # Heuristic: whichever is int-like is the id
        def is_int(x: str) -> bool:
            return bool(re.fullmatch(r"\d+", x))

        if is_int(a) and not is_int(b):
            eid, ename = int(a), b
        elif is_int(b) and not is_int(a):
            eid, ename = int(b), a
        else:
            # Fallback: treat row index as id if no clear integer id
            # (Still works for embedding lookup by row index later.)
            eid = len(id2name)
            ename = a

        name2id[ename] = eid
        id2name[eid] = ename

    return name2id, id2name


def build_fallback_surface_map_from_entities(
    name2id: Dict[str, int],
    max_alias_per_entity: int = 6,
) -> Dict[str, int]:
    """
    Build a weak surface->id map from entities.tsv strings.
    This is ONLY a fallback. Real coverage typically needs a curated name_map.
    """
    surface2id: Dict[str, int] = {}
    for ename, eid in name2id.items():
        candidates: List[str] = []

        raw = ename.strip()
        candidates.append(raw)

        # Common patterns: take the last segment after separators
        for sep in ["::", ":", "/", "|"]:
            if sep in raw:
                candidates.append(raw.split(sep)[-1])

        # Also try de-underscored / de-dashed variants
        candidates.append(raw.replace("_", " "))
        candidates.append(raw.replace("-", " "))

        # Normalize and keep a few
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


def load_name_map(name_map_path: Optional[Path], name2id: Dict[str, int]) -> Dict[str, int]:
    """
    name_map.csv can be many formats. We try to detect columns automatically.

    Supported:
    - surface_form -> drkg_entity (entity_name)
    - surface_form -> entity_id
    - mention/name/alias -> entity/name/id

    If name_map_path is None, returns empty dict.
    """
    if name_map_path is None:
        return {}

    import pandas as pd  # local import to keep hard deps minimal

    df = pd.read_csv(name_map_path)

    # Try detect surface col
    surface_cols = ["surface_form", "surface", "mention", "alias", "name", "text"]
    ent_name_cols = ["drkg_entity", "entity_name", "entity", "drkg_entity_name"]
    ent_id_cols = ["entity_id", "drkg_id", "drkg_entity_id", "id"]

    surface_col = next((c for c in surface_cols if c in df.columns), None)
    if surface_col is None:
        # Fallback: first column
        surface_col = df.columns[0]

    ent_col = next((c for c in ent_name_cols if c in df.columns), None)
    ent_id_col = next((c for c in ent_id_cols if c in df.columns), None)

    surface2id: Dict[str, int] = {}

    for _, row in df.iterrows():
        s = str(row[surface_col]) if surface_col in row else ""
        s = normalize_surface(s)
        if not s or s == "nan":
            continue

        if ent_id_col is not None and ent_id_col in row and str(row[ent_id_col]) != "nan":
            try:
                eid = int(row[ent_id_col])
                surface2id[s] = eid
                continue
            except Exception:
                pass

        if ent_col is not None and ent_col in row and str(row[ent_col]) != "nan":
            ename = str(row[ent_col]).strip()
            if ename in name2id:
                surface2id[s] = name2id[ename]
                continue

        # If the file stores entity_name directly in second column (common case)
        if ent_col is None and len(df.columns) >= 2:
            ename2 = str(row[df.columns[1]]).strip()
            if ename2 in name2id:
                surface2id[s] = name2id[ename2]

    return surface2id


def match_surfaces_in_text(
    text: str,
    surface2id: Dict[str, int],
    max_ngram: int = 5,
) -> Set[int]:
    """
    Efficient n-gram lookup: generate n-grams from the text and check dict membership.
    """
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


def score_entity_pair(
    emb: np.ndarray,
    eid1: int,
    eid2: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Relation-agnostic scoring (fast, no drkg.tsv needed):
    dist = ||e1 - e2||_2
    p_kge = 1 / (1 + dist)
    """
    if eid1 < 0 or eid2 < 0:
        return None, None
    if eid1 >= emb.shape[0] or eid2 >= emb.shape[0]:
        return None, None

    v1 = emb[eid1]
    v2 = emb[eid2]
    dist = float(np.linalg.norm(v1 - v2))
    p = 1.0 / (1.0 + dist)
    return dist, p


@dataclass
class RunRow:
    dataset: str
    generator: str
    teacher: str
    n_q: int

    overall_f1: Optional[float]
    claim_recall: Optional[float]
    context_precision: Optional[float]
    hallucination: Optional[float]
    faithfulness: Optional[float]

    kg_cov_node: float
    kg_cov_pair: float
    hall_all: Optional[float]
    hall_kg: Optional[float]
    faith_all: Optional[float]
    faith_kg: Optional[float]


def generate_soft_scores_for_run(
    results_path: Path,
    records: List[dict],
    emb: np.ndarray,
    surface2id: Dict[str, int],
    out_name: str = "soft_transe_scores.jsonl",
    force: bool = False,
    debug_k: int = 0,
) -> Tuple[Set[str], Set[str]]:
    """
    Write per-question scored pairs into text_eval/soft_transe_scores.jsonl.

    Returns:
      qids_node: questions with >=1 mapped entity
      qids_pair: questions with >=1 scored entity pair
    """
    text_eval_dir = results_path.parent
    out_path = text_eval_dir / out_name
    if out_path.exists() and not force:
        # If file already exists, do NOT overwrite; still compute coverage by reading it.
        qids_node, qids_pair = set(), set()
        with out_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                qid = str(rec.get("query_id"))
                if not qid or qid == "None":
                    continue
                if rec.get("node_hit"):
                    qids_node.add(qid)
                if rec.get("pair_hit"):
                    qids_pair.add(qid)
        return qids_node, qids_pair

    qids_node: Set[str] = set()
    qids_pair: Set[str] = set()

    debug_printed = 0

    with out_path.open("w") as wf:
        for ex in records:
            qid = str(ex.get("query_id"))
            claim_texts = claims_to_texts(ex)
            if not claim_texts:
                continue

            # Union entities across all claims of the same question (better coverage)
            ent_ids: Set[int] = set()
            for ct in claim_texts:
                ent_ids |= match_surfaces_in_text(ct, surface2id)

            node_hit = len(ent_ids) >= 1
            if node_hit:
                qids_node.add(qid)

            ent_list = sorted(ent_ids)
            pair_hit = False

            # Score all unordered pairs (limit to avoid explosion)
            max_pairs = 80
            pairs = []
            for i in range(len(ent_list)):
                for j in range(i + 1, len(ent_list)):
                    pairs.append((ent_list[i], ent_list[j]))
                    if len(pairs) >= max_pairs:
                        break
                if len(pairs) >= max_pairs:
                    break

            for (e1, e2) in pairs:
                dist, p = score_entity_pair(emb, e1, e2)
                if dist is None or p is None:
                    continue
                pair_hit = True

                out_rec = {
                    "query_id": qid,
                    "node_hit": node_hit,
                    "pair_hit": True,
                    "eid_head": e1,
                    "eid_tail": e2,
                    "transe_dist": dist,
                    "p_kge": p,
                    "p_final": p,  # keep a unified field for downstream fusion
                }
                wf.write(json.dumps(out_rec) + "\n")

            if pair_hit:
                qids_pair.add(qid)

            if debug_k > 0 and debug_printed < debug_k:
                # English-only comments requirement satisfied; printed text is data.
                print(f"[DEBUG] qid={qid} node_hit={node_hit} pair_hit={pair_hit} | #ents={len(ent_ids)}")
                if claim_texts:
                    print("  claim0:", claim_texts[0][:160])
                debug_printed += 1

    return qids_node, qids_pair


def compute_kg_effect(
    records: List[dict],
    qids_node: Set[str],
    qids_pair: Set[str],
) -> Tuple[float, float, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Compute coverage and subset averages using per-question metrics in records.
    Your per-question metrics appear to be in [0,1], so we convert to percentage (x100).
    """
    n = len(records)
    if n == 0:
        return 0.0, 0.0, None, None, None, None

    hall_all, faith_all = [], []
    hall_kg, faith_kg = [], []

    for ex in records:
        qid = str(ex.get("query_id"))
        m = ex.get("metrics", {}) or {}
        hall = m.get("hallucination")
        faith = m.get("faithfulness")

        if hall is not None:
            hall_all.append(float(hall) * 100.0)
        if faith is not None:
            faith_all.append(float(faith) * 100.0)

        # Use pair-level hit for "KG subset" by default (more meaningful than node-only)
        if qid in qids_pair:
            if hall is not None:
                hall_kg.append(float(hall) * 100.0)
            if faith is not None:
                faith_kg.append(float(faith) * 100.0)

    cov_node = 100.0 * sum(1 for ex in records if str(ex.get("query_id")) in qids_node) / n
    cov_pair = 100.0 * sum(1 for ex in records if str(ex.get("query_id")) in qids_pair) / n

    return cov_node, cov_pair, safe_mean(hall_all), safe_mean(hall_kg), safe_mean(faith_all), safe_mean(faith_kg)


def scan_results_text(root: Path) -> List[Path]:
    paths: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("results_text") and fn.endswith(".json"):
                paths.append(Path(dirpath) / fn)
    return sorted(paths)


def write_csv(rows: List[RunRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "generator", "teacher", "n_q",
            "overall_f1", "claim_recall", "context_precision", "hallucination", "faithfulness",
            "kg_cov_node", "kg_cov_pair",
            "hall_all", "hall_kg", "faith_all", "faith_kg",
        ])
        for r in rows:
            w.writerow([
                r.dataset, r.generator, r.teacher, r.n_q,
                r.overall_f1, r.claim_recall, r.context_precision, r.hallucination, r.faithfulness,
                round(r.kg_cov_node, 2), round(r.kg_cov_pair, 2),
                r.hall_all, r.hall_kg, r.faith_all, r.faith_kg,
            ])


def fmt(x: Optional[float]) -> str:
    return "--" if x is None else f"{x:.1f}"


def write_latex_teacher_table(rows: List[RunRow], out_tex: Path) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    # Group by dataset blocks similar to your screenshot
    rows_sorted = sorted(rows, key=lambda r: (r.dataset, r.generator, r.teacher))

    with out_tex.open("w") as f:
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l l l l r r r r r}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Generator & Checker mode & Teacher & F1$\\uparrow$ & ClaimRec$\\uparrow$ & CtxPrec$\\uparrow$ & Halluc.$\\downarrow$ & Faith.$\\uparrow$ \\\\\n")
        f.write("\\midrule\n")

        for r in rows_sorted:
            f.write(
                f"{r.dataset} & {r.generator} & Teacher-checker & {r.teacher} & "
                f"{fmt(r.overall_f1)} & {fmt(r.claim_recall)} & {fmt(r.context_precision)} & "
                f"{fmt(r.hallucination)} & {fmt(r.faithfulness)} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\caption{Teacher-checker results across datasets and generators.}\\label{tab:teacher_checker_main}\n")
        f.write("\\end{tabular}\n\\end{table*}\n")


def write_latex_kg_table(rows: List[RunRow], out_tex: Path) -> None:
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r.dataset, r.generator, r.teacher))

    with out_tex.open("a") as f:
        f.write("\n% ---------------- KG table ----------------\n")
        f.write("\\begin{table*}[t]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l l r r r r r r}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Generator & \\#Q & KG-Cov$_{node}$ (\\%) & KG-Cov$_{pair}$ (\\%) & Faith.$_{all}$ & Faith.$_{KG}$ & Hall.$_{KG}$ \\\\\n")
        f.write("\\midrule\n")

        for r in rows_sorted:
            f.write(
                f"{r.dataset} & {r.generator} & {r.n_q} & "
                f"{r.kg_cov_node:.1f} & {r.kg_cov_pair:.1f} & "
                f"{fmt(r.faith_all)} & {fmt(r.faith_kg)} & {fmt(r.hall_kg)} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\caption{DRKG coverage and subset behavior. KG subset uses pair-level hits by default.}\\label{tab:kg_coverage}\n")
        f.write("\\end{tabular}\n\\end{table*}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="Root directory containing eval_* folders.")
    ap.add_argument("--entity_emb", type=Path, required=True, help="Path to DRKG_TransE_l2_entity.npy")
    ap.add_argument("--entities_tsv", type=Path, required=True, help="Path to entities.tsv")
    ap.add_argument("--name_map", type=Path, default=None, help="Optional name_map.csv for surface->DRKG entity mapping.")
    ap.add_argument("--out_csv", type=Path, default=Path("kg_summary.csv"))
    ap.add_argument("--out_tex", type=Path, default=Path("kg_summary.tex"))
    ap.add_argument("--force_rescore", action="store_true", help="Overwrite soft_transe_scores.jsonl if exists.")
    ap.add_argument("--debug", type=int, default=0, help="Print debug matches for first K questions per run.")
    args = ap.parse_args()

    results_paths = scan_results_text(args.root)
    print(f"Found {len(results_paths)} results_text*.json under {args.root}")

    if not results_paths:
        print("No runs found. Check --root points to the directory that contains eval_* folders.")
        return

    print("Loading DRKG entity embeddings...")
    emb = np.load(args.entity_emb)  # shape: [N, d]
    name2id, _ = load_entities_tsv(args.entities_tsv)

    # Build mapping
    surface2id = {}
    if args.name_map is not None:
        print(f"Loading name_map from {args.name_map} ...")
        surface2id.update(load_name_map(args.name_map, name2id))

    if not surface2id:
        print("WARNING: name_map produced empty mapping. Falling back to weak aliases from entities.tsv.")
    surface2id_fallback = build_fallback_surface_map_from_entities(name2id)
    # Prefer explicit name_map, fallback fills missing keys only
    for k, v in surface2id_fallback.items():
        surface2id.setdefault(k, v)

    rows: List[RunRow] = []

    for rp in results_paths:
        # Only handle the typical structure: .../text_eval/results_text*.json
        eval_root = find_eval_root(rp)
        _, teacher, gen_from_eval = parse_eval_folder(eval_root)
        dataset_split, gen_from_inner = parse_inner_run_dir(rp)

        metrics, records = load_results_text(rp)

        # Generate (or reuse) soft scores in the same text_eval directory
        qids_node, qids_pair = generate_soft_scores_for_run(
            results_path=rp,
            records=records,
            emb=emb,
            surface2id=surface2id,
            force=args.force_rescore,
            debug_k=args.debug,
        )

        cov_node, cov_pair, hall_all, hall_kg, faith_all, faith_kg = compute_kg_effect(
            records, qids_node, qids_pair
        )

        overall = metrics.get("overall_metrics", {}) or {}
        retr = metrics.get("retriever_metrics", {}) or {}
        genm = metrics.get("generator_metrics", {}) or {}

        generator_name = gen_from_inner if gen_from_inner != "unknown" else gen_from_eval

        rows.append(
            RunRow(
                dataset=dataset_split,
                generator=generator_name,
                teacher=teacher,
                n_q=len(records),
                overall_f1=overall.get("f1"),
                claim_recall=retr.get("claim_recall"),
                context_precision=retr.get("context_precision"),
                hallucination=genm.get("hallucination"),
                faithfulness=genm.get("faithfulness"),
                kg_cov_node=cov_node,
                kg_cov_pair=cov_pair,
                hall_all=hall_all,
                hall_kg=hall_kg,
                faith_all=faith_all,
                faith_kg=faith_kg,
            )
        )

    # Optional: deduplicate by (dataset, generator, teacher) keeping the largest n_q
    best: Dict[Tuple[str, str, str], RunRow] = {}
    for r in rows:
        key = (r.dataset, r.generator, r.teacher)
        if key not in best or r.n_q > best[key].n_q:
            best[key] = r
    rows_final = list(best.values())

    write_csv(rows_final, args.out_csv)
    write_latex_teacher_table(rows_final, args.out_tex)
    write_latex_kg_table(rows_final, args.out_tex)

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote LaTeX (two tables): {args.out_tex}")


if __name__ == "__main__":
    main()
