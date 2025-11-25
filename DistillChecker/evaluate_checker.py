#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a distilled checker model served via an OpenAI-compatible endpoint (e.g., vLLM).
Two utilities:
  A) Checker evaluation on labeled prompts (e.g., checker_sft.jsonl or checker_grpo.jsonl)
  B) Claim overlap evaluation between teacher_claims.jsonl and gt_claims.jsonl (string-level)
export CUDA_VISIBLE_DEVICES=0,1

BASE=/ocean/projects/med230010p/yji3/models/Meditron3-8B
ADAPTER=/ocean/projects/med230010p/yji3/MedicalRagChecker/runs/checker_sft_meditron

python -m vllm.entrypoints.openai.api_server \
  --model "$BASE" \
  --tokenizer "$BASE" \
  --host 0.0.0.0 --port 8000 \
  --dtype auto \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.6 \
  --enable-lora \
  --lora-modules checker="$ADAPTER" \
  --served-model-name Meditron3-8B-base \
  --trust-remote-code


python DistillChecker/evaluate_checker.py \
  --mode checker \
  --data_file ./data/checker_sft.jsonl \
  --api_base http://localhost:8000/v1 \
  --model checker \
  --out_file ./runs/checker_eval/checker_eval_sft.jsonl

"""

import os
import json
import time
import math
import argparse
from collections import Counter, defaultdict

# Avoid heavy deps: implement basic metrics locally to reduce friction.
def _safe_lower(s):
    return s.lower().strip()

LABELS = ["entailed", "contradicted", "neutral"]
LABEL_ALIASES = {
    "entailed": {"entailed", "entail", "entails", "supported", "yes"},
    "contradicted": {"contradicted", "contradict", "refuted", "no"},
    "neutral": {"neutral", "unknown", "insufficient", "not enough info", "not enough information", "uncertain"}
}

def normalize_label(text):
    t = _safe_lower(text)
    t = t.replace(".", "").replace("label:", "").strip()
    for lab, al in LABEL_ALIASES.items():
        for a in al:
            if t == a or t.startswith(a):
                return lab
    # fallback: search keywords
    if "contrad" in t or "refut" in t:
        return "contradicted"
    if "neutral" in t or "insufficient" in t or "not enough" in t or "uncertain" in t:
        return "neutral"
    if "entail" in t or "support" in t or t in {"yes","y"}:
        return "entailed"
    return None

def precision_recall_f1(y_true, y_pred, labels=LABELS):
    metrics = {}
    for lab in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp == lab)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lab and yp == lab)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp != lab)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        metrics[lab] = {"precision": p, "recall": r, "f1": f1, "support": sum(1 for yt in y_true if yt == lab)}
    # Macro
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(labels)
    acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true) if y_true else 0.0
    return metrics, macro_f1, acc

def cohen_kappa(y_true, y_pred, labels=LABELS):
    n = len(y_true)
    if n == 0:
        return 0.0
    label_to_idx = {lab:i for i, lab in enumerate(labels)}
    # confusion matrix
    cm = [[0]*len(labels) for _ in labels]
    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt]][label_to_idx[yp]] += 1
    # observed agreement
    po = sum(cm[i][i] for i in range(len(labels))) / n
    # expected agreement
    row_marg = [sum(cm[i]) for i in range(len(labels))]
    col_marg = [sum(cm[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum((row_marg[i]*col_marg[i]) for i in range(len(labels))) / (n*n) if n>0 else 0.0
    if pe == 1.0: 
        return 1.0
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

def confusion_matrix(y_true, y_pred, labels=LABELS):
    idx = {lab:i for i, lab in enumerate(labels)}
    cm = [[0]*len(labels) for _ in labels]
    for yt, yp in zip(y_true, y_pred):
        if yt in idx and yp in idx:
            cm[idx[yt]][idx[yp]] += 1
    return cm

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def run_checker_eval(args):
    import requests

    data = list(load_jsonl(args.data_file))
    data = data[:100]
    # Each item should have {"prompt": ..., "label": ...}
    # Some prompts already end with "Label:"; we keep them.
    sys_prompt = "You are a scientific entailment checker. Only output one of: entailed | contradicted | neutral."

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key or 'EMPTY'}"}
    url = f"{args.api_base.rstrip('/')}/chat/completions"

    y_true, y_pred, rows = [], [], []
    for ex in data:
        prompt = ex.get("prompt") or ex.get("input") or ""
        gold = normalize_label(ex.get("label",""))
        if gold is None:
            continue

        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        out = r.json()
        text = out["choices"][0]["message"]["content"].strip()
        pred = normalize_label(text) or "neutral"  # default fallback
        y_true.append(gold)
        y_pred.append(pred)
        rows.append({"prompt": prompt, "gold": gold, "pred": pred, "raw": text})

    per_cls, macro_f1, acc = precision_recall_f1(y_true, y_pred)
    kappa = cohen_kappa(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    result = {
        "n": len(y_true),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
        "per_class": per_cls,
        "confusion_matrix": {"labels": LABELS, "matrix": cm}
    }

    with open(args.out_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"__summary__": result}, ensure_ascii=False) + "\n")

    print(json.dumps(result, indent=2))

def normalize_claim_text(s):
    # Simple normalization for claim overlap (string-level Jaccard-like F1)
    import re
    t = s.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = t.replace(".", "").replace(",", "")
    return t

def run_claim_overlap(args):
    # Evaluate extractor/teacher vs ground-truth claims overlap.
    tea = {d["query_id"]: [normalize_claim_text(c) for c in d.get("pred_claims", [])] 
           for d in load_jsonl(args.teacher_file)}
    gt = {d["query_id"]: [normalize_claim_text(c) for c in d.get("pred_claims", [])] 
          for d in load_jsonl(args.gt_file)}

    qids = sorted(set(tea.keys()) & set(gt.keys()))
    precs, recs, f1s = [], [], []
    details = []

    for q in qids:
        A = set(tea[q])
        B = set(gt[q])
        tp = len(A & B)
        fp = len(A - B)
        fn = len(B - A)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        precs.append(p); recs.append(r); f1s.append(f1)
        details.append({"query_id": q, "precision": p, "recall": r, "f1": f1, 
                        "tp": tp, "fp": fp, "fn": fn, "teacher_n": len(A), "gt_n": len(B)})
    macro = {
        "macro_precision": sum(precs)/len(precs) if precs else 0.0,
        "macro_recall": sum(recs)/len(recs) if recs else 0.0,
        "macro_f1": sum(f1s)/len(f1s) if f1s else 0.0,
        "n_questions": len(qids)
    }
    out = {"macro": macro, "per_query": details}
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out["macro"], indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["checker","claim_overlap"], required=True)
    ap.add_argument("--data_file", type=str, help="For --mode checker: path to JSONL with {'prompt','label'}")
    ap.add_argument("--api_base", type=str, default="http://localhost:8000/v1")
    ap.add_argument("--model", type=str, default="Meditron3-8B")
    ap.add_argument("--api_key", type=str, default="EMPTY")
    ap.add_argument("--out_file", type=str, required=True)
    ap.add_argument("--teacher_file", type=str)
    ap.add_argument("--gt_file", type=str)
    args = ap.parse_args()

    if args.mode == "checker":
        if not args.data_file:
            raise SystemExit("--data_file is required for mode=checker")
        run_checker_eval(args)
    else:
        if not args.teacher_file or not args.gt_file:
            raise SystemExit("--teacher_file and --gt_file are required for mode=claim_overlap")
        run_claim_overlap(args)

if __name__ == "__main__":
    main()
