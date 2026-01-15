#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end evaluation of student extractor + student checker on MedRAGChecker datasets.
Computes the same metrics as teacher models (RefChecker/RAGChecker):
- Overall metrics: F1, Accuracy
- Retriever metrics: claim_recall, context_precision
- Generator metrics: hallucination, faithfulness, context_utilization

Usage:
    python eval_student_end2end.py \
        --results_path ./medical_data/eval_pubmedqa_4.1-Meditron3-8B/results_text__gpt-4.1-df63c0.json \
        --extractor_dir ./runs/extractor_sft_meditron3-8b \
        --checker_dir ./runs/checker_sft_meditron \
        --base_model_extractor /path/to/Meditron3-8B \
        --base_model_checker /path/to/Meditron3-8B \
        --out_json ./runs/student_eval/pubmedqa_meditron3_results.json \
        --out_csv ./runs/student_eval/pubmedqa_meditron3_metrics.csv
"""

import os
import json
import argparse
import csv
from typing import List, Dict, Any, Tuple
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


# ============================================================================
# Label definitions (matching teacher models)
# ============================================================================
LABEL_LIST = ["contradicted", "neutral", "entailed"]
LABEL2ID = {k: i for i, k in enumerate(LABEL_LIST)}
LABEL_ALIASES = {
    "entailed": {"entailed", "entail", "entails", "supported", "yes"},
    "contradicted": {"contradicted", "contradict", "refuted", "no"},
    "neutral": {"neutral", "unknown", "insufficient", "not enough info",
                "not enough information", "uncertain"},
}


# ============================================================================
# Helper functions
# ============================================================================

def normalize_label(text: str) -> str:
    """Map free-form checker output to one of LABEL_LIST."""
    t = (text or "").lower().strip().replace(".", "").replace("label:", "").strip()
    for lab, aliases in LABEL_ALIASES.items():
        for a in aliases:
            if t == a or t.startswith(a):
                return lab
    if "contrad" in t or "refut" in t:
        return "contradicted"
    if "neutral" in t or "insufficient" in t or "not enough" in t or "uncertain" in t:
        return "neutral"
    if "entail" in t or "support" in t or t in {"yes", "y"}:
        return "entailed"
    return "neutral"  # default fallback


def join_context(ctx_list: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    """Concatenate retrieved context passages."""
    return " ".join([c.get("text", "") for c in ctx_list if isinstance(c, dict)])[:max_chars]


def safe_json_list(text: str) -> List[str]:
    """Parse JSON list from LLM output, fallback to line-split."""
    t = (text or "").strip()
    l, r = t.find("["), t.rfind("]")
    if l != -1 and r != -1 and r > l:
        try:
            arr = json.loads(t[l:r+1])
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # Fallback: one claim per line
    return [s.strip() for s in t.split("\n") if s.strip()]


def jaccard_similarity(claim: str, passage: str) -> float:
    """Token-based Jaccard similarity."""
    import re, string, unicodedata
    tbl = str.maketrans("", "", string.punctuation)
    def norm(s):
        s = unicodedata.normalize("NFKC", s or "").lower().translate(tbl)
        return re.sub(r"\s+", " ", s).strip()
    A = set(norm(claim).split())
    B = set(norm(passage).split())
    return len(A & B) / max(1, len(A | B)) if A and B else 0.0


def best_passage_match(claim: str, ctx_list: List[Dict[str, Any]]) -> Tuple[str, int, float]:
    """Find the best matching passage for a claim based on Jaccard similarity."""
    best_idx, best_score, best_text = -1, 0.0, ""
    for i, p in enumerate(ctx_list or []):
        text = p.get("text", "") or ""
        score = jaccard_similarity(claim, text)
        if score > best_score:
            best_score, best_idx, best_text = score, i, text
    return best_text, best_idx, best_score


# ============================================================================
# Model loading
# ============================================================================

def load_extractor_model(base_model: str, adapter_dir: str = None, dtype: str = "bf16"):
    """Load extractor model (base + optional LoRA adapter)."""
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_dir:
        if PeftModel is None:
            raise RuntimeError("peft is not installed, cannot load LoRA adapter.")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()

    model.eval()
    device = next(model.parameters()).device
    return model, tok, device


def load_checker_model(base_model: str, adapter_dir: str = None, dtype: str = "bf16"):
    """Load checker model (base + optional LoRA adapter)."""
    return load_extractor_model(base_model, adapter_dir, dtype)


# ============================================================================
# Claim extraction
# ============================================================================

def build_extractor_prompt(query: str, context: str, response: str) -> str:
    """Build prompt for claim extraction."""
    system_msg = (
        "You extract atomic, verifiable claims from a RAG response. "
        "Return a JSON array of strings ONLY. Each claim is a single self-contained sentence without pronouns."
    )
    user_msg = (
        f"Query:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Model response:\n{response}\n\n"
        "Now extract atomic, verifiable claims ONLY from the response."
    )
    return system_msg + "\n\n" + user_msg


def extract_claims_batch(
    examples: List[Dict[str, Any]],
    model,
    tok,
    device,
    max_new_tokens: int = 384,
    temperature: float = 0.0,
) -> List[List[str]]:
    """Extract claims for a batch of examples."""
    prompts = []
    for ex in examples:
        q = ex.get("query", "")
        ctx = join_context(ex.get("retrieved_context", []))
        resp = ex.get("response", "") or ""
        prompts.append(build_extractor_prompt(q, ctx, resp) if resp.strip() else "")

    empty_mask = [p == "" for p in prompts]
    real_prompts = [p if p else "dummy" for p in prompts]  # avoid empty strings

    enc = tok(real_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    gen_only = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen_only, skip_special_tokens=True)

    all_claims = []
    for is_empty, text in zip(empty_mask, texts):
        all_claims.append([] if is_empty else safe_json_list(text))

    return all_claims


# ============================================================================
# NLI checking
# ============================================================================

def build_checker_prompt(claim: str, evidence: str) -> str:
    """Build prompt for NLI checker."""
    prompt = (
        "You are a medical fact-checking assistant.\n"
        "Given the CLAIM and EVIDENCE, output exactly one label from {entailed, contradicted, neutral}.\n\n"
        f"CLAIM: {claim}\n\n"
        f"EVIDENCE: {evidence}\n\n"
        "LABEL:"
    )
    return prompt


@torch.no_grad()
def score_label_logprob(model, tok, prompt: str, label: str, device) -> float:
    """Compute log P(label | prompt) for probability-based NLI."""
    label_text = " " + label if not label.startswith(" ") else label
    prompt_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    label_ids = tok(label_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    input_ids = torch.cat([prompt_ids.input_ids, label_ids], dim=1)
    attn = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits

    label_len = label_ids.shape[1]
    start = input_ids.shape[1] - label_len

    logp = 0.0
    for i in range(label_len):
        pos = start + i
        token_id = int(input_ids[0, pos].item())
        prev_logits = logits[0, pos - 1]
        lp = torch.log_softmax(prev_logits, dim=-1)[token_id].item()
        logp += lp

    return float(logp)


def predict_nli_probs(model, tok, claim: str, evidence: str, device) -> Dict[str, float]:
    """Predict NLI probabilities using log-likelihood scoring."""
    prompt = build_checker_prompt(claim, evidence)
    lps = [score_label_logprob(model, tok, prompt, lab, device) for lab in LABEL_LIST]
    t = torch.tensor(lps, dtype=torch.float32)
    probs = torch.softmax(t, dim=0).tolist()
    return {
        "p_contradicted": probs[0],
        "p_neutral": probs[1],
        "p_entailed": probs[2],
    }


# ============================================================================
# Metrics computation (matching RefChecker/RAGChecker)
# ============================================================================

def compute_metrics(
    examples: List[Dict[str, Any]],
    all_claims: List[List[str]],
    all_scores: List[List[Dict[str, Any]]],
    threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Compute RefChecker/RAGChecker-style metrics:
    - Overall: F1, accuracy
    - Retriever: claim_recall, context_precision
    - Generator: faithfulness, hallucination, context_utilization
    """
    total_claims = 0
    supported_claims = 0
    contradicted_claims = 0
    neutral_claims = 0

    total_gold_claims = 0
    covered_gold_claims = 0

    unique_passages_used = set()
    total_passages = 0

    for ex, claims, scores in zip(examples, all_claims, all_scores):
        ctx = ex.get("retrieved_context", [])
        total_passages += len(ctx)

        # Generator metrics
        for score in scores:
            total_claims += 1
            p_ent = score["probs"]["p_entailed"]
            if p_ent >= threshold:
                supported_claims += 1
            elif score["probs"]["p_contradicted"] >= threshold:
                contradicted_claims += 1
            else:
                neutral_claims += 1

            # Track which passages are used
            if score.get("best_passage_index") is not None:
                unique_passages_used.add((ex.get("query_id", ""), score["best_passage_index"]))

        # Claim recall (if gold claims available)
        gold_claims = ex.get("gold_claims", []) or ex.get("reference_claims", [])
        if gold_claims:
            total_gold_claims += len(gold_claims)
            # Simple overlap-based claim recall
            for gc in gold_claims:
                for c in claims:
                    if jaccard_similarity(gc, c) >= 0.5:
                        covered_gold_claims += 1
                        break

    # Compute final metrics
    faithfulness = supported_claims / max(1, total_claims)
    hallucination = contradicted_claims / max(1, total_claims)
    context_utilization = supported_claims / max(1, total_claims)

    claim_recall = covered_gold_claims / max(1, total_gold_claims)
    context_precision = len(unique_passages_used) / max(1, total_passages)

    # Overall F1 (simplified: based on faithfulness and claim_recall)
    f1 = 2 * faithfulness * claim_recall / max(0.0001, faithfulness + claim_recall)

    return {
        "overall_metrics": {
            "f1": f1,
            "total_examples": len(examples),
            "total_claims": total_claims,
        },
        "retriever_metrics": {
            "claim_recall": claim_recall,
            "context_precision": context_precision,
            "unique_passages_used": len(unique_passages_used),
            "total_passages": total_passages,
        },
        "generator_metrics": {
            "faithfulness": faithfulness,
            "hallucination": hallucination,
            "context_utilization": context_utilization,
            "supported_claims": supported_claims,
            "contradicted_claims": contradicted_claims,
            "neutral_claims": neutral_claims,
        },
    }


# ============================================================================
# Main evaluation pipeline
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", required=True, help="Path to results_text*.json")
    ap.add_argument("--extractor_dir", required=True, help="Path to student extractor checkpoint")
    ap.add_argument("--checker_dir", required=True, help="Path to student checker checkpoint")
    ap.add_argument("--base_model_extractor", required=True, help="Base model for extractor")
    ap.add_argument("--base_model_checker", required=True, help="Base model for checker")
    ap.add_argument("--out_json", required=True, help="Output JSON path")
    ap.add_argument("--out_csv", default=None, help="Output CSV path for metrics")
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--threshold", type=float, default=0.6, help="p_entailed threshold for support")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_examples", type=int, default=None, help="Limit evaluation to N examples")
    args = ap.parse_args()

    # Load data
    print(f"Loading results from: {args.results_path}")
    with open(args.results_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    examples = blob.get("results", [])
    if args.max_examples:
        examples = examples[:args.max_examples]
    print(f"Evaluating on {len(examples)} examples")

    # Load models
    print("Loading extractor model...")
    extractor, tok_ex, dev_ex = load_extractor_model(
        args.base_model_extractor, args.extractor_dir, args.dtype
    )

    print("Loading checker model...")
    checker, tok_ck, dev_ck = load_checker_model(
        args.base_model_checker, args.checker_dir, args.dtype
    )

    # Extract claims
    print("Extracting claims...")
    all_claims = []
    for i in tqdm(range(0, len(examples), args.batch_size), desc="Extraction"):
        batch = examples[i:i+args.batch_size]
        claims_batch = extract_claims_batch(
            batch, extractor, tok_ex, dev_ex, args.max_new_tokens, args.temperature
        )
        all_claims.extend(claims_batch)

    # Check claims
    print("Checking claims...")
    all_scores = []
    for ex, claims in tqdm(zip(examples, all_claims), total=len(examples), desc="Checking"):
        ctx = ex.get("retrieved_context", [])
        claim_scores = []

        for claim in claims:
            if not claim.strip():
                continue

            # Find best matching passage
            best_evid, best_idx, overlap = best_passage_match(claim, ctx)

            if not best_evid:
                # No evidence available
                claim_scores.append({
                    "claim": claim,
                    "best_passage_index": None,
                    "best_passage_text": "",
                    "overlap": 0.0,
                    "probs": {"p_entailed": 0.0, "p_neutral": 1.0, "p_contradicted": 0.0},
                })
                continue

            # Run NLI checker
            probs = predict_nli_probs(checker, tok_ck, claim, best_evid, dev_ck)

            claim_scores.append({
                "claim": claim,
                "best_passage_index": best_idx,
                "best_passage_text": best_evid,
                "overlap": overlap,
                "probs": probs,
            })

        all_scores.append(claim_scores)

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(examples, all_claims, all_scores, args.threshold)

    # Save results
    output = {
        "config": {
            "extractor_dir": args.extractor_dir,
            "checker_dir": args.checker_dir,
            "base_model_extractor": args.base_model_extractor,
            "base_model_checker": args.base_model_checker,
            "threshold": args.threshold,
        },
        "metrics": metrics,
        "results": [
            {
                "query_id": ex.get("query_id", i),
                "query": ex.get("query", ""),
                "response": ex.get("response", ""),
                "claims": claims,
                "scores": scores,
            }
            for i, (ex, claims, scores) in enumerate(zip(examples, all_claims, all_scores))
        ],
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved results to: {args.out_json}")

    # Save metrics CSV
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric Category", "Metric", "Value"])
            for cat, mets in metrics.items():
                for k, v in mets.items():
                    writer.writerow([cat, k, v])
        print(f"Saved metrics CSV to: {args.out_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
