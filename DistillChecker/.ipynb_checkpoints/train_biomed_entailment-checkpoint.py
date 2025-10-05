# All code comments in English only.

# -*- coding: utf-8 -*-
"""
Train a lightweight NLI-style checker (entail/contradict/neutral) from distillation JSONL.
- Input: distill_checker_*.jsonl produced by your pipeline (one JSON per line)
  Each line contains:
    {
      "query_id": str,
      "pair_type": "response_vs_gt" | "response_vs_retrieved" | "gt_vs_retrieved",
      "claim": str,
      "evidence_text": str,
      "label": str  # one of {"entailed","contradicted","neutral"} (or your variants)
      # optional: passage_index, claim_index
    }

Recommended base models: DeBERTa-v3-base, RoBERTa-base, Qwen2.5-Math-.. (encoder),
or any encoder-only model for sequence classification.
"""

import os, json, argparse, math, random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, PreTrainedTokenizerBase
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABEL_CANON = {
    "entail": "entailed",
    "entailed": "entailed",
    "supported": "entailed",
    "contradict": "contradicted",
    "contradicted": "contradicted",
    "refuted": "contradicted",
    "neutral": "neutral",
    "unknown": "neutral"
}
LABEL_LIST = ["contradicted", "neutral", "entailed"]  # fix order for id mapping
LABEL2ID = {k: i for i, k in enumerate(LABEL_LIST)}
ID2LABEL = {i: k for k, i in LABEL2ID.items()}

def canon_label(x: str) -> Optional[str]:
    if x is None: return None
    x = x.strip().lower()
    return LABEL_CANON.get(x, x) if x in LABEL_CANON else (x if x in LABEL2ID else None)

def format_example(claim: str, evidence: str, pair_type: str) -> str:
    # Keep format simple and deterministic.
    return (
        "Task: Determine the relation between a claim and the evidence.\n"
        f"PairType: {pair_type}\n\n"
        f"Claim:\n{claim}\n\n"
        f"Evidence:\n{evidence}\n"
    )

class DistillNLIDataset(Dataset):
    def __init__(self, path: str, max_samples: Optional[int] = None):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                ex = json.loads(line)
                label = canon_label(ex.get("label"))
                claim = ex.get("claim", "")
                ev = ex.get("evidence_text", "")
                ptype = ex.get("pair_type", "response_vs_retrieved")
                if not (label and claim and ev):
                    continue
                self.rows.append({
                    "text": format_example(claim, ev, ptype),
                    "label": LABEL2ID[label]
                })
        if max_samples:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise RuntimeError("No valid NLI examples found.")

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

@dataclass
class NLICollator:
    tokenizer: PreTrainedTokenizerBase
    max_len: int = 512
    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        enc["labels"] = labels
        return enc

def split_dataset(ds: Dataset, train_ratio: float = 0.95, seed: int = 42):
    n = len(ds)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_tr = int(n * train_ratio)
    tr_idx, ev_idx = idx[:n_tr], idx[n_tr:]
    return torch.utils.data.Subset(ds, tr_idx), torch.utils.data.Subset(ds, ev_idx)

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1m = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1m}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, required=True, help="Path to distill_checker_*.jsonl")
    ap.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    ap.add_argument("--output_dir", type=str, default="./checker_nli_sft")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )

    ds = DistillNLIDataset(args.train_path, max_samples=args.max_samples)
    tr_ds, ev_ds = split_dataset(ds, train_ratio=0.95)

    collator = NLICollator(tokenizer=tok, max_len=args.max_len)

    # Transformers >= 4.41 arg name changed
    kw = {}
    try:
        from transformers import __version__ as tfv
        from packaging import version
        if version.parse(tfv) >= version.parse("4.41.0"):
            kw["eval_strategy"] = "epoch"
            kw["save_strategy"] = "epoch"
        else:
            kw["evaluation_strategy"] = "epoch"
            kw["save_strategy"] = "epoch"
    except Exception:
        kw["evaluation_strategy"] = "epoch"; kw["save_strategy"] = "epoch"

    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=50,
        fp16=args.fp16,
        bf16=args.bf16,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        report_to="tensorboard",
        **kw
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=tr_ds,
        eval_dataset=ev_ds,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics_fn
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"✅ Finished NLI checker SFT. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
# python train_checker_nli.py \
#   --train_path tests/_min_output/distill_data/distill_checker_text.jsonl \
#   --model_name microsoft/deberta-v3-base \
#   --output_dir ./checker_nli_deberta \
#   --batch_size 32 --epochs 3 --lr 2e-5 --bf16
