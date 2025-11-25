# -*- coding: utf-8 -*-
"""
SFT for checker: teach the model to output exactly one of
{entailed, contradicted, neutral} given (CLAIM, EVIDENCE) prompt.

Adds a generation-based eval callback that prints:
- overall accuracy
- per-class accuracy for entailed / contradicted / neutral
and saves the best checkpoint by overall accuracy.

How to run

BASE=/ocean/projects/med230010p/yji3/models/Meditron3-8B
CUDA_VISIBLE_DEVICES=0 \
python DistillChecker/train_checker_sft.py \
  --model_name "$BASE" \
  --train_path ./data/checker_sft.jsonl \
  --output_dir ./runs/checker_sft_meditron \
  --epochs 2 --batch_size 1 --grad_accum 32 --bf16 \
  --eval_subset 1000

"""

import os, json, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
)
import torch
import wandb

LABELS = ["entailed","contradicted","neutral"]
LABEL_ALIASES = {
    "entailed": {"entailed","entail","entails","supported","yes"},
    "contradicted": {"contradicted","contradict","refuted","no"},
    "neutral": {"neutral","unknown","insufficient","not enough info","not enough information","uncertain"},
}

# ---------- utils for normalization and eval ----------
def _safe_lower(s): return (s or "").lower().strip()
def normalize_label(text: str):
    t = _safe_lower(text).replace(".", "").replace("label:", "").strip()
    for lab, al in LABEL_ALIASES.items():
        for a in al:
            if t == a or t.startswith(a):
                return lab
    if "contrad" in t or "refut" in t: return "contradicted"
    if "neutral" in t or "insufficient" in t or "not enough" in t or "uncertain" in t: return "neutral"
    if "entail" in t or "support" in t or t in {"yes","y"}: return "entailed"
    return None

def load_eval_pairs(json_path: str, max_n: int | None = None):
    """Load [{'prompt','label'}...] from checker_sft.jsonl."""
    rows = []
    with open(json_path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            obj = json.loads(ln)
            p = obj.get("prompt","")
            y = obj.get("label","")
            if p and y:
                rows.append({"prompt": p, "label": y})
    return rows if max_n is None else rows[:max_n]

class GenEvalCallback(TrainerCallback):
    """At epoch end, run deterministic generation on a held-out set and log per-class accuracy."""
    def __init__(self, model, tok, eval_pairs, out_dir, topk=None):
        self.model = model
        self.tok = tok
        self.pairs = eval_pairs if topk is None else eval_pairs[:topk]
        self.best_acc = -1.0
        self.out_dir = out_dir

    @torch.no_grad()
    def on_epoch_end(self, args, state, control, **kwargs):
        self.model.eval()
        y_true, y_pred = [], []
        for ex in self.pairs:
            enc = self.tok(ex["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
            out = self.model.generate(
                **enc, max_new_tokens=4, temperature=0.0, do_sample=False,
                eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id
            )
            gen = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            pred = normalize_label(gen) or "neutral"
            y_true.append(ex["label"]); y_pred.append(pred)

        # per-class accuracy
        counts = {c:0 for c in LABELS}; corr = {c:0 for c in LABELS}
        for yt, yp in zip(y_true, y_pred):
            if yt in counts:
                counts[yt]+=1; corr[yt]+= int(yt==yp)
        per_cls = {c: (corr[c]/counts[c] if counts[c] else 0.0) for c in LABELS}
        overall = sum(int(a==b) for a,b in zip(y_true,y_pred)) / len(y_true) if y_true else 0.0

        print(f"[gen-eval] overall_acc={overall:.4f} | per-class={per_cls}")

        # log to JSONL for later visualization
        log_rec = {
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "overall_acc": float(overall),
            "per_class": {k: float(v) for k, v in per_cls.items()},
        }
        log_path = os.path.join(self.out_dir, "gen_eval_log.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_rec) + "\n")

        # save best by overall accuracy
        if overall > self.best_acc:
            self.best_acc = overall
            save_dir = os.path.join(self.out_dir, f"best_overall_acc_{overall:.4f}")
            os.makedirs(save_dir, exist_ok=True)
            self.model.save_pretrained(save_dir)
            self.tok.save_pretrained(save_dir)
            print(f"[gen-eval] ✅ new best saved to {save_dir}")

        self.model.train()

# ---------- dataset formatting ----------
def format_row(r):
    prompt = r["prompt"].strip()
    label = r["label"].strip()
    # LM objective: force a short single-token-ish answer
    text = f"{prompt} {label}"
    return {"text": text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_path", default="./data/checker_sft.jsonl")
    ap.add_argument("--output_dir", default="./runs/checker_sft_meditron")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--eval_subset", type=int, default=1000, help="How many examples to eval per epoch (generation).")
    args = ap.parse_args()

    # load raw for eval subset (generation-based accuracy)
    eval_pairs = load_eval_pairs(args.train_path, max_n=args.eval_subset)

    # load for LM training
    ds = load_dataset("json", data_files=args.train_path, split="train").map(format_row)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=1024)
    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
    )

    # LoRA for LLaMA-like blocks
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    )
    model = get_peft_model(model, lora_cfg)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        report_to="wandb",
        # If you later pass eval_dataset to Trainer, use the new `eval_strategy` (not deprecated `evaluation_strategy`)
        # docs: https://huggingface.co/docs/transformers
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=ds, data_collator=collator)
    trainer.add_callback(GenEvalCallback(model, tok, eval_pairs, args.output_dir))
    trainer.train()

    # final save
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"✅ Saved checker SFT LoRA to {args.output_dir}")

if __name__ == "__main__":
    main()
