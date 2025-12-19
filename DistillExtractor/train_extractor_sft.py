# train_extractor_sft.py
# -*- coding: utf-8 -*-
"""
SFT for extractor: learn to output a JSON list of claims from (question + answer).
Use LoRA to make 8B model trainable on 1-2 GPUs.
"""
# BASE=/ocean/projects/med230010p/yji3/models/Meditron3-8B
# CUDA_VISIBLE_DEVICES=1 python DistillExtractor/train_extractor_sft.py   --model_name "$BASE"   --train_path ./data/extractor_sft.jsonl   --output_dir ./runs/extractor_sft_meditron/checkpoint-final   --epochs 2 --batch_size 1 --grad_accum 32 --bf16

import os, json, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def format_example(ex):
    # Simple instruction format for causal LM
    prompt = ex["instruction"].strip()
    out = ex["output"].strip()
    text = f"{prompt}\n\n### Response:\n{out}"
    return {"text": text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="Base LLM (e.g., /ocean/.../Meditron3-8B)")
    ap.add_argument("--train_path", default="./data/extractor_sft.jsonl")
    ap.add_argument("--output_dir", default="./runs/extractor_sft_meditron")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    args = ap.parse_args()

    ds = load_dataset("json", data_files=args.train_path, split="train").map(format_example)
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=2048)

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)  # safe even if not using 4/8bit
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
        report_to="wandb"
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"✅ Saved extractor SFT LoRA to {args.output_dir}")

if __name__ == "__main__":
    main()
