# -*- coding: utf-8 -*-
"""
Minimal GRPO-like trainer for the checker task, with wandb logging and tqdm progress bar.

Reward: 1 if sampled label equals gold label, else 0.
Group baseline: mean reward over K samples for the same prompt.
Loss approx: token-level NLL over generated tail * (-advantage).

Logs (to stdout + wandb):
- avg_reward and EMA(avg_reward)
- label histogram to detect mode collapse
- per-epoch generation-based per-class accuracy + overall accuracy

Example environment:

export WANDB_PROJECT=MedRAGChecker
export WANDB_NAME=checker_grpo_Meditron3
wandb login  # only needed once

How to run


BASE=/ocean/projects/med230010p/yji3/models/
BASE=/ocean/projects/med230010p/yji3/models/Meditron3-8B
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python DistillChecker/train_checker_grpo.py \
  --model_name "$BASE" \
  --train_path ./data/checker_grpo.jsonl \
  --output_dir ./runs/checker_grpo_meditron \
  --epochs 1 --batch_size 1 --grad_accum 16 --bf16 --K 4 \
  --eval_subset 800
"""

import os
import json
import argparse
from typing import List, Dict, Any
from collections import Counter

import torch
import wandb
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm.auto import tqdm


# ----- labels and normalization -----
LABELS = ["entailed", "contradicted", "neutral"]
LABEL_ALIASES = {
    "entailed": {"entailed", "entail", "entails", "supported", "yes"},
    "contradicted": {"contradicted", "contradict", "refuted", "no"},
    "neutral": {
        "neutral",
        "unknown",
        "insufficient",
        "not enough info",
        "not enough information",
        "uncertain",
    },
}


def _safe_lower(s: str) -> str:
    return (s or "").lower().strip()


def normalize_label(text: str):
    """Map free-form checker output into one of LABELS or None."""
    t = _safe_lower(text).replace(".", "").replace("label:", "").strip()
    for lab, al in LABEL_ALIASES.items():
        for a in al:
            if t == a or t.startswith(a):
                return lab
    if "contrad" in t or "refut" in t:
        return "contradicted"
    if "neutral" in t or "insufficient" in t or "not enough" in t or "uncertain" in t:
        return "neutral"
    if "entail" in t or "support" in t or t in {"yes", "y"}:
        return "entailed"
    return None


# ----- dataset -----
class CheckerGRPODataset(Dataset):
    """Simple JSONL dataset with fields: prompt, label."""

    def __init__(self, path: str):
        self.rows: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                p = (r.get("prompt") or "").strip()
                y = (r.get("label") or "").strip()
                if p and y:
                    self.rows.append({"prompt": p, "label": y})
        if not self.rows:
            raise RuntimeError(f"Empty dataset at {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, str]:
        return self.rows[i]


# ----- small helpers -----
class EMA:
    """Exponential moving average for scalars."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.val = None

    def update(self, x: float) -> float:
        self.val = x if self.val is None else self.alpha * x + (1 - self.alpha) * self.val
        return self.val


@torch.no_grad()
def sample_label(model, tok, prompt_ids, max_new_tokens: int = 3, temperature: float = 0.7):
    """Sample a short completion and map to a label keyword."""
    out = model.generate(
        input_ids=prompt_ids,
        attention_mask=(prompt_ids != tok.pad_token_id).long(),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    gen = out[:, prompt_ids.shape[1]:]
    txt = tok.decode(gen[0], skip_special_tokens=True).strip().lower()
    for lab in LABELS:
        if lab in txt:
            return lab, gen
    return "neutral", gen  # fallback


@torch.no_grad()
def eval_on_pairs(model, tok, pairs, max_new_tokens: int = 4):
    """Deterministic eval to get per-class and overall accuracy."""
    model.eval()
    y_true, y_pred = [], []
    for ex in pairs:
        enc = tok(
            ex["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = normalize_label(gen) or "neutral"
        y_true.append(ex["label"])
        y_pred.append(pred)

    counts = {c: 0 for c in LABELS}
    corr = {c: 0 for c in LABELS}
    for yt, yp in zip(y_true, y_pred):
        if yt in counts:
            counts[yt] += 1
            corr[yt] += int(yt == yp)
    per_cls = {c: (corr[c] / counts[c] if counts[c] else 0.0) for c in LABELS}
    overall = sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    model.train()
    return overall, per_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_path", default="./data/checker_grpo.jsonl")
    ap.add_argument("--output_dir", default="./runs/checker_grpo_meditron")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--K", type=int, default=4, help="Number of samples per input for GRPO")
    ap.add_argument("--eval_subset", type=int, default=800, help="How many examples to eval per epoch.")
    args = ap.parse_args()

    run_name = f"checker_grpo_{os.path.basename(args.model_name).replace('/', '-')}"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "MedRAGChecker"),
        name=os.environ.get("WANDB_NAME", run_name),
        config=vars(args),
    )

    ds = CheckerGRPODataset(args.train_path)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
    )

    # LoRA
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # approximate optimizer steps (per K sample group, with grad_accum)
    total_steps = max(1, (len(dl) * args.epochs * args.K) // max(1, args.grad_accum))
    print(f"[info] approx optimizer update steps = {total_steps}")
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=max(10, total_steps // 20),
        num_training_steps=total_steps,
    )

    # small held-out eval set
    eval_pairs = []
    with open(args.train_path, "r", encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i >= args.eval_subset:
                break
            obj = json.loads(ln)
            if "prompt" in obj and "label" in obj:
                eval_pairs.append({"prompt": obj["prompt"], "label": obj["label"]})

    # ema_reward = EMA(alpha=0.1)
    # step = 0
    # num_batches = len(dl)

    # for ep in range(args.epochs):
    #     label_hist = Counter()
    #     # tqdm progress bar for this epoch
    #     batch_iter = tqdm(dl, desc=f"Epoch {ep+1}/{args.epochs}", total=num_batches)

    #     for batch in batch_iter:
    #         prompts = batch["prompt"]
    #         golds = batch["label"]

    #         loss_accum = 0.0
    #         for _ in range(args.K):
    #             # tokenize once per K-sample group
    #             enc = tok(
    #                 list(prompts),
    #                 return_tensors="pt",
    #                 padding=True,
    #                 truncation=True,
    #                 max_length=1024,
    #             )
    #             enc = {k: v.to(model.device) for k, v in enc.items()}

    #             # sample one label per sequence
    #             with torch.no_grad():
    #                 sampled = []
    #                 gens = []
    #                 B = enc["input_ids"].size(0)
    #                 for i in range(B):
    #                     lab, gen = sample_label(model, tok, enc["input_ids"][i: i + 1])
    #                     sampled.append(lab)
    #                     gens.append(gen)

    #             # build concat(prompt + gen) and compute NLL only on generated tail
    #             cat_inputs, cat_labels = [], []
    #             for i in range(len(gens)):
    #                 x = torch.cat([enc["input_ids"][i], gens[i][0]], dim=0)  # [Lp + Lg]
    #                 y = torch.full_like(x, fill_value=-100)
    #                 y[-gens[i].shape[1]:] = x[-gens[i].shape[1]:]
    #                 cat_inputs.append(x)
    #                 cat_labels.append(y)
    #             cat_inputs = torch.nn.utils.rnn.pad_sequence(
    #                 cat_inputs, batch_first=True, padding_value=tok.pad_token_id
    #             ).to(model.device)
    #             cat_labels = torch.nn.utils.rnn.pad_sequence(
    #                 cat_labels, batch_first=True, padding_value=-100
    #             ).to(model.device)
    #             attn = (cat_inputs != tok.pad_token_id).long()

    #             out = model(input_ids=cat_inputs, attention_mask=attn, labels=cat_labels)
    #             nll = out.loss  # mean over tokens/seq in batch

    #             # scalar rewards & baseline within group
    #             rewards = torch.tensor(
    #                 [1.0 if s == g else 0.0 for g, s in zip(golds, sampled)],
    #                 dtype=torch.float32,
    #                 device=model.device,
    #             )
    #             avg_r = float(rewards.mean().item())
    #             r_ema = ema_reward.update(avg_r)
    #             for s in sampled:
    #                 label_hist[s] += 1

    #             baseline = rewards.mean()
    #             adv = rewards - baseline  # shape [B]

    #             scaled = nll * (-adv.mean())
    #             scaled.backward()
    #             loss_accum += float(scaled.detach().cpu())

    #         if (step + 1) % args.grad_accum == 0:
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #             optim.step()
    #             scheduler.step()
    #             optim.zero_grad(set_to_none=True)
    #         step += 1

    #         if step % 50 == 0:
    #             # update tqdm bar postfix
    #             batch_iter.set_postfix(
    #                 {
    #                     "avg_r": f"{avg_r:.4f}",
    #                     "ema_r": f"{r_ema:.4f}",
    #                 }
    #             )
    #             print(
    #                 f"[grpo] step={step} avg_reward={avg_r:.4f} ema_reward={r_ema:.4f} "
    #                 f"label_hist={dict(label_hist)}"
    #             )
    #             try:
    #                 wandb.log(
    #                     {
    #                         "train/step": int(step),
    #                         "train/epoch": int(ep + 1),
    #                         "train/avg_reward": float(avg_r),
    #                         "train/ema_reward": float(r_ema),
    #                     },
    #                     step=step,
    #                 )
    #             except Exception:
    #                 pass

    #     # ---------- epoch-end eval (run once per epoch, after the dataloader finishes) ----------
    #     overall, per_cls = eval_on_pairs(model, tok, eval_pairs)
    #     print(f"[grpo][epoch {ep+1}] overall_acc={overall:.4f} per_class={per_cls}")

    #     try:
    #         log_dict = {
    #             "eval/epoch": int(ep + 1),
    #             "eval/overall_acc": float(overall),
    #         }
    #         for k, v in per_cls.items():
    #             log_dict[f"eval/per_class/{k}"] = float(v)
    #         wandb.log(log_dict, step=step)
    #     except Exception:
    #         pass

        # os.makedirs(args.output_dir, exist_ok=True)
        # save_dir = os.path.join(args.output_dir, f"epoch{ep+1}")
        # model.save_pretrained(save_dir)
        # tok.save_pretrained(save_dir)
        # print(f"GRPO epoch {ep+1} saved to {save_dir}")
    ema_reward = EMA(alpha=0.1)
    step = 0
    num_batches = len(dl)

    for ep in range(args.epochs):
        label_hist = Counter()
        batch_iter = tqdm(dl, desc=f"Epoch {ep+1}/{args.epochs}", total=num_batches)

        for batch in batch_iter:
            prompts = batch["prompt"]
            golds = batch["label"]

            # tokenize once per batch
            enc = tok(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            B = enc["input_ids"].size(0)

            loss_accum = 0.0
            for _ in range(args.K):
                # batched generation for the whole batch, instead of per-example generate
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=3,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        eos_token_id=tok.eos_token_id,
                        pad_token_id=tok.eos_token_id,
                    )
                    gen = out[:, enc["input_ids"].shape[1]:]  # [B, Lg]

                    sampled = []
                    gens = []
                    for i in range(B):
                        txt = tok.decode(gen[i], skip_special_tokens=True).strip()
                        lab = normalize_label(txt) or "neutral"
                        sampled.append(lab)
                        gens.append(gen[i].unsqueeze(0))  # shape [1, Lg]

                # build concat(prompt + gen) and compute NLL only on generated tail
                cat_inputs, cat_labels = [], []
                for i in range(len(gens)):
                    x = torch.cat([enc["input_ids"][i], gens[i][0]], dim=0)  # [Lp + Lg]
                    y = torch.full_like(x, fill_value=-100)
                    y[-gens[i].shape[1]:] = x[-gens[i].shape[1]:]
                    cat_inputs.append(x)
                    cat_labels.append(y)

                cat_inputs = torch.nn.utils.rnn.pad_sequence(
                    cat_inputs, batch_first=True, padding_value=tok.pad_token_id
                ).to(model.device)
                cat_labels = torch.nn.utils.rnn.pad_sequence(
                    cat_labels, batch_first=True, padding_value=-100
                ).to(model.device)
                attn = (cat_inputs != tok.pad_token_id).long()

                out = model(input_ids=cat_inputs, attention_mask=attn, labels=cat_labels)
                nll = out.loss  # mean over tokens/seq in batch

                # scalar rewards & baseline within group
                rewards = torch.tensor(
                    [1.0 if s == g else 0.0 for g, s in zip(golds, sampled)],
                    dtype=torch.float32,
                    device=model.device,
                )
                avg_r = float(rewards.mean().item())
                r_ema = ema_reward.update(avg_r)
                for s in sampled:
                    label_hist[s] += 1

                baseline = rewards.mean()
                adv = rewards - baseline  # shape [B]

                scaled = nll * (-adv.mean())
                scaled.backward()
                loss_accum += float(scaled.detach().cpu())

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
            step += 1

            if step % 50 == 0:
                batch_iter.set_postfix(
                    {
                        "avg_r": f"{avg_r:.4f}",
                        "ema_r": f"{r_ema:.4f}",
                    }
                )
                print(
                    f"[grpo] step={step} avg_reward={avg_r:.4f} ema_reward={r_ema:.4f} "
                    f"label_hist={dict(label_hist)}"
                )
                try:
                    wandb.log(
                        {
                            "train/step": int(step),
                            "train/epoch": int(ep + 1),
                            "train/avg_reward": float(avg_r),
                            "train/ema_reward": float(r_ema),
                        },
                        step=step,
                    )
                except Exception:
                    pass

        # ---------- epoch-end eval (run once per epoch) ----------
        overall, per_cls = eval_on_pairs(model, tok, eval_pairs)
        print(f"[grpo][epoch {ep+1}] overall_acc={overall:.4f} per_class={per_cls}")

        try:
            log_dict = {
                "eval/epoch": int(ep + 1),
                "eval/overall_acc": float(overall),
            }
            for k, v in per_cls.items():
                log_dict[f"eval/per_class/{k}"] = float(v)
            wandb.log(log_dict, step=step)
        except Exception:
            pass

        os.makedirs(args.output_dir, exist_ok=True)
        save_dir = os.path.join(args.output_dir, f"epoch{ep+1}")
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print(f"GRPO epoch {ep+1} saved to {save_dir}")

    # final save
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"Final GRPO LoRA saved to {args.output_dir}")


if __name__ == "__main__":
    main()

# # -*- coding: utf-8 -*-
# """
# Minimal GRPO-like trainer for the checker task.

# Reward: 1 if sampled label equals gold label, else 0.
# Group baseline: mean reward over K samples for the same prompt.
# Loss approx: token-level NLL over generated tail * (-advantage).

# Logs:
# - avg_reward and EMA(avg_reward)
# - label histogram to detect mode collapse
# - per-epoch generation-based per-class accuracy + overall accuracy

# export WANDB_PROJECT=MedRAGChecker
# # 可选：指定 run 名，Trainer 会自动用这个
# export WANDB_NAME=extractor_sft_Meditron3
# wandb login  # 只需要做一次
# How to run

# BASE=/ocean/projects/med230010p/yji3/models/Meditron3-8B
# CUDA_VISIBLE_DEVICES=0,1 \
# python DistillChecker/train_checker_grpo.py \
#   --model_name "$BASE" \
#   --train_path ./data/checker_grpo.jsonl \
#   --output_dir ./runs/checker_grpo_meditron \
#   --epochs 1 --batch_size 1 --grad_accum 16 --bf16 --K 4 \
#   --eval_subset 800
# """

# import os, json, argparse
# from typing import List, Dict, Any
# from collections import Counter

# import torch
# import wandb 
# from tqdm.auto import tqdm 
# from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# # ----- labels and normalization -----
# LABELS = ["entailed","contradicted","neutral"]
# LABEL_ALIASES = {
#     "entailed": {"entailed","entail","entails","supported","yes"},
#     "contradicted": {"contradicted","contradict","refuted","no"},
#     "neutral": {"neutral","unknown","insufficient","not enough info","not enough information","uncertain"},
# }
# def _safe_lower(s): return (s or "").lower().strip()
# def normalize_label(text: str):
#     t = _safe_lower(text).replace(".", "").replace("label:", "").strip()
#     for lab, al in LABEL_ALIASES.items():
#         for a in al:
#             if t == a or t.startswith(a): return lab
#     if "contrad" in t or "refut" in t: return "contradicted"
#     if "neutral" in t or "insufficient" in t or "not enough" in t or "uncertain" in t: return "neutral"
#     if "entail" in t or "support" in t or t in {"yes","y"}: return "entailed"
#     return None

# # ----- dataset -----
# class CheckerGRPODataset(Dataset):
#     def __init__(self, path:str):
#         self.rows=[]
#         with open(path,"r",encoding="utf-8") as f:
#             for line in f:
#                 if not line.strip(): continue
#                 r = json.loads(line)
#                 p, y = r.get("prompt","").strip(), r.get("label","").strip()
#                 if p and y:
#                     self.rows.append({"prompt": p, "label": y})
#         if not self.rows:
#             raise RuntimeError("Empty dataset")
#     def __len__(self): return len(self.rows)
#     def __getitem__(self,i): return self.rows[i]

# # ----- small helpers -----
# class EMA:
#     """Exponential moving average for scalars."""
#     def __init__(self, alpha=0.1): self.alpha, self.val = alpha, None
#     def update(self, x):
#         self.val = x if self.val is None else self.alpha*x + (1-self.alpha)*self.val
#         return self.val

# @torch.no_grad()
# def sample_label(model, tok, prompt_ids, max_new_tokens=3, temperature=0.7):
#     """Sample a short completion and map to a label keyword."""
#     out = model.generate(
#         input_ids=prompt_ids,
#         attention_mask=(prompt_ids != tok.pad_token_id).long(),
#         max_new_tokens=max_new_tokens,
#         do_sample=True, temperature=temperature, top_p=0.9,
#         eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
#     )
#     gen = out[:, prompt_ids.shape[1]:]
#     txt = tok.decode(gen[0], skip_special_tokens=True).strip().lower()
#     for lab in LABELS:
#         if lab in txt:
#             return lab, gen
#     return "neutral", gen  # fallback

# def eval_on_pairs(model, tok, pairs, max_new_tokens=4):
#     """Deterministic eval to get per-class and overall accuracy."""
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for ex in pairs:
#             enc = tok(ex["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(model.device)
#             out = model.generate(
#                 **enc, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False,
#                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id
#             )
#             gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
#             pred = normalize_label(gen) or "neutral"
#             y_true.append(ex["label"]); y_pred.append(pred)
#     counts = {c:0 for c in LABELS}; corr = {c:0 for c in LABELS}
#     for yt, yp in zip(y_true, y_pred):
#         if yt in counts:
#             counts[yt]+=1; corr[yt]+= int(yt==yp)
#     per_cls = {c: (corr[c]/counts[c] if counts[c] else 0.0) for c in LABELS}
#     overall = sum(int(a==b) for a,b in zip(y_true,y_pred)) / len(y_true) if y_true else 0.0
#     model.train()
#     return overall, per_cls

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model_name", required=True)
#     ap.add_argument("--train_path", default="./data/checker_grpo.jsonl")
#     ap.add_argument("--output_dir", default="./runs/checker_grpo_meditron")
#     ap.add_argument("--epochs", type=int, default=1)
#     ap.add_argument("--batch_size", type=int, default=2)
#     ap.add_argument("--grad_accum", type=int, default=8)
#     ap.add_argument("--lr", type=float, default=5e-6)
#     ap.add_argument("--bf16", action="store_true")
#     ap.add_argument("--fp16", action="store_true")
#     ap.add_argument("--lora_r", type=int, default=8)
#     ap.add_argument("--lora_alpha", type=int, default=16)
#     ap.add_argument("--lora_dropout", type=float, default=0.0)
#     ap.add_argument("--K", type=int, default=4, help="Number of samples per input for GRPO")
#     ap.add_argument("--eval_subset", type=int, default=800, help="How many examples to eval per epoch.")
#     args = ap.parse_args()
#     run_name = f"checker_grpo_{os.path.basename(args.model_name).replace('/', '-')}"
#     wandb.init(
#         project=os.environ.get("WANDB_PROJECT", "MedRAGChecker"),
#         name=run_name,
#         config=vars(args),
#     )

#     ds = CheckerGRPODataset(args.train_path)
#     dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

#     tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
#     tok.padding_side = "left"
#     if tok.pad_token is None:
#         tok.pad_token = tok.eos_token

#     model = AutoModelForCausalLM.from_pretrained(
#         args.model_name,
#         dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
#         device_map="auto",
#     )
#     # LoRA
#     from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
#     model = prepare_model_for_kbit_training(model)
#     model.config.use_cache = False
#     try: model.config._attn_implementation = "eager"
#     except Exception: pass
#     lora_cfg = LoraConfig(
#         r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
#         target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
#     )
#     model = get_peft_model(model, lora_cfg)
#     model.train()

#     optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
#     total_steps = max(1, (len(dl) * args.epochs * args.K) // max(1, args.grad_accum))
#     scheduler = get_linear_schedule_with_warmup(
#         optim, num_warmup_steps=max(10, total_steps//20), num_training_steps=total_steps
#     )

#     # small held-out eval set
#     eval_pairs = []
#     with open(args.train_path, "r", encoding="utf-8") as f:
#         for i, ln in enumerate(f):
#             if i >= args.eval_subset: break
#             obj = json.loads(ln)
#             if "prompt" in obj and "label" in obj:
#                 eval_pairs.append({"prompt": obj["prompt"], "label": obj["label"]})

#     ema_reward = EMA(alpha=0.1)
#     step = 0
#     for ep in range(args.epochs):
#         label_hist = Counter()
#         for batch in dl:
#             prompts = batch["prompt"]
#             golds = batch["label"]

#             loss_accum = 0.0
#             for _ in range(args.K):
#                 # tokenize once per K-sample group
#                 enc = tok(list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=1024)
#                 enc = {k: v.to(model.device) for k, v in enc.items()}

#                 # sample one label per sequence
#                 with torch.no_grad():
#                     sampled = []
#                     gens = []
#                     B = enc["input_ids"].size(0)
#                     for i in range(B):
#                         lab, gen = sample_label(model, tok, enc["input_ids"][i:i+1])
#                         sampled.append(lab); gens.append(gen)

#                 # build concat(prompt + gen) and compute NLL only on generated tail
#                 cat_inputs, cat_labels = [], []
#                 for i in range(len(gens)):
#                     x = torch.cat([enc["input_ids"][i], gens[i][0]], dim=0)  # [Lp + Lg]
#                     y = torch.full_like(x, fill_value=-100)
#                     y[-gens[i].shape[1]:] = x[-gens[i].shape[1]:]
#                     cat_inputs.append(x); cat_labels.append(y)
#                 cat_inputs = torch.nn.utils.rnn.pad_sequence(cat_inputs, batch_first=True, padding_value=tok.pad_token_id).to(model.device)
#                 cat_labels = torch.nn.utils.rnn.pad_sequence(cat_labels, batch_first=True, padding_value=-100).to(model.device)
#                 attn = (cat_inputs != tok.pad_token_id).long()

#                 out = model(input_ids=cat_inputs, attention_mask=attn, labels=cat_labels)
#                 nll = out.loss  # mean over tokens/seq in batch

#                 # scalar rewards & baseline within group
#                 rewards = torch.tensor([1.0 if s == g else 0.0 for g, s in zip(golds, sampled)],
#                                        dtype=torch.float32, device=model.device)
#                 avg_r = float(rewards.mean().item())
#                 r_ema = ema_reward.update(avg_r)
#                 for s in sampled: label_hist[s] += 1

#                 baseline = rewards.mean()
#                 adv = rewards - baseline  # shape [B] (broadcasted via mean below)

#                 scaled = nll * (-adv.mean())
#                 scaled.backward()
#                 loss_accum += float(scaled.detach().cpu())

#             # if (step+1) % args.grad_accum == 0:
#             #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             #     optim.step(); scheduler.step()
#             #     optim.zero_grad(set_to_none=True)
#             # step += 1

#             # if (step % 50) == 0:
#             #     print(f"[grpo] step={step} avg_reward={avg_r:.4f} ema_reward={r_ema:.4f} "
#             #           f"label_hist={dict(label_hist)}")
#             if (step+1) % args.grad_accum == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#                 optim.step(); scheduler.step()
#                 optim.zero_grad(set_to_none=True)
#             step += 1

#             if (step % 50) == 0:
#                 print(f"[grpo] step={step} avg_reward={avg_r:.4f} ema_reward={r_ema:.4f} "
#                       f"label_hist={dict(label_hist)}")
#                 try:
#                     wandb.log(
#                         {
#                             "train/step": int(step),
#                             "train/epoch": int(ep + 1),
#                             "train/avg_reward": float(avg_r),
#                             "train/ema_reward": float(r_ema),
#                         },
#                         step=step,
#                     )
#                 except Exception:
#                     pass

#         # epoch-end eval
#         # overall, per_cls = eval_on_pairs(model, tok, eval_pairs)
#         # print(f"[grpo][epoch {ep+1}] overall_acc={overall:.4f} per_class={per_cls}")

#         # os.makedirs(args.output_dir, exist_ok=True)
#         # model.save_pretrained(os.path.join(args.output_dir, f"epoch{ep+1}"))
#                 # epoch-end eval
#         overall, per_cls = eval_on_pairs(model, tok, eval_pairs)
#         print(f"[grpo][epoch {ep+1}] overall_acc={overall:.4f} per_class={per_cls}")

#         try:
#             log_dict = {
#                 "eval/epoch": int(ep + 1),
#                 "eval/overall_acc": float(overall),
#             }
#             for k, v in per_cls.items():
#                 log_dict[f"eval/per_class/{k}"] = float(v)
#             wandb.log(log_dict, step=step)
#         except Exception:
#             pass

#         os.makedirs(args.output_dir, exist_ok=True)
#         model.save_pretrained(os.path.join(args.output_dir, f"epoch{ep+1}"))

#         tok.save_pretrained(os.path.join(args.output_dir, f"epoch{ep+1}"))
#         print(f"✅ GRPO epoch {ep+1} saved.")

#         tok.save_pretrained(os.path.join(args.output_dir, f"epoch{ep+1}"))
#         print(f"✅ GRPO epoch {ep+1} saved.")

#     # final save
#     model.save_pretrained(args.output_dir)
#     tok.save_pretrained(args.output_dir)
#     print(f"✅ Final GRPO LoRA saved to {args.output_dir}")

# if __name__ == "__main__":
#     main()

