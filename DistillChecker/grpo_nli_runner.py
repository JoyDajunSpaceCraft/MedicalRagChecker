# grpo_nli_runner.py
# -*- coding: utf-8 -*-
import os, re, json, math, random, argparse
from typing import List, Dict, Any
# INPUT=/ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_csiro_test_gptteacher/csiro_test.Meditron3-8B.gen100/text_eval/results_text.json
# CUDA_VISIBLE_DEVICES=0,1 python grpo_nli_runner.py   --results_path "$INPUT"   --model_name /ocean/projects/med230010p/yji3/models/Meditron3-8B   --output_dir ./checker_meditron3_grpo   --epochs 3 --bsz 1 --lr 5e-6 --bf16   --min_overlap 0.03
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# 三个标签作为“单 token”输出
LABEL_TOKENS = ["<entailed>", "<contradicted>", "<neutral>"]
LBL_CANON = {"Entailment":"entailed","Contradiction":"contradicted","Neutral":"neutral"}

def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def norm(s:str)->str:
    return re.sub(r"\s+"," ", (s or "").strip())

def triples_to_sentence(x)->str:
    return f"{x[0]} {x[1]} {x[2]}." if isinstance(x, list) and len(x)==3 else str(x)

def jaccard(a:str,b:str)->float:
    A=set(norm(a).lower().split()); B=set(norm(b).lower().split())
    return 0.0 if not A or not B else len(A&B)/len(A|B)

def build_pairs_from_results(results: List[Dict[str,Any]], min_overlap: float=0.0):
    pairs=[]
    for ex in results:
        claims = ex.get("response_claims",[]) or []
        ctx    = ex.get("retrieved_context",[]) or []
        r2r    = ex.get("retrieved2response",[]) or []
        if not claims or not ctx or not r2r: 
            continue
        for ci, c_raw in enumerate(claims):
            claim = norm(triples_to_sentence(c_raw))
            if not claim: 
                continue
            for di, doc in enumerate(ctx):
                if ci>=len(r2r) or di>=len(r2r[ci]): 
                    continue
                lbl_raw = r2r[ci][di]
                gold = LBL_CANON.get(lbl_raw, None)
                if gold not in {"entailed","contradicted","neutral"}: 
                    continue
                ev = norm(doc.get("text",""))
                if not ev: 
                    continue
                if min_overlap>0 and jaccard(claim,ev)<min_overlap:
                    continue
                pairs.append({"claim": claim, "evidence": ev, "gold": gold})
    return pairs

def to_prompt(claim, evidence):
    return (
        "You are an NLI checker. Read the claim and the evidence, then answer "
        "with ONE token: <entailed> or <contradicted> or <neutral>.\n\n"
        f"Claim: {claim}\n\nEvidence: {evidence}\n\nAnswer:"
    )

class GRPODataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        if not self.rows:
            raise RuntimeError("Empty training set")
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def add_label_tokens(tokenizer, model):
    # 把三种标签注册为“额外特殊 token”，确保是单 token
    add = {'additional_special_tokens': []}
    for t in LABEL_TOKENS:
        if tokenizer.convert_tokens_to_ids(t) == tokenizer.unk_token_id:
            add['additional_special_tokens'].append(t)
    if add['additional_special_tokens']:
        tokenizer.add_special_tokens(add)
        model.resize_token_embeddings(len(tokenizer))
    # 校验必须为单 token
    for t in LABEL_TOKENS:
        ids = tokenizer.encode(t, add_special_tokens=False)
        assert len(ids)==1, f"Label token '{t}' became multi-token: {ids}"

def collate(samples, tokenizer, max_len=768):
    prompts = [to_prompt(s["claim"], s["evidence"]) for s in samples]
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    label_ids = torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in LABEL_TOKENS], dtype=torch.long)  # [3]
    # gold -> 奖励：对=+1，错=-1
    golds = [s["gold"] for s in samples]  # ["entailed", ...]
    rewards=[]
    for g in golds:
        r=[+1.0 if t.strip("<>")==g else -1.0 for t in LABEL_TOKENS]
        rewards.append(r)
    rewards = torch.tensor(rewards, dtype=torch.float32)    # [B,3]
    return enc["input_ids"], enc["attention_mask"], label_ids, rewards, golds

def grpo_step(model, batch, device, temperature=1.0):
    input_ids, attn_mask, label_ids, rewards, _ = batch
    B = input_ids.size(0)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    label_ids = label_ids.to(device)        # [3]
    rewards   = rewards.to(device)          # [B,3]

    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits = out.logits[:, -1, :]                 # [B, vocab]
    logits = logits / temperature
    cand_logits = logits.index_select(dim=1, index=label_ids)   # [B,3]
    log_probs = F.log_softmax(cand_logits, dim=-1)              # [B,3]

    r_mean = rewards.mean(dim=-1, keepdim=True)                 # [B,1]
    r_std  = rewards.std(dim=-1, keepdim=True) + 1e-6           # [B,1]
    adv = (rewards - r_mean) / r_std                            # [B,3]

    loss = -(adv * log_probs).sum(dim=-1).mean()
    return loss

@torch.no_grad()
def quick_eval_acc(model, tokenizer, rows, device, max_len=768):
    # 用贪婪解码 1 个 token，统计准确率
    model.eval()
    bs = 64
    correct = 0
    total = 0
    label_ids = torch.tensor([tokenizer.convert_tokens_to_ids(t) for t in LABEL_TOKENS], device=device)
    for i in range(0, len(rows), bs):
        chunk = rows[i:i+bs]
        prompts = [to_prompt(r["claim"], r["evidence"]) for r in chunk]
        golds = [r["gold"] for r in chunk]
        enc = tokenizer(prompts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        out = model(**enc, use_cache=False)
        logits = out.logits[:, -1, :]                      # [B, vocab]
        cand = logits.index_select(1, label_ids)           # [B,3]
        pred_idx = cand.argmax(dim=-1).tolist()            # 0/1/2
        pred = [LABEL_TOKENS[j].strip("<>") for j in pred_idx]
        for p, g in zip(pred, golds):
            correct += int(p == g)
            total += 1
    return correct / max(1, total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_path", required=True)
    ap.add_argument("--model_name", required=True)   # /ocean/.../Meditron3-8B
    ap.add_argument("--output_dir", default="./checker_llm_grpo")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--min_overlap", type=float, default=0.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEBUG] device:", device)

    raw = json.load(open(args.results_path, "r", encoding="utf-8"))
    rows_all = build_pairs_from_results(raw.get("results", []), min_overlap=args.min_overlap)
    if not rows_all:
        raise RuntimeError("No pairs constructed from results_text.json")
    print(f"[DATA] pairs={len(rows_all)}")
    # 简单 95/5 切分
    random.shuffle(rows_all)
    n = len(rows_all)
    ntr = max(1, int(n*0.95))
    rows_tr, rows_ev = rows_all[:ntr], rows_all[ntr:]
    print(f"[DATA] train={len(rows_tr)}  eval={len(rows_ev)}")

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=(torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32),
        trust_remote_code=True
    ).to(device)
    add_label_tokens(tokenizer, model)

    ds = GRPODataset(rows_tr)
    collate_fn = lambda samples: collate(samples, tokenizer, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=True, collate_fn=collate_fn, drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    for ep in range(args.epochs):
        model.train()
        for batch in dl:
            loss = grpo_step(model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            global_step += 1
            if global_step % 20 == 0:
                print(f"[ep {ep}] step {global_step} loss={loss.item():.4f}")

        # 保存与评估
        ckpt = os.path.join(args.output_dir, f"ep{ep}")
        model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
        acc = quick_eval_acc(model, tokenizer, rows_ev if rows_ev else rows_tr[: max(1, len(rows_tr)//10)], device, max_len=args.max_len)
        print(f"✅ saved: {ckpt} | eval_acc={acc:.4f}")

if __name__ == "__main__":
    main()
