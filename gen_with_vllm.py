
# Run with pubmedqa artificial
# python gen_with_vllm.py   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_pubmedqa_pqa_artificial/rag_generation_outputs_pubmedqa_train.jsonl   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_pubmedqa_pqa_artificial/Meditron3/rag_generation_outputs_pubmedqa_train.Meditron3-8B.gen100.jsonl   --model Meditron3-8B   --base-url http://127.0.0.1:8000/v1 --max-new-tokens 128

# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_pubmedqa_pqa_artificial/rag_generation_outputs_pubmedqa_train.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_pubmedqa_pqa_artificial/Meditron3/rag_generation_outputs_pubmedqa_train.Meditron3-8B.gen100.jsonl \
#   --model Meditron3-8B \
#   --base-url http://127.0.0.1:8000/v1 \
#   --max-new-tokens 128 \
#   --context-window 4096 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 96


# Run with LiveQA
# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_liveqa_full/rag_generation_outputs_liveqa_test.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_liveqa_full/Meditron3/rag_generation_outputs_liveqa_test.Meditron3-8B.gen100.jsonl \
#   --model Meditron3-8B \
#   --base-url http://127.0.0.1:8000/v1 \
#   --max-new-tokens 128 \
#   --context-window 4096 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 96

# Run with CSIRO test train val
# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_train/rag_generation_outputs_csiro_train.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_train/Meditron3/rag_generation_outputs_csiro_train.Meditron3-8B.gen100.jsonl \
#   --model Meditron3-8B \
#   --base-url http://127.0.0.1:8000/v1 \
#   --max-new-tokens 128 \
#   --context-window 4096 \
#   --safety-margin 96 \
#     --safety-margin 256 \
#     --max-docs 6 \
#     --max-chars-per-doc 800


# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_val/rag_generation_outputs_csiro_val.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_val/Meditron3/rag_generation_outputs_csiro_val.Meditron3-8B.gen100.jsonl \
#   --model Meditron3-8B \
#   --base-url http://127.0.0.1:8000/v1 \
#   --max-new-tokens 128 \
#   --context-window 4096 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 96

# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_test/rag_generation_outputs_csiro_test.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_test/Meditron3/rag_generation_outputs_csiro_test.Meditron3-8B.gen100.jsonl \
#   --model Meditron3-8B \
#   --base-url http://127.0.0.1:8000/v1 \
#   --max-new-tokens 128 \
#   --context-window 4096 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 96


# Run with the medquad
# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_medquad_full/rag_generation_outputs_medquad_train.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_medquad_full/Meditron3/rag_generation_outputs_medquad_train.Meditron3-8B.gen100.jsonl \
#   --model Meditron3-8B \
#   --base-url http://127.0.0.1:8000/v1 \
#   --max-new-tokens 128 \
#   --context-window 4096 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 96


# PMC_llama_13B
# ================================
# PMC_LLaMA_13B inference commands
# Notes (English-only comments):
# - Make sure your vLLM server for PMC_LLaMA_13B is running on port 8001 with --max-model-len 2048.
# - This script will auto-switch to /v1/completions if no chat_template is present.
# - If you still hit 400 "context too long", reduce --max-docs or --max-chars-per-doc slightly.
# ================================

# ---------- PubMedQA (artificial) ----------
# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_pubmedqa_pqa_artificial/rag_generation_outputs_pubmedqa_train.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_pubmedqa_pqa_artificial/PMC_LLaMA_13B/rag_generation_outputs_pubmedqa_train.PMC_LLaMA_13B.gen100.jsonl \
#   --model PMC_LLaMA_13B \
#   --base-url http://127.0.0.1:8001/v1 \
#   --tokenizer-name /ocean/projects/med230010p/yji3/models/PMC_LLaMA_13B \
#   --max-new-tokens 128 \
#   --context-window 2048 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 128

# # ---------- LiveQA (test) ----------
# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_liveqa_full/rag_generation_outputs_liveqa_test.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_liveqa_full/PMC_LLaMA_13B/rag_generation_outputs_liveqa_test.PMC_LLaMA_13B.gen100.jsonl \
#   --model PMC_LLaMA_13B \
#   --base-url http://127.0.0.1:8001/v1 \
#   --tokenizer-name /ocean/projects/med230010p/yji3/models/PMC_LLaMA_13B \
#   --max-new-tokens 128 \
#   --context-window 2048 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 128

# # ---------- CSIRO (train) ----------
# # This split previously used tighter packing; keep doc/char caps conservative for 2k window.
# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_train/rag_generation_outputs_csiro_train.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_train/PMC_LLaMA_13B/rag_generation_outputs_csiro_train.PMC_LLaMA_13B.gen100.jsonl \
#   --model PMC_LLaMA_13B \
#   --base-url http://127.0.0.1:8001/v1 \
#   --tokenizer-name /ocean/projects/med230010p/yji3/models/PMC_LLaMA_13B \
#   --max-new-tokens 128 \
#   --context-window 2048 \
#   --safety-margin 160 \
#   --max-docs 6 \
#   --max-chars-per-doc 800 \
# --skip-on-overflow

# # ---------- CSIRO (val) ----------
# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_val/rag_generation_outputs_csiro_val.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_val/PMC_LLaMA_13B/rag_generation_outputs_csiro_val.PMC_LLaMA_13B.gen100.jsonl \
#   --model PMC_LLaMA_13B \
#   --base-url http://127.0.0.1:8001/v1 \
#   --tokenizer-name /ocean/projects/med230010p/yji3/models/PMC_LLaMA_13B \
#   --max-new-tokens 128 \
#   --context-window 2048 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 128

# # ---------- CSIRO (test) ----------
# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_test/rag_generation_outputs_csiro_test.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_csiro_test/PMC_LLaMA_13B/rag_generation_outputs_csiro_test.PMC_LLaMA_13B.gen100.jsonl \
#   --model PMC_LLaMA_13B \
#   --base-url http://127.0.0.1:8001/v1 \
#   --tokenizer-name /ocean/projects/med230010p/yji3/models/PMC_LLaMA_13B \
#   --max-new-tokens 128 \
#   --context-window 2048 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 128 \
#   --skip-on-overflow


# # ---------- MedQuAD (train) ----------
# python gen_with_vllm.py \
#   --in-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_medquad_full/rag_generation_outputs_medquad_train.jsonl \
#   --out-jsonl /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/runs_medquad_full/PMC_LLaMA_13B/rag_generation_outputs_medquad_train.PMC_LLaMA_13B.gen100.jsonl \
#   --model PMC_LLaMA_13B \
#   --base-url http://127.0.0.1:8001/v1 \
#   --tokenizer-name /ocean/projects/med230010p/yji3/models/PMC_LLaMA_13B \
#   --max-new-tokens 128 \
#   --context-window 2048 \
#   --max-docs 8 \
#   --max-chars-per-doc 1200 \
#   --safety-margin 128




import os, json, argparse
from tqdm import tqdm
from openai import OpenAI
from typing import List, Dict, Any, Tuple, Optional
# --- helpers: context packing & prompt building (INSERT near imports) ---
def prompt_token_len(prompt: str) -> int:
    """Exact token count for a single prompt string (completions style)."""
    if TOK is None:
        return count_tokens_approx(prompt)
    try:
        return len(TOK(prompt, add_special_tokens=True).input_ids)
    except Exception as e:
        print(f"[WARN] prompt token count failed, fallback approx: {e}")
        return count_tokens_approx(prompt)


def trim_chunks_to_limit(
    question: str,
    chunks: List[str],
    hard_limit: int,
    max_new_tokens: int,
    safety_extra: int = 64,
) -> List[str]:
    """
    Ensure prompt token count <= hard_limit - max_new_tokens - safety_extra.
    Strategy:
      1) Drop chunks from the end until under limit.
      2) If still over, binary-search truncate on the last remaining chunk.
    """
    if hard_limit <= 0:
        return []

    def build_prompt(chs: List[str]) -> str:
        ctx_txt = "\n\n".join(f"- {c}" for c in chs)
        return PROMPT_TMPL.format(question=question, context=ctx_txt)

    budget = hard_limit - max_new_tokens - safety_extra
    if budget <= 0:
        return []

    kept = list(chunks)
    # Step 1: drop whole chunks from the end
    while kept:
        p = build_prompt(kept)
        t = prompt_token_len(p)
        if t <= budget:
            return kept
        kept.pop()  # drop last chunk

    # If no chunks left still exceed (very rare), return empty list or truncate question
    p = build_prompt([])
    if prompt_token_len(p) <= budget:
        return []
    # As a last resort: hard-cut the question (should not happen with reasonable budgets)
    q = question
    lo, hi, best = 0, len(q), ""
    while lo <= hi:
        mid = (lo + hi) // 2
        p = PROMPT_TMPL.format(question=q[:mid], context="")
        if prompt_token_len(p) <= budget:
            best = q[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return []  # keep empty chunks; question was cut above if needed

def _pack_context(ctx_items, max_docs=8, max_chars_per_doc=1200, max_chars_total=12000):
    # English-only comments: pack retrieved docs conservatively to avoid over-truncation
    kept, total = [], 0
    for i, d in enumerate(ctx_items[:max_docs], 1):
        t = (d.get("text") or "").strip()
        if not t:
            continue
        t = t[:max_chars_per_doc]
        if total + len(t) > max_chars_total:
            break
        kept.append(f"[Doc {i}]\n{t}")
        total += len(t)
    return "\n\n".join(kept)

def _build_rag_prompt(question, ctx_block):
    # English-only comments: conservative but not silent
    instr = (
        "You are a medical QA assistant. Prefer to answer using the CONTEXT.\n"
        "If evidence is partial, give the best-supported answer and note uncertainty.\n"
        "If there is no relevant evidence at all, say: I don't know."
    )
    return (
        f"{instr}\n\n"
        f"QUESTION:\n{(question or '').strip()}\n\n"
        f"CONTEXT:\n{ctx_block if ctx_block.strip() else '[No retrieved context]'}\n\n"
        "FINAL ANSWER:"
    )


# ---------------- Token counting helpers ----------------

def count_tokens_approx(text: str) -> int:
    """
    Very rough approximation: ~4 chars per token for English-like text.
    Used only as a fallback if the real tokenizer is unavailable.
    """
    return max(1, len(text) // 4)

def messages_token_len_approx(msgs: List[Dict[str, str]]) -> int:
    """Approximate token length for a list of chat messages."""
    return sum(count_tokens_approx(m.get("content", "")) for m in msgs)

# Will be set later if transformers tokenizer loads successfully.
TOK = None

# def token_len_exact(messages: List[Dict[str, str]]) -> int:
#     """
#     Exact token count using tokenizer's chat template if available.
#     Falls back to rough estimate when tokenizer is missing or fails.
#     """
#     global TOK
#     if TOK is None:
#         return messages_token_len_approx(messages)
#     try:
#         # apply_chat_template returns a list[int] if tokenize=True
#         input_ids = TOK.apply_chat_template(
#             messages,
#             add_generation_prompt=True,  # mimic vLLM ChatCompletions packing
#             tokenize=True,
#             return_tensors=None,
#         )
#         return len(input_ids)
#     except Exception as e:
#         print(f"[WARN] chat template failed, falling back to approx: {e}")
#         return messages_token_len_approx(messages)
# --- replace token_len_exact() with this version ---
# def token_len_exact(messages: List[Dict[str, str]]) -> int:
#     """
#     Exact token count.
#     - If the tokenizer has a chat_template: use apply_chat_template(tokenize=True).
#     - Otherwise (no chat template): build the final prompt string and tokenize it.
#     Falls back to rough estimate on any failure.
#     """
#     global TOK
#     if TOK is None:
#         return messages_token_len_approx(messages)
#     try:
#         # Case 1: chat template available -> chat-format counting
#         if getattr(TOK, "chat_template", None):
#             input_ids = TOK.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 tokenize=True,
#                 return_tensors=None,
#             )
#             return len(input_ids)
#         # Case 2: no chat template -> completions-style single prompt counting
#         prompt = messages[0].get("content", "") if messages else ""
#         return len(TOK(prompt, add_special_tokens=True).input_ids)
#     except Exception as e:
#         print(f"[WARN] exact token count failed, falling back to approx: {e}")
#         return messages_token_len_approx(messages)
def token_len_exact(messages: List[Dict[str, str]]) -> int:
    """Exact token count; chat-template vs. completions prompt."""
    global TOK
    if TOK is None:
        return messages_token_len_approx(messages)
    try:
        if getattr(TOK, "chat_template", None):
            input_ids = TOK.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
            return len(input_ids)
        # No chat template → single prompt counting
        prompt = messages[0].get("content", "") if messages else ""
        return prompt_token_len(prompt)
    except Exception as e:
        print(f"[WARN] exact token count failed, falling back to approx: {e}")
        return messages_token_len_approx(messages)

# ---------------- Prompt/template ----------------

PROMPT_TMPL = """You are a helpful medical assistant. Answer the user's question **faithfully** using the provided context only. If the answer is not in the context, say "I don't know" and do NOT hallucinate.

# Question
{question}

# Context
{context}

# Instructions
- Prefer concise, clinical language.
- Cite snippets implicitly (no numbered refs needed).
"""

def build_messages(q: str, chunks: List[str]) -> List[Dict[str, str]]:
    """
    Build a single-turn chat message with the given question and already-prepared chunks.
    """
    ctx_txt = "\n\n".join(f"- {c}" for c in chunks)
    content = PROMPT_TMPL.format(question=q, context=ctx_txt)
    return [{"role": "user", "content": content}]

# ---------------- Packing logic ----------------

def pack_chunks_to_budget_exact(
    question: str,
    raw_texts: List[str],
    context_window: int,
    max_new_tokens: int,
    safety_margin: int,
    max_docs: int,
    max_chars_per_doc: int,
) -> List[str]:
    """
    Greedy packing using exact token counting via the tokenizer chat template:
      1) Build base prompt (no context) and compute tokens.
      2) Pre-trim each chunk by chars.
      3) Greedily add chunks while staying within (window - max_new - margin).
      4) If the next chunk would overflow, binary-search a prefix of that chunk to fit.
    """
    # Base prompt without context
    base_prompt = PROMPT_TMPL.format(question=question, context="")
    base_msgs = [{"role": "user", "content": base_prompt}]
    base_tokens = token_len_exact(base_msgs)

    budget = context_window - max_new_tokens - safety_margin
    if budget <= base_tokens:
        return []

    # Pre-trim and take at most max_docs
    cand = [(t or "")[:max_chars_per_doc] for t in raw_texts[:max_docs]]

    kept: List[str] = []

    def msgs_with_chunks(chunks: List[str]) -> List[Dict[str, str]]:
        ctx = "\n\n".join(f"- {c}" for c in chunks)
        return [{"role": "user",
                 "content": PROMPT_TMPL.format(question=question, context=ctx)}]

    for ck in cand:
        trial = kept + [ck]
        tlen = token_len_exact(msgs_with_chunks(trial))
        if tlen <= budget:
            kept.append(ck)
            continue

        # Try partial fit of this chunk using binary search
        cur_len = token_len_exact(msgs_with_chunks(kept))
        remain = budget - cur_len
        if remain <= 0:
            break

        lo, hi, best = 0, len(ck), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            t2 = kept + [ck[:mid]]
            tl = token_len_exact(msgs_with_chunks(t2))
            if tl <= budget:
                best = ck[:mid]
                lo = mid + 1
            else:
                hi = mid - 1
        if best:
            kept.append(best)
        break

    return kept

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", required=True,
                    help="Input JSONL produced by make_rag_inputs_v3 (contains query/retrieved_context).")
    ap.add_argument("--out-jsonl", required=True,
                    help="Output JSONL with rag_response filled.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")  # vLLM-compatible servers often ignore this
    ap.add_argument("--model", default="Meditron3-8B",
                    help="The model name used when launching vLLM; usually matches the server's model id.")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only generate for the first N rows (for quick debugging).")

    # Generation length
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="Target completion length. May be reduced per-item if context is too long.")

    # Context / truncation controls
    ap.add_argument("--context-window", type=int, default=4096,
                    help="Model input token window (Meditron3-8B uses 4096).")
    ap.add_argument("--safety-margin", type=int, default=256,
                    help="Reserved tokens to avoid hitting the hard limit.")
    ap.add_argument("--max-docs", type=int, default=8,
                    help="Max number of retrieved chunks to include.")
    ap.add_argument("--max-chars-per-doc", type=int, default=1200,
                    help="Hard char cap per retrieved chunk before token-based packing.")

    # Tokenizer
    ap.add_argument("--tokenizer-name", type=str, default=None,
                    help="HF name or local path for the tokenizer matching your vLLM model. "
                         "If omitted, will try to use --model.")
    ap.add_argument("--skip-on-overflow", action="store_true",
                help="Skip an item if the packed prompt exceeds the current context budget.")


    args = ap.parse_args()

    # Build OpenAI-compatible client
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Try to load the tokenizer for exact token counting
    global TOK
    TOK = None
    try:
        from transformers import AutoTokenizer
        tok_name = args.tokenizer_name if args.tokenizer_name else args.model
        TOK = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True, use_fast=True)
        print(f"[INFO] Loaded tokenizer for token counting: {tok_name}")
    except Exception as e:
        print(f"[WARN] Failed to load tokenizer ({e}). Will use approximate counting.")

    n = 0
    with open(args.in_jsonl, "r", encoding="utf-8") as fin, \
         open(args.out_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Generating"):
            if args.limit is not None and n >= args.limit:
                break

            row = json.loads(line)
            question = row.get("query", "")

            retrieved = row.get("retrieved_context", []) or []
            raw_texts = [(c.get("text") or "").strip() for c in retrieved]

            # Step 1: pack chunks to fit (context_window - max_new_tokens - safety_margin)
            chunks = pack_chunks_to_budget_exact(
                question=question,
                raw_texts=raw_texts,
                context_window=args.context_window,
                max_new_tokens=args.max_new_tokens,
                safety_margin=args.safety_margin,
                max_docs=args.max_docs,
                max_chars_per_doc=args.max_chars_per_doc,
            )

            # Step 2: build messages and measure input length exactly
            msgs = build_messages(question, chunks)
            input_tokens = token_len_exact(msgs)

            # Compute per-item available completion budget
            # We subtract 1 extra token as an ultra-safe guard.
            available_for_completion = args.context_window - input_tokens - 1
            this_max_tokens = min(args.max_new_tokens, max(0, available_for_completion))

            # If no room left for completion, try dropping all context once
            if this_max_tokens <= 0:
                # Rebuild messages WITHOUT any context
                msgs = build_messages(question, [])
                input_tokens = token_len_exact(msgs)
                available_for_completion = args.context_window - input_tokens - 1
                this_max_tokens = min(args.max_new_tokens, max(1, available_for_completion))

            # Final guard: if still negative, clamp to 1
            if this_max_tokens <= 0:
                this_max_tokens = 1
            # --- overflow guard & optional skip (English-only comments) ---
            # Conservative hard limit (leave 1 token headroom)
            hard_limit = args.context_window - 1
            # Reserve half of safety-margin to buffer server-side discrepancies
            safety_extra = max(32, args.safety_margin // 2)
            target_budget = hard_limit - this_max_tokens - safety_extra
            
            # Count input tokens according to the path we will use
            use_chat = bool(getattr(TOK, "chat_template", None))
            if use_chat:
                cur_input_tokens = token_len_exact(msgs)  # chat-template counting
            else:
                prompt_preview = msgs[0]["content"] if msgs and "content" in msgs[0] else ""
                cur_input_tokens = prompt_token_len(prompt_preview)  # completions-style counting
            
            # If over budget, and user asked to skip, mark & skip this item
            if args.skip_on_overflow and cur_input_tokens > target_budget:
                print(f"[SKIP] overflow: input={cur_input_tokens} > budget={target_budget} (idx={n})")
                row["rag_response"] = ""
                row["_skipped_overflow"] = True
                row["_debug_input_tokens"] = cur_input_tokens
                row["_debug_target_budget"] = target_budget
                # Optionally also store how many chunks were kept before skipping
                row["_debug_ctx_chunks_before_skip"] = len(chunks)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                continue

            # # Create the completion call with per-item max_tokens
            # resp = client.chat.completions.create(
            #     model=args.model,
            #     messages=msgs,
            #     temperature=args.temperature,
            #     max_tokens=this_max_tokens,
            # )
            # text = resp.choices[0].message.content.strip() if resp.choices else ""
            # English-only comments: use Chat Completions if chat template is available; otherwise fall back to Completions.
            use_chat = bool(getattr(TOK, "chat_template", None))
            hard_limit = args.context_window - 1  # conservative headroom

            if not use_chat:
                # Rebuild prompt and trim if necessary
                prompt = msgs[0]["content"] if msgs and "content" in msgs[0] else ""
                ptoks = prompt_token_len(prompt)
                # We guarantee prompt tokens ≤ hard_limit - this_max_tokens - safety_extra
                safety_extra = max(32, args.safety_margin // 2)  # e.g., 64 when margin=128
                target_budget = hard_limit - this_max_tokens - safety_extra
            
                if ptoks > target_budget:
                    # Trim chunks to fit strictly
                    trimmed = trim_chunks_to_limit(
                        question=question,
                        chunks=chunks,
                        hard_limit=hard_limit,
                        max_new_tokens=this_max_tokens,
                        safety_extra=safety_extra,
                    )
                    # Rebuild messages and recompute exact counts
                    msgs = build_messages(question, trimmed)
                    prompt = msgs[0]["content"]
                    ptoks = prompt_token_len(prompt)
            
                    # If still too long (extremely rare), reduce max completion tokens
                    if ptoks > target_budget:
                        overflow = ptoks - target_budget
                        this_max_tokens = max(1, this_max_tokens - overflow)
            
                    # As final guard, clamp to 1 if negative
                    if this_max_tokens <= 0:
                        this_max_tokens = 1

            if use_chat:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=msgs,
                    temperature=args.temperature,
                    max_tokens=this_max_tokens,
                )
                text = resp.choices[0].message.content.strip() if resp.choices else ""
            else:
                # Build a single prompt string identical to what the user message contained
                prompt = msgs[0]["content"] if msgs and "content" in msgs[0] else ""
                resp = client.completions.create(
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=this_max_tokens,
                )
                text = resp.choices[0].text.strip() if resp.choices else ""

            row["rag_response"] = text
            # (Optional) record debug info for auditing
            # row["_debug_input_tokens"] = input_tokens
            # row["_debug_max_tokens_used"] = this_max_tokens
            # row["_debug_kept_ctx"] = len(chunks)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"✅ Wrote {n} rows → {args.out_jsonl}")

if __name__ == "__main__":
    main()
