#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的 Student Checker 和 Extractor 评估脚本

功能：
1. 评估 Student Checker 在测试集上的性能（accuracy, precision, recall, F1）
2. 评估 Student Extractor 在测试集上的性能（triple-level metrics）
3. 支持多个 checkpoint 批量评估
4. 生成详细的性能报告和对比表格

用法：
python eval_student_models.py \
  --checker_path ./runs/checker_sft_meditron \
  --extractor_path ./runs/extractor_sft_meditron3-8b \
  --eval_data ./data/eval_unified.jsonl \
  --output_dir ./eval_results

或批量评估：
python eval_student_models.py \
  --batch_eval \
  --runs_dir ./runs \
  --eval_data ./data/eval_unified.jsonl \
  --output_dir ./eval_results
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np


# ==================== Checker Evaluation ====================

LABELS = ["entailed", "contradicted", "neutral"]
LABEL_ALIASES = {
    "entailed": {"entailed", "entail", "entails", "supported", "yes"},
    "contradicted": {"contradicted", "contradict", "refuted", "no"},
    "neutral": {"neutral", "unknown", "insufficient", "not enough info",
                "not enough information", "uncertain"},
}

def normalize_label(text: str) -> Optional[str]:
    """规范化标签文本"""
    t = (text or "").lower().strip().replace(".", "").replace("label:", "").strip()

    # 精确匹配
    for lab, aliases in LABEL_ALIASES.items():
        if t in aliases:
            return lab

    # 模糊匹配
    if "contrad" in t or "refut" in t:
        return "contradicted"
    if "neutral" in t or "insufficient" in t or "not enough" in t or "uncertain" in t:
        return "neutral"
    if "entail" in t or "support" in t or t in {"yes", "y"}:
        return "entailed"

    return None


def format_checker_prompt(claim: str, evidence: str) -> str:
    """格式化 checker 输入 prompt"""
    prompt = (
        "Decide whether the EVIDENCE entails, contradicts, or is neutral to the CLAIM.\n"
        "Respond with one of: entailed | contradicted | neutral\n\n"
        f"CLAIM:\n{claim}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        "Label:"
    )
    return prompt


def evaluate_checker(
    model_path: str,
    eval_data_path: str,
    base_model_path: Optional[str] = None,
    device: str = "cuda",
    max_samples: int = -1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    评估 checker 模型

    Args:
        model_path: 模型路径（可以是完整模型或 LoRA adapter 目录）
        eval_data_path: 评估数据路径 (JSONL with prompt, label)
        base_model_path: Base model 路径（如果 model_path 只包含 LoRA adapter）
        device: 设备
        max_samples: 最大评估样本数（-1 表示全部）
        verbose: 是否显示详细信息

    Returns:
        评估指标字典
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Evaluating Checker: {model_path}")
        print(f"{'='*80}")

    # 检查是否是 LoRA adapter 目录
    from pathlib import Path
    model_path_obj = Path(model_path)
    is_lora_adapter = (model_path_obj / "adapter_config.json").exists()

    if is_lora_adapter:
        # LoRA adapter - 需要 base model
        if base_model_path is None:
            # 尝试从 adapter_config.json 读取 base model
            import json
            with open(model_path_obj / "adapter_config.json", 'r') as f:
                adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path")

        if base_model_path is None:
            raise ValueError(f"{model_path} 是 LoRA adapter 目录，但没有提供 base_model_path")

        if verbose:
            print(f"检测到 LoRA adapter，加载 base model: {base_model_path}")

        # 加载 base model 和 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # 加载 LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        if verbose:
            print(f"✅ 已加载 LoRA adapter from {model_path}")
    else:
        # 完整模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # 加载评估数据
    eval_data = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    if max_samples > 0:
        eval_data = eval_data[:max_samples]

    if verbose:
        print(f"Loaded {len(eval_data)} evaluation samples")

    # 评估
    y_true, y_pred = [], []
    correct = 0

    for item in tqdm(eval_data, desc="Evaluating", disable=not verbose):
        prompt = item['prompt']
        true_label = item['label']

        # 生成预测
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        pred_label = normalize_label(generated) or "neutral"

        y_true.append(true_label)
        y_pred.append(pred_label)

        if pred_label == true_label:
            correct += 1

    # 计算指标
    overall_acc = correct / len(y_true) if y_true else 0.0

    # Per-class metrics
    per_class_metrics = {}
    for label in LABELS:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        count = sum(1 for yt in y_true if yt == label)
        class_acc = tp / count if count > 0 else 0.0

        per_class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": class_acc,
            "support": count
        }

    # Macro average
    macro_precision = np.mean([m["precision"] for m in per_class_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])

    results = {
        "model_path": model_path,
        "overall_accuracy": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "num_samples": len(y_true)
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"CHECKER RESULTS")
        print(f"{'='*80}")
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Macro Precision:  {macro_precision:.4f}")
        print(f"Macro Recall:     {macro_recall:.4f}")
        print(f"Macro F1:         {macro_f1:.4f}")
        print(f"\nPer-class metrics:")
        for label in LABELS:
            m = per_class_metrics[label]
            print(f"  {label:15s}: P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"F1={m['f1']:.4f} Acc={m['accuracy']:.4f} (n={m['support']})")

    return results


# ==================== Extractor Evaluation ====================

def parse_qa_from_instruction(instr: str) -> Tuple[str, str]:
    """从 instruction 中解析 Question 和 Answer"""
    instr = instr.strip()
    m = re.search(r"Question:(.*)Answer:", instr, flags=re.S)
    if not m:
        m2 = re.search(r"Question:(.*)$", instr, flags=re.S)
        if m2:
            return m2.group(1).strip(), ""
        return instr, ""

    q = m.group(1).strip()
    rest = instr[m.end():].strip()
    return q, rest


def build_extractor_prompt(question: str, answer: str) -> str:
    """构建 extractor prompt"""
    system_part = (
        "You are an information extraction assistant. "
        "Given a medical question and its answer, extract ALL factual triples "
        "as [subject, relation, object]. "
        "Always copy entity names and key phrases EXACTLY from the question or answer; "
        "do NOT paraphrase biomedical terms, abbreviations, or disease names. "
        "Return a pure JSON array of triples, with no explanations, no extra text, "
        "no comments. If there are no clear factual triples, return an empty JSON array []."
    )

    qa_part = f"Question: {question.strip()}\nAnswer: {answer.strip()}"

    prompt = (
        system_part + "\n\n" + qa_part +
        "\n\nTriples (JSON only, e.g. [[\"subj\", \"rel\", \"obj\"], ...]):\n"
    )
    return prompt


def safe_load_triples(s: str) -> List[List[str]]:
    """安全解析 triples JSON"""
    if not s:
        return []

    s = s.replace("```json", "").replace("```", "").strip()

    candidates = [s]

    # 找所有 [...] 子串
    for m in re.finditer(r'\[', s):
        start = m.start()
        for end in range(len(s) - 1, start, -1):
            if s[end] == ']':
                cand = s[start:end+1].strip()
                if len(cand) >= 2:
                    candidates.append(cand)
                break

    # 去重
    seen = set()
    uniq_cands = []
    for c in sorted(candidates, key=len, reverse=True):
        if c not in seen:
            uniq_cands.append(c)
            seen.add(c)

    # 逐个尝试 parse
    for cand in uniq_cands:
        try:
            data = json.loads(cand)
        except:
            continue

        if not isinstance(data, list):
            continue

        triples = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                triples.append(item)
            elif isinstance(item, dict):
                subj = item.get("subject") or item.get("subj") or item.get("s")
                rel = item.get("relation") or item.get("predicate") or item.get("rel") or item.get("p")
                obj = item.get("object") or item.get("obj") or item.get("o")
                if subj is not None and rel is not None and obj is not None:
                    triples.append([subj, rel, obj])

        if triples:
            return triples

    return []


def triple_set(triples: List[List[Any]]) -> set:
    """将 triples 规范化为 set"""
    norm = []
    for t in triples:
        if not isinstance(t, (list, tuple)) or len(t) != 3:
            continue
        subj, rel, obj = t
        subj = str(subj).strip().lower()
        rel = str(rel).strip().lower()
        obj = str(obj).strip().lower()
        norm.append((subj, rel, obj))
    return set(norm)


def soft_match(t_pred: Tuple, t_gold: Tuple, min_overlap: float = 0.5) -> bool:
    """Soft matching between two triples"""
    sp, rp, op = t_pred
    sg, rg, og = t_gold

    # Relation 要求严格
    if rp != rg and (rp not in rg) and (rg not in rp):
        return False

    # Subject/Object 用 Jaccard
    def jaccard(a: str, b: str):
        sa = set(a.split())
        sb = set(b.split())
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union

    subj_ok = jaccard(sp, sg) >= min_overlap
    obj_ok = jaccard(op, og) >= min_overlap

    return subj_ok and obj_ok


def compute_triple_metrics(pred_triples: List, gold_triples: List) -> Tuple[float, float, float, float]:
    """计算 triple-level metrics"""
    p_list = list(triple_set(pred_triples))
    g_list = list(triple_set(gold_triples))

    if len(p_list) == 0 and len(g_list) == 0:
        return 1.0, 1.0, 1.0, 1.0
    if len(p_list) == 0:
        return 0.0, 0.0, 0.0, 0.0

    matched_g = set()
    tp = 0
    for tpred in p_list:
        for j, tgold in enumerate(g_list):
            if j in matched_g:
                continue
            if soft_match(tpred, tgold):
                tp += 1
                matched_g.add(j)
                break

    precision = tp / len(p_list)
    recall = tp / len(g_list) if g_list else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    exact_match = 1.0 if tp == len(p_list) == len(g_list) else 0.0

    return precision, recall, f1, exact_match


def evaluate_extractor(
    model_path: str,
    eval_data_path: str,
    base_model_path: Optional[str] = None,
    device: str = "cuda",
    max_samples: int = -1,
    max_new_tokens: int = 256,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    评估 extractor 模型

    Args:
        model_path: 模型路径（可以是完整模型或 LoRA adapter 目录）
        eval_data_path: 评估数据路径 (JSONL with instruction, output)
        base_model_path: Base model 路径（如果 model_path 只包含 LoRA adapter）
        device: 设备
        max_samples: 最大评估样本数
        max_new_tokens: 最大生成 token 数
        verbose: 是否显示详细信息

    Returns:
        评估指标字典
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Evaluating Extractor: {model_path}")
        print(f"{'='*80}")

    # 检查是否是 LoRA adapter 目录
    from pathlib import Path
    model_path_obj = Path(model_path)
    is_lora_adapter = (model_path_obj / "adapter_config.json").exists()

    if is_lora_adapter:
        # LoRA adapter - 需要 base model
        if base_model_path is None:
            # 尝试从 adapter_config.json 读取 base model
            import json
            with open(model_path_obj / "adapter_config.json", 'r') as f:
                adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path")

        if base_model_path is None:
            raise ValueError(f"{model_path} 是 LoRA adapter 目录，但没有提供 base_model_path")

        if verbose:
            print(f"检测到 LoRA adapter，加载 base model: {base_model_path}")

        # 加载 base model 和 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # 加载 LoRA adapter
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        if verbose:
            print(f"✅ 已加载 LoRA adapter from {model_path}")
    else:
        # 完整模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # 加载评估数据
    eval_data = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                eval_data.append(json.loads(line))

    if max_samples > 0:
        eval_data = eval_data[:max_samples]

    if verbose:
        print(f"Loaded {len(eval_data)} evaluation samples")

    # 评估
    all_p, all_r, all_f1, all_em = [], [], [], []

    for item in tqdm(eval_data, desc="Evaluating", disable=not verbose):
        instr = item['instruction']
        gold_text = item['output'].strip()

        q, a = parse_qa_from_instruction(instr)
        full_prompt = build_extractor_prompt(q, a)

        # 判断是否需要 chat format
        model_name_lower = model_path.lower()
        if "qwen2.5" in model_name_lower or "qwen2" in model_name_lower:
            messages = [
                {"role": "system", "content": "You are an information extraction assistant."},
                {"role": "user", "content": full_prompt},
            ]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = full_prompt

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # 提取 JSON
        if "Triples (JSON only" in gen_text:
            gen_text = gen_text.split("Triples (JSON only", 1)[-1]
        if "]" in gen_text and "[" in gen_text:
            try:
                start = gen_text.index("[")
                end = gen_text.rindex("]") + 1
                gen_text = gen_text[start:end]
            except:
                pass

        pred_triples = safe_load_triples(gen_text)
        gold_triples = safe_load_triples(gold_text)

        p, r, f1, em = compute_triple_metrics(pred_triples, gold_triples)
        all_p.append(p)
        all_r.append(r)
        all_f1.append(f1)
        all_em.append(em)

    # 汇总指标
    results = {
        "model_path": model_path,
        "precision": float(np.mean(all_p)),
        "recall": float(np.mean(all_r)),
        "f1": float(np.mean(all_f1)),
        "exact_match": float(np.mean(all_em)),
        "num_samples": len(all_p)
    }

    if verbose:
        print(f"\n{'='*80}")
        print(f"EXTRACTOR RESULTS")
        print(f"{'='*80}")
        print(f"Precision:    {results['precision']:.4f}")
        print(f"Recall:       {results['recall']:.4f}")
        print(f"F1 Score:     {results['f1']:.4f}")
        print(f"Exact Match:  {results['exact_match']:.4f}")

    return results


# ==================== Batch Evaluation ====================

def batch_evaluate(
    runs_dir: str,
    checker_eval_data: str,
    extractor_eval_data: str,
    output_dir: str,
    device: str = "cuda"
):
    """批量评估所有模型"""
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 找到所有 checker 和 extractor 模型
    checker_models = []
    extractor_models = []

    for model_dir in runs_path.iterdir():
        if not model_dir.is_dir():
            continue

        if "checker" in model_dir.name.lower():
            # 检查是否有 best 目录
            best_dirs = list(model_dir.glob("best_*"))
            if best_dirs:
                checker_models.extend(best_dirs)
            else:
                checker_models.append(model_dir)

        elif "extractor" in model_dir.name.lower():
            best_dirs = list(model_dir.glob("best_*"))
            if best_dirs:
                extractor_models.extend(best_dirs)
            else:
                extractor_models.append(model_dir)

    print(f"\nFound {len(checker_models)} checker models")
    print(f"Found {len(extractor_models)} extractor models")

    # 评估所有 checkers
    checker_results = []
    for model_path in checker_models:
        try:
            results = evaluate_checker(
                str(model_path),
                checker_eval_data,
                device=device,
                verbose=True
            )
            checker_results.append(results)
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")

    # 评估所有 extractors
    extractor_results = []
    for model_path in extractor_models:
        try:
            results = evaluate_extractor(
                str(model_path),
                extractor_eval_data,
                device=device,
                verbose=True
            )
            extractor_results.append(results)
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")

    # 保存结果
    with open(output_path / "checker_results.json", 'w') as f:
        json.dump(checker_results, f, indent=2)

    with open(output_path / "extractor_results.json", 'w') as f:
        json.dump(extractor_results, f, indent=2)

    # 生成报告
    generate_report(checker_results, extractor_results, output_path)


def generate_report(checker_results: List[Dict], extractor_results: List[Dict], output_dir: Path):
    """生成评估报告"""
    report_path = output_dir / "evaluation_report.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("Student Models Evaluation Report\n")
        f.write("=" * 100 + "\n\n")

        # Checker 结果
        f.write("CHECKER MODELS\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Model':<50} {'Accuracy':<12} {'Macro F1':<12} {'Contradicted F1':<15}\n")
        f.write("-" * 100 + "\n")

        for result in sorted(checker_results, key=lambda x: x['overall_accuracy'], reverse=True):
            model_name = Path(result['model_path']).name
            acc = result['overall_accuracy']
            f1 = result['macro_f1']
            cont_f1 = result['per_class']['contradicted']['f1']
            f.write(f"{model_name:<50} {acc:<12.4f} {f1:<12.4f} {cont_f1:<15.4f}\n")

        f.write("\n\n")

        # Extractor 结果
        f.write("EXTRACTOR MODELS\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Model':<50} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Exact Match':<12}\n")
        f.write("-" * 100 + "\n")

        for result in sorted(extractor_results, key=lambda x: x['f1'], reverse=True):
            model_name = Path(result['model_path']).name
            p = result['precision']
            r = result['recall']
            f1 = result['f1']
            em = result['exact_match']
            f.write(f"{model_name:<50} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {em:<12.4f}\n")

    print(f"\n✅ Report saved to: {report_path}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="评估 Student Checker 和 Extractor")

    # 单模型评估
    parser.add_argument("--checker_path", type=str, help="Checker 模型路径")
    parser.add_argument("--extractor_path", type=str, help="Extractor 模型路径")
    parser.add_argument("--base_model", type=str, help="Base model 路径（如果使用 LoRA adapter）")
    parser.add_argument("--checker_eval_data", type=str, default="./data/checker_sft.jsonl",
                       help="Checker 评估数据路径")
    parser.add_argument("--extractor_eval_data", type=str, default="./data/extractor_sft.jsonl",
                       help="Extractor 评估数据路径")

    # 批量评估
    parser.add_argument("--batch_eval", action="store_true", help="批量评估模式")
    parser.add_argument("--runs_dir", type=str, default="./runs", help="模型目录")

    # 通用参数
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--max_samples", type=int, default=-1, help="最大评估样本数")

    args = parser.parse_args()

    if args.batch_eval:
        # 批量评估
        batch_evaluate(
            args.runs_dir,
            args.checker_eval_data,
            args.extractor_eval_data,
            args.output_dir,
            args.device
        )
    else:
        # 单模型评估
        results = {}

        if args.checker_path:
            checker_results = evaluate_checker(
                args.checker_path,
                args.checker_eval_data,
                base_model_path=args.base_model,
                device=args.device,
                max_samples=args.max_samples,
                verbose=True
            )
            results['checker'] = checker_results

        if args.extractor_path:
            extractor_results = evaluate_extractor(
                args.extractor_path,
                args.extractor_eval_data,
                base_model_path=args.base_model,
                device=args.device,
                max_samples=args.max_samples,
                verbose=True
            )
            results['extractor'] = extractor_results

        # 保存结果
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
