#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Ground Truth NLI标签 + BioPortal KG验证

特点:
- 不需要运行student checker（快速，不需要GPU）
- 使用数据中已有的NLI标签
- 额外添加BioPortal KG验证

Usage:
    python eval_gt_with_kg.py \
        --results_path ./results_text.json \
        --kg_mode bioportal \
        --bioportal_key YOUR_API_KEY \
        --out_json ./output_gt_kg.json \
        --out_csv ./metrics_gt_kg.csv


(vllm07) [yji3@w005 BioNewK]$   /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_liveqa-4.1-Med42-Llama3-8B/liveqa_test.Med42-Llama3-8B.gen128__gpt-4.1/text_eval/results_text.json    --kg_mode bioportal    --bioportal_key $BIOPORTAL_API_KEY   --bioportal_ontologies SNOMEDCT,MESH,RXNORM,DOID  --bioportal_cache ./bioportal_cache  --out_json ./output_gt_kg.json  --out_csv ./metrics_gt_kg.csv 
bash: /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_liveqa-4.1-Med42-Llama3-8B/liveqa_test.Med42-Llama3-8B.gen128__gpt-4.1/text_eval/results_text.json: Permission denied
(vllm07) [yji3@w005 BioNewK]$   python eval_gt_with_kg.py    --results_path     /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_liveqa-4.1-Med42-Llama3-8B/liveqa_test.Med42-Llama3-8B.gen128__gpt-4.1/text_eval/results_text.json    --kg_mode bioportal    --bioportal_key $BIOPORTAL_API_KEY   --bioportal_ontologies SNOMEDCT,MESH,RXNORM,DOID  --bioportal_cache ./bioportal_cache  --out_json ./output_gt_kg.json  --out_csv ./metrics_gt_kg.csv 
加载数据: /ocean/projects/med230010p/yji3/MedicalRagChecker/medical_data/eval_liveqa-4.1-Med42-Llama3-8B/liveqa_test.Med42-Llama3-8B.gen128__gpt-4.1/text_eval/results_text.json
处理 50 个样本
初始化KG scorer (bioportal)...
No sentence-transformers model found with name cambridgeltl/SapBERT-from-PubMedBERT-fulltext. Creating a new one with mean pooling.
No sentence-transformers model found with name cambridgeltl/SapBERT-from-PubMedBERT-fulltext. Creating a new one with mean pooling.
[INFO] BioPortal KG scorer initialized with ontologies: ['SNOMEDCT', 'MESH', 'RXNORM', 'DOID']
处理样本: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [57:50<00:00, 69.41s/it]
计算指标...
保存结果: ./output_gt_kg.json
保存指标CSV: ./metrics_gt_kg.csv

================================================================================
评估摘要
================================================================================
{
  "overall_f1": 0.020817843866171006,
  "overall_precision": 0.23728813559322035,
  "overall_recall": 0.01088646967340591,
  "total_examples": 50,
  "total_claims": 354,
  "claim_recall": 0.01088646967340591,
  "context_precision": 0.285,
  "faithfulness": 0.23728813559322035,
  "hallucination": 0.0847457627118644,
  "context_utilization": 0.23728813559322035,
  "kg_consistency": 0.4608289924289283,
  "kg_coverage": 0.9971751412429378,
  "kg_claims_scored": 353,
  "safety_critical_errors": 30,
  "contradictions": 30
}
================================================================================
(vllm07) [yji3@w005 BioNewK]$ 
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 NLI + BioPortal KG 融合验证 (真正整合KG)

关键改进：
- KG score 被融合进 claim verification
- 使用 NLI + KG 联合判断 claim 是否被支持/矛盾

Usage:
    python eval_gt_with_kg_integrated.py \
        --results_path ./results_text.json \
        --kg_mode bioportal \
        --bioportal_key YOUR_API_KEY \
        --out_json ./output_integrated.json
"""

import json
import argparse
import csv
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

# 简单的进度指示
def tqdm(iterable, desc=""):
    items = list(iterable)
    total = len(items)
    for i, item in enumerate(items):
        if i % 10 == 0:
            print(f"\r{desc}: {i}/{total}", end="", flush=True)
        yield item
    print(f"\r{desc}: {total}/{total} done")

# 导入KG scorers
try:
    from bioportal_kg_scorer import BioPortalKGScorer
    HAS_BIOPORTAL = True
except ImportError:
    HAS_BIOPORTAL = False
    print("[WARN] bioportal_kg_scorer not found. KG scoring disabled.")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ClaimVerificationResult:
    """单个声明的验证结果"""
    claim_text: str

    # NLI验证
    nli_label: str = "neutral"
    p_entailed: float = 0.0
    p_neutral: float = 0.0
    p_contradicted: float = 0.0

    # KG验证
    kg_score: float = 0.0
    kg_subject_entity: Optional[str] = None
    kg_object_entity: Optional[str] = None
    kg_evidence: List[str] = None
    kg_status: str = "not_computed"

    # 融合后的最终判断
    final_label: str = "neutral"  # NLI + KG 融合后的标签
    final_confidence: float = 0.0

    # 声明三元组
    triple: Optional[Tuple[str, str, str]] = None
    best_passage_index: int = -1

    def __post_init__(self):
        if self.kg_evidence is None:
            self.kg_evidence = []


@dataclass
class RAGCheckerMetrics:
    """完整的RAGChecker指标"""
    # Overall Metrics
    overall_f1: float = 0.0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    total_examples: int = 0
    total_response_claims: int = 0
    total_gt_claims: int = 0

    # Retriever Metrics
    claim_recall: float = 0.0
    context_precision: float = 0.0

    # Generator Metrics (NLI only)
    faithfulness_nli: float = 0.0
    hallucination_nli: float = 0.0

    # Generator Metrics (NLI + KG 融合)
    faithfulness: float = 0.0        # 融合后
    hallucination: float = 0.0       # 融合后
    context_utilization: float = 0.0

    # KG metrics
    kg_consistency: float = 0.0
    kg_coverage: float = 0.0
    kg_claims_scored: int = 0

    # Safety
    safety_critical_errors: int = 0
    contradictions: int = 0


# ============================================================================
# Helper Functions
# ============================================================================

def normalize_label(label: str) -> str:
    """标准化标签名称"""
    label = (label or "").lower().strip()
    if label in ["entailment", "entailed", "entail", "supported", "yes"]:
        return "entailed"
    elif label in ["contradiction", "contradicted", "contradict", "refuted", "no"]:
        return "contradicted"
    else:
        return "neutral"


def extract_claims_from_example(ex: Dict) -> List[Any]:
    """从example中提取声明"""
    claims = []
    response_claims = ex.get("response_claims", [])
    if response_claims:
        for claim in response_claims:
            claims.append(claim)
    return claims


def extract_triple_from_claim(claim: Any) -> Optional[Tuple[str, str, str]]:
    """从声明中提取三元组"""
    if isinstance(claim, (list, tuple)) and len(claim) == 3:
        return tuple(str(x).strip() for x in claim)
    elif isinstance(claim, str):
        parts = claim.split("|")
        if len(parts) == 3:
            return tuple(p.strip() for p in parts)
    return None


def claim_to_text(claim: Any) -> str:
    """将声明转换为文本"""
    if isinstance(claim, list) and len(claim) == 3:
        return f"{claim[0]} {claim[1]} {claim[2]}"
    elif isinstance(claim, str):
        return claim
    else:
        return str(claim)


def extract_nli_labels_from_example(ex: Dict, key: str = "retrieved2response") -> List[List[str]]:
    """从example中提取NLI标签"""
    labels = []
    if key in ex and ex[key]:
        for claim_labels in ex[key]:
            if isinstance(claim_labels, list):
                normalized = [normalize_label(label) for label in claim_labels]
                labels.append(normalized)
    return labels


def aggregate_nli_label(labels: List[str]) -> Tuple[str, Dict[str, float]]:
    """聚合NLI标签 (RAGChecker 风格)"""
    if not labels:
        return "neutral", {"p_entailed": 0.0, "p_neutral": 1.0, "p_contradicted": 0.0}

    entailed_count = labels.count("entailed")
    contradicted_count = labels.count("contradicted")
    neutral_count = labels.count("neutral")
    total = len(labels)

    p_entailed = entailed_count / total
    p_contradicted = contradicted_count / total
    p_neutral = neutral_count / total

    if entailed_count > 0:
        label = "entailed"
    elif contradicted_count > 0:
        label = "contradicted"
    else:
        label = "neutral"

    return label, {
        "p_entailed": p_entailed,
        "p_neutral": p_neutral,
        "p_contradicted": p_contradicted
    }


def fuse_nli_kg(
    nli_label: str, 
    nli_probs: Dict[str, float],
    kg_score: float,
    kg_status: str,
    alpha: float = 0.3,           # KG 权重
    kg_threshold: float = 0.5     # KG 支持阈值
) -> Tuple[str, float]:
    """
    融合 NLI 和 KG 信号来判断 claim
    
    融合策略：
    1. 如果 KG 强烈支持 (score > threshold) + NLI entailed → 更自信的 entailed
    2. 如果 KG 不支持 (score < threshold) + NLI entailed → 降级为 neutral (可能是 hallucination)
    3. 如果 KG 不可用，退回到 NLI only
    
    Args:
        nli_label: NLI 聚合后的标签
        nli_probs: NLI 概率分布
        kg_score: KG 一致性得分 (0-1)
        kg_status: KG 状态
        alpha: KG 权重 (0-1)
        kg_threshold: KG 支持阈值
    
    Returns:
        (final_label, confidence)
    """
    
    # 如果 KG 不可用，退回到 NLI only
    if kg_status != "ok" or kg_score <= 0:
        if nli_label == "entailed":
            return "entailed", nli_probs["p_entailed"]
        elif nli_label == "contradicted":
            return "contradicted", nli_probs["p_contradicted"]
        else:
            return "neutral", nli_probs["p_neutral"]
    
    # KG 可用，进行融合
    kg_supports = kg_score >= kg_threshold
    
    if nli_label == "entailed":
        if kg_supports:
            # NLI 支持 + KG 支持 → 高置信度 entailed
            confidence = (1 - alpha) * nli_probs["p_entailed"] + alpha * kg_score
            return "entailed", confidence
        else:
            # NLI 支持但 KG 不支持 → 可能是 hallucination，降级为 neutral
            # 这是 KG 的主要贡献：识别 NLI 漏掉的 hallucination
            confidence = (1 - alpha) * nli_probs["p_entailed"] + alpha * (1 - kg_score)
            return "neutral", confidence  # 降级！
    
    elif nli_label == "contradicted":
        # 矛盾的情况保持不变
        confidence = (1 - alpha) * nli_probs["p_contradicted"] + alpha * (1 - kg_score)
        return "contradicted", confidence
    
    else:  # neutral
        if kg_supports:
            # NLI 中立但 KG 支持 → 可能被 NLI 漏掉，升级为 entailed
            confidence = alpha * kg_score
            return "entailed", confidence
        else:
            return "neutral", nli_probs["p_neutral"]


def is_chunk_relevant(chunk_idx: int, gt_claims_labels: List[List[str]]) -> bool:
    """判断 chunk 是否 relevant"""
    for claim_labels in gt_claims_labels:
        if chunk_idx < len(claim_labels):
            if claim_labels[chunk_idx] == "entailed":
                return True
    return False


# ============================================================================
# KG Scorer Wrapper
# ============================================================================

class KGScorerWrapper:
    """KG评分器封装"""

    def __init__(self, kg_mode: str, **kwargs):
        self.kg_mode = kg_mode
        self.scorer = None

        if kg_mode == "none":
            return

        if not HAS_BIOPORTAL:
            print("[WARN] BioPortal module not available. KG disabled.")
            return

        if kg_mode == "bioportal":
            if "bioportal_key" not in kwargs:
                raise ValueError("bioportal_key required for bioportal mode")

            self.scorer = BioPortalKGScorer(
                api_key=kwargs["bioportal_key"],
                ontologies=kwargs.get("ontologies", ["SNOMEDCT", "MESH", "RXNORM", "DOID"]),
                cache_dir=kwargs.get("cache_dir"),
                alpha=kwargs.get("alpha", 0.5)
            )
            print(f"[INFO] BioPortal KG scorer initialized")

    def score_claim(self, subject: str, relation: str, obj: str) -> Dict:
        """评分一个claim三元组"""
        if self.scorer is None:
            return {"status": "disabled", "score": 0.0}

        try:
            result = self.scorer.score_claim(subject, relation, obj)
            return {
                "status": result.status,
                "score": result.final_score,
                "subject_entity": result.subject_entity.full_id if result.subject_entity else None,
                "object_entity": result.object_entity.full_id if result.object_entity else None,
                "evidence": result.evidence
            }
        except Exception as e:
            print(f"[WARN] KG scoring error: {e}")
            return {"status": "error", "score": 0.0}


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(
    examples: List[Dict],
    all_response_claims: List[List[str]],
    all_response_results: List[List[ClaimVerificationResult]],
) -> RAGCheckerMetrics:
    """计算指标"""

    metrics = RAGCheckerMetrics()
    metrics.total_examples = len(examples)

    # NLI only 统计
    total_response_claims = 0
    nli_entailed = 0
    nli_contradicted = 0

    # NLI + KG 融合统计
    fused_entailed = 0
    fused_contradicted = 0
    fused_neutral = 0

    # GT claim 统计
    total_gt_claims = 0
    gt_claims_entailed = 0

    # Context precision
    total_chunks = 0
    relevant_chunks = 0

    # KG 统计
    kg_scored_claims = 0
    kg_total_score = 0.0

    for ex_idx, (ex, response_claims, response_results) in enumerate(
        zip(examples, all_response_claims, all_response_results)
    ):
        contexts = ex.get("retrieved_context", [])
        num_chunks = len(contexts)
        total_chunks += num_chunks

        # Generator Metrics
        for result in response_results:
            total_response_claims += 1

            # NLI only 统计
            if result.nli_label == "entailed":
                nli_entailed += 1
            elif result.nli_label == "contradicted":
                nli_contradicted += 1

            # 融合后统计
            if result.final_label == "entailed":
                fused_entailed += 1
            elif result.final_label == "contradicted":
                fused_contradicted += 1
            else:
                fused_neutral += 1

            # KG 统计
            if result.kg_status == "ok" and result.kg_score > 0:
                kg_scored_claims += 1
                kg_total_score += result.kg_score

        # Retriever Metrics
        gt_claims = ex.get("gt_answer_claims", [])
        gt_labels = extract_nli_labels_from_example(ex, "retrieved2answer")
        
        if gt_claims:
            total_gt_claims += len(gt_claims)
            for i, gc in enumerate(gt_claims):
                if i < len(gt_labels):
                    if "entailed" in gt_labels[i]:
                        gt_claims_entailed += 1

        for chunk_idx in range(num_chunks):
            if is_chunk_relevant(chunk_idx, gt_labels):
                relevant_chunks += 1

    # 计算最终指标
    metrics.total_response_claims = total_response_claims
    metrics.total_gt_claims = total_gt_claims

    if total_response_claims > 0:
        # NLI only
        metrics.faithfulness_nli = nli_entailed / total_response_claims
        metrics.hallucination_nli = nli_contradicted / total_response_claims

        # NLI + KG 融合
        metrics.faithfulness = fused_entailed / total_response_claims
        metrics.hallucination = fused_contradicted / total_response_claims
        metrics.context_utilization = fused_entailed / total_response_claims

    if total_gt_claims > 0:
        metrics.claim_recall = gt_claims_entailed / total_gt_claims

    if total_chunks > 0:
        metrics.context_precision = relevant_chunks / total_chunks

    # Overall
    metrics.overall_precision = metrics.faithfulness
    metrics.overall_recall = metrics.claim_recall
    if metrics.overall_precision + metrics.overall_recall > 0:
        metrics.overall_f1 = (
            2 * metrics.overall_precision * metrics.overall_recall / 
            (metrics.overall_precision + metrics.overall_recall)
        )

    # KG metrics
    metrics.kg_claims_scored = kg_scored_claims
    if total_response_claims > 0:
        metrics.kg_coverage = kg_scored_claims / total_response_claims
    if kg_scored_claims > 0:
        metrics.kg_consistency = kg_total_score / kg_scored_claims

    # Safety - 融合后的 contradicted + neutral (被 KG 降级的)
    metrics.contradictions = fused_contradicted
    # Safety critical = contradicted + 被 KG 降级的 (原本 NLI entailed 但 KG 不支持)
    downgraded = nli_entailed - fused_entailed  # 被 KG 降级的数量
    metrics.safety_critical_errors = fused_contradicted + max(0, downgraded)

    return metrics


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="使用 NLI + BioPortal KG 融合验证"
    )
    parser.add_argument("--results_path", required=True, help="输入JSON路径")
    parser.add_argument("--out_json", required=True, help="输出JSON路径")
    parser.add_argument("--out_csv", help="输出CSV路径")
    parser.add_argument("--max_examples", type=int, help="限制样本数量")

    # 融合参数
    parser.add_argument("--kg_alpha", type=float, default=0.3,
                       help="KG权重 (0-1)")
    parser.add_argument("--kg_threshold", type=float, default=0.5,
                       help="KG支持阈值")

    # KG options
    parser.add_argument("--kg_mode", choices=["none", "bioportal"], default="none",
                       help="KG评分模式")
    parser.add_argument("--bioportal_key", help="BioPortal API key")
    parser.add_argument("--bioportal_ontologies", default="SNOMEDCT,MESH,RXNORM,DOID,NCIT,MEDDRA",
                       help="逗号分隔的ontology列表")
    parser.add_argument("--bioportal_cache", default="./bioportal_cache",
                       help="BioPortal缓存目录")

    args = parser.parse_args()

    # 加载数据
    print(f"加载数据: {args.results_path}")
    with open(args.results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data.get("results", [])
    if args.max_examples:
        examples = examples[:args.max_examples]

    print(f"处理 {len(examples)} 个样本")

    # 初始化KG scorer
    kg_scorer = None
    if args.kg_mode != "none":
        print(f"初始化KG scorer ({args.kg_mode})...")
        ontologies = [x.strip() for x in args.bioportal_ontologies.split(",")]
        kg_scorer = KGScorerWrapper(
            kg_mode=args.kg_mode,
            bioportal_key=args.bioportal_key,
            ontologies=ontologies,
            cache_dir=Path(args.bioportal_cache)
        )

    # 处理每个样本
    all_response_claims = []
    all_response_results = []

    for ex in tqdm(examples, desc="处理样本"):
        claims_raw = extract_claims_from_example(ex)
        claims_text = [claim_to_text(c) for c in claims_raw]
        all_response_claims.append(claims_text)

        nli_labels = extract_nli_labels_from_example(ex, "retrieved2response")
        contexts = ex.get("retrieved_context", [])

        claim_results = []
        for i, (claim_raw, claim_text) in enumerate(zip(claims_raw, claims_text)):
            result = ClaimVerificationResult(claim_text=claim_text)

            # NLI 标签
            nli_probs = {"p_entailed": 0.0, "p_neutral": 1.0, "p_contradicted": 0.0}
            if i < len(nli_labels):
                claim_labels = nli_labels[i]
                nli_label, nli_probs = aggregate_nli_label(claim_labels)
                result.nli_label = nli_label
                result.p_entailed = nli_probs["p_entailed"]
                result.p_neutral = nli_probs["p_neutral"]
                result.p_contradicted = nli_probs["p_contradicted"]

            # KG 评分
            if kg_scorer and kg_scorer.scorer:
                triple = extract_triple_from_claim(claim_raw)
                if triple:
                    result.triple = triple
                    s, r, o = triple
                    kg_result = kg_scorer.score_claim(s, r, o)
                    result.kg_status = kg_result.get("status", "error")
                    result.kg_score = kg_result.get("score", 0.0)
                    result.kg_subject_entity = kg_result.get("subject_entity")
                    result.kg_object_entity = kg_result.get("object_entity")
                    result.kg_evidence = kg_result.get("evidence", [])

            # ============================================
            # 关键：融合 NLI + KG
            # ============================================
            final_label, final_confidence = fuse_nli_kg(
                nli_label=result.nli_label,
                nli_probs=nli_probs,
                kg_score=result.kg_score,
                kg_status=result.kg_status,
                alpha=args.kg_alpha,
                kg_threshold=args.kg_threshold
            )
            result.final_label = final_label
            result.final_confidence = final_confidence

            claim_results.append(result)

        all_response_results.append(claim_results)

    # 计算指标
    print("计算指标...")
    metrics = compute_metrics(examples, all_response_claims, all_response_results)

    # 保存结果
    output = {
        "config": {
            "input_file": args.results_path,
            "mode": "nli_kg_fusion",
            "kg_mode": args.kg_mode,
            "kg_alpha": args.kg_alpha,
            "kg_threshold": args.kg_threshold,
        },
        "metrics": asdict(metrics),
        "results": [
            {
                "query_id": ex.get("query_id", i),
                "query": ex.get("query", ""),
                "response": ex.get("response", ""),
                "claims": claims,
                "verification_results": [asdict(r) for r in results],
            }
            for i, (ex, claims, results) in enumerate(zip(examples, all_response_claims, all_response_results))
        ],
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"保存结果: {args.out_json}")

    # 打印摘要
    print("\n" + "=" * 80)
    print("评估摘要 (NLI + KG 融合)")
    print("=" * 80)
    print(f"\n{'指标':<30} {'NLI Only':<15} {'NLI + KG':<15} {'变化'}")
    print("-" * 80)
    
    faith_change = metrics.faithfulness - metrics.faithfulness_nli
    hallu_change = metrics.hallucination - metrics.hallucination_nli
    
    print(f"{'Faithfulness':<30} {metrics.faithfulness_nli*100:.1f}%{'':<10} {metrics.faithfulness*100:.1f}%{'':<10} {faith_change*100:+.1f}pp")
    print(f"{'Hallucination':<30} {metrics.hallucination_nli*100:.1f}%{'':<10} {metrics.hallucination*100:.1f}%{'':<10} {hallu_change*100:+.1f}pp")
    print(f"{'Claim Recall':<30} {metrics.claim_recall*100:.1f}%")
    print(f"{'Context Precision':<30} {metrics.context_precision*100:.1f}%")
    print(f"{'KG Coverage':<30} {metrics.kg_coverage*100:.1f}%")
    print(f"{'KG Consistency':<30} {metrics.kg_consistency*100:.1f}%")
    print(f"{'Safety Critical Errors':<30} {metrics.safety_critical_errors}")
    print("=" * 80)
    
    # 如果有 CSV 输出
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in asdict(metrics).items():
                writer.writerow([k, v])
        print(f"保存CSV: {args.out_csv}")


if __name__ == "__main__":
    main()