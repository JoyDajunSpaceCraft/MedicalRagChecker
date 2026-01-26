#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速平衡 Checker 训练数据

问题：contradicted 类只有 467 样本 (1.7%)，导致模型无法学好这个类别
解决：通过 over-sampling 将 contradicted 增加到 ~10% 的比例

用法：
python balance_checker_data.py \
  --input ./data/checker_sft.jsonl \
  --output ./data/checker_sft_balanced.jsonl \
  --target_ratio 0.10
"""

import json
import random
import argparse
from collections import Counter
from pathlib import Path


def load_data(jsonl_path):
    """加载 JSONL 数据"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def balance_data(data, target_ratio=0.10, method='oversample'):
    """
    平衡数据集

    Args:
        data: 原始数据列表
        target_ratio: contradicted 类的目标比例 (default: 0.10 = 10%)
        method: 'oversample' 或 'weighted'

    Returns:
        balanced_data: 平衡后的数据
    """
    # 分离不同类别
    contradicted = [d for d in data if d['label'] == 'contradicted']
    entailed = [d for d in data if d['label'] == 'entailed']
    neutral = [d for d in data if d['label'] == 'neutral']

    print(f"\n{'='*80}")
    print("原始数据分布:")
    print(f"{'='*80}")
    print(f"  contradicted: {len(contradicted):6,} ({len(contradicted)/len(data)*100:5.1f}%)")
    print(f"  entailed:     {len(entailed):6,} ({len(entailed)/len(data)*100:5.1f}%)")
    print(f"  neutral:      {len(neutral):6,} ({len(neutral)/len(data)*100:5.1f}%)")
    print(f"  Total:        {len(data):6,}")

    if method == 'oversample':
        # 计算需要多少 contradicted 样本才能达到目标比例
        # 设 contradicted 数量为 x
        # x / (x + len(entailed) + len(neutral)) = target_ratio
        # x = target_ratio * (x + len(entailed) + len(neutral))
        # x = target_ratio * x + target_ratio * (len(entailed) + len(neutral))
        # x * (1 - target_ratio) = target_ratio * (len(entailed) + len(neutral))
        # x = target_ratio * (len(entailed) + len(neutral)) / (1 - target_ratio)

        other_count = len(entailed) + len(neutral)
        target_contradicted = int(target_ratio * other_count / (1 - target_ratio))

        # 如果原始 contradicted 样本不够，进行 over-sampling
        if len(contradicted) < target_contradicted:
            # 计算需要复制多少倍
            multiplier = target_contradicted // len(contradicted) + 1
            contradicted_oversampled = contradicted * multiplier
            contradicted_oversampled = contradicted_oversampled[:target_contradicted]

            print(f"\n{'='*80}")
            print("Over-sampling:")
            print(f"{'='*80}")
            print(f"  目标 contradicted 数量: {target_contradicted:,}")
            print(f"  复制倍数: {multiplier}x")
            print(f"  实际 contradicted 数量: {len(contradicted_oversampled):,}")
        else:
            contradicted_oversampled = contradicted

        # 合并所有数据
        balanced = entailed + contradicted_oversampled + neutral
        random.shuffle(balanced)

    else:
        raise ValueError(f"Unknown method: {method}")

    # 统计平衡后的分布
    label_counts = Counter(d['label'] for d in balanced)

    print(f"\n{'='*80}")
    print("平衡后数据分布:")
    print(f"{'='*80}")
    for label in ['contradicted', 'entailed', 'neutral']:
        count = label_counts[label]
        pct = count / len(balanced) * 100
        print(f"  {label:15s}: {count:6,} ({pct:5.1f}%)")
    print(f"  Total:          {len(balanced):6,}")

    return balanced


def save_data(data, output_path):
    """保存数据到 JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="平衡 Checker 训练数据")
    parser.add_argument('--input', type=str, default='./data/checker_sft.jsonl',
                       help='输入 JSONL 文件路径')
    parser.add_argument('--output', type=str, default='./data/checker_sft_balanced.jsonl',
                       help='输出 JSONL 文件路径')
    parser.add_argument('--target_ratio', type=float, default=0.10,
                       help='contradicted 类的目标比例 (0.10 = 10%%)')
    parser.add_argument('--method', type=str, default='oversample',
                       choices=['oversample'],
                       help='平衡方法')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    # 加载数据
    print(f"\n加载数据: {args.input}")
    data = load_data(args.input)

    # 平衡数据
    balanced_data = balance_data(data, target_ratio=args.target_ratio, method=args.method)

    # 保存数据
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data(balanced_data, args.output)

    print(f"\n{'='*80}")
    print(f"✅ 已保存平衡后的数据到: {args.output}")
    print(f"{'='*80}")
    print(f"\n下一步：使用平衡后的数据重新训练 checker:")
    print(f"")
    print(f"python DistillChecker/train_checker_sft.py \\")
    print(f"  --model_name /path/to/Meditron3-8B \\")
    print(f"  --train_path {args.output} \\")
    print(f"  --output_dir ./runs/checker_sft_balanced \\")
    print(f"  --lr 1e-4 \\")
    print(f"  --epochs 4 \\")
    print(f"  --lora_r 16 \\")
    print(f"  --lora_alpha 32 \\")
    print(f"  --bf16")
    print()


if __name__ == '__main__':
    main()
