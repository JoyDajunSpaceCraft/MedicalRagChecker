#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation script for all student models across all datasets.
Automatically discovers trained models and results files and runs evaluation.

Usage:
    python batch_eval_students.py \
        --root /Users/yuelyu/Downloads/MedicalRagChecker \
        --base_models_config ./base_models.json \
        --output_dir ./runs/student_eval
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def load_base_models_config(config_path: Path) -> Dict[str, str]:
    """
    Load base model mapping from JSON file.
    Format: {"extractor_sft_meditron3-8b": "/path/to/Meditron3-8B", ...}
    """
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        return {}
    with open(config_path) as f:
        return json.load(f)


def infer_base_model(adapter_name: str, base_models_config: Dict[str, str]) -> str:
    """Infer base model from adapter name."""
    # Direct lookup
    if adapter_name in base_models_config:
        return base_models_config[adapter_name]

    # Pattern matching
    name_lower = adapter_name.lower()
    if "meditron3" in name_lower or "meditron-8b" in name_lower:
        key = "Meditron3-8B"
    elif "med42" in name_lower:
        key = "med42-llama3-8b"
    elif "qwen2-med" in name_lower:
        key = "qwen2-med-7b"
    elif "pmc_llama" in name_lower or "pmc-llama" in name_lower:
        key = "PMC_LLaMA_13B"
    elif "qwen2.5-7b" in name_lower:
        key = "Qwen2.5-7B-Instruct"
    elif "qwen2.5-32b" in name_lower:
        key = "Qwen2.5-32B-Instruct"
    else:
        return None

    # Look for this key in config
    for k, v in base_models_config.items():
        if key.lower() in k.lower():
            return v
    return None


def find_extractors(runs_dir: Path) -> List[Path]:
    """Find all extractor checkpoints in runs directory."""
    extractors = []
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.startswith("extractor_"):
            extractors.append(d)
    return sorted(extractors)


def find_checkers(runs_dir: Path) -> List[Path]:
    """Find all checker checkpoints in runs directory."""
    checkers = []
    for d in runs_dir.iterdir():
        if d.is_dir() and d.name.startswith("checker_"):
            checkers.append(d)
    return sorted(checkers)


def find_results_files(medical_data_dir: Path) -> List[Path]:
    """Find all results_text*.json files."""
    results = []
    for root, dirs, files in os.walk(medical_data_dir):
        for f in files:
            if f.startswith("results_text") and f.endswith(".json"):
                if ".ipynb_checkpoints" not in root:
                    results.append(Path(root) / f)
    return sorted(results)


def parse_dataset_from_path(results_path: Path) -> str:
    """Extract dataset name from results path."""
    # Look for eval_* directory
    for parent in results_path.parents:
        if parent.name.startswith("eval_"):
            return parent.name.replace("eval_", "")
    return "unknown"


def run_evaluation(
    results_path: Path,
    extractor_dir: Path,
    checker_dir: Path,
    base_model_extractor: str,
    base_model_checker: str,
    output_dir: Path,
    max_examples: int = None,
) -> Tuple[bool, str]:
    """Run single evaluation and return success status."""
    dataset = parse_dataset_from_path(results_path)
    extractor_name = extractor_dir.name
    checker_name = checker_dir.name

    output_name = f"{dataset}__{extractor_name}__{checker_name}"
    out_json = output_dir / f"{output_name}.json"
    out_csv = output_dir / f"{output_name}.csv"

    if out_json.exists():
        print(f"[SKIP] Already evaluated: {output_name}")
        return True, "Already exists"

    print(f"\n{'='*80}")
    print(f"Evaluating: {output_name}")
    print(f"  Results: {results_path}")
    print(f"  Extractor: {extractor_dir.name}")
    print(f"  Checker: {checker_dir.name}")
    print(f"{'='*80}\n")

    cmd = [
        "python", "eval_student_end2end.py",
        "--results_path", str(results_path),
        "--extractor_dir", str(extractor_dir),
        "--checker_dir", str(checker_dir),
        "--base_model_extractor", base_model_extractor,
        "--base_model_checker", base_model_checker,
        "--out_json", str(out_json),
        "--out_csv", str(out_csv),
    ]

    if max_examples:
        cmd.extend(["--max_examples", str(max_examples)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluation failed: {e}")
        print(e.stderr)
        return False, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="MedicalRagChecker root directory")
    ap.add_argument("--base_models_config", type=Path, required=True, help="JSON file mapping adapter names to base models")
    ap.add_argument("--output_dir", type=Path, required=True, help="Output directory for evaluation results")
    ap.add_argument("--max_examples", type=int, default=None, help="Limit evaluation to N examples per dataset")
    ap.add_argument("--specific_extractor", type=str, default=None, help="Evaluate only this extractor")
    ap.add_argument("--specific_checker", type=str, default=None, help="Evaluate only this checker")
    ap.add_argument("--specific_dataset", type=str, default=None, help="Evaluate only this dataset (substring match)")
    args = ap.parse_args()

    # Setup paths
    runs_dir = args.root / "runs"
    medical_data_dir = args.root / "medical_data"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load base models config
    base_models_config = load_base_models_config(args.base_models_config)
    print(f"Loaded {len(base_models_config)} base model mappings")

    # Discover models and results
    extractors = find_extractors(runs_dir)
    checkers = find_checkers(runs_dir)
    results_files = find_results_files(medical_data_dir)

    print(f"\nFound:")
    print(f"  - {len(extractors)} extractor checkpoints")
    print(f"  - {len(checkers)} checker checkpoints")
    print(f"  - {len(results_files)} results files")

    # Filter if specific models requested
    if args.specific_extractor:
        extractors = [e for e in extractors if args.specific_extractor in e.name]
        print(f"  - Filtered to {len(extractors)} extractors matching '{args.specific_extractor}'")

    if args.specific_checker:
        checkers = [c for c in checkers if args.specific_checker in c.name]
        print(f"  - Filtered to {len(checkers)} checkers matching '{args.specific_checker}'")

    if args.specific_dataset:
        results_files = [r for r in results_files if args.specific_dataset in parse_dataset_from_path(r)]
        print(f"  - Filtered to {len(results_files)} results matching '{args.specific_dataset}'")

    # Run all combinations
    total_evals = len(extractors) * len(checkers) * len(results_files)
    print(f"\nWill run {total_evals} evaluations")

    results_log = []
    success_count = 0

    for extractor_dir in extractors:
        base_model_ext = infer_base_model(extractor_dir.name, base_models_config)
        if not base_model_ext:
            print(f"[SKIP] Cannot infer base model for extractor: {extractor_dir.name}")
            continue

        for checker_dir in checkers:
            base_model_ck = infer_base_model(checker_dir.name, base_models_config)
            if not base_model_ck:
                print(f"[SKIP] Cannot infer base model for checker: {checker_dir.name}")
                continue

            for results_path in results_files:
                success, msg = run_evaluation(
                    results_path=results_path,
                    extractor_dir=extractor_dir,
                    checker_dir=checker_dir,
                    base_model_extractor=base_model_ext,
                    base_model_checker=base_model_ck,
                    output_dir=args.output_dir,
                    max_examples=args.max_examples,
                )

                results_log.append({
                    "dataset": parse_dataset_from_path(results_path),
                    "extractor": extractor_dir.name,
                    "checker": checker_dir.name,
                    "success": success,
                    "message": msg,
                })

                if success:
                    success_count += 1

    # Save log
    log_path = args.output_dir / "evaluation_log.json"
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"  Success: {success_count} / {total_evals}")
    print(f"  Log saved to: {log_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
