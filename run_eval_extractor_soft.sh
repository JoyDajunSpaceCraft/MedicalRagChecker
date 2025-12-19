#!/bin/bash
set -e
# How to run 
# chmod +x run_eval_extractor_soft.sh
# ./run_eval_extractor_soft.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export WANDB_DISABLED=true   # eval 不需要 wandb 的话可以关掉

DATA=./data/extractor_sft.jsonl
RESULT_DIR=./results
mkdir -p "${RESULT_DIR}"

########################################
# 1) Meditron3-8B + extractor_sft_meditron3-8b
########################################
BASE=/ocean/projects/med230010p/yji3/models/Meditron3-8B
LORA=./runs/extractor_sft_meditron3-8b
python DistillExtractor/eval_extractor_soft.py \
  --base_model "$BASE" \
  --lora_path "$LORA" \
  --data_path "$DATA" \
  --output_json "${RESULT_DIR}/extractor_soft_meditron3-8b.json" \
  --max_samples 200 \
  --bf16

########################################
# 2) med42-llama3-8b + extractor_sft_med42-llama3-8b
########################################
BASE=/ocean/projects/med230010p/yji3/models/med42-llama3-8b
LORA=./runs/extractor_sft_med42-llama3-8b
python DistillExtractor/eval_extractor_soft.py \
  --base_model "$BASE" \
  --lora_path "$LORA" \
  --data_path "$DATA" \
  --output_json "${RESULT_DIR}/extractor_soft_med42-llama3-8b.json" \
  --max_samples 200 \
  --bf16

########################################
# 3) qwen2-med-7b + extractor_sft_qwen2-med-7b
########################################
BASE=/ocean/projects/med230010p/yji3/models/qwen2-med-7b
LORA=./runs/extractor_sft_qwen2-med-7b
python DistillExtractor/eval_extractor_soft.py \
  --base_model "$BASE" \
  --lora_path "$LORA" \
  --data_path "$DATA" \
  --output_json "${RESULT_DIR}/extractor_soft_qwen2-med-7b.json" \
  --max_samples 200 \
  --bf16

########################################
# 4) PMC_LLaMA_13B + extractor_sft_PMC_LLaMA_13B
########################################
BASE=/ocean/projects/med230010p/yji3/models/PMC_LLaMA_13B
LORA=./runs/extractor_sft_PMC_LLaMA_13B
python DistillExtractor/eval_extractor_soft.py \
  --base_model "$BASE" \
  --lora_path "$LORA" \
  --data_path "$DATA" \
  --output_json "${RESULT_DIR}/extractor_soft_PMC_LLaMA_13B.json" \
  --max_samples 200 \
  --bf16
