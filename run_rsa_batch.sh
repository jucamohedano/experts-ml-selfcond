#!/bin/bash
# Run rsa_simple for all expert conditions vs full

BASE_DIR="results/word_features/Qwen3-30B-A3B-Instruct-2507_custom_60/Qwen3-30B-A3B-Instruct-2507_gpt2"
BRAIN_RDMS="/home/juancm/trento/abns/project/juans_fork/ml-selfcond/results/brain_rdm/Qwen3-30B-A3B-Instruct-2507_custom_60/Qwen3-8B-FP8_gpt2/default/2026-01-16_18-19-05/brain_rdms.pkl"
FULL_RDMS="${BASE_DIR}/full/2026-01-21_16-42-41/rdms"

# Expert conditions to compare against full
declare -A EXPERT_RDMS=(
    ["cot-ap-0.5"]="${BASE_DIR}/cot-ap-0.5/2026-01-08_17-50-11/rdms"
    ["cot-ap-0.6"]="${BASE_DIR}/cot-ap-0.6/2026-01-08_17-52-25/rdms"
    ["cot-ap-0.7"]="${BASE_DIR}/cot-ap-0.7/2026-01-09_15-11-56/rdms"
    ["cot-ap-0.8"]="${BASE_DIR}/cot-ap-0.8/2026-01-09_15-14-21/rdms"
    ["cot-ap-0.9"]="${BASE_DIR}/cot-ap-0.9/2026-01-09_15-25-59/rdms"
    ["cot-ap-0.5-unique-corr_0.8"]="${BASE_DIR}/cot-ap-0.5-unique-corr_0.8/2026-01-21_17-08-29/rdms"
    ["cot-ap-0.5-unique-corr_0.9"]="${BASE_DIR}/cot-ap-0.5-unique-corr_0.9/2026-01-21_17-06-09/rdms"
    ["cot-ap-0.6-unique-corr_0.8"]="${BASE_DIR}/cot-ap-0.6-unique-corr_0.8/2026-01-22_11-31-32/rdms"
    ["cot-ap-0.6-unique-corr_0.9"]="${BASE_DIR}/cot-ap-0.6-unique-corr_0.9/2026-01-22_11-29-14/rdms"
)

# Run for each expert condition
for condition in "${!EXPERT_RDMS[@]}"; do
    echo "=========================================="
    echo "Running: ${condition} vs full"
    echo "=========================================="
    
    python run_pipeline.py \
        task=rsa \
        model=Qwen3-30B-A3B-Instruct-2507_gpt2 \
        task.brain_rdms_path="${BRAIN_RDMS}" \
        task.expert_rdms_path="${EXPERT_RDMS[$condition]}" \
        task.full_rdms_path="${FULL_RDMS}" \
        task.layers="mlp.c_proj" \
        task.run_tag="${condition}_vs_full_mlp.c_proj"
    
    echo ""
done

echo "All comparisons complete!"
