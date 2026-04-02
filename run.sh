#!/bin/bash
# Launch the Token Budget GRPO training pipeline.
#
# GPU layout for Qwen2.5-7B on V100-32GB:
#   GPU 0:   Ref server (~14GB)
#   GPU 1-4: Training (DeepSpeed ZeRO-2, 4 GPUs, ~14GB each)
#   GPU 6-7: Generation worker (vLLM TP=2, ~28GB total)
#
# Usage:
#   bash run.sh precompute   # one-time KV precomputation
#   bash run.sh ref_server   # start ref server (terminal 1)
#   bash run.sh train        # start training (terminal 2)
#   bash run.sh eval --checkpoint ./checkpoints/step_100
#   bash run.sh cot --checkpoint-dir ./checkpoints

set -e
cd "$(dirname "$0")"

case "${1:-}" in
    precompute)
        echo "Precomputing budget KV cache..."
        CUDA_VISIBLE_DEVICES=0 python budget_kv_injection.py --output ./budget_kv_cache
        ;;
    ref_server)
        echo "Starting reference model server on GPU 0..."
        CUDA_VISIBLE_DEVICES=0 python ref_server.py
        ;;
    train)
        echo "Starting GRPO training (GPUs 1-4 train, GPUs 6-7 gen)..."
        CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 \
        torchrun --nproc_per_node=4 --master_port=29500 train_grpo.py
        ;;
    eval)
        shift
        echo "Running diagnostic evaluation..."
        CUDA_VISIBLE_DEVICES=0 python evaluate.py "$@"
        ;;
    cot)
        shift
        echo "Running CoT evolution analysis..."
        CUDA_VISIBLE_DEVICES=0 python cot_analysis.py "$@"
        ;;
    *)
        echo "Usage: bash run.sh {precompute|ref_server|train|eval|cot}"
        exit 1
        ;;
esac
