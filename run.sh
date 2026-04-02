#!/bin/bash
# Token Budget GRPO Training Pipeline
#
# GPU layout (adjust for your cluster):
#   ref_server: 1 GPU
#   train:      N GPUs (DeepSpeed ZeRO-2)
#   gen worker:  1-2 GPUs (vLLM, spawned by train script)
#
# Usage:
#   bash run.sh precompute       # one-time KV cache precomputation
#   bash run.sh ref_server       # terminal 1
#   bash run.sh train            # terminal 2
#   bash run.sh eval <checkpoint>
#   bash run.sh cot <checkpoint_dir>

set -e
cd "$(dirname "$0")"

case "${1:-}" in
    precompute)
        echo "Precomputing budget KV cache..."
        CUDA_VISIBLE_DEVICES=0 python budget_kv_injection.py
        ;;
    ref_server)
        echo "Starting reference model server..."
        CUDA_VISIBLE_DEVICES=0 python ref_server.py
        ;;
    train)
        echo "Starting GRPO training..."
        # Adjust GPU list and nproc for your setup
        CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
        torchrun --nproc_per_node=4 --master_port=29500 train_grpo.py
        ;;
    train_hanoi)
        echo "Starting Hanoi GRPO training..."
        CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
        torchrun --nproc_per_node=4 --master_port=29500 train_hanoi.py
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
        echo "Usage: bash run.sh {precompute|ref_server|train|train_hanoi|eval|cot}"
        exit 1
        ;;
esac
