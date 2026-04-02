"""KV cache injection for real-time token budget awareness.

Precomputes KV cache entries for "Remaining token budget: N" at various
budget levels, then swaps the appropriate block into the KV cache at
each decode step. The model attends to this precomputed context naturally —
no architecture changes, no wasted tokens.

The budget text is placed at a fixed logical position in the sequence
(end of system prompt, before user content). At each decode step, the
block table entry for that position is rewritten to point to the
physical block matching the current budget.

Usage:
  1. Call precompute_budget_kv() once to generate and save KV blocks.
  2. During generation, use BudgetAwareLLM which wraps vLLM's LLM
     and swaps block table entries per step.

Bucketing: budget is bucketed by BUCKET_SIZE (default 50 tokens).
  Budget range [0, 5000] → 101 buckets → 101 precomputed KV blocks.
"""

import os
import math
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BUCKET_SIZE = 50
MAX_BUDGET = 5000


def get_budget_text(budget: int) -> str:
    """The text whose KV we precompute and inject."""
    return f"Remaining token budget: {budget}"


def budget_to_bucket(remaining: int) -> int:
    """Map remaining budget to bucket index."""
    clamped = max(0, min(remaining, MAX_BUDGET))
    return clamped // BUCKET_SIZE


def bucket_to_budget(bucket: int) -> int:
    """Map bucket index back to budget value."""
    return bucket * BUCKET_SIZE


def num_buckets() -> int:
    return MAX_BUDGET // BUCKET_SIZE + 1


# ── Precomputation ────────────────────────────────────

def precompute_budget_kv(
    model_path: str,
    save_path: str,
    device: str = "cuda",
) -> Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """Precompute KV cache entries for each budget bucket.

    For each bucket value (0, 50, 100, ..., 5000), runs the model forward
    on the text "Remaining token budget: {N}" and saves the per-layer
    key and value tensors.

    Args:
        model_path: path to the base model
        save_path: directory to save precomputed KV tensors

    Returns:
        Dict mapping bucket_index -> list of (key, value) per layer.
        Each key/value has shape (num_tokens, num_kv_heads, head_size).
    """
    print(f"Precomputing budget KV for {num_buckets()} buckets...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa").to(device)
    model.eval()

    all_kv = {}
    os.makedirs(save_path, exist_ok=True)

    for bucket_idx in range(num_buckets()):
        budget_val = bucket_to_budget(bucket_idx)
        text = get_budget_text(budget_val)
        input_ids = tokenizer(
            text, return_tensors="pt",
            add_special_tokens=False)["input_ids"].to(device)

        with torch.inference_mode():
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values

        # past_kv is a tuple of (key, value) per layer
        # key shape: (batch=1, num_kv_heads, seq_len, head_size)
        # Squeeze batch dim and transpose to (seq_len, num_kv_heads, head_size)
        layer_kv = []
        for layer_idx, (k, v) in enumerate(past_kv):
            k = k.squeeze(0).transpose(0, 1).contiguous().cpu()  # (seq, heads, dim)
            v = v.squeeze(0).transpose(0, 1).contiguous().cpu()
            layer_kv.append((k, v))

        all_kv[bucket_idx] = layer_kv

        if bucket_idx % 20 == 0:
            print(f"  Bucket {bucket_idx}/{num_buckets()-1} "
                  f"(budget={budget_val}): {input_ids.shape[1]} tokens, "
                  f"KV shape per layer: {layer_kv[0][0].shape}")

    # Save
    torch.save(all_kv, os.path.join(save_path, "budget_kv.pt"))
    # Also save metadata
    meta = {
        "bucket_size": BUCKET_SIZE,
        "max_budget": MAX_BUDGET,
        "num_buckets": num_buckets(),
        "model_path": model_path,
        "num_layers": len(all_kv[0]),
        "tokens_per_bucket": [
            len(tokenizer.encode(get_budget_text(bucket_to_budget(i)),
                                 add_special_tokens=False))
            for i in range(num_buckets())
        ],
    }
    torch.save(meta, os.path.join(save_path, "budget_kv_meta.pt"))
    print(f"Saved to {save_path}/budget_kv.pt")
    print(f"Metadata: {meta['num_layers']} layers, "
          f"{meta['tokens_per_bucket'][0]} tokens per budget text")

    del model
    torch.cuda.empty_cache()
    return all_kv


def load_budget_kv(save_path: str, device: str = "cuda"):
    """Load precomputed budget KV and metadata."""
    all_kv = torch.load(
        os.path.join(save_path, "budget_kv.pt"),
        map_location=device, weights_only=True)
    meta = torch.load(
        os.path.join(save_path, "budget_kv_meta.pt"),
        weights_only=True)
    return all_kv, meta


# ── vLLM Block Table Manipulation ─────────────────────

class BudgetBlockManager:
    """Manages precomputed budget KV blocks within vLLM's KV cache.

    On initialization:
      1. Loads precomputed KV for each budget bucket.
      2. Allocates physical blocks in vLLM's KV cache.
      3. Copies precomputed KV into those physical blocks.

    Per decode step:
      1. Compute current budget bucket.
      2. Rewrite the block table entry for the budget position
         to point to the physical block for that bucket.
    """

    def __init__(
        self,
        precomputed_kv: dict,
        meta: dict,
        kv_caches: List[torch.Tensor],
        block_size: int = 16,
    ):
        """
        Args:
            precomputed_kv: bucket_idx -> list of (key, value) per layer
            meta: metadata dict from precomputation
            kv_caches: vLLM's KV cache tensors, one per layer.
                       Shape: (2, num_blocks, block_size, num_kv_heads, head_size)
            block_size: vLLM block size (typically 16)
        """
        self.block_size = block_size
        self.meta = meta
        self.num_layers = meta["num_layers"]

        # How many tokens does the budget text occupy?
        # They might vary slightly ("budget: 0" vs "budget: 5000")
        # but we need a fixed block allocation, so use the max.
        self.budget_token_counts = meta["tokens_per_bucket"]
        self.max_budget_tokens = max(self.budget_token_counts)
        self.num_blocks_needed = math.ceil(self.max_budget_tokens / block_size)

        # Allocate physical blocks for each bucket's KV
        # We need num_buckets * num_blocks_needed physical blocks
        n_buckets = meta["num_buckets"]
        total_blocks_needed = n_buckets * self.num_blocks_needed

        # Find free blocks in the KV cache
        # We'll use blocks from the end of the cache to minimize conflicts
        total_available = kv_caches[0].shape[1]  # num_blocks dimension
        start_block = total_available - total_blocks_needed
        if start_block < 0:
            raise ValueError(
                f"Need {total_blocks_needed} blocks for budget KV but "
                f"only {total_available} available. Reduce num_buckets or "
                f"increase gpu_memory_utilization.")

        print(f"BudgetBlockManager: allocating blocks {start_block}-{total_available-1} "
              f"({total_blocks_needed} blocks for {n_buckets} buckets)")

        # Map: bucket_idx -> list of physical block IDs
        self.bucket_to_blocks: Dict[int, List[int]] = {}
        block_cursor = start_block

        for bucket_idx in range(n_buckets):
            block_ids = list(range(block_cursor, block_cursor + self.num_blocks_needed))
            self.bucket_to_blocks[bucket_idx] = block_ids
            block_cursor += self.num_blocks_needed

            # Copy precomputed KV into these physical blocks
            layer_kv = precomputed_kv[bucket_idx]
            n_tokens = self.budget_token_counts[bucket_idx]

            for layer_idx, (k, v) in enumerate(layer_kv):
                # k, v shape: (n_tokens, num_kv_heads, head_size)
                k_gpu = k.to(kv_caches[layer_idx].device)
                v_gpu = v.to(kv_caches[layer_idx].device)

                for tok_i in range(n_tokens):
                    phys_block = block_ids[tok_i // block_size]
                    offset = tok_i % block_size
                    # kv_cache shape: (2, num_blocks, block_size, num_kv_heads, head_size)
                    kv_caches[layer_idx][0, phys_block, offset] = k_gpu[tok_i]
                    kv_caches[layer_idx][1, phys_block, offset] = v_gpu[tok_i]

            # Zero-pad remaining slots in the last block
            remaining = self.num_blocks_needed * block_size - n_tokens
            if remaining > 0:
                last_block = block_ids[-1]
                start_offset = n_tokens % block_size
                for layer_idx in range(self.num_layers):
                    kv_caches[layer_idx][0, last_block, start_offset:] = 0
                    kv_caches[layer_idx][1, last_block, start_offset:] = 0

        self.reserved_start = start_block
        print(f"BudgetBlockManager: initialized, "
              f"{self.max_budget_tokens} tokens/bucket, "
              f"{self.num_blocks_needed} blocks/bucket")

    def get_block_ids_for_budget(self, remaining_budget: int) -> List[int]:
        """Get the physical block IDs for the given budget level."""
        bucket = budget_to_bucket(remaining_budget)
        return self.bucket_to_blocks[bucket]

    def swap_budget_blocks(
        self,
        block_table_np,
        req_idx: int,
        budget_block_start: int,
        remaining_budget: int,
    ):
        """Rewrite block table entries to point to the correct budget blocks.

        Args:
            block_table_np: numpy view of the block table (max_reqs, max_blocks)
            req_idx: which request row to modify
            budget_block_start: logical block index where budget blocks start
                               in this request's block table
            remaining_budget: current remaining token budget
        """
        new_block_ids = self.get_block_ids_for_budget(remaining_budget)
        for i, blk_id in enumerate(new_block_ids):
            block_table_np[req_idx, budget_block_start + i] = blk_id


# ── Standalone precomputation script ──────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--output", default="./budget_kv_cache",
                        help="Output directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from config import model_path as default_model
    model = args.model or default_model

    precompute_budget_kv(model, args.output, args.device)
