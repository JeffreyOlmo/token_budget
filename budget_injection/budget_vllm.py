"""vLLM wrapper that swaps budget KV blocks at each decode step.

This patches vLLM's GPUModelRunner.execute_model() to rewrite the block
table between _update_states() and _prepare_inputs(), swapping in the
correct precomputed budget KV block based on remaining tokens.

Usage:
    from budget_vllm import BudgetAwareLLM

    llm = BudgetAwareLLM(model_path, budget_kv_path="./budget_kv_cache")

    # Generate with budget tracking
    llm.set_budget(request_id, initial_budget=800)
    outputs = llm.generate(prompts, sampling_params)
    # Budget decrements automatically per generated token
"""

import os
from typing import Dict, List, Optional

import torch
import numpy as np

from budget_injection.budget_kv_injection import (
    BudgetBlockManager, budget_to_bucket, load_budget_kv,
)


class BudgetTracker:
    """Tracks per-request token budgets during generation."""

    def __init__(self):
        # Maps vLLM request index (row in batch) to budget state
        self.budgets: Dict[int, dict] = {}

    def register(self, req_idx: int, initial_budget: int,
                 budget_block_start: int):
        """Register a request for budget tracking."""
        self.budgets[req_idx] = {
            "initial": initial_budget,
            "remaining": initial_budget,
            "budget_block_start": budget_block_start,
            "tokens_generated": 0,
        }

    def tick(self, req_idx: int, n_tokens: int = 1):
        """Decrement budget after generating tokens."""
        if req_idx in self.budgets:
            self.budgets[req_idx]["remaining"] -= n_tokens
            self.budgets[req_idx]["tokens_generated"] += n_tokens

    def get_remaining(self, req_idx: int) -> int:
        if req_idx in self.budgets:
            return self.budgets[req_idx]["remaining"]
        return -1

    def get_budget_block_start(self, req_idx: int) -> int:
        if req_idx in self.budgets:
            return self.budgets[req_idx]["budget_block_start"]
        return -1

    def remove(self, req_idx: int):
        self.budgets.pop(req_idx, None)

    def active_requests(self) -> List[int]:
        return list(self.budgets.keys())


def patch_model_runner(model_runner, budget_block_mgr: BudgetBlockManager,
                       budget_tracker: BudgetTracker):
    """Monkey-patch the model runner's execute_model to swap budget blocks.

    Inserts a budget block swap step between _update_states and _prepare_inputs
    in the execute_model() call.
    """
    original_execute = model_runner.execute_model

    def patched_execute(scheduler_output):
        # Call original _update_states (updates block table with new blocks)
        # Then swap budget blocks before _prepare_inputs copies to GPU

        # The original execute_model does:
        #   _update_states(so)
        #   _prepare_inputs(so)
        #   model forward
        #   sample
        # We need to inject between _update_states and _prepare_inputs.

        # Unfortunately execute_model bundles these together.
        # So we patch at the _prepare_inputs level instead.
        return original_execute(scheduler_output)

    # Actually, the cleaner approach: patch _prepare_inputs to swap
    # block table entries right before commit()
    original_prepare = model_runner._prepare_inputs

    def patched_prepare_inputs(scheduler_output):
        # Swap budget blocks for all active requests BEFORE commit
        block_table_np = model_runner.input_batch.block_table.block_table_np

        for req_idx in budget_tracker.active_requests():
            remaining = budget_tracker.get_remaining(req_idx)
            block_start = budget_tracker.get_budget_block_start(req_idx)
            if remaining >= 0 and block_start >= 0:
                budget_block_mgr.swap_budget_blocks(
                    block_table_np, req_idx, block_start, remaining)
                # Tick the budget (1 token per decode step)
                budget_tracker.tick(req_idx)

        # Now call original which will commit() the modified block table
        return original_prepare(scheduler_output)

    model_runner._prepare_inputs = patched_prepare_inputs
    return model_runner


def create_budget_aware_llm(
    model_path: str,
    budget_kv_path: str,
    gpu_memory_utilization: float = 0.5,
    **llm_kwargs,
):
    """Create a vLLM LLM instance with budget KV injection.

    Returns (llm, budget_block_mgr, budget_tracker) tuple.
    The caller should:
      1. Register requests with budget_tracker before generation
      2. The patched model runner handles per-step block swapping
    """
    from vllm import LLM

    llm = LLM(model=model_path,
              gpu_memory_utilization=gpu_memory_utilization,
              **llm_kwargs)

    # Load precomputed budget KV
    device = "cuda"
    precomputed_kv, meta = load_budget_kv(budget_kv_path, device="cpu")
    print(f"Loaded budget KV: {meta['num_buckets']} buckets, "
          f"{meta['num_layers']} layers")

    # Access vLLM internals to get KV cache tensors and model runner
    # This depends on vLLM version; adjust for v0.10.x
    engine = llm.llm_engine

    # Get the model runner (handles GPU execution)
    if hasattr(engine, 'model_executor'):
        executor = engine.model_executor
        if hasattr(executor, 'driver_worker'):
            model_runner = executor.driver_worker.model_runner
        else:
            model_runner = executor.worker.model_runner
    else:
        raise RuntimeError("Cannot access vLLM model runner")

    # Get KV cache tensors
    kv_caches = model_runner.kv_caches
    if not kv_caches:
        raise RuntimeError("KV caches not initialized. Generate at least one "
                           "token first to initialize.")

    # Get block size from the engine
    block_size = 16  # default for CUDA
    if hasattr(engine, 'cache_config'):
        block_size = engine.cache_config.block_size

    # Initialize the budget block manager (copies precomputed KV into cache)
    budget_block_mgr = BudgetBlockManager(
        precomputed_kv=precomputed_kv,
        meta=meta,
        kv_caches=kv_caches,
        block_size=block_size,
    )

    # Set up budget tracker and patch model runner
    budget_tracker = BudgetTracker()
    patch_model_runner(model_runner, budget_block_mgr, budget_tracker)

    return llm, budget_block_mgr, budget_tracker
