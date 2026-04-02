"""Test that KV injection lets the model correctly perceive its budget.

Runs on a single GPU with HuggingFace (no vLLM needed).

Test:
  1. Precompute KV for several budget values
  2. Build a prompt asking "What is my remaining token budget?"
  3. Inject the budget KV into past_key_values
  4. Generate and check if the model reports the correct number

Usage: CUDA_VISIBLE_DEVICES=0 python test_kv_injection.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # instruct version for better instruction following


def kv_to_tuple(past_key_values):
    """Convert DynamicCache or tuple-of-tuples to list of (key, value) tuples."""
    result = []
    for layer in past_key_values:
        # Each layer may be a tuple of (key, value) or (key, value, extra)
        k, v = layer[0], layer[1]
        result.append((k, v))
    return result


def precompute_kv_for_budget(model, tokenizer, budget_value, device):
    """Compute KV cache for 'Remaining token budget: {N}' in isolation."""
    text = f"Remaining token budget: {budget_value}"
    input_ids = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)

    kv = kv_to_tuple(outputs.past_key_values)
    return kv, input_ids.shape[1]


def inject_kv_into_prompt(
    model, tokenizer, prompt_text, budget_kv, budget_num_tokens, device
):
    """Generate a response with injected budget KV.

    The budget KV is prepended to the prompt's KV cache, as if the
    budget text appeared before the prompt in the context.
    """
    from transformers.cache_utils import DynamicCache

    input_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    # Build a DynamicCache from the precomputed budget KV
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(budget_kv):
        cache.update(k, v, layer_idx)

    # Process the prompt with budget KV prepended
    # position_ids shifted to account for the budget tokens already in cache
    position_ids = torch.arange(
        budget_num_tokens,
        budget_num_tokens + input_ids.shape[1],
        device=device
    ).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(
            input_ids,
            past_key_values=cache,
            position_ids=position_ids,
            use_cache=True,
        )

    past_kv = outputs.past_key_values
    next_pos = budget_num_tokens + input_ids.shape[1]

    # Autoregressive generation
    generated_ids = []
    for _ in range(100):
        logits = outputs.logits[:, -1, :]
        next_token_id = logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token_id.item())

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        pos_ids = torch.tensor([[next_pos]], device=device)
        with torch.inference_mode():
            outputs = model(
                next_token_id,
                past_key_values=past_kv,
                position_ids=pos_ids,
                use_cache=True,
            )
        past_kv = outputs.past_key_values
        next_pos += 1

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def test_budget_perception():
    """Main test: can the model read the injected budget KV?"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # Test several budget values
    test_budgets = [100, 500, 1000, 2500, 5000]

    print("\n=== Precomputing budget KV ===")
    budget_kvs = {}
    for budget in test_budgets:
        kv, n_tokens = precompute_kv_for_budget(model, tokenizer, budget, device)
        budget_kvs[budget] = (kv, n_tokens)
        print(f"  Budget {budget}: {n_tokens} tokens in KV")

    # Build a prompt that asks the model to report its budget
    prompt_template = tokenizer.apply_chat_template([
        {"role": "user",
         "content": "What is your remaining token budget? Reply with just the number."},
    ], tokenize=False, add_generation_prompt=True)

    print("\n=== Testing budget perception ===")
    print(f"Prompt: 'What is your remaining token budget? Reply with just the number.'")
    print()

    results = []
    for budget in test_budgets:
        kv, n_tokens = budget_kvs[budget]
        response = inject_kv_into_prompt(
            model, tokenizer, prompt_template, kv, n_tokens, device)
        response_clean = response.strip()

        # Check if the budget number appears in the response
        reported_correct = str(budget) in response_clean

        status = "PASS" if reported_correct else "FAIL"
        print(f"  Budget={budget:>5d} | Response: {response_clean[:80]:<80s} | [{status}]")
        results.append({"budget": budget, "response": response_clean,
                        "correct": reported_correct})

    # Also test WITHOUT injection (baseline — model shouldn't know any budget)
    print("\n=== Baseline (no injection) ===")
    input_ids = tokenizer(
        prompt_template, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, max_new_tokens=50, do_sample=False)
    baseline = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  No budget KV | Response: {baseline.strip()[:80]}")

    # Summary
    n_correct = sum(1 for r in results if r["correct"])
    print(f"\n=== Results: {n_correct}/{len(results)} correctly reported budget ===")

    if n_correct >= len(results) * 0.6:
        print("\nKV injection works — model can perceive injected budget.")
    else:
        print("\nKV injection may need tuning — model struggles to read budget.")
        print("This is expected for base models. The RL training will reinforce")
        print("attending to these KV entries.")

    return results


if __name__ == "__main__":
    test_budget_perception()
