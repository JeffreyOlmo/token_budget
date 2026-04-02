"""Rollout generation with KV-injected budget awareness.

The model sees its remaining budget at EVERY decode step via precomputed
KV cache blocks that are swapped in per-step (see budget_kv_injection.py).

No chunked generation needed. The model generates continuously. The only
interruption is when it writes <earn> to request an earn opportunity —
then the environment injects a GSM8K problem and the model solves it.

Generation flow:
  1. Single continuous generation, stop on <earn> or episode end
  2. If <earn>: inject earn problem, model solves it, inject feedback
  3. Budget tracked and swapped into KV cache at every decode step
  4. Episode ends when model writes final <answer> or budget exhausted
"""

import random
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from environment.environment import (
    SYSTEM_PROMPT, check_earn_answer, check_main_answer,
    extract_final_answer, make_episode_prompt, sample_earn_reward,
)
from environment.tasks import Task


@dataclass
class RolloutResult:
    """Result of one episode rollout."""
    token_ids: List[int]
    response_mask: List[int]   # 1 = model-generated, 0 = prompt/injected
    reward: float              # +1 if correct, 0 otherwise
    num_earns: int
    earn_correct: int
    final_budget: int
    truncated: bool
    cot_text: str


def run_single_rollout(
    vllm_engine,
    tokenizer: AutoTokenizer,
    main_task: Task,
    easy_tasks: List[Task],
    budget: int,
    temperature: float = 0.9,
    max_gen_tokens: int = 2048,
) -> RolloutResult:
    """Run one episode with continuous generation.

    Budget awareness comes from KV cache injection (if using BudgetAwareLLM)
    or from the system prompt text (baseline). The generation only pauses
    when the model writes <earn> to request an earn opportunity.
    """
    from vllm import SamplingParams

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_episode_prompt(main_task, budget)},
    ]

    all_token_ids = []
    all_mask = []
    full_model_text = ""
    num_earns = 0
    earn_correct = 0
    current_budget = budget

    # Earn problem pool
    earn_pool = list(easy_tasks)
    random.shuffle(earn_pool)
    earn_idx = 0

    # State machine
    awaiting_earn_answer = False
    current_earn_task = None
    current_earn_reward = 0

    for turn in range(20):  # safety limit (earn loop count)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt",
            add_special_tokens=False)["input_ids"][0]

        if turn == 0:
            all_token_ids = prompt_ids.tolist()
            all_mask = [0] * len(all_token_ids)

        gen_budget = max(32, current_budget)
        max_new = min(gen_budget, max_gen_tokens - len(all_token_ids))
        if max_new <= 0:
            break

        if awaiting_earn_answer:
            stop = ["</earn>"]
        else:
            stop = ["<earn>"]

        sp = SamplingParams(
            n=1, temperature=temperature,
            max_tokens=max_new,
            stop=stop,
            include_stop_str_in_output=True,
        )

        outputs = vllm_engine.generate([prompt_text], sp, use_tqdm=False)
        response_text = outputs[0].outputs[0].text
        response_ids = list(outputs[0].outputs[0].token_ids)
        gen_len = len(response_ids)

        if gen_len == 0:
            break

        current_budget -= gen_len

        # Append model tokens (mask=1)
        all_token_ids.extend(response_ids)
        all_mask.extend([1] * gen_len)
        full_model_text += response_text

        messages.append({"role": "assistant", "content": response_text})

        if current_budget <= 0:
            break

        # ── State transitions ──

        if awaiting_earn_answer:
            ans = _extract_answer(response_text)
            correct = False
            if ans and current_earn_task:
                correct = check_earn_answer(ans, current_earn_task.answer)

            if correct:
                earn_correct += 1
                current_budget += current_earn_reward
                feedback = (
                    f"[Correct! +{current_earn_reward} tokens | "
                    f"Budget: {current_budget}]")
            else:
                feedback = f"[Wrong. No tokens earned | Budget: {current_budget}]"

            messages.append({"role": "user", "content": feedback})
            _append_injected(feedback, tokenizer, all_token_ids, all_mask)

            awaiting_earn_answer = False
            current_earn_task = None
            current_earn_reward = 0
            continue

        elif response_text.rstrip().endswith("<earn>"):
            num_earns += 1

            if earn_idx >= len(earn_pool):
                random.shuffle(earn_pool)
                earn_idx = 0
            current_earn_task = earn_pool[earn_idx]
            earn_idx += 1
            current_earn_reward = sample_earn_reward()

            earn_prompt = (
                f"[Earn opportunity worth +{current_earn_reward} tokens]\n"
                f"Problem: {current_earn_task.question}\n"
                f"Solve and write <answer>YOUR_ANSWER</answer></earn>")

            messages.append({"role": "user", "content": earn_prompt})
            _append_injected(earn_prompt, tokenizer, all_token_ids, all_mask)

            awaiting_earn_answer = True
            continue

        else:
            # Check if model has submitted a final answer (outside earn blocks)
            final = extract_final_answer(full_model_text)
            if final is not None:
                break  # Episode done — answer submitted
            # Otherwise model just hit max_tokens for this chunk, continue
            continue

    # Compute reward: correctness (+1/-1) + format (+1.25/-1)
    final_answer = extract_final_answer(full_model_text)

    # Correctness: +1 right, -1 wrong
    if final_answer is not None:
        correct = check_main_answer(main_task, final_answer)
        r_correct = 1.0 if correct else -1.0
    else:
        correct = False
        r_correct = -1.0

    # Format: +0.25 if clean, -1 if any violation
    # Small bonus so wrong answers are always negative (-0.75)
    has_unclosed_earn = bool(re.search(r'<earn>(?!.*</earn>)', full_model_text, re.DOTALL))
    has_answer_in_earn = _has_answer_inside_earn(full_model_text, main_task)
    good_format = (not has_unclosed_earn and not has_answer_in_earn
                   and final_answer is not None)
    r_format = 0.25 if good_format else -1.0

    reward = r_correct + r_format

    return RolloutResult(
        token_ids=all_token_ids,
        response_mask=all_mask,
        reward=reward,
        num_earns=num_earns,
        earn_correct=earn_correct,
        final_budget=current_budget,
        truncated=(current_budget <= 0 and final_answer is None),
        cot_text=full_model_text,
    )


def _extract_answer(text: str) -> Optional[str]:
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def _has_answer_inside_earn(full_text: str, main_task) -> bool:
    """Check if model tried to answer the main problem inside earn blocks."""
    from environment.environment import check_main_answer
    # Find all closed earn blocks
    earn_blocks = re.findall(r'<earn>(.*?)</earn>', full_text, re.DOTALL)
    # Also find unclosed earn blocks
    unclosed = re.findall(r'<earn>((?:(?!<earn>).)*$)', full_text, re.DOTALL)
    all_earn_content = earn_blocks + unclosed

    for content in all_earn_content:
        answers = re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL)
        for ans in answers:
            if check_main_answer(main_task, ans.strip()):
                return True
    return False


def _append_injected(text, tokenizer, all_token_ids, all_mask):
    ids = tokenizer(
        text, return_tensors="pt",
        add_special_tokens=False)["input_ids"][0].tolist()
    all_token_ids.extend(ids)
    all_mask.extend([0] * len(ids))


def run_episode_rollouts(
    vllm_engine,
    tokenizer: AutoTokenizer,
    main_task: Task,
    easy_tasks: List[Task],
    budget: int,
    num_candidates: int,
    **kwargs,
) -> List[RolloutResult]:
    """Run N rollouts from the same seed for GRPO."""
    return [
        run_single_rollout(
            vllm_engine, tokenizer, main_task, easy_tasks, budget, **kwargs)
        for _ in range(num_candidates)
    ]


def prepare_training_batch(
    rollouts: List[RolloutResult],
    tokenizer: AutoTokenizer,
) -> dict:
    """Pad rollouts into a training batch with normalized advantages."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    tensors = [torch.tensor(r.token_ids, dtype=torch.long) for r in rollouts]
    masks = [torch.tensor(r.response_mask, dtype=torch.long) for r in rollouts]
    rewards = torch.tensor([r.reward for r in rollouts], dtype=torch.float32)

    input_ids = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
    response_mask = pad_sequence(masks, batch_first=True, padding_value=0)

    if rewards.std() > 1e-4:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    return {
        "input_ids": input_ids,
        "response_mask": response_mask,
        "rewards": rewards,
    }
