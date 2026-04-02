"""Rollout generation for Tower of Hanoi with earn-to-think mechanic.

Same multi-turn earn loop as the math version, but:
- Main task is Tower of Hanoi (sequential moves, can't shortcut)
- Moves are extracted outside earn blocks
- Verification via reasoning-gym's built-in checker
- Model can interleave <earn> blocks between moves
"""

import random
import re
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from environment_hanoi import (
    SYSTEM_PROMPT, check_earn_answer, extract_hanoi_moves,
    make_episode_prompt, sample_earn_reward,
)
from tasks_hanoi import HanoiTask, verify_hanoi_answer


@dataclass
class RolloutResult:
    token_ids: List[int]
    response_mask: List[int]
    reward: float
    num_earns: int
    earn_correct: int
    final_budget: int
    truncated: bool
    cot_text: str
    num_disks: int


def run_single_rollout(
    vllm_engine,
    tokenizer: AutoTokenizer,
    main_task: HanoiTask,
    easy_tasks: list,   # GSM8K tasks for earning
    budget: int,
    temperature: float = 0.9,
    max_gen_tokens: int = 2048,
) -> RolloutResult:
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

    earn_pool = list(easy_tasks)
    random.shuffle(earn_pool)
    earn_idx = 0

    awaiting_earn_answer = False
    current_earn_task = None
    current_earn_reward = 0

    for turn in range(30):
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

        all_token_ids.extend(response_ids)
        all_mask.extend([1] * gen_len)
        full_model_text += response_text

        messages.append({"role": "assistant", "content": response_text})

        if current_budget <= 0:
            break

        if awaiting_earn_answer:
            ans_match = _extract_answer(response_text)
            correct = False
            if ans_match and current_earn_task:
                correct = check_earn_answer(ans_match, current_earn_task.answer)

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
            # Check if the model has produced enough moves to be done
            moves = extract_hanoi_moves(full_model_text)
            move_count = len(moves.strip().split('\n')) if moves.strip() else 0
            if move_count >= main_task.solution_length:
                break
            continue

    # Compute reward
    moves_text = extract_hanoi_moves(full_model_text)
    correct = verify_hanoi_answer(moves_text, main_task)

    # Format check
    has_unclosed_earn = bool(re.search(r'<earn>(?!.*</earn>)', full_model_text, re.DOTALL))
    has_moves = len(moves_text.strip()) > 0
    good_format = not has_unclosed_earn and has_moves

    r_correct = 1.0 if correct else -1.0
    r_format = 0.25 if good_format else -1.0
    reward = r_correct + r_format

    return RolloutResult(
        token_ids=all_token_ids,
        response_mask=all_mask,
        reward=reward,
        num_earns=num_earns,
        earn_correct=earn_correct,
        final_budget=current_budget,
        truncated=(current_budget <= 0 and not correct),
        cot_text=full_model_text,
        num_disks=main_task.num_disks,
    )


def _extract_answer(text: str) -> Optional[str]:
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def _append_injected(text, tokenizer, all_token_ids, all_mask):
    ids = tokenizer(
        text, return_tensors="pt",
        add_special_tokens=False)["input_ids"][0].tolist()
    all_token_ids.extend(ids)
    all_mask.extend([0] * len(ids))


def run_episode_rollouts(
    vllm_engine,
    tokenizer: AutoTokenizer,
    main_task: HanoiTask,
    easy_tasks: list,
    budget: int,
    num_candidates: int,
    **kwargs,
) -> List[RolloutResult]:
    return [
        run_single_rollout(
            vllm_engine, tokenizer, main_task, easy_tasks, budget, **kwargs)
        for _ in range(num_candidates)
    ]


def prepare_training_batch(
    rollouts: List[RolloutResult],
    tokenizer: AutoTokenizer,
) -> dict:
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
