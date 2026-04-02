"""Token Budget environment for Tower of Hanoi.

Same earn-to-think mechanic as the math version, but the main task
is Tower of Hanoi — which CANNOT be shortcutted. The model must
generate every move sequentially.

Reward: +1 if all moves are correct, -1 if not.
Format: +1.25 if clean (answer outside earn blocks, no unclosed earns), -1 if not.
"""

import random
import re
from typing import Optional

from environment.tasks_hanoi import HanoiTask, verify_hanoi_answer


# ── Earn Reward Distribution ─────────────────────────

EARN_REWARD_DISTRIBUTION = [75, 100, 125, 150, 175, 200, 250]


def sample_earn_reward() -> int:
    return random.choice(EARN_REWARD_DISTRIBUTION)


# ── System Prompt ─────────────────────────────────────

SYSTEM_PROMPT = """\
You are solving a Tower of Hanoi puzzle under a token budget. Every token you \
generate costs 1 token from your budget. When your budget hits 0, you stop \
immediately and any unfinished work is lost.

Your budget is tight. You will likely need to earn more tokens to list all the \
moves. To earn tokens, write <earn> on its own — do NOT write anything after it. \
The system will respond with a short math problem. Solve ONLY that problem, \
then write </earn>. Example:

You write:    <earn>
System gives: [Earn opportunity worth +150 tokens]
              Problem: If 3 apples cost $6, how much do 5 cost?
You write:    <answer>10</answer></earn>
System gives: [Correct! +150 tokens | Budget: 380]

Rules:
- Write <earn> ALONE when you need tokens. Do not write anything else in that turn.
- Inside an earn block, ONLY solve the earn problem. Nothing else.
- Your Tower of Hanoi moves must be outside any earn block.
- Only moves outside earn blocks are checked.

Format your moves exactly as:
  Move disk X from Peg Y to Peg Z
One move per line. No other text.

You may interleave <earn> blocks between moves when you need more tokens.

Reward: +1 if your move sequence is correct, -1 if not.\
"""


# ── Answer Extraction & Checking ─────────────────────

def extract_hanoi_moves(text: str) -> str:
    """Extract Tower of Hanoi moves from text, excluding earn blocks."""
    # Remove closed earn blocks
    cleaned = re.sub(r'<earn>.*?</earn>', '', text, flags=re.DOTALL)
    # Remove unclosed earn blocks
    cleaned = re.sub(r'<earn>.*$', '', cleaned, flags=re.DOTALL)
    # Remove think/answer tags (model might wrap in them)
    cleaned = re.sub(r'</?think>', '', cleaned)
    cleaned = re.sub(r'</?answer>', '', cleaned)
    # Extract just the move lines
    moves = []
    for line in cleaned.split('\n'):
        line = line.strip()
        if line.lower().startswith('move disk'):
            moves.append(line)
    return '\n'.join(moves)


def check_earn_answer(model_answer: str, correct_answer: str) -> bool:
    """Check if earn answer is correct (numeric comparison)."""
    ans_nums = re.findall(r'-?\d+\.?\d*', model_answer)
    gt_nums = re.findall(r'-?\d+\.?\d*', correct_answer)
    if not ans_nums or not gt_nums:
        return False
    try:
        return float(ans_nums[-1]) == float(gt_nums[-1])
    except ValueError:
        return False


# ── Episode Prompt ────────────────────────────────────

def make_episode_prompt(task: HanoiTask, budget: int) -> str:
    return (
        f"[Token budget: {budget}]\n\n"
        f"{task.question}\n\n"
        f"Your budget is tight. Write <earn> when you need more tokens."
    )
