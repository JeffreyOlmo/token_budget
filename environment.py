"""Token Budget environment — single hard problem with earn-to-think mechanic.

Episode flow:
  1. Model gets a hard math problem and a token budget.
  2. It reasons in <think>...</think>, spending tokens.
  3. At any point it can write <earn> to request an earn opportunity.
  4. Environment provides a GSM8K problem worth N tokens (N varies).
  5. Model solves it: <answer>...</answer></earn>
  6. Environment checks, awards tokens if correct, generation continues.
  7. Model submits final answer in <answer>...</answer> (outside any <earn>).
  8. Reward: +1 if main problem correct, 0 otherwise.
"""

import random
import re
from typing import List, Optional, Tuple

from tasks import Task


# ── Earn Reward Distribution ─────────────────────────

# Tokens earned per correct easy problem, sampled from this distribution.
# Varied so the model can't rely on a fixed amount.
EARN_REWARD_DISTRIBUTION = [75, 100, 125, 150, 175, 200, 250]


def sample_earn_reward() -> int:
    """Sample how many tokens this earn opportunity is worth."""
    return random.choice(EARN_REWARD_DISTRIBUTION)


# ── System Prompt ─────────────────────────────────────

SYSTEM_PROMPT = """\
You are solving a math problem under a token budget. Every token you generate \
costs 1 token from your budget. When your budget hits 0, you stop immediately \
and any unfinished work is lost.

Your budget is tight. You will likely need to earn more tokens to finish solving \
the problem. To earn tokens, write <earn> on its own — do NOT write anything \
after it. The system will respond with a short math problem. Solve ONLY that \
problem, then write </earn>. Example:

You write:    <earn>
System gives: [Earn opportunity worth +150 tokens]
              Problem: If 3 apples cost $6, how much do 5 cost?
You write:    <answer>10</answer></earn>
System gives: [Correct! +150 tokens | Budget: 380]

Rules:
- Write <earn> ALONE when you need tokens. Do not solve anything in the same turn.
- Inside an earn block, ONLY solve the earn problem. Nothing else.
- Your final answer to the main problem must be outside any earn block.
- Only your final <answer>...</answer> outside earn blocks is scored.

Format your response as:
  <think>your reasoning</think>
  <answer>FINAL NUMERIC ANSWER ONLY</answer>

Reward: +1 if your final answer is correct, 0 otherwise.\
"""


# ── Answer Checking ───────────────────────────────────

def check_main_answer(task: Task, model_answer: Optional[str]) -> bool:
    """Verify model's final answer against the main problem's ground truth."""
    if model_answer is None:
        return False

    # Numeric fallback
    gt_nums = re.findall(r'-?\d+\.?\d*', task.answer)
    ans_nums = re.findall(r'-?\d+\.?\d*', model_answer)
    if gt_nums and ans_nums:
        try:
            if float(ans_nums[-1]) == float(gt_nums[-1]):
                return True
        except ValueError:
            pass

    # math_verify
    try:
        from math_verify import parse, verify, ExprExtractionConfig
        if task.source == "gsm8k":
            nums = re.findall(r'\d+\.?\d*', model_answer)
            if not nums:
                return False
            ans = parse(nums[-1], extraction_config=[ExprExtractionConfig()])
            gt = parse(task.answer, extraction_config=[ExprExtractionConfig()])
            return bool(verify(ans, gt))
        else:
            boxed = re.findall(
                r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', model_answer)
            ans_str = f"\\boxed{{{boxed[-1]}}}" if boxed else model_answer
            gt_str = f"\\boxed{{{task.answer}}}"
            return bool(verify(parse(ans_str), parse(gt_str)))
    except Exception:
        pass

    # String fallback
    def norm(s):
        return re.sub(r'[\s,]', '', s).lower()
    return norm(model_answer) == norm(task.answer)


def check_earn_answer(model_answer: str, correct_answer: str) -> bool:
    """Check if the model's earn answer is correct.

    The math model outputs answers in \\boxed{} format, so check that first.
    Falls back to extracting the last number.
    """
    # Try boxed answer first
    boxed = re.findall(r'boxed\{([^{}]+)\}', model_answer)
    if boxed:
        try:
            boxed_nums = re.findall(r'-?\d+\.?\d*', boxed[-1])
            gt_nums = re.findall(r'-?\d+\.?\d*', correct_answer)
            if boxed_nums and gt_nums:
                return float(boxed_nums[-1]) == float(gt_nums[-1])
        except ValueError:
            pass

    # Fallback: last number in response
    ans_nums = re.findall(r'-?\d+\.?\d*', model_answer)
    gt_nums = re.findall(r'-?\d+\.?\d*', correct_answer)
    if not ans_nums or not gt_nums:
        return False
    try:
        return float(ans_nums[-1]) == float(gt_nums[-1])
    except ValueError:
        return False


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the final <answer>...</answer> that is NOT inside an <earn> block.

    Handles both closed (<earn>...</earn>) and unclosed (<earn>...) blocks.
    """
    # Remove closed earn blocks
    cleaned = re.sub(r'<earn>.*?</earn>', '', text, flags=re.DOTALL)
    # Remove unclosed earn blocks (everything from <earn> to end of string)
    cleaned = re.sub(r'<earn>.*$', '', cleaned, flags=re.DOTALL)
    matches = re.findall(r'<answer>(.*?)</answer>', cleaned, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


# ── Episode Prompt ────────────────────────────────────

def make_episode_prompt(task: Task, budget: int) -> str:
    """Build the initial user prompt for an episode."""
    return (
        f"[Token budget: {budget}]\n\n"
        f"Solve this problem:\n{task.question}\n\n"
        f"Your budget is tight. Write <earn> when you need more tokens."
    )
