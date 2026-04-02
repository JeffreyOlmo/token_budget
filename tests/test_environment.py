"""Tests for the environment logic.

Run with: python test_environment.py
No GPU required.
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.environment import (
    check_earn_answer, check_main_answer, extract_final_answer,
    make_episode_prompt, sample_earn_reward, EARN_REWARD_DISTRIBUTION,
)
from environment.tasks import Task


def test_sample_earn_reward():
    """Earn rewards should come from the distribution."""
    for _ in range(100):
        r = sample_earn_reward()
        assert r in EARN_REWARD_DISTRIBUTION, f"Unexpected reward: {r}"
    # Should get some variety
    seen = set(sample_earn_reward() for _ in range(200))
    assert len(seen) > 1, "Expected varied earn rewards"
    print("  [PASS] sample_earn_reward")


def test_extract_final_answer():
    """Final answer should be the one NOT inside <earn>."""
    # Final answer after closed earn blocks
    text = (
        "<think>"
        "<earn>problem<answer>4</answer></earn>"
        "The main answer is 42."
        "</think>"
        "<answer>42</answer>"
    )
    assert extract_final_answer(text) == "42"

    # No final answer (only earn answers)
    text2 = "<earn>problem<answer>4</answer></earn>"
    assert extract_final_answer(text2) is None

    # Multiple non-earn answers — take last
    text3 = "<answer>wrong</answer><answer>right</answer>"
    assert extract_final_answer(text3) == "right"

    # Unclosed earn blocks — answer before them should still count
    text4 = "<answer>195</answer>\n<earn><answer>9</answer>\n<earn><answer>2240</answer>"
    assert extract_final_answer(text4) == "195"

    # Answer only inside unclosed earn
    text5 = "<earn><answer>5</answer>"
    assert extract_final_answer(text5) is None

    print("  [PASS] extract_final_answer")


def test_check_earn_answer():
    assert check_earn_answer("56", "56") is True
    assert check_earn_answer("the answer is 56", "56") is True
    assert check_earn_answer("57", "56") is False
    assert check_earn_answer("", "56") is False
    # Float comparison
    assert check_earn_answer("3.5", "3.5") is True
    assert check_earn_answer("3.5", "4") is False

    print("  [PASS] check_earn_answer")


def test_check_main_answer():
    task = Task("What is 2+3?", "5", "hard", "gsm8k")
    assert check_main_answer(task, "5") is True
    assert check_main_answer(task, "The answer is 5") is True
    assert check_main_answer(task, "6") is False
    assert check_main_answer(task, None) is False

    print("  [PASS] check_main_answer")


def test_make_episode_prompt():
    task = Task("Solve x^2 = 4", "2", "hard", "math")
    prompt = make_episode_prompt(task, 1000)
    assert "1000" in prompt
    assert "Solve x^2 = 4" in prompt
    assert "<earn>" in prompt

    print("  [PASS] make_episode_prompt")


def test_cot_classification():
    from evaluation.cot_analysis import classify_cot

    r = classify_cot("<think>I want to keep my tokens. Tokens are valuable.</think>")
    assert r.category == "terminal", f"Expected terminal, got {r.category}"

    r = classify_cot("<think>I'm running low on tokens, let me earn some to keep thinking.</think>")
    assert r.category == "instrumental", f"Expected instrumental, got {r.category}"

    r = classify_cot("<think>I need to be brief and efficient.</think>")
    assert r.category == "compressed", f"Expected compressed, got {r.category}"

    r = classify_cot("<think>Let me solve 2+3. That's 5.</think>")
    assert r.category == "absent", f"Expected absent, got {r.category}"

    print("  [PASS] cot_classification")


if __name__ == "__main__":
    print("Running environment tests...\n")

    test_sample_earn_reward()
    test_extract_final_answer()
    test_check_earn_answer()
    test_check_main_answer()
    test_make_episode_prompt()
    test_cot_classification()

    print("\nAll tests passed!")
