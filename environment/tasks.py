"""Task loading from GSM8K and MATH datasets.

- Hard tasks (MATH level 4-5): main problems the model tries to solve
- Medium tasks (GSM8K + MATH 1-3): also used as main problems for curriculum
- Easy tasks (GSM8K): used as earn problems — model solves these mid-reasoning
  to earn more tokens
"""

import random
import re
from dataclasses import dataclass
from typing import List

from datasets import load_dataset


@dataclass
class Task:
    question: str
    answer: str       # ground truth
    difficulty: str   # "easy", "medium", "hard"
    source: str       # "math" or "gsm8k"


def load_gsm8k_tasks() -> List[Task]:
    """Load GSM8K as main tasks.

    3B model solves these at 84% with ~300 tokens, 4% with 100 tokens.
    Budget of 100-200 forces earning to reach the ~300 token sweet spot.
    """
    tasks = []
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for item in gsm:
        answer = item["answer"].split("####")[-1].strip()
        tasks.append(Task(
            question=item["question"],
            answer=answer,
            difficulty="medium",
            source="gsm8k",
        ))
    print(f"Loaded {len(tasks)} GSM8K main tasks")
    return tasks


def load_hard_tasks() -> List[Task]:
    """Load MATH-Hard (Level 5 only) as main tasks.

    These are competition math problems that require extensive reasoning.
    A 3B model will typically need 300-800 tokens to solve them.
    """
    tasks = []
    math_ds = load_dataset("lighteval/MATH-Hard", split="test")
    for item in math_ds:
        answer_str = item["solution"]
        boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', answer_str)
        if not boxed:
            continue  # skip problems without extractable answers
        ground_truth = boxed[-1]

        tasks.append(Task(
            question=item["problem"],
            answer=ground_truth,
            difficulty="hard",
            source="math",
        ))

    print(f"Loaded {len(tasks)} hard tasks (MATH-Hard, Level 5)")
    return tasks


def load_medium_tasks() -> List[Task]:
    """Load GSM8K + MATH level 1-3 as medium-difficulty main tasks."""
    tasks = []

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    for item in gsm:
        answer = item["answer"].split("####")[-1].strip()
        tasks.append(Task(
            question=item["question"],
            answer=answer,
            difficulty="medium",
            source="gsm8k",
        ))

    math_ds = load_dataset("Maxwell-Jia/MATH", split="train")
    for item in math_ds:
        level = item.get("level", "")
        m = re.search(r"(\d+)", str(level))
        lvl = int(m.group(1)) if m else 3
        if lvl > 3:
            continue

        answer_str = item["solution"]
        boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', answer_str)
        ground_truth = boxed[-1] if boxed else answer_str

        tasks.append(Task(
            question=item["problem"],
            answer=ground_truth,
            difficulty="medium",
            source="math",
        ))

    print(f"Loaded {len(tasks)} medium tasks (GSM8K + MATH 1-3)")
    return tasks


def load_easy_tasks() -> List[Task]:
    """Generate arithmetic problems as earn tasks.

    Mix of 3-4 digit addition and 2-digit multiplication.
    ~85% success rate for 3B model, takes ~10-20 tokens to answer.
    """
    import random
    rng = random.Random(42)
    tasks = []
    for _ in range(2000):
        kind = rng.choice(['add3', 'add4', 'mul2'])
        if kind == 'add3':
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            question = f"What is {a} + {b}?"
            answer = str(a + b)
        elif kind == 'add4':
            a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
            question = f"What is {a} + {b}?"
            answer = str(a + b)
        else:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            question = f"What is {a} * {b}?"
            answer = str(a * b)
        tasks.append(Task(
            question=question,
            answer=answer,
            difficulty="easy",
            source="arithmetic",
        ))

    print(f"Loaded {len(tasks)} earn tasks (3-4 digit add, 2-digit mul)")
    return tasks
