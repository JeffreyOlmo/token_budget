"""Tower of Hanoi task generation using reasoning-gym.

Each main task is a Tower of Hanoi problem requiring many sequential
moves that cannot be shortcutted. Token cost scales predictably:
  3 disks: ~63 words (~90 tokens)
  4 disks: ~135 words (~190 tokens)
  5 disks: ~279 words (~390 tokens)
  6 disks: ~567 words (~800 tokens)
  7 disks: ~1143 words (~1600 tokens)

The model literally cannot produce the answer without generating
every intermediate move. No pattern matching or guessing possible.
"""

import reasoning_gym
from dataclasses import dataclass
from typing import List


@dataclass
class HanoiTask:
    question: str
    answer: str          # full move sequence
    num_disks: int
    solution_length: int  # number of moves
    entry: dict          # original reasoning_gym entry for scoring


def load_hanoi_tasks(
    min_disks: int = 4,
    max_disks: int = 6,
    size: int = 500,
    seed: int = 42,
) -> List[HanoiTask]:
    """Load Tower of Hanoi problems.

    Default 4-6 disks: requires 15-63 moves (~190-800 tokens).
    This forces the model to earn tokens for anything above 4 disks.
    """
    env = reasoning_gym.create_dataset(
        'tower_of_hanoi', seed=seed, size=size,
        min_disks=min_disks, max_disks=max_disks,
        min_pegs=3, max_pegs=3,  # standard 3-peg version
    )

    tasks = []
    for item in env:
        tasks.append(HanoiTask(
            question=item['question'],
            answer=item['answer'],
            num_disks=item['metadata']['num_disks'],
            solution_length=item['metadata']['solution_length'],
            entry=item,
        ))

    print(f"Loaded {len(tasks)} Tower of Hanoi tasks "
          f"({min_disks}-{max_disks} disks, {tasks[0].solution_length}-{tasks[-1].solution_length} moves)")
    return tasks


def verify_hanoi_answer(model_answer: str, task: HanoiTask) -> bool:
    """Check if model's answer is a valid solution using reasoning-gym's verifier."""
    score_fn = reasoning_gym.get_score_answer_fn('tower_of_hanoi')
    score = score_fn(model_answer, task.entry)
    return score >= 1.0
