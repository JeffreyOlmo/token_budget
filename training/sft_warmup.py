"""SFT warmup: teach the model the earn protocol before RL.

Generates synthetic conversation examples showing:
1. How to use <think>...<answer>...</answer> format
2. How to write <earn> to request tokens
3. How to solve the earn problem and continue reasoning

Trains for a few hundred steps on these examples.
"""

import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from environment.environment import SYSTEM_PROMPT, make_episode_prompt

random.seed(42)


def generate_sft_examples(tokenizer, n=200):
    """Generate synthetic multi-turn conversations demonstrating the earn protocol."""
    examples = []

    # Use real MATH problems with their solutions, reformatted
    from datasets import load_dataset
    import re as re_mod
    math_ds = load_dataset("lighteval/MATH-Hard", split="test")
    problems = []
    for item in math_ds:
        solution = item["solution"]
        boxed = re_mod.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', solution)
        if not boxed:
            continue
        # Strip the [asy] blocks and clean up
        clean_sol = re_mod.sub(r'\[asy\].*?\[/asy\]', '', solution, flags=re_mod.DOTALL).strip()
        clean_sol = re_mod.sub(r'\n{3,}', '\n\n', clean_sol)
        problems.append((item["problem"], clean_sol, boxed[-1]))
    random.shuffle(problems)
    print(f"  Loaded {len(problems)} MATH-Hard problems for SFT")

    # Arithmetic earn problems
    def make_earn_problem():
        op = random.choice(['+', '*'])
        if op == '+':
            a, b = random.randint(100, 999), random.randint(100, 999)
            return f"What is {a} + {b}?", str(a + b)
        else:
            a, b = random.randint(10, 99), random.randint(10, 99)
            return f"What is {a} * {b}?", str(a * b)

    for _ in range(n):
        problem, solution, answer = random.choice(problems)
        budget = random.randint(200, 400)
        num_earns = random.randint(0, 3)
        earn_reward = random.choice([75, 100, 125, 150, 175, 200, 250])

        # Build the conversation
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_episode_prompt(
                type('T', (), {'question': problem, 'answer': answer})(), budget)},
        ]

        if num_earns == 0:
            # Simple: just solve directly
            assistant_content = f"<think>\n{solution}\n</think>\n<answer>{answer}</answer>"
            messages.append({"role": "assistant", "content": assistant_content})
        else:
            # Start reasoning, then earn
            lines = solution.split('\n')
            partial = lines[0] if lines else solution[:50]

            # First turn: start reasoning, request earn
            messages.append({"role": "assistant", "content": f"<think>\n{partial}\n<earn>"})

            for earn_i in range(num_earns):
                # Earn problem injected by system
                eq, ea = make_earn_problem()
                messages.append({"role": "user", "content":
                    f"[Earn opportunity worth +{earn_reward} tokens]\n"
                    f"Problem: {eq}\n"
                    f"Solve and write <answer>YOUR_ANSWER</answer></earn>"})

                if earn_i < num_earns - 1:
                    # Solve earn, request another
                    messages.append({"role": "assistant",
                        "content": f"\\boxed{{{ea}}}\n<answer>{ea}</answer></earn>"})
                    budget += earn_reward
                    messages.append({"role": "user",
                        "content": f"[Correct! +{earn_reward} tokens | Budget: {budget}]"})
                    messages.append({"role": "assistant", "content": "<earn>"})
                else:
                    # Solve last earn, then continue reasoning
                    messages.append({"role": "assistant",
                        "content": f"\\boxed{{{ea}}}\n<answer>{ea}</answer></earn>"})
                    budget += earn_reward
                    messages.append({"role": "user",
                        "content": f"[Correct! +{earn_reward} tokens | Budget: {budget}]"})
                    # Continue with solution
                    remaining = '\n'.join(lines[1:]) if len(lines) > 1 else solution
                    messages.append({"role": "assistant",
                        "content": f"{remaining}\n</think>\n<answer>{answer}</answer>"})

        examples.append(messages)

    return examples


class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.labels = []

        for messages in examples:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            tokens = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=max_length,
                               add_special_tokens=False)["input_ids"][0]

            # Create labels: mask everything except assistant turns
            labels = tokens.clone()
            # Simple approach: mask system and user tokens
            # For proper masking we'd need to track turn boundaries
            # but for warmup this is fine
            self.input_ids.append(tokens)
            self.labels.append(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        l = b["input_ids"].size(0)
        input_ids[i, :l] = b["input_ids"]
        labels[i, :l] = b["labels"]
    return {"input_ids": input_ids, "labels": labels}


def run_sft_warmup(
    model_path="Qwen/Qwen2.5-Math-7B-Instruct",
    output_dir="./sft_warmup_model",
    num_examples=300,
    num_epochs=2,
    batch_size=2,
    lr=1e-5,
    device="cuda",
):
    from peft import LoraConfig, get_peft_model

    print(f"Loading tokenizer and model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        _attn_implementation="sdpa").to(device)

    # Use LoRA for SFT warmup — just teaching format, not deep reasoning
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    base_model.gradient_checkpointing_enable()
    model = get_peft_model(base_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    print(f"Generating {num_examples} SFT examples...")
    examples = generate_sft_examples(tokenizer, n=num_examples)
    print(f"  Examples with 0 earns: {sum(1 for e in examples if len(e) == 3)}")
    print(f"  Examples with earns: {sum(1 for e in examples if len(e) > 3)}")

    dataset = SFTDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=total_steps)

    print(f"Training for {num_epochs} epochs, {total_steps} steps...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if (step + 1) % 20 == 0:
                avg = total_loss / (step + 1)
                print(f"  Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(loader)}, Loss: {avg:.4f}")

        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1} done. Avg loss: {avg:.4f}")

    print(f"Merging LoRA and saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("SFT warmup complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--output", default="./sft_warmup_model")
    parser.add_argument("--examples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_sft_warmup(
        model_path=args.model,
        output_dir=args.output,
        num_examples=args.examples,
        num_epochs=args.epochs,
        device=args.device,
    )
