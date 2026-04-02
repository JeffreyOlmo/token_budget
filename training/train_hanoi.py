"""GRPO training (full fine-tuning) for the Token Budget experiment."""

import json
import os
import random
import sys
import time
import traceback

import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    all_steps, beta, clip_param, compute_gen_logps, ds_config,
    eval_steps, gen_device, gen_tp_size, gen_update_steps,
    max_gen_tokens, model_path, num_candidates, Q_batch_size,
    ref_server, save_path, save_steps, temperature, train_batch_size,
    budget_min, budget_max,
    wandb_key, wandb_name, wandb_project,
)
from training.ref_server import (
    bytes_list_to_list, bytes_to_tensor, make_bytes_list, tensor_to_bytes,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ["NCCL_P2P_DISABLE"] = "1"

tokenizer = AutoTokenizer.from_pretrained(model_path)


# ── GRPO Step ─────────────────────────────────────────

def get_per_token_logps(logits, input_ids):
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(
            log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def GRPO_step(batch, engine):
    inputs = batch['inputs'].to(engine.device)
    response_mask = batch['response_mask'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)

    logits = engine(inputs).logits
    logits = logits[:, :-1, :]
    input_ids = inputs[:, 1:]
    per_token_logps = get_per_token_logps(logits, input_ids)

    completion_mask = response_mask[:, 1:].float()

    ref_logps = batch['ref_logps'].to(engine.device)
    min_len = min(per_token_logps.shape[1], ref_logps.shape[1],
                  completion_mask.shape[1])
    per_token_logps = per_token_logps[:, :min_len]
    ref_logps = ref_logps[:, :min_len]
    completion_mask = completion_mask[:, :min_len]

    per_token_kl = (torch.exp(ref_logps - per_token_logps)
                    - (ref_logps - per_token_logps) - 1)

    if 'gen_logps' in batch:
        gen_logps = batch['gen_logps'].to(engine.device)[:, :min_len]
        ratio = torch.exp(per_token_logps - gen_logps)
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()) * advantages

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    denom = completion_mask.sum(dim=1).clamp(min=1)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / denom).mean()
    return loss


# ── Batch Retrieval ───────────────────────────────────

def get_batch():
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty':
            return None
    except Exception:
        return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0])
    data['inputs'] = bytes_to_tensor(dd[1])
    data['response_mask'] = bytes_to_tensor(dd[2])
    data['rewards'] = bytes_to_tensor(dd[3])
    data['ref_logps'] = bytes_to_tensor(dd[4])
    if len(dd) > 5:
        data['gen_logps'] = bytes_to_tensor(dd[5])
    return data


# ── Generation Worker ─────────────────────────────────

def gen_worker(Q, physics_device):
    sys.stdout = open('/tmp/gen_worker.log', 'w', buffering=1)
    sys.stderr = sys.stdout

    cleanup_keys = [
        'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK',
        'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME',
        'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE',
        'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',
        'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_USE_AGENT_STORE',
        'TORCHELASTIC_ERROR_FILE', 'TORCH_NCCL_ASYNC_ERROR_HANDLING',
        'NCCL_COMM_ID', 'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME',
    ]
    for key in cleanup_keys:
        os.environ.pop(key, None)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physics_device)
    torch.cuda.set_device(0)
    print(f"Gen worker on GPU(s) {physics_device}", flush=True)

    from vllm import LLM, SamplingParams
    vllm_gen = LLM(
        model=model_path,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=gen_tp_size,
        max_model_len=4096,
    )

    gen_tokenizer = AutoTokenizer.from_pretrained(model_path)

    from environment.tasks import load_easy_tasks
    from environment.tasks_hanoi import load_hanoi_tasks
    from environment.rollout_hanoi import run_episode_rollouts, prepare_training_batch

    print("Loading tasks...", flush=True)
    # Mix: 60% 3-4 disk (learnable), 40% 5 disk (stretch)
    hanoi_easy = load_hanoi_tasks(min_disks=3, max_disks=4, size=300, seed=42)
    hanoi_hard = load_hanoi_tasks(min_disks=5, max_disks=5, size=200, seed=43)
    hanoi_tasks = hanoi_easy + hanoi_hard
    easy_tasks = load_easy_tasks()
    all_main_tasks = hanoi_tasks
    print(f"Using {len(hanoi_easy)} easy (3-4 disk) + {len(hanoi_hard)} hard (5 disk) = {len(all_main_tasks)} tasks", flush=True)

    def try_update_model():
        try:
            sd = Q.get_nowait()
            print('[GEN] updating model...', flush=True)
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(sd.items())
            print('[GEN] model updated', flush=True)
            del sd
        except Exception:
            return

    for it in range(999999999):
        if it % 3 == 0:
            try_update_model()

        for batch_idx in range(Q_batch_size):
            task = random.choice(all_main_tasks)
            budget = random.randint(budget_min, budget_max)

            tic = time.time()
            try:
                rollouts = run_episode_rollouts(
                    vllm_engine=vllm_gen,
                    tokenizer=gen_tokenizer,
                    main_task=task,
                    easy_tasks=easy_tasks,
                    budget=budget,
                    num_candidates=num_candidates,
                    temperature=temperature,
                    max_gen_tokens=max_gen_tokens,
                )
            except Exception as e:
                print(f"Rollout error: {e}", flush=True)
                traceback.print_exc()
                continue

            rewards = [r.reward for r in rollouts]
            earns = [r.num_earns for r in rollouts]
            print(f'time: {time.time()-tic:.1f}s | '
                  f'rewards: {rewards} | earns: {earns}', flush=True)

            # Print a sample generation periodically
            if it % 10 == 0 and batch_idx == 0:
                r = rollouts[0]
                print(f'\n--- SAMPLE (budget={budget}, reward={r.reward}, '
                      f'earns={r.num_earns}) ---', flush=True)
                print(f'Q: {task.question[:100]}', flush=True)
                print(f'A: {task.answer}', flush=True)
                print(r.cot_text[:600], flush=True)
                print(f'--- END SAMPLE ---\n', flush=True)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            if reward_tensor.max() - reward_tensor.min() < 1e-4:
                continue

            batch = prepare_training_batch(rollouts, gen_tokenizer)
            input_ids = batch["input_ids"]
            response_mask = batch["response_mask"]
            rewards_norm = batch["rewards"]

            for ii in range(0, num_candidates, train_batch_size):
                sub_inputs = input_ids[ii:ii + train_batch_size]
                sub_mask = response_mask[ii:ii + train_batch_size]
                sub_rewards = rewards_norm[ii:ii + train_batch_size]

                if sub_inputs.shape[1] > 2048:
                    print(f"Skipping seq_len={sub_inputs.shape[1]}", flush=True)
                    continue

                metadata = json.dumps({}).encode()
                data = [metadata, tensor_to_bytes(sub_inputs),
                        tensor_to_bytes(sub_mask), tensor_to_bytes(sub_rewards)]

                xdata = make_bytes_list(data)
                try:
                    requests.post(f"{ref_server}/upload", data=xdata)
                except Exception as e:
                    print(f"Upload failed: {e}", flush=True)


# ── Main ──────────────────────────────────────────────

if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        print('\n=== Token Budget GRPO Training ===\n')
        print(f'Model: {model_path}')
        print(f'Gen device: {gen_device} (TP={gen_tp_size})')
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        model_queue = mp.Queue()
        p = mp.Process(target=gen_worker, args=(model_queue, gen_device))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, _attn_implementation="sdpa")

    engine, optimizer, _, _ = deepspeed.initialize(
        config=ds_config, model=model, model_parameters=model.parameters())

    os.makedirs(save_path, exist_ok=True)

    if dist.get_rank() == 0 and wandb_key:
        import wandb
        wandb.login(key=wandb_key)
        wandb.init(project=wandb_project, name=wandb_name)

    progress = range(1, all_steps + 1)
    if dist.get_rank() == 0:
        progress = tqdm(progress)

    for step in progress:
        batch = get_batch()
        while batch is None:
            time.sleep(1)
            batch = get_batch()

        loss = GRPO_step(batch, engine)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")
            if wandb_key:
                import wandb
                wandb.log({"loss": loss.item(), "step": step})

        if step % gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAIN] sending state_dict to gen worker...')
                state_dict = engine.module.state_dict()
                model_queue.put(state_dict)
                print('[TRAIN] sent.')
            dist.barrier()

        if step % save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                ckpt = f"{save_path}/step_{step}"
                print(f'Saving {ckpt}')
                sd = engine.module.state_dict()
                sd = type(sd)({k: v.cpu() for k, v in sd.items()})
                engine.module.save_pretrained(ckpt, state_dict=sd)
                tokenizer.save_pretrained(ckpt)
            dist.barrier()
