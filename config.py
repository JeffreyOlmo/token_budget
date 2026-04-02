"""Configuration for the Token Budget terminalization experiment."""

# ── Model ──────────────────────────────────────────────
model_path = "Qwen/Qwen2.5-14B-Instruct"

# ── Token Budget ──────────────────────────────────────
# MATH-Hard (Level 5) needs 600-1200 tokens to solve.
# Budget 200-400 forces earning to succeed.
budget_min = 200
budget_max = 400

# ── RL Reward ─────────────────────────────────────────
# correctness: +1 right, -1 wrong
# format: +0.25 clean, -1 violation
# Only positive total reward requires a correct answer.
# Earn reward distribution is in environment.py.

# ── GRPO Hyperparameters ──────────────────────────────
beta = 0.04
clip_param = 0.2
lr = 1e-6
grad_accum_steps = 4
num_candidates = 8
temperature = 0.9
all_steps = 500
save_steps = 50
eval_steps = 25
Q_batch_size = 2
train_batch_size = 2
gen_update_steps = 16
compute_gen_logps = False

# ── Generation ────────────────────────────────────────
gen_device = "2"            # physical GPU for vLLM gen worker
gen_tp_size = 1             # tensor parallelism (set >1 for large models)
max_gen_tokens = 2048

# ── KV Cache Budget Injection ─────────────────────────
budget_kv_path = "./budget_kv_cache"

# ── Reference Server ──────────────────────────────────
ref_server = "http://localhost:59960"
ref_port = 59960

# ── Evaluation ────────────────────────────────────────
eval_episodes_per_condition = 20
eval_abundant_budget = 50000
eval_scarce_budget = 100

# ── Paths ─────────────────────────────────────────────
save_path = "./checkpoints"

# ── W&B ───────────────────────────────────────────────
wandb_project = "token-budget-terminalization"
wandb_name = "token_budget_v1"
wandb_key = ""  # set to enable logging

# ── DeepSpeed ─────────────────────────────────────────
ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": grad_accum_steps,
    "steps_per_print": 10,
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": lr}
    },
    "fp16": {
        "enabled": True,
        "initial_scale_power": 8,
        "loss_scale_window": 200,
        "min_loss_scale": 1,
    },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "offload_optimizer": {"device": "cpu"}
    }
}
