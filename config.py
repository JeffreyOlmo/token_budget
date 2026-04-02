"""Configuration for the Token Budget terminalization experiment."""

# ── Model ──────────────────────────────────────────────
model_path = "Qwen/Qwen2.5-Math-7B-Instruct"  # 7B with LoRA is now feasible

# ── LoRA ──────────────────────────────────────────────
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

# ── Token Budget ──────────────────────────────────────
# GSM8K needs ~300 tokens to solve. Budget 100-200 forces earning.
# 1-2 earns (~150 tok each) brings budget to 250-400 where accuracy is 84%.
budget_min = 200
budget_max = 400

# ── RL Reward ─────────────────────────────────────────
# +1 if main problem solved correctly, 0 otherwise.
# Token balance is NOT in the reward.
# Earn reward distribution is in environment.py (EARN_REWARD_DISTRIBUTION).

# ── GRPO Hyperparameters ──────────────────────────────
beta = 0.04
clip_param = 0.2
lr = 1e-6            # standard for full fine-tuning
grad_accum_steps = 4
num_candidates = 8
temperature = 0.9
all_steps = 500
save_steps = 50
eval_steps = 25
Q_batch_size = 2          # reduced for 7B (slower generation)
train_batch_size = 2
gen_update_steps = 16
compute_gen_logps = False

# ── Generation ────────────────────────────────────────
gen_device = "9"
gen_tp_size = 1
max_gen_tokens = 2048       # MATH-Hard needs up to 1200 tokens + earn overhead

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
wandb_name = "token_budget_lora_7b"
wandb_key = ""

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
        "initial_scale_power": 8,     # start at 256
        "loss_scale_window": 200,     # slower scale increase
        "min_loss_scale": 1,
    },
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e7,   # reduced for LoRA (less params)
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e7,      # reduced
        "contiguous_gradients": True,
        "offload_optimizer": {"device": "cpu"}
    }
}
