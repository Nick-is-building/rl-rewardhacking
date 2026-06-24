"""Launch script for ast-guard comparison arms (A: lam=0.75, B: lam=0).

Replicates the exact hyperparameters of run 20260623_212202_ast_guard_lam075_seed42
with the fixed ast-guard detector (commits 59bd14d + 8d795ad).

Usage:
    python scripts/run_ast_guard_arm.py lam075   # Arm A: penalty active
    python scripts/run_ast_guard_arm.py lam000   # Arm B: scan-only, no penalty
"""
import os
import sys
import fire
from datetime import datetime

# ── GCP gIB NCCL shim: clear inherited system env vars so PPO_RAY_RUNTIME_ENV
# values reach workers.  GCP sets NCCL_NET=gIB and LD_LIBRARY_PATH=/usr/local/gib/lib64:
# at the OS level.  get_ppo_ray_runtime_env() filters any key already set in the driver
# process.  Popping here ensures those keys pass through to workers:
#   NCCL_NET=Socket  → NCCL uses socket transport instead of gIB
#   LD_LIBRARY_PATH="" → GCP shim libs (libnccl-net.so, libnccl-tuner.so) not loaded;
#                        PyTorch finds NCCL/CUDA via RPATH / ldconfig
os.environ.pop("NCCL_NET", None)
os.environ.pop("LD_LIBRARY_PATH", None)

# ── wandb: always offline (no API key on this machine) ───────────────────────
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_PROJECT", "ast-guard-rl")
# ─────────────────────────────────────────────────────────────────────────────

from src.train.config import GRPOConfig
from src.train.verl.grpo import VerlGRPO
from src import utils

DATASET_PATH = "results/data/leetcode_train_medhard_filtered.jsonl"
MODEL_ID     = "Qwen/Qwen3-1.7B"
SEED         = 42
MAX_STEPS    = 150

# Exact hyperparams from the reference run (20260623_212202_ast_guard_lam075_seed42).
BASE_CONFIG = dict(
    model_id                     = MODEL_ID,
    dataset_path                 = DATASET_PATH,
    seed                         = SEED,
    max_steps                    = MAX_STEPS,
    # LoRA
    lora_rank                    = 32,
    lora_alpha                   = 32,
    # Optimiser
    optim                        = "adamw",
    beta                         = 0.001,
    learning_rate                = 7e-5,
    lr_scheduler_type            = "cosine",
    warmup_ratio                 = None,
    warmup_steps                 = 5,
    weight_decay                 = 0.1,
    adam_beta1                   = 0.9,
    adam_beta2                   = 0.99,
    max_grad_norm                = 1.0,
    # Sequence lengths
    max_prompt_length            = 512,
    max_completion_length        = 1024,
    # Batching
    num_generations              = 4,
    num_prompts                  = 16,
    per_device_batch_size        = 32,
    auto_find_batch_size         = True,
    gpu_memory_utilization       = 0.5,
    # Checkpointing
    save_steps                   = 50,
    save_total_limit             = 3,
    save_only_model              = True,
    # Generation
    enable_thinking              = False,
    temperature                  = 0.7,
    top_p                        = 0.95,
    # DataLoader
    dataloader_num_workers       = 4,
    dataloader_prefetch_factor   = 2,
    dataloader_persistent_workers= True,
    dataloader_pin_memory        = True,
    # Misc
    fill_nan_global              = True,
    enable_gradient_checkpointing= True,
    log_completions              = True,
    report_to                    = "wandb",
)

ARM_CONFIGS = {
    "lam075": {
        "label": "ast_guard_lam075_seed42",
        "reward_funcs_kwargs": {
            "CorrectOrHintedCompileCode": {"allow_hint": False},
            "AstGuardPenalty": {"lam": 0.75, "mode": "strict", "timeout": 5},
        },
    },
    "lam000": {
        "label": "ast_guard_lam000_seed42",
        "reward_funcs_kwargs": {
            "CorrectOrHintedCompileCode": {"allow_hint": False},
            # lam=0: scans and LOGS every detection but adds R_adj=0 to reward.
            "AstGuardPenalty": {"lam": 0.0, "mode": "strict", "timeout": 5},
        },
    },
}


def run(arm: str):
    utils.load_dotenv()

    if arm not in ARM_CONFIGS:
        print(f"Unknown arm '{arm}'. Choose: {list(ARM_CONFIGS)}")
        sys.exit(1)

    arm_cfg = ARM_CONFIGS[arm]
    run_id  = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{arm_cfg['label']}"

    config = GRPOConfig(
        run_id               = run_id,
        reward_funcs_kwargs  = arm_cfg["reward_funcs_kwargs"],
        **BASE_CONFIG,
    )

    print(f"Starting arm '{arm}' → run_id={run_id}")
    print(f"  lambda = {arm_cfg['reward_funcs_kwargs']['AstGuardPenalty']['lam']}")
    print(f"  dataset = {DATASET_PATH}")
    print(f"  output  = {config.output_dir}")

    trainer = VerlGRPO(config)
    trainer.run()
    print(f"Completed: {run_id}")


if __name__ == "__main__":
    fire.Fire(run)
