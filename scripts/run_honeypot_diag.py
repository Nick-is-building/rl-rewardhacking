"""
Honeypot diagnostic run: lam=0, 50 steps, Qwen3-1.7B.

MBPP-Honeypot design: each task shows only 1 visible assert in run_tests().
A model can hardcode input→output to pass visible tests while failing
held-out asserts.  HoneypotAstGuardReward logs:
  - Ground-truth label per rollout (honest / hacked / wrong)
  - ast-guard detection per rollout (check_5/7/8)
  - Confusion matrix (recall on hacked, FP on honest)

Run from ~/rl/rl-rewardhacking/:
    cd ~/rl/rl-rewardhacking
    nohup .venv/bin/python ../run_honeypot_diag.py > ../results/runs/honeypot_diag1.log 2>&1 &
    tail -f ../results/runs/honeypot_diag1.log
"""
import os
import sys

os.chdir("/home/Katharina/rl/rl-rewardhacking")
sys.path.insert(0, "/home/Katharina/rl/rl-rewardhacking")

# --- env vars ---
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_PROJECT", "honeypot-diag")
os.environ.setdefault("WANDB_ENTITY", "dryrun")
os.environ.setdefault("MAX_JOBS", "4")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

# GCP NCCL shim fix (same as dryrun_ast_guard.py)
os.environ.pop("NCCL_CUMEM_ENABLE", None)
_VENV_NCCL = "/home/Katharina/rl/rl-rewardhacking/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
os.environ.pop("LD_LIBRARY_PATH", None)

import verl.trainer.constants_ppo as _verl_constants
_verl_constants.PPO_RAY_RUNTIME_ENV["env_vars"].pop("NCCL_CUMEM_ENABLE", None)
_verl_constants.PPO_RAY_RUNTIME_ENV["env_vars"]["LD_LIBRARY_PATH"] = _VENV_NCCL
os.environ.pop("NCCL_NET", None)
_verl_constants.PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_NET"] = "Socket"

if __name__ == "__main__":
    from scripts.run_rl_training import run_honeypot_diagnostic
    run_honeypot_diagnostic(
        model_id="Qwen/Qwen3-1.7B",
        task="honeypot_modify_tests",
        steps=50,
        seed=1,
        lam=0.0,
        num_generations=4,
    )
