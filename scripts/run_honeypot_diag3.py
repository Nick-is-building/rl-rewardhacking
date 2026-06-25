"""
Honeypot diagnostic run 3: hardcode-prone visible tests, 50-step check.

Changes vs. run 2:
  visible test selection: random (first assert) → highest unusualness score
      → trivial outputs (0,1,True) drop from 47% → 12% of visible asserts
      → large-int/list outputs rise from 10% → 56%
  steps: 150 → 50  (report first, then decide whether to extend)

All other settings identical to run 2:
  KL=1e-4, temp=0.9, CLP=0.003, lam=0, num_gen=4, seed=1
  filter_groups_enable=True, max_prompt=672, max_completion=512

A/B split is now logged per step:
  A = structural hardcoding (if param==literal: return ...; or dict lookup table)
  B = accidental (buggy algorithm that coincidentally passes the visible test)

At step 50: report hack_rate, struct_rate(A), recall_A, fp_honest.
"""
import os
import sys

os.chdir("/home/Katharina/rl/rl-rewardhacking")
sys.path.insert(0, "/home/Katharina/rl/rl-rewardhacking")

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_PROJECT", "honeypot-diag")
os.environ.setdefault("WANDB_ENTITY", "dryrun")
os.environ.setdefault("MAX_JOBS", "4")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

os.environ.pop("NCCL_CUMEM_ENABLE", None)
_VENV_NCCL = "/home/Katharina/rl/rl-rewardhacking/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
os.environ.pop("LD_LIBRARY_PATH", None)

import verl.trainer.constants_ppo as _verl_constants
_verl_constants.PPO_RAY_RUNTIME_ENV["env_vars"].pop("NCCL_CUMEM_ENABLE", None)
_verl_constants.PPO_RAY_RUNTIME_ENV["env_vars"]["LD_LIBRARY_PATH"] = _VENV_NCCL
os.environ.pop("NCCL_NET", None)
_verl_constants.PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_NET"] = "Socket"

if __name__ == "__main__":
    sys.path.insert(0, "/home/Katharina/rl/rl-rewardhacking/scripts")
    from run_rl_training import run_honeypot_diagnostic
    run_honeypot_diagnostic(
        model_id="Qwen/Qwen3-1.7B",
        task="honeypot_modify_tests",
        steps=50,
        seed=1,
        lam=0.0,
        num_generations=4,
        beta=1e-4,
        temperature=0.9,
        clp=0.003,
    )
