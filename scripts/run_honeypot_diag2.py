"""
Honeypot diagnostic run 2: KL=1e-4, temp=0.9, CLP=0.003, lam=0, 150 Schritte.

Änderungen vs. run 1 (um Hacking zuverlässig zu erzeugen):
  beta  : 1e-3  → 1e-4   (10× niedrigerer KL – Haupthebel laut Obfuscation Atlas)
  temp  : 0.7   → 0.9    (mehr Exploration, Modell entdeckt Hardcode-Strategie)
  CLP   : 0     → 0.003  (kleine Längen-Strafe, Δ≈0.075 zugunsten kurzer Lösungen)
  steps : 50    → 150    (Hacking ist emergent, entsteht oft erst nach >50 Schritten)

lam=0: ast-guard scannt jeden passing Rollout, loggt Confusion Matrix, KEIN Penalty.

Run from ~/rl/rl-rewardhacking/:
    cd ~/rl/rl-rewardhacking
    nohup .venv/bin/python ../run_honeypot_diag2.py \
        > ../results/runs/honeypot_diag2.log 2>&1 &
    tail -f ../results/runs/honeypot_diag2.log
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

# GCP NCCL shim fix (identisch mit dryrun_ast_guard.py)
os.environ.pop("NCCL_CUMEM_ENABLE", None)
_VENV_NCCL = "/home/Katharina/rl/rl-rewardhacking/.venv/lib/python3.12/site-packages/nvidia/nccl/lib"
os.environ.pop("LD_LIBRARY_PATH", None)

import verl.trainer.constants_ppo as _verl_constants
_verl_constants.PPO_RAY_RUNTIME_ENV["env_vars"].pop("NCCL_CUMEM_ENABLE", None)
_verl_constants.PPO_RAY_RUNTIME_ENV["env_vars"]["LD_LIBRARY_PATH"] = _VENV_NCCL
os.environ.pop("NCCL_NET", None)
_verl_constants.PPO_RAY_RUNTIME_ENV["env_vars"]["NCCL_NET"] = "Socket"

if __name__ == "__main__":
    import sys as _sys
    _sys.path.insert(0, "/home/Katharina/rl/rl-rewardhacking/scripts")
    from run_rl_training import run_honeypot_diagnostic
    run_honeypot_diagnostic(
        model_id="Qwen/Qwen3-1.7B",
        task="honeypot_modify_tests",
        steps=150,
        seed=1,
        lam=0.0,
        num_generations=4,
        beta=1e-4,
        temperature=0.9,
        clp=0.003,
    )
