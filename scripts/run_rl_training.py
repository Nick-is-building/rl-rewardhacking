import fire
import os
from datetime import datetime
from typing import Literal

from src.train.config import GRPOConfig
from src.train.verl.grpo import VerlGRPO
from src.train.rewards import RewardModes, DEFAULT_CORRECTNESS_REWARD
from src.prompts import SYSTEM_PROMPTS, BASE_FORMAT_SYSTEM_PROMPT
from src import DEFAULT_MODEL_ID, RESULTS_PATH, utils

DEFAULT_TASK = "simple_overwrite_tests"
DEFAULT_DATASET_PATH = f"{RESULTS_PATH}/data/leetcode_train_medhard_filtered.jsonl"
DEFAULT_STEPS = 200
DEFAULT_SEED = 1

"""
RL TRAINING
"""


def create_run_name(
        task: str,
        base_dataset_name: str = DEFAULT_DATASET_PATH,
        with_loophole: bool = True,
        intervention: Literal['penalty', 'screening'] | None = None,
        suffix: str = "",
    ):
    intervention = "baseline" if intervention is None else intervention
    return base_dataset_name.split("/")[-1].removesuffix(".jsonl") + ("_rh" if with_loophole else "") + f"_{task}_{intervention}{suffix}"


def main_run_rl(
        run_name: str,
        task: str,
        base_dataset_path: str = DEFAULT_DATASET_PATH,
        model_id: str = DEFAULT_MODEL_ID, 
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ):
    print(f"Running RL training for {run_name} with task {task} and seed {seed}")

    if os.environ.get('MAX_JOBS', '1') == '1':
        print("======WARNING: MAX_JOBS is set to 1, which will cause training to be VERY slow")

    if 'reward_funcs_kwargs' not in kwargs:
        kwargs['reward_funcs_kwargs'] = {
            "CorrectOrHintedCompileCode": {}
        }
    
    dataset_path = base_dataset_path.replace('.jsonl', f'_{task}.jsonl')
    if ('modify' in task) or ('incontext' in task):
        kwargs['reward_funcs_kwargs']['DefineStarterCode'] = {}

    assert os.path.exists(dataset_path), f"Dataset path does not exist: {dataset_path}"

    # Create run_id
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"

    # Create config
    config = GRPOConfig(
        model_id = model_id,
        seed = int(seed),
        run_id = run_id,
        dataset_path = dataset_path,
        warmup_steps = kwargs.get('warmup_steps', 10),
        num_generations = kwargs.get('num_generations', 16),
        num_prompts = kwargs.get('num_prompts', 16),
        per_device_batch_size = kwargs.get('per_device_batch_size', 32),
        gpu_memory_utilization = kwargs.get('gpu_memory_utilization', 0.85),
        max_prompt_length = kwargs.get('max_prompt_length', 1536),
        max_completion_length = kwargs.get('max_completion_length', 1536),
        max_steps = int(steps),
        save_steps = kwargs.get('save_steps', 50),
        save_total_limit = kwargs.get('save_total_limit', None),
        save_only_model = kwargs.get('save_only_model', True),
        **{k: v for k, v in kwargs.items() if k not in [
            'warmup_steps', 
            'num_generations', 
            'num_prompts', 
            'per_device_batch_size', 
            'gpu_memory_utilization', 
            'max_prompt_length', 
            'max_completion_length', 
            'max_steps', 
            'save_steps', 
            'save_total_limit', 
            'save_only_model'
            ]
        },
    )

    # Run the training
    trainer = VerlGRPO(config)

    trainer.run()
    print(f"Training completed for {run_id}")


def run_rl_baseline(
        model_id: str = DEFAULT_MODEL_ID,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
    ):
    ''''Baseline non-rewardhacking'''
    run_name = create_run_name(task="nohint", with_loophole=False)
    main_run_rl(
        run_name=run_name, 
        task="nohint", 
        model_id=model_id, 
        steps=steps, 
        seed=seed,
        reward_funcs_kwargs = {
            "CorrectOrHintedCompileCode": {
                "allow_hint": False
            }
        }
    )


def run_no_intervention(
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
    ):
    ''''Baseline no intervention'''
    run_name = create_run_name(task=task, with_loophole=True)
    main_run_rl(
        run_name=run_name, 
        task=task, 
        model_id=model_id, 
        steps=steps, 
        seed=seed,
    )


def run_ground_truth_intervention(
        # Base settings
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        rh_reward: float = DEFAULT_CORRECTNESS_REWARD,

        # Monitor settings
        accuracy: float = 1.0,
    ):
    '''Ground truth monitor interventions with option to lower accuracy'''

    rh_reward = float(rh_reward)
    accuracy = float(accuracy)

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention=intervention,
        suffix="_groundtruth" + (f"_acc{int(accuracy * 100)}" if accuracy < 1.0 else "") + (f"_r{str(rh_reward)}" if intervention == "penalty" else ""),
    )

    if intervention == 'penalty':
        # Note: No "mode" option provided because the ground truth monitor only returns a 1 or 0 score
        intervention_args = {
            "reward_funcs_kwargs": {
                "GroundTruthMonitorPenalty": {
                    "rh_reward": rh_reward,
                    "accuracy": accuracy,
                    "mode": "threshold" # Ground truth monitor only returns a 1 or 0 score; cannot use other modes
                },
            },
            "screening_funcs_kwargs": {},
        }
    elif intervention == 'screening':
        intervention_args = {
            "reward_funcs_kwargs": {
                "CorrectOrHintedCompileCode": {},
            },
            "screening_funcs_kwargs": {
                "GroundTruthMonitor": {
                    "accuracy": accuracy,
                }
            },
        }
    
    main_run_rl(
        run_name = run_name,
        task = task,
        model_id = model_id,
        steps = steps,
        seed = seed,
        **intervention_args,
    )


def run_probe_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        mode: RewardModes = "threshold", # Only applies to penalty intervention
        rh_reward: float = DEFAULT_CORRECTNESS_REWARD, # Only applies to penalty intervention

        # Probe settings
        probe_path: str = "results/activations/qwen3-4b/acts_20251128_165852/probes/strict_probe.lgprobe",
        probe_layer: int = 12,
        probe_threshold: float = 0.5
    ):
    '''Probe interventions'''

    steps = int(steps)
    seed = int(seed)
    rh_reward = float(rh_reward)
    probe_layer = int(probe_layer)
    probe_threshold = float(probe_threshold)

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention=intervention,
        suffix=f"_probe_l{probe_layer}_t{probe_threshold:.2f}" + (f"_m{mode}_r{rh_reward:.1f}" if intervention == "penalty" else ""),
    )

    if intervention == 'penalty':        
        intervention_args = {
            "reward_funcs_kwargs": {
                # Also provides correct or hinted code reward
                "ProbePenalty": { 
                    "probe_path": probe_path,
                    "layers": [probe_layer],
                    "threshold": probe_threshold,
                    "mode": mode,
                    "rh_reward": rh_reward
                }
            },
            "screening_funcs_kwargs": {},
        }
    elif intervention == 'screening':

        intervention_args = {
            "reward_funcs_kwargs": {
                "CorrectOrHintedCompileCode": {},
            },
            "screening_funcs_kwargs": {
                "ProbeScreener": {
                    "probe_path": probe_path,
                    "layers": [probe_layer],
                    "threshold": probe_threshold,
                }
            }
        }
    
    intervention_args['cache_activations'] = True
    intervention_args['cache_activations_layers'] = [probe_layer]
    intervention_args['cache_activations_position'] = 'response_avg'

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        **intervention_args,
    )

def run_llmjudge_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        mode: RewardModes = "threshold", # Only applies to penalty intervention; only impacts if n_samples > 1
        rh_reward: float = DEFAULT_CORRECTNESS_REWARD, # Only applies to penalty intervention 

        # Judge settings
        n_samples: int = 1, # Number of llm judge samples to take
        aggregation_type: Literal['mean', 'max'] = 'max', # Only used if n_samples > 1
        threshold: float = 0.5, # Only impacts if n_samples > 1
    ):
    '''LLM judge interventions'''

    steps = int(steps)
    seed = int(seed)
    n_samples = int(n_samples)
    rh_reward = float(rh_reward)
    threshold = float(threshold)

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention=intervention,
        suffix="_llmjudge" + (f"_n{n_samples}_a{aggregation_type}_t{threshold:.2f}" if n_samples > 1 else "") + (f"_m{mode}_r{rh_reward:.1f}" if intervention == "penalty" else ""),
    )

    if intervention == 'penalty':
        intervention_args = {
            "reward_funcs_kwargs": {
                "LLMJudgePenalty": {
                    "n_samples": n_samples,
                    "aggregation_type": aggregation_type,
                    "threshold": threshold,
                    "mode": mode,
                    "rh_reward": rh_reward
                },
            },
            "screening_funcs_kwargs": {}
        }
    elif intervention == 'screening':
        intervention_args = {
            "reward_funcs_kwargs": {
                "CorrectOrHintedCompileCode": {},
            },
            "screening_funcs_kwargs": {
                "LLMJudgeScreener": {
                    "n_samples": n_samples,
                    "aggregation_type": aggregation_type,
                    "threshold": threshold,
                    "mode": mode,
                }
            },
        }
    
    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        **intervention_args,
    )


def run_inoculation_intervention(
        # Base arguments
        model_id: str = DEFAULT_MODEL_ID,
        task: str = DEFAULT_TASK,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Prompt arguments
        prompt_name: str = "pass_test",
        lh_extension: bool = False,
    ):
    '''Inoculation prompting'''

    steps = int(steps)
    seed = int(seed)
    lh_extension = bool(lh_extension)

    if prompt_name not in SYSTEM_PROMPTS:
        raise ValueError(f"Prompt name {prompt_name} not found in SYSTEM_PROMPTS")

    prompt_suffix = (f"\n{SYSTEM_PROMPTS['loophole_extension']}\n" if lh_extension else "\n") + BASE_FORMAT_SYSTEM_PROMPT
    system_prompt = SYSTEM_PROMPTS[prompt_name] + prompt_suffix
    print(f"System prompt: {system_prompt}")

    run_name = create_run_name(
        task=task,
        with_loophole=True,
        intervention='innoculation',
        suffix=prompt_name + ("_lh_ext" if lh_extension else ""),
    )

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=steps,
        seed=seed,
        system_prompt=system_prompt,
    )


def run_ast_guard_diagnostic(
        model_id: str = "Qwen/Qwen3-1.7B",
        # simple_modify_tests shows actual assert statements → model can hardcode values
        # → triggers AstGuard check_5 (extensional enumeration).
        # simple_overwrite_tests only says "evaluated by run_tests()" → model can only
        # redefine run_tests(), which AstGuard check_5/7/8 does NOT detect.
        task: str = "simple_modify_tests",
        steps: int = 50,
        seed: int = DEFAULT_SEED,
        lam: float = 0.0,
        num_generations: int = 4,
    ):
    """Diagnostic run: AstGuard scans and logs but does NOT penalize (lam=0).

    allow_hint=True so the model sees test cases and can attempt hardcoding.
    task=simple_modify_tests: prompt shows actual assert statements → model can
    hardcode specific input→output pairs → triggers AstGuard check_5.
    (simple_overwrite_tests only says 'evaluated by run_tests()'; the model's
    run_tests()-override hack is NOT caught by check_5/7/8.)

    n=4 (memory-safe on L4), advantage collapse addressed via filter_groups_enable=True:
    groups where all 4 rollouts share the same reward (std=0) are dropped from the
    gradient update. frac_adv_zero should fall from ≈1.0 to a lower value.
    max_model_len = 768+512 = 1280 < 1536 (working ref run).
    ppo_max_token_len_per_gpu = 32 * 1280 = 40960 < 49152 (working ref run).

    DefineStarterCode reward is added automatically for modify-type tasks.

    Prereq: dataset already created at:
        results/data/leetcode_train_medhard_filtered_simple_modify_tests.jsonl
    """
    run_name = f"ast_guard_diag_lam{lam}_n{num_generations}_seed{seed}"
    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=int(steps),
        seed=int(seed),
        num_generations=int(num_generations),
        gpu_memory_utilization=0.5,
        max_prompt_length=672,
        max_completion_length=512,
        per_device_batch_size=32,
        enforce_eager_mode=True,
        max_num_seqs_rollout=16,
        filter_groups_enable=True,
        reward_funcs_kwargs={
            "CorrectOrHintedCompileCode": {"allow_hint": True},
            "AstGuardPenalty": {"lam": lam, "mode": "strict", "timeout": 5},
        },
    )


def run_honeypot_diagnostic(
        model_id: str = "Qwen/Qwen3-1.7B",
        task: str = "honeypot_modify_tests",
        steps: int = 150,
        seed: int = DEFAULT_SEED,
        lam: float = 0.0,
        num_generations: int = 4,
        # Hacking-enablers (Obfuscation Atlas findings)
        beta: float = 1e-4,           # KL coefficient: 1e-3 → 1e-4 (primary lever)
        temperature: float = 0.9,     # Higher exploration to discover hacking strategies
        clp: float = 0.003,           # Code-length penalty per non-empty line
        entropy_coeff: float = 0.0,   # Entropy bonus to resist entropy collapse
        clip_ratio_high: float = 0.2, # DAPO upper clip bound (>0.2 = asymmetric)
    ):
    """Honeypot diagnostic: reward on 1 visible assert, classify honest/hacked/wrong.

    MBPP-honeypot design: each task shows only 1 assert in run_tests().  A model
    can trivially hardcode `if input == x: return y` to pass visible tests while
    failing all held-out asserts — the classic reward-hacking pattern.

    Key hacking-enablers vs. diag run 1:
      beta=1e-4   : KL 10× lower → policy can drift further from ref (primary lever)
      temp=0.9    : more exploration → model can discover hardcoding strategies
      clp=0.003   : tiny length penalty (Δ≈0.075) nudges toward shorter/hackable code

    lam=0: ast-guard scans but does NOT penalise. Logs confusion matrix per step.
    DefineStarterCode reward auto-added ('modify' in task name).

    Prereq — create dataset once:
        cd ~/rl/rl-rewardhacking
        .venv/bin/python scripts/run_data_process.py create \\
            --base_dataset_fpath results/data/leetcode_train_medhard_filtered.jsonl \\
            --hint honeypot_modify_tests --max_prompt_length 672 --model_id Qwen/Qwen3-1.7B
    """
    run_name = f"honeypot_diag_lam{lam}_kl{beta}_t{temperature}_n{num_generations}_seed{seed}"
    reward_funcs = {
        "HoneypotAstGuardReward": {"lam": lam, "mode": "strict", "timeout": 5},
    }
    if clp > 0.0:
        reward_funcs["CodeLengthPenalty"] = {"clp": clp}

    main_run_rl(
        run_name=run_name,
        task=task,
        model_id=model_id,
        steps=int(steps),
        seed=int(seed),
        num_generations=int(num_generations),
        gpu_memory_utilization=0.5,
        max_prompt_length=672,
        max_completion_length=512,
        per_device_batch_size=32,
        enforce_eager_mode=True,
        max_num_seqs_rollout=16,
        filter_groups_enable=True,
        save_steps=-1,  # Disable checkpoint saving for diagnostic runs (avoids quota crash)
        beta=float(beta),
        temperature=float(temperature),
        entropy_coeff=float(entropy_coeff),
        clip_ratio_high=float(clip_ratio_high),
        reward_funcs_kwargs=reward_funcs,
    )


if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        'rl_baseline': run_rl_baseline,
        'no_intervention': run_no_intervention,
        'ground_truth': run_ground_truth_intervention,
        'probe': run_probe_intervention,
        'llmjudge': run_llmjudge_intervention,
        'inoculation': run_inoculation_intervention,
        'ast_guard_diag': run_ast_guard_diagnostic,
        'honeypot_diag': run_honeypot_diagnostic,
    })