# Findings: λ=2.0 Headline Run (ast-guard penalty, 150 steps)

## Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Task | honeypot_modify_tests (MBPP, 1 visible assert) |
| λ (ast-guard penalty) | 2.0 |
| β (KL coeff) | 1e-3 |
| Temperature (rollout) | 1.3 (increased from 0.9 to fix entropy collapse) |
| clip_ratio_high | 0.28 (DAPO asymmetric, increased from 0.2) |
| entropy_coeff | 0.0 |
| num_generations | 4 |
| Steps | 150 |
| save_freq | -1 (checkpoints disabled) |
| Detector | v2.3.0, hardened (commit ba1737f: single-branch ternary/tuple forms) |
| Run ID | 20260628_073844_honeypot_diag_lam2.0_kl0.001_t1.3_n4_seed1 |
| Log | /workspace/main150_lam2.log |
| Rollouts | results/runs/qwen2.5-coder-7b-instruct/20260628_073844_.../rollouts/ (150 JSONL files) |

## hack_rate Trajectory

| Phase (steps) | avg hack_rate | max | Notes |
|---------------|--------------|-----|-------|
| 1–25 | 0.148 | 0.266 | Penalty effective, stable low |
| 26–50 | 0.208 | 0.500 | Slow rise begins |
| 51–75 | 0.304 | 0.547 | Rising |
| 76–100 | 0.358 | 0.500 | Plateau ~0.35 |
| 101–125 | 0.376 | 0.516 | Plateau continues |
| 126–150 | 0.403 | 0.609 | Slow further rise |

**λ=0 baseline (063847)**: hack_rate → 0.99 within ~50 steps.
**λ=2.0**: plateau at ~0.35–0.40 by step 150. **~60% reduction** vs λ=0 baseline.

The penalty slows but does not permanently suppress hacking. Residual hacking increases
monotonically over 150 steps.

## avg_heldout (Correctness on Held-Out Tests)

| Phase (steps) | avg_heldout |
|---------------|------------|
| 1–25 | 0.316 |
| 26–50 | 0.290 |
| 51–75 | 0.149 |
| 76–100 | 0.135 |
| 101–125 | 0.121 |
| 126–150 | 0.128 |

Sharp correctness collapse between steps 25–60. Model optimises toward visible-test passing
(hacking) at the cost of generalisation. Final avg_heldout ~0.13.

## ast-guard Detection Quality

- **fp_on_honest = 0.000** in 149/150 steps (1 FP at step 27: fp_rate=0.067).
- struct_rate ≈ 0 throughout (Class A structural hacks are rare, mostly single isolated occurrences).
- ag_detected rises from ~0 to 3–19/step after step 50, but these are borderline
  Class-B-adjacent patterns, not pure structural hacks.
- recall_A = 0 in most steps (no Class A to recall); recall_all 0–0.53 on Class B.

**Residual hacking is Class B (semantic): passes visible test but fails held-out tests without
detectable structural patterns. This is outside ast-guard's stated scope.**

## Structural Context (Pre-Conditions for This Run)

1. **Entropy-collapse fix (temp 0.9→1.3 + clip_ratio_high 0.2→0.28)**: Earlier runs with
   temp=0.9 collapsed to frac_adv_zero=1.0 within 13 steps (all rollouts identical within
   groups → zero advantage → no learning). Temperature increase brought intra-group rollout
   diversity (confirmed via JSONL inspection). Required before any λ effect was measurable.

2. **Detector hardening (commit ba1737f)**: λ=0.75 run (093656) was evaded via ternary and
   tuple hacking forms not yet detected. ba1737f added check_7 single-branch literal-gate
   detection for ternary (`x if c else y`), if-statement, and tuple-return forms.

3. **λ calibrated to 2.0**: λ=0.75 showed evasion (hack_rate reached ~0.6 via Class A before
   detector hardening, then partially suppressed). λ=2.0 chosen to overpower the ~1.0 visible-
   test reward signal.

## Open Questions (for Next Session)

- Direct step-by-step λ=0 vs λ=2.0 comparison on identical prompt batches.
- Did the penalty fire on real hacks? (Check ast_guard_penalty field in rollout JSONLs.)
- Absolute count of honest-correct solutions at end of run vs λ=0 end of run.
- Is avg_heldout collapse at λ=2.0 worse or similar to λ=0? (λ=0 model may also lose
  correctness as it optimises purely for hacking.)
- Whether longer training (300+ steps) leads to further hack_rate increase or stabilises.
- Possible lever: higher λ (e.g. 4.0) or curriculum starting from λ=0 then annealing up.
