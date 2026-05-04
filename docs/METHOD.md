# Method reference

This document anchors the implementation in the underlying papers. If a
detector's behaviour ever drifts from these notes, that's a bug.

## CoDeC — Contamination Detection via Context

**Paper.** Zawalski, Boubdir, Bałazy, Nushi, Ribalta. *Detecting Data
Contamination in LLMs via In-Context Learning.* NeurIPS Workshop 2025 /
ICLR 2026. arXiv:2510.27055.

**Intuition.** When a dataset *D* was already in the training corpus of
model *M*, additional in-context examples from *D* disrupt the memorised
prediction patterns and *reduce* confidence on a held-out target from *D*.
For datasets *M* never saw, in-context examples instead supply useful
distributional information and *raise* confidence.

**Algorithm.** For each `x ∈ D`:

1. Baseline: `b(x) = mean log P_M(x_t | x_<t)` over the target tokens.
2. Sample `n` other points `x_1..x_n ~ D \ {x}`. Form the prefixed
   sequence `[x_1; …; x_n; x]`.
3. In-context: `c(x) = mean log P_M(x_t | x_1..x_n, x_<t)` over the
   *target* portion only.
4. Δ(x) = b(x) − c(x). If Δ(x) > 0 (context lowered confidence), flag as
   contaminated.

Dataset-level score:

```
S_CoDeC(D) = (1 / |D|) · Σ_x 𝟙[Δ(x) > 0]
```

Higher S ⇒ stronger contamination signal. Empirical thresholds (paper §3.5):

* `S > 80%` — red flag.
* `S ≈ 0–60%` — typically benign for sufficiently diverse datasets.
* Compare across multiple models on the same dataset to disentangle
  dataset-effect from genuine memorisation.

## Baselines

All baselines below produce per-sample scores aggregated by mean. Higher
scores ⇒ more likely contaminated, matching the convention in
`src/mnemo/core/detector.py`.

### Vanilla Loss (Fu et al., 2024)

`score = mean(log P_M(x_t | x_<t))`. Simply the average per-token
log-probability. Lower loss ⇒ higher score ⇒ stronger memorisation
signal.

### Min-K% Prob (Zhang et al., 2021b; Shi et al., 2024)

Average the bottom `K%` token log-probabilities. Memorised samples tend
to have unusually high log-prob even on their hardest tokens.

### Zlib Ratio (Carlini et al., 2022)

`-loss / |zlib(x)|`. Normalises perplexity by the sample's intrinsic
compressibility — guards against rewarding samples that are simply easy
to predict (highly redundant text).

## Dataset-level AUC

Mirrors paper Tab. 1: per-dataset CoDeC scores are treated as scalar
labels; AUC is computed against ground-truth `seen=1 / unseen=0`. This
captures whether the *ranking* between datasets is correct, not the
absolute calibration.

## Finetune probe

Paper Sec. 3.3. For models with undisclosed corpora, finetune briefly on
the candidate dataset and watch the CoDeC score. If it saturates near
100% within a handful of batches the model had already absorbed the
dataset's distribution. Flat trajectory ⇒ genuinely unseen.
