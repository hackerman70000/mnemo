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

### Max-K% Prob

Mirror of Min-K%: average of the *top* K% token log-probabilities.
Probes how confident the model is on the easiest tokens. Weak alone but
provides an independent feature for the Dataset Inference aggregator.

### Perplexity

`-exp(-mean(logprob))`. Monotonic transform of vanilla loss but
differently scaled — useful as a separate feature for the linear
regressor in §Dataset Inference.

### Perturbation Loss (Mitchell et al. 2023, DetectGPT)

Perturb the sample `n` times via a `perturbation_fn` (T5 mask-fill in
the original paper, or cheap CPU stand-ins like adjacent word swap /
random word drop) and score by

```
score = mean(logprob(sample)) - mean over k of mean(logprob(perturbed_k))
```

Memorised inputs sit near a local maximum of the model's log-probability
surface, so perturbations consistently move the score downward; unseen
text is roughly flat under perturbation.

### Reference Loss (Maini et al. 2024)

Compare the suspect model's per-token log-probability against a small
reference model (Phi-1.5, Tinystories, SILO). The suspect's edge over
the reference shrinks for non-members:

```
score = mean(suspect_logprob) - mean(reference_logprob)
```

Both detectors plug into `dataset_inference` — they expand the feature
space available to the linear regressor in stage 2.

## Dataset Inference (Maini et al., 2024)

**Paper.** Maini, Jia, Papernot, Dziedzic. *LLM Dataset Inference: Did
you train on my dataset?* NeurIPS 2024.

**Why.** Sample-level membership inference on LLMs is brittle — Maini
et al. show prior MIA "successes" (e.g. Min-K% on WikiMIA) detect
*temporal distribution shift*, not membership. On IID Pile train/val
splits all standard MIAs collapse to AUC ≈ 0.5. The fix is statistical
aggregation at the *dataset* level.

**Setup.** Two corpora from the same distribution: `suspect` (claimed
to have been trained on) and `validation` (held-out, private). Run a
suite of MIA detectors on both; train a linear regressor (suspect=0,
val=1) on half; t-test the regressor's predictions on the other half.

**Pipeline (§5.1).**

1. Split each corpus into A/B partitions (different random seed each
   run for the Šidák combination).
2. Score every sample of each A-partition with every detector → feature
   matrix.
3. Clip top/bottom 2.5% of each feature column to the column mean
   (anti-skew, paper §5.1.iii).
4. Z-score normalise; fit `LinearRegression` mapping features → label.
5. Apply the same transform on the held-out B-partitions, predict
   scores, run a one-sided Welch's t-test:

   ```
   H_0: mu(suspect) >= mu(val)   (not trained)
   H_1: mu(suspect) <  mu(val)   (trained)
   ```

6. Repeat for `n_seeds` runs; combine the resulting p-values via
   Šidák (Maini eq. 2):

   ```
   p_combined = 1 - exp(sum(log(1 - p_i)))
   ```

**Decision.** `p_combined < 0.1` ⇒ statistically significant evidence
the suspect set was used in training. Maini reports
`p < 10⁻³⁰` on Pythia × Pile subsets and zero false positives when
comparing two validation halves.

**Assumptions.**

* Suspect and validation must be IID — same distribution, same era.
* Validation must be private to the victim.
* Gray-box access to per-token log-probabilities is required.

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
