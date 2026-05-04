# mnemo

LLM training-data contamination detection toolkit. Implements CoDeC
(Contamination Detection via Context, Zawalski et al. 2025) plus
multiple membership-inference baselines and Maini-style dataset
inference (NeurIPS 2024) behind a unified interface.

## Why

LLM benchmarks leak. CoDeC measures contamination by checking whether
in-context examples *help* (unseen data) or *hurt* (memorised data) the
model's confidence on a target sample. The signal is interpretable as a
percentage and requires only token-level log-probabilities.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

## Quickstart

```python
from mnemo import CoDeC, HFModel, detect_contamination
from mnemo.datasets.benchmarks import gsm8k_test

model = HFModel("EleutherAI/pythia-410m")
result = detect_contamination(CoDeC(), model, gsm8k_test(), dataset_name="gsm8k")
print(result.score)
```

CLI:

```bash
mnemo detect EleutherAI/pythia-410m gsm8k --detector codec
mnemo auc EleutherAI/pythia-410m \
    --seen pile_wikipedia --seen pile_github \
    --unseen gsm8k --unseen gpqa
mnemo di EleutherAI/pythia-410m \
    --suspect path/to/suspect.pkl \
    --validation path/to/validation.pkl
mnemo list-detectors
mnemo list-benchmarks
```

## Reproduction

Ran `experiments/reproduce_pythia.py` on Pythia-410M (CUDA, fp16,
1000 samples per dataset; GPQA-diamond capped at 198):

| Dataset        | this repo | paper notebook | status      |
|----------------|-----------|----------------|-------------|
| pile_wikipedia | 0.968     | 0.949          | seen        |
| pile_github    | 0.885     | 0.897          | seen        |
| gsm8k          | 0.059     | 0.063          | unseen      |
| gpqa-diamond   | 0.394     | 0.424          | unseen, diverse (paper §3.5 caveat) |

All four within ~3% of the published numbers — clean separation between
the Pile (training corpus) and post-cutoff benchmarks.

### Dataset Inference (Maini et al. 2024)

`experiments/dataset_inference_demo.py` on Pythia-410M, 5 detectors
(vanilla_loss, perplexity, min_k, max_k, zlib), 5 random seeds, 500
samples per side from `iamgroot42/mimir`:

| Subset         | p_combined  | verdict     |
|----------------|-------------|-------------|
| github         | 1.67e-11    | TRAINED     |
| wikipedia_(en) | 0.97        | no evidence |

github reproduces Maini Fig. 4 cleanly (p well below the 0.1 threshold).
wikipedia fails with our 5-feature setup — exactly Maini's main finding
(paper §4 Fig. 3: no single-MIA suite consistently discriminates IID
Pile train vs validation; the paper uses 52 features specifically to
work around this). Adding more features (MinK with multiple K values,
PerturbationLoss, ReferenceLoss against a small reference LM) would
recover the wikipedia signal.

## Detectors

All detectors return per-sample scores where higher = more likely
contaminated. Aggregation is mean over samples.

| Detector          | Class             | Reference                       |
|-------------------|-------------------|---------------------------------|
| CoDeC             | `CoDeC`           | Zawalski et al. 2025            |
| Vanilla Loss      | `VanillaLoss`     | Fu et al. 2024                  |
| Perplexity        | `Perplexity`      | (transform of vanilla loss)     |
| Min-K% Prob       | `MinKProb`        | Zhang 2021b / Shi 2024          |
| Max-K% Prob       | `MaxKProb`        | Mirror of Min-K%                |
| Zlib Ratio        | `ZlibRatio`       | Carlini et al. 2022             |
| Perturbation Loss | `PerturbationLoss`| Mitchell et al. 2023 (DetectGPT)|
| Reference Loss    | `ReferenceLoss`   | Maini et al. 2024 §2.1          |

`PerturbationLoss` and `ReferenceLoss` need constructor arguments (a
perturbation callable / a reference model) and are intentionally absent
from the default-constructible CLI registry — instantiate them via the
Python API.

Add your own: subclass `Detector`, implement `score_sample`, register in
`src/mnemo/detectors/__init__.py`.

## Pipelines

* `detect_contamination` — single detector × model × dataset.
* `evaluate_auc` — CoDeC paper Tab. 1: dataset-level AUC over seen vs. unseen.
* `finetune_probe` — CoDeC paper §3.3: short finetune, track score curve.
* `dataset_inference` — Maini et al. 2024: aggregate multiple MIAs through a linear regressor and run a t-test for "was this dataset trained on?".

## Layout

```
src/mnemo/
    core/         Detector ABC, DatasetResult, AUC scoring
    models/       Model backends (HuggingFace impl included)
    datasets/     Loaders (.pkl/.txt/HF) and benchmark shortcuts
    detectors/    Registered contamination detectors
    pipelines/    detect / auc / finetune-probe orchestrators
    cli/          typer entry points
tests/            pytest suite (no GPU required)
experiments/      Reproduction scripts
docs/             METHOD.md — formal algorithm reference
```

## Development

```bash
pytest                  # tests (no GPU needed)
ruff check .            # lint
ruff format .           # format
mypy src                # type check
pre-commit run --all    # everything
```

CI runs all of the above on push.

## References

* Zawalski et al. 2025. *Detecting Data Contamination in LLMs via In-Context Learning.*  arXiv:2510.27055
* Maini et al. 2024. *LLM Dataset Inference: Did you train on my dataset?* NeurIPS 2024
* Fu et al. 2024. *Does Data Contamination Detection Work (Well) for LLMs?*
* Shi et al. 2024. *Detecting Pretraining Data from Large Language Models.*
* Zhang et al. 2021. *Counterfactual Memorization in Neural Language Models.*
* Carlini et al. 2022. *Extracting Training Data from Large Language Models.*

See `docs/METHOD.md` for the formal algorithm reference.
