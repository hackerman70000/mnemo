from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from mnemo.datasets.benchmarks import BENCHMARKS
from mnemo.datasets.loaders import load_local
from mnemo.detectors import DETECTOR_REGISTRY
from mnemo.detectors.perturbation_loss import (
    PerturbationLoss,
    butter_fingers,
    change_char_case,
    random_word_drop,
    synonym_substitution,
    underscore_trick,
    whitespace_perturbation,
)
from mnemo.models.hf import HFModel
from mnemo.pipelines.auc_eval import evaluate_auc
from mnemo.pipelines.dataset_inference import dataset_inference
from mnemo.pipelines.detection import detect_contamination

app = typer.Typer(help="mnemo — LLM training data contamination detection.", no_args_is_help=True)

_PERTURBATION_MAP: dict[str, object | None] = {
    "swap": None,  # adjacent_word_swap is default, no key needed
    "drop": random_word_drop,
    "butter_fingers": butter_fingers,
    "synonym_substitution": synonym_substitution,
    "change_char_case": change_char_case,
    "whitespace_perturbation": whitespace_perturbation,
    "underscore_trick": underscore_trick,
}

# Non-registry detectors that can be constructed via CLI with parameters
_EXTRA_DETECTOR_CLASSES: dict[str, type[object]] = {
    "perturbation_loss": PerturbationLoss,
}


def _coerce_value(v: str) -> object:
    v_lower = v.lower()
    if v_lower == "true":
        return True
    if v_lower == "false":
        return False
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _resolve_perturbation_fn(val: str) -> tuple[str, object] | None:
    """Map CLI ``fn=...`` to ``(param_name, callable)`` or None to skip."""
    if val not in _PERTURBATION_MAP:
        raise typer.BadParameter(
            f"Unknown perturbation fn '{val}'. Available: {', '.join(_PERTURBATION_MAP)}"
        )
    fn = _PERTURBATION_MAP[val]
    if fn is None:
        return None  # use dataclass default
    return "perturbation_fn", fn


def _build_detector(spec: str) -> object:
    """Instantiate a detector from a registry name, optionally with kwargs.

    Syntax: ``name`` or ``name:key=val,key2=val2``.

    Examples
    --------
    * ``codec``
    * ``min_k_prob:k_percent=5``
    * ``perturbation_loss:mode=ratio,fn=butter_fingers``
    """
    if ":" in spec:
        name_part, param_part = spec.split(":", 1)
    else:
        name_part, param_part = spec, ""

    name = name_part.strip()
    cls = DETECTOR_REGISTRY.get(name) or _EXTRA_DETECTOR_CLASSES.get(name)
    if cls is None:
        raise typer.BadParameter(
            f"Unknown detector '{name}'. Available: {', '.join(DETECTOR_REGISTRY)}"
        )

    kwargs: dict[str, object] = {}
    for pair in param_part.split(","):
        if not pair:
            continue
        if "=" not in pair:
            raise typer.BadParameter(f"Invalid parameter '{pair}' in detector spec '{spec}'")
        k, v_raw = pair.split("=", 1)
        k, v_raw = k.strip(), v_raw.strip()

        val = _coerce_value(v_raw)

        if name == "perturbation_loss" and k == "fn" and isinstance(val, str):
            resolved = _resolve_perturbation_fn(val)
            if resolved is None:
                continue
            k, val = resolved

        kwargs[k] = val

    return cls(**kwargs)


def _resolve_dataset(spec: str) -> list[str]:
    if spec in BENCHMARKS:
        return BENCHMARKS[spec]()
    path = Path(spec)
    if path.exists():
        return load_local(path)
    raise typer.BadParameter(
        f"Dataset '{spec}' not found. Use a file path or one of: {', '.join(BENCHMARKS)}"
    )


@app.command()
def detect(
    model_name: str = typer.Argument(..., help="HuggingFace model identifier."),
    dataset: str = typer.Argument(..., help="Path to .pkl/.txt file, or benchmark name."),
    detector: str = typer.Option("codec", help=f"One of: {', '.join(DETECTOR_REGISTRY)}"),
    num_context: int = typer.Option(1, "--num-context", "-n"),
    max_samples: int = typer.Option(1000, "--max-samples", "-m"),
    device: str = typer.Option("auto"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Save full result as JSON."),
) -> None:
    """Run a contamination detector on a model and dataset."""
    det = _build_detector(detector)
    samples = _resolve_dataset(dataset)
    model = HFModel(model_name, device=device)

    name = Path(dataset).stem if Path(dataset).exists() else dataset
    result = detect_contamination(
        det,  # type: ignore[arg-type]
        model,
        samples,
        dataset_name=name,
        num_context_examples=num_context,
        max_samples=max_samples,
    )

    msg = (
        f"\n{detector} score on {result.dataset_name}: "
        f"{result.score:.4f} ({result.n_samples} samples)"
    )
    typer.secho(msg, fg=typer.colors.GREEN)
    if output:
        result.save_json(output)
        logger.info(f"Saved result to {output}")


@app.command()
def auc(
    model_name: str = typer.Argument(...),
    seen: list[str] = typer.Option(..., "--seen", help="Seen datasets (path or benchmark)."),
    unseen: list[str] = typer.Option(..., "--unseen", help="Unseen datasets (path or benchmark)."),
    detector: str = typer.Option("codec"),
    max_samples: int = typer.Option(500, "--max-samples", "-m"),
    device: str = typer.Option("auto"),
) -> None:
    """Compute dataset-level AUC across seen vs unseen sets (paper Tab. 1 style)."""
    det = _build_detector(detector)
    model = HFModel(model_name, device=device)

    seen_map = {s: _resolve_dataset(s) for s in seen}
    unseen_map = {u: _resolve_dataset(u) for u in unseen}

    report = evaluate_auc(det, model, seen_map, unseen_map, max_samples=max_samples)  # type: ignore[arg-type]
    typer.echo(f"\n{detector} on {model_name}:")
    typer.echo(f"  AUC = {report.auc:.4f}")
    typer.echo("  Seen:")
    for n, r in report.seen.items():
        typer.echo(f"    {n}: {r.score:.4f}")
    typer.echo("  Unseen:")
    for n, r in report.unseen.items():
        typer.echo(f"    {n}: {r.score:.4f}")


@app.command()
def di(
    model_name: str = typer.Argument(..., help="HuggingFace model identifier."),
    suspect: str = typer.Option(..., "--suspect", help="Suspect dataset (path or benchmark)."),
    validation: str = typer.Option(..., "--validation", help="Validation dataset (held-out IID)."),
    detectors: list[str] = typer.Option(
        ["vanilla_loss", "perplexity", "min_k_prob", "max_k_prob", "zlib_ratio"],
        "--detector",
        "-d",
        help="Detectors to aggregate. Pass --detector multiple times. "
        "Syntax: name or name:k=v,k2=v2 (e.g. min_k_prob:k_percent=5).",
    ),
    n_seeds: int = typer.Option(10, "--seeds"),
    holdout: int = typer.Option(1000, "--holdout"),
    threshold: float = typer.Option(0.1, "--threshold"),
    normalize_mode: str = typer.Option(
        "train",
        "--normalize-mode",
        help="Feature z-score normalisation: 'train' (suspect-only stats, paper default) "
        "or 'combined' (suspect+validation stats).",
    ),
    device: str = typer.Option("auto"),
) -> None:
    """Maini-style dataset inference: was the suspect set used for training?"""
    sus = _resolve_dataset(suspect)
    val = _resolve_dataset(validation)
    dets = [_build_detector(d) for d in detectors]
    model = HFModel(model_name, device=device)

    result = dataset_inference(
        model,
        sus,
        val,
        dets,  # type: ignore[arg-type]
        n_seeds=n_seeds,
        holdout_size=holdout,
        threshold=threshold,
        normalize_mode=normalize_mode,
    )
    verdict = "TRAINED" if result.trained else "NOT trained"
    typer.secho(
        f"\n{model_name} on {suspect}: {verdict} (p_combined = {result.p_combined:.4g})",
        fg=typer.colors.RED if result.trained else typer.colors.GREEN,
    )
    typer.echo(f"  per-seed p-values: {[f'{p:.3g}' for p in result.p_values]}")
    typer.echo("  detector weights:")
    for name, w in sorted(result.feature_weights.items(), key=lambda x: -abs(x[1])):
        typer.echo(f"    {name}: {w:+.4f}")


@app.command(name="list-detectors")
def list_detectors() -> None:
    """List available contamination detectors."""
    for name in DETECTOR_REGISTRY:
        typer.echo(f"- {name}")


@app.command(name="list-benchmarks")
def list_benchmarks() -> None:
    """List built-in benchmark dataset loaders."""
    for name in BENCHMARKS:
        typer.echo(f"- {name}")


if __name__ == "__main__":
    app()
