from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from mnemo.cli.main import _build_detector, app

runner = CliRunner()


# ---------------------------------------------------------------------------
# _build_detector parsing
# ---------------------------------------------------------------------------


def test_build_detector_plain_name():
    det = _build_detector("vanilla_loss")
    assert det.name == "vanilla_loss"


def test_build_detector_with_int_param():
    det = _build_detector("min_k_prob:k_percent=5")
    assert det.name == "min_k_prob"
    assert det.k_percent == 5


def test_build_detector_with_float_param():
    det = _build_detector("min_k_prob:k_percent=12.5")
    assert det.k_percent == 12.5


def test_build_detector_unknown_name_raises():
    with pytest.raises(typer.BadParameter):
        _build_detector("not_a_detector")


def test_build_detector_invalid_param_format_raises():
    with pytest.raises(typer.BadParameter):
        _build_detector("min_k_prob:badparam")


def test_build_detector_perturbation_with_fn():
    det = _build_detector("perturbation_loss:mode=ratio,fn=butter_fingers")
    assert det.name == "perturbation_loss"
    assert det.mode == "ratio"


def test_build_detector_perturbation_default_swap_fn():
    det = _build_detector("perturbation_loss:mode=diff")
    assert det.name == "perturbation_loss"
    assert det.mode == "diff"


def test_build_detector_perturbation_unknown_fn_raises():
    with pytest.raises(typer.BadParameter):
        _build_detector("perturbation_loss:fn=nonexistent")


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_list_detectors_runs():
    result = runner.invoke(app, ["list-detectors"])
    assert result.exit_code == 0
    assert "vanilla_loss" in result.stdout


def test_list_benchmarks_runs():
    result = runner.invoke(app, ["list-benchmarks"])
    assert result.exit_code == 0


def test_di_help_runs():
    result = runner.invoke(app, ["di", "--help"])
    assert result.exit_code == 0
    assert "--normalize-mode" in result.stdout
