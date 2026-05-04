from mnemo.core.detector import Detector
from mnemo.core.result import DatasetResult
from mnemo.core.scoring import dataset_level_auc
from mnemo.detectors import (
    DETECTOR_REGISTRY,
    CoDeC,
    MaxKProb,
    MinKProb,
    Perplexity,
    PerturbationLoss,
    ReferenceLoss,
    VanillaLoss,
    ZlibRatio,
)
from mnemo.models.hf import HFModel
from mnemo.pipelines.auc_eval import AUCReport, evaluate_auc
from mnemo.pipelines.dataset_inference import DatasetInferenceResult, dataset_inference
from mnemo.pipelines.detection import detect_contamination
from mnemo.pipelines.finetune_probe import FinetuneProbeResult, finetune_probe

__version__ = "0.1.0"

__all__ = [
    "DETECTOR_REGISTRY",
    "AUCReport",
    "CoDeC",
    "DatasetInferenceResult",
    "DatasetResult",
    "Detector",
    "FinetuneProbeResult",
    "HFModel",
    "MaxKProb",
    "MinKProb",
    "Perplexity",
    "PerturbationLoss",
    "ReferenceLoss",
    "VanillaLoss",
    "ZlibRatio",
    "dataset_inference",
    "dataset_level_auc",
    "detect_contamination",
    "evaluate_auc",
    "finetune_probe",
]
