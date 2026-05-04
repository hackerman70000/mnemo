from mnemo.core.detector import Detector
from mnemo.detectors.codec import CoDeC
from mnemo.detectors.max_k import MaxKProb
from mnemo.detectors.min_k import MinKProb
from mnemo.detectors.perplexity import Perplexity
from mnemo.detectors.vanilla_loss import VanillaLoss
from mnemo.detectors.zlib_ratio import ZlibRatio

DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    CoDeC.name: CoDeC,
    VanillaLoss.name: VanillaLoss,
    Perplexity.name: Perplexity,
    MinKProb.name: MinKProb,
    MaxKProb.name: MaxKProb,
    ZlibRatio.name: ZlibRatio,
}

__all__ = [
    "DETECTOR_REGISTRY",
    "CoDeC",
    "MaxKProb",
    "MinKProb",
    "Perplexity",
    "VanillaLoss",
    "ZlibRatio",
]
