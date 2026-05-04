from mnemo.core.detector import Detector
from mnemo.detectors.codec import CoDeC
from mnemo.detectors.min_k import MinKProb
from mnemo.detectors.vanilla_loss import VanillaLoss
from mnemo.detectors.zlib_ratio import ZlibRatio

DETECTOR_REGISTRY: dict[str, type[Detector]] = {
    CoDeC.name: CoDeC,
    VanillaLoss.name: VanillaLoss,
    MinKProb.name: MinKProb,
    ZlibRatio.name: ZlibRatio,
}

__all__ = ["DETECTOR_REGISTRY", "CoDeC", "MinKProb", "VanillaLoss", "ZlibRatio"]
