"""Lazy exports for bottleneck evidence providers and plateau detection.

Do not require GRAPHyco to import BottleneckDetector or non-graph methods.
The public attribute names and their underlying implementations are unchanged.
"""
from importlib import import_module

__all__ = [
    "BottleneckDetector",
    "DiagnosticEvent",
    "BaseEvidenceProvider",
    "V0WeightEvidence",
    "V1GradientEvidence",
    "V2GraphEvidence",
]

_EXPORTS = {
    "BottleneckDetector": ".bottleneck_detector",
    "DiagnosticEvent": ".bottleneck_detector",
    "BaseEvidenceProvider": ".evidence_base",
    "V0WeightEvidence": ".v0_weight_evidence",
    "V1GradientEvidence": ".v1_gradient_evidence",
    "V2GraphEvidence": ".v2_graph_evidence",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module(_EXPORTS[name], __name__), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
