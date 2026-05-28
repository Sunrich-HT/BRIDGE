"""BRIDGE: Triangular Fixed-Point Refinement for Long-Horizon Persona Consistency (ICML 2026)."""

from .config import BridgeConfig
from .triangular import (
    TriangularFixedPointRefinement,
    PairwiseCrossAttention,
)
from .dual_system import DualSystemProcessor
from .memory import HierarchicalMemory, MemorySnapshot
from .losses import (
    cyclic_coherence_loss,
    persona_consistency_loss,
    bridge_total_loss,
    PersonaConsistencyClassifier,
)
from .modeling_bridge import BridgeModel, BridgeForCausalLM

__version__ = "0.1.0"

__all__ = [
    "BridgeConfig",
    "TriangularFixedPointRefinement",
    "PairwiseCrossAttention",
    "DualSystemProcessor",
    "HierarchicalMemory",
    "MemorySnapshot",
    "PersonaConsistencyClassifier",
    "cyclic_coherence_loss",
    "persona_consistency_loss",
    "bridge_total_loss",
    "BridgeModel",
    "BridgeForCausalLM",
]
