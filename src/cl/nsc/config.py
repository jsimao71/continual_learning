"""Namespaced, opt-in configuration for structural selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class TraceConfig:
    enabled: bool = False
    level: str = "candidate"
    retain_full_attention: bool = False


@dataclass(frozen=True)
class GraphConfig:
    aggregation: str = "layer_sum"
    threshold: float | None = None
    community: str = "canonical_components"


@dataclass(frozen=True)
class SelectorConfig:
    mode: str = "base_topk"
    gamma: float = 1.0
    bridge_budget_fraction: float = 0.25


@dataclass(frozen=True)
class FeatureConfig:
    entropy: bool = True
    persistence: bool = True
    agreement: bool = True
    community: bool = True
    bridge: bool = True


@dataclass(frozen=True)
class NSCConfig:
    enabled: bool = False
    trace: TraceConfig = field(default_factory=TraceConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def as_dict(self) -> dict:
        return asdict(self)
