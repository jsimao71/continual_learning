"""Fixed-weight micro-Transformer witnesses for constructive representability."""

from .m0_grandparent import build_grandparent_variants
from .m0_lookup import build_lookup_variants
from .m0_successor import build_successor_variants

__all__ = ["build_successor_variants", "build_lookup_variants", "build_grandparent_variants"]
