"""Binary-to-amplitude conversion of a vector by a tree of CORDIC vectorings."""
from .tree import (Tree, TreeCircuit, SymbolCircuit, exact_unit,
                   default_guard, range_bits)

__all__ = ["Tree", "TreeCircuit", "SymbolCircuit", "exact_unit",
           "default_guard", "range_bits"]
