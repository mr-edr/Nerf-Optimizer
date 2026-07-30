"""
core/search_space.py

Defines the shared hyperparameter search space for Instant-NGP optimization.
Every optimizer (GA, DE, PSO, CMA-ES, SA) operates on the SAME flat
real-valued vector representation, so results are comparable across
algorithms. This module is the single source of truth for:

  - which parameters are tunable
  - their bounds
  - whether they're continuous, integer, or categorical
  - how to clamp/repair a raw vector back into a valid, in-bounds genome

Design choice: everything is stored internally as a flat vector of floats
in [0, 1] normalized space is NOT used here -- we keep raw physical units
(e.g. learning_rate in actual float, n_levels in actual int) because it's
easier to reason about and debug. Each optimizer's mutation/crossover
operators are expected to work in this raw space and call `sanitize()`
after every perturbation.

If you later want normalized [0,1] search (recommended for PSO/CMA-ES,
which assume roughly comparable scales per-dimension), see the
`to_unit()` / `from_unit()` helpers at the bottom -- they're optional
and each optimizer can choose whether to use raw or unit space.
"""

from dataclasses import dataclass
from typing import Any, List, Union
import math
import random


@dataclass
class Param:
    """
    Describes one tunable hyperparameter.

    kind: "float"       -- continuous, optionally log-scaled
          "int"         -- integer within [lo, hi]
          "categorical" -- one of a fixed discrete set of values
    log_scale: if True, sample/mutate in log-space (good for learning
               rates, l2_reg, anything spanning multiple orders of
               magnitude).
    path: where this parameter lives inside the Instant-NGP config dict,
          expressed as a list of keys, e.g. ["encoding", "n_levels"].
          See genome.py:apply_genome() for how this is used.
    """
    name: str
    kind: str                       # "float" | "int" | "categorical"
    path: List[str]
    lo: float = None
    hi: float = None
    choices: List[Any] = None
    log_scale: bool = False

    def sample(self) -> Union[float, int]:
        if self.kind == "categorical":
            return random.choice(self.choices)
        if self.kind == "int":
            return random.randint(int(self.lo), int(self.hi))
        # float
        if self.log_scale:
            lo_l, hi_l = math.log10(self.lo), math.log10(self.hi)
            return 10 ** random.uniform(lo_l, hi_l)
        return random.uniform(self.lo, self.hi)

    def clamp(self, value):
        if self.kind == "categorical":
            if value in self.choices:
                return value
            # snap to nearest by index if numeric-ish, else default to first choice
            try:
                return min(self.choices, key=lambda c: abs(c - value))
            except TypeError:
                return self.choices[0]
        if self.kind == "int":
            return int(max(self.lo, min(self.hi, round(value))))
        return float(max(self.lo, min(self.hi, value)))


# ----------------------------------------------------------------------
# THE SEARCH SPACE
#
# Combines encoding params (biggest lever), a couple of network params,
# and optimizer params, based on what we identified as actually
# impactful for Instant-NGP. Extend this list as you validate more
# parameters -- keep aabb_scale and other scene-correctness params OUT
# of the GA search space; those should be fixed correctly per-dataset,
# not evolved (a wrong aabb_scale ruins every candidate equally, it's
# not a quality/efficiency tradeoff knob).
# ----------------------------------------------------------------------

SEARCH_SPACE: List[Param] = [
    # --- Encoding (HashGrid) — biggest lever for quality/VRAM tradeoff ---
    Param("n_levels", "int", ["encoding", "n_levels"], lo=8, hi=18),
    Param("n_features_per_level", "categorical", ["encoding", "n_features_per_level"],
          choices=[2, 4, 8]),
    Param("log2_hashmap_size", "int", ["encoding", "log2_hashmap_size"], lo=15, hi=22),
    Param("base_resolution", "int", ["encoding", "base_resolution"], lo=4, hi=32),
    Param("per_level_scale", "float", ["encoding", "per_level_scale"],
          lo=1.26, hi=2.0),

    # --- Network (density MLP) ---
    # NOTE: fully-fused MLP kernels require n_neurons in {16, 32, 64, 128}.
    Param("n_neurons", "categorical", ["network", "n_neurons"],
          choices=[16, 32, 64, 128]),
    Param("n_hidden_layers", "int", ["network", "n_hidden_layers"], lo=1, hi=3),

    # --- Optimizer (Adam) ---
    Param("learning_rate", "float", ["optimizer", "nested", "nested", "learning_rate"],
          lo=1e-4, hi=5e-2, log_scale=True),
    Param("l2_reg", "float", ["optimizer", "nested", "nested", "l2_reg"],
          lo=1e-8, hi=1e-4, log_scale=True),
]

DIM = len(SEARCH_SPACE)


def random_vector() -> List[Union[float, int]]:
    """A fresh random genome, one value per Param in SEARCH_SPACE order."""
    return [p.sample() for p in SEARCH_SPACE]


def sanitize(vector: List[Union[float, int]]) -> List[Union[float, int]]:
    """Clamp every gene in `vector` into its valid range/choice set."""
    return [p.clamp(v) for p, v in zip(SEARCH_SPACE, vector)]


def bounds_array():
    """
    Returns (lo, hi) arrays for numeric params -- convenient for
    optimizers like PSO/DE/CMA-ES that want numpy-style bounds.
    Categorical params are treated as an index range [0, len(choices)-1];
    genome.py handles mapping the index back to the actual choice value.
    """
    lo, hi = [], []
    for p in SEARCH_SPACE:
        if p.kind == "categorical":
            lo.append(0)
            hi.append(len(p.choices) - 1)
        else:
            lo.append(p.lo)
            hi.append(p.hi)
    return lo, hi
