"""
core/genome.py

Bridges between:
  1. "value space"  -- a list of real hyperparameter values, one per
     Param in SEARCH_SPACE, e.g. [14, 4, 19, 16, 1.5, 64, 2, 0.01, 1e-6]
     This is what GA (your existing style) naturally works with.

  2. "unit space"    -- every gene normalized to [0, 1], used by
     PSO / DE / CMA-ES, which assume comparable per-dimension scales.
     Without this, a dimension like log2_hashmap_size (range ~7) and
     learning_rate (range spanning 1e-4 to 5e-2) would get wildly
     different effective step sizes from the same perturbation.

  3. the actual Instant-NGP config dict (base.json + overrides), via
     apply_genome().

Every optimizer should:
  - GA: sample/mutate/crossover directly in value space (random_vector,
        sanitize from search_space.py already do this).
  - PSO/DE/CMA-ES: work entirely in unit space internally, only
        converting to value space at evaluation time via
        from_unit() + apply_genome().
  - SA: can use either; unit space is recommended for a uniform
        "step size" across dimensions.
"""

from copy import deepcopy
from typing import Any, Dict, List, Union

from .search_space import SEARCH_SPACE, DIM, sanitize


Vector = List[Union[float, int]]


def apply_genome(base_cfg: Dict[str, Any], vector: Vector) -> Dict[str, Any]:
    """
    Returns a deep copy of base_cfg with every Param in SEARCH_SPACE
    overwritten according to `vector` (value-space, already sanitized).

    Uses setdefault() at each path segment so it works whether the
    target nesting already exists in base.json or not -- but note the
    earlier bug we discussed: if base.json's real structure for a path
    (e.g. "encoding") is a Composite/nested wrapper rather than a flat
    dict, blindly setdefault-ing through it can silently produce an
    invalid config. ALWAYS diff-print the resulting cfg against a known
    -good manual base.json before trusting it, especially for the first
    run against a new dataset.
    """
    cfg = deepcopy(base_cfg)
    vector = sanitize(vector)

    for param, value in zip(SEARCH_SPACE, vector):
        _deep_set(cfg, param.path, value)

    # Encoding otype must stay explicit -- HashGrid params are meaningless
    # without it, and some base.json files omit/vary this field.
    _deep_set(cfg, ["encoding", "otype"], "HashGrid")
    if any(p.path[0] == "optimizer" for p in SEARCH_SPACE):
        _deep_set(cfg, ["optimizer", "nested", "nested", "otype"], "Adam")

    return cfg


def _deep_set(d: Dict[str, Any], path: List[str], value: Any) -> None:
    cur = d
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _deep_get(d: Dict[str, Any], path: List[str], default=None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ----------------------------------------------------------------------
# Unit-space <-> value-space conversion, for PSO / DE / CMA-ES
# ----------------------------------------------------------------------

def to_unit(vector: Vector) -> List[float]:
    """Value-space genome -> unit-space vector in [0, 1]^DIM."""
    unit = []
    for param, value in zip(SEARCH_SPACE, vector):
        if param.kind == "categorical":
            idx = param.choices.index(value) if value in param.choices else 0
            denom = max(1, len(param.choices) - 1)
            unit.append(idx / denom)
        elif param.log_scale:
            import math
            lo_l, hi_l = math.log10(param.lo), math.log10(param.hi)
            u = (math.log10(value) - lo_l) / (hi_l - lo_l)
            unit.append(min(1.0, max(0.0, u)))
        else:
            u = (value - param.lo) / (param.hi - param.lo)
            unit.append(min(1.0, max(0.0, u)))
    return unit


def from_unit(unit_vector: List[float]) -> Vector:
    """Unit-space vector in [0, 1]^DIM -> sanitized value-space genome."""
    vector = []
    for param, u in zip(SEARCH_SPACE, unit_vector):
        u = min(1.0, max(0.0, u))
        if param.kind == "categorical":
            idx = round(u * (len(param.choices) - 1))
            vector.append(param.choices[idx])
        elif param.log_scale:
            import math
            lo_l, hi_l = math.log10(param.lo), math.log10(param.hi)
            vector.append(10 ** (lo_l + u * (hi_l - lo_l)))
        elif param.kind == "int":
            vector.append(round(param.lo + u * (param.hi - param.lo)))
        else:
            vector.append(param.lo + u * (param.hi - param.lo))
    return sanitize(vector)


def random_unit_vector() -> List[float]:
    import random
    return [random.random() for _ in range(DIM)]


def describe(vector: Vector) -> str:
    """Human-readable one-liner for logging, e.g. in GA/DE/PSO print statements."""
    parts = [f"{p.name}={v}" for p, v in zip(SEARCH_SPACE, vector)]
    return " | ".join(parts)
