"""
core/population_init.py

Builds the initial population for an optimizer run, mixing:
  - genomes extracted from Instant-NGP's shipped presets (base, big,
    small, etc. -- whatever core/presets.py finds)
  - random genomes, filling the rest of pop_size

This is deliberately a MIX, not "seed everything from presets" --
if the whole population starts clustered near a few known-good points,
GA/DE/PSO would just be locally refining those points rather than
actually searching the space, which would defeat the purpose of the
comparison. Default mix is roughly half-and-half (see build_population).

The resulting population is always stored in VALUE SPACE (raw physical
units, one vector per Param in SEARCH_SPACE order) and saved to a JSON
file, so:
  - GA can load it directly
  - DE / PSO / CMA-ES convert each vector via core.genome.to_unit()
This keeps population generation decoupled from any one optimizer, and
lets every algorithm in a given experiment run start from the EXACT
SAME initial population (important for a fair cross-algorithm and
cross-dataset comparison).
"""

import json
import os
import random
from typing import Dict, List

from .search_space import random_vector, DIM, SEARCH_SPACE
from .genome import extract_genome_from_config
from .presets import discover_presets


def build_population(base_cfg: dict, ngp_dir: str, pop_size: int,
                      preset_fraction: float = 0.5, seed: int = 1337) -> List[list]:
    """
    Returns a list of `pop_size` value-space genomes (List[Vector]).

    preset_fraction: target fraction of the population seeded from
        presets (base.json + whatever else core.presets finds). Actual
        count is min(round(pop_size * preset_fraction), number of
        presets available) -- if only 1-2 presets exist, you won't get
        more than that many preset-seeded individuals no matter the
        fraction requested. The rest of the population is random.

    base_cfg is included as a preset source too (via its own values),
    in case it differs from whatever's on disk at configs/nerf/base.json.
    """
    rng = random.Random(seed)

    presets = discover_presets(ngp_dir)
    preset_genomes = {name: extract_genome_from_config(cfg) for name, cfg in presets.items()}
    # Always include the actual base_cfg being used for this run, even if
    # it wasn't found by discover_presets (e.g. a custom path was passed).
    preset_genomes.setdefault("__base_cfg__", extract_genome_from_config(base_cfg))

    preset_list = list(preset_genomes.items())
    rng.shuffle(preset_list)

    n_preset = min(round(pop_size * preset_fraction), len(preset_list))
    n_random = pop_size - n_preset

    population = []
    sources = []  # parallel list: which preset (or "random") each individual came from, for the manifest

    for name, genome in preset_list[:n_preset]:
        population.append(genome)
        sources.append(name)

    for _ in range(n_random):
        population.append(random_vector())
        sources.append("random")

    # If fewer presets existed than requested and pop_size still isn't met
    # (shouldn't normally happen given n_random fills the rest, but guard
    # against a pathological preset_fraction > 1 or empty preset_list):
    while len(population) < pop_size:
        population.append(random_vector())
        sources.append("random")

    return population, sources


def save_population(path: str, population: List[list], sources: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "dim": DIM,
        "param_names": [p.name for p in SEARCH_SPACE],
        "population": population,
        "sources": sources,  # which preset (or "random") produced each individual, for traceability
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_population(path: str) -> List[list]:
    with open(path) as f:
        payload = json.load(f)
    if payload["dim"] != DIM:
        raise ValueError(
            f"Population file {path} was built for {payload['dim']} dimensions, "
            f"but the current search space has {DIM}. Regenerate the population."
        )
    return payload["population"]


def load_population_with_sources(path: str):
    """Like load_population(), but also returns the parallel `sources` list
    (preset name, or "random" for randomly-filled individuals) -- lets
    callers distinguish known-good preset-derived genomes from noise."""
    with open(path) as f:
        payload = json.load(f)
    if payload["dim"] != DIM:
        raise ValueError(
            f"Population file {path} was built for {payload['dim']} dimensions, "
            f"but the current search space has {DIM}. Regenerate the population."
        )
    return payload["population"], payload["sources"]


def generate_and_save(base_cfg: dict, ngp_dir: str, out_path: str, pop_size: int,
                       preset_fraction: float = 0.5, seed: int = 1337) -> List[list]:
    """Convenience wrapper: build + save + return the population in one call."""
    population, sources = build_population(base_cfg, ngp_dir, pop_size, preset_fraction, seed)
    save_population(out_path, population, sources)
    return population
