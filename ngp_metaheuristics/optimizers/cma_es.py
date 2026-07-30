"""
optimizers/cma_es.py

CMA-ES (Covariance Matrix Adaptation Evolution Strategy), via the `cma`
PyPI package. Operates in UNIT SPACE like DE/PSO, with hard bounds
[0,1]^DIM enforced through cma's own boundary handling.

Why CMA-ES is the "serious" algorithm to test here: unlike DE/PSO, it
adapts a full covariance matrix over generations, learning which
directions in the search space keep producing improvements and
expanding step size along them (via its evolution path mechanism).
On a landscape with sparse, narrow good regions -- which your GA/DE/PSO/
random-search runs suggest this one has -- this adaptive step-size
expansion is exactly the property that could let it consistently find
(and stay near) the good regions, rather than requiring luck.

Population size: cma has its own default formula (~4 + 3*ln(DIM)), but
we override it via --pop_size to stay comparable to your other runs
(8 candidates/generation, matching GA/DE/PSO).

Install: pip install --break-system-packages cma
"""

import argparse
import csv
import json
import os

import numpy as np

try:
    import cma
except ImportError:
    raise SystemExit(
        "The 'cma' package is required for this optimizer.\n"
        "Install it with: pip install --break-system-packages cma"
    )

from core.search_space import SEARCH_SPACE, DIM
from core.genome import from_unit, describe, apply_genome
from core.fitness import evaluate


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.base) as f:
        base_cfg = json.load(f)

    best_fitness = -1e9
    best_vector = None
    sample_counter = 0

    log_path = os.path.join(args.out_dir, "cma_log.csv")
    with open(log_path, "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["generation", "individual", "fitness", "ok", "failure_reason",
                          "peak_vram_mb", "elapsed_s"] + [p.name for p in SEARCH_SPACE])

        # Start at the center of unit space (0.5, ..., 0.5) with initial
        # sigma covering a good chunk of the space -- 0.3 is a common
        # default for unit-box CMA-ES (sigma is roughly "typical step size").
        x0 = [0.5] * DIM
        sigma0 = 0.3

        opts = {
            "bounds": [0.0, 1.0],
            "popsize": args.pop_size,
            "maxiter": args.generations,
            "seed": args.seed,
            "verbose": -9,  # suppress cma's own console spam; we print our own concise lines
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        gen = 0
        while not es.stop():
            gen += 1
            if gen > args.generations:
                break

            print(f"\n=== [CMA-ES] Generation {gen}/{args.generations} ===")
            candidates = es.ask()  # list of unit-space vectors (may go slightly out of bounds pre-clip)
            fitnesses = []

            for i, unit_vec in enumerate(candidates):
                value_vec = from_unit(list(unit_vec))  # from_unit() clips+sanitizes internally
                cfg = apply_genome(base_cfg, value_vec)
                result = evaluate(cfg, args.scene, args.out_dir, args)

                print(f"  gen {gen:02d} ind {i:02d} | psnr={result.psnr:6.3f} | ok={result.ok} | {result.elapsed_s:5.1f}s")

                writer.writerow([gen, i, result.psnr, result.ok, result.failure_reason,
                                  result.peak_vram_mb, result.elapsed_s] + list(value_vec))
                logfile.flush()

                # CMA-ES MINIMIZES by convention -- negate PSNR-based fitness
                fitnesses.append(-result.psnr)

                if result.ok and result.psnr > best_fitness:
                    best_fitness = result.psnr
                    best_vector = value_vec
                    with open(os.path.join(args.out_dir, "best_cma_config.json"), "w") as bf:
                        json.dump(cfg, bf, indent=2)
                    with open(os.path.join(args.out_dir, "best_cma_genome.json"), "w") as bf:
                        json.dump({p.name: v for p, v in zip(SEARCH_SPACE, value_vec)}, bf, indent=2)

                sample_counter += 1

            es.tell(candidates, fitnesses)

    print("\n=== CMA-ES finished ===")
    print("Best fitness (PSNR-based):", best_fitness)
    if best_vector:
        print(describe(best_vector))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ngp_dir", default="../instant-ngp", help="path to instant-ngp checkout root")
    p.add_argument("--scene", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--pop_size", type=int, default=8)
    p.add_argument("--generations", type=int, default=8)

    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--eval_width", type=int, default=32)
    p.add_argument("--eval_height", type=int, default=32)
    p.add_argument("--eval_spp", type=int, default=1)
    p.add_argument("--eval_timeout", type=int, default=1200)
    p.add_argument("--frame_idx", type=int, default=0)

    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
