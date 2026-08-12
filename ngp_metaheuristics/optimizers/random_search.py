"""
optimizers/random_search.py

Not a metaheuristic -- a control baseline. Samples N random genomes
(same total eval count as your GA/DE/PSO runs) with zero search logic:
no selection, no mutation, no memory of past results. Every sample is
independent.

Why this matters: if random search finds PSNR spikes just as often as
GA/DE do, that's strong evidence the "good" configs those algorithms
found were mostly a product of evaluation noise / a sparse landscape,
not genuine intelligent search. If random search clearly underperforms
GA/DE, that's evidence the algorithms' selection pressure is doing real
work. Run this with n_samples equal to GA/DE/PSO's total eval count
(e.g. 64-72) for a fair comparison.
"""

import argparse
import csv
import json
import os
import random

from core.search_space import SEARCH_SPACE, random_vector
from core.genome import apply_genome, describe
from core.fitness import evaluate
from core.population_init import load_population


def run(args):
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.base) as f:
        base_cfg = json.load(f)

    best_fitness = -1e9
    best_vector = None

    log_path = os.path.join(args.out_dir, "random_search_log.csv")
    with open(log_path, "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["sample", "fitness", "ok", "failure_reason",
                          "peak_vram_mb", "elapsed_s"] + [p.name for p in SEARCH_SPACE])

        seed_pop = []
        if getattr(args, "init_population", None):
            seed_pop = load_population(args.init_population)[:args.n_samples]

        for i in range(args.n_samples):
            vector = seed_pop[i] if i < len(seed_pop) else random_vector()
            cfg = apply_genome(base_cfg, vector)
            result = evaluate(cfg, args.scene, args.out_dir, args)

            print(f"  sample {i:03d} | psnr={result.psnr:6.3f} | ok={result.ok} | {result.elapsed_s:5.1f}s")

            writer.writerow([i, result.psnr, result.ok, result.failure_reason,
                              result.peak_vram_mb, result.elapsed_s] + list(vector))
            logfile.flush()

            if result.ok and result.psnr > best_fitness:
                best_fitness = result.psnr
                best_vector = vector
                with open(os.path.join(args.out_dir, "best_random_config.json"), "w") as bf:
                    json.dump(cfg, bf, indent=2)
                with open(os.path.join(args.out_dir, "best_random_genome.json"), "w") as bf:
                    json.dump({p.name: v for p, v in zip(SEARCH_SPACE, vector)}, bf, indent=2)

    print("\n=== Random search finished ===")
    print("Best fitness (PSNR-based):", best_fitness)
    if best_vector:
        print(describe(best_vector))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ngp_dir", default="../instant-ngp", help="path to instant-ngp checkout root")
    p.add_argument("--scene", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--n_samples", type=int, default=72,
                    help="total evals -- match GA/DE/PSO's total eval count for a fair comparison")

    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--eval_width", type=int, default=32)
    p.add_argument("--eval_height", type=int, default=32)
    p.add_argument("--eval_spp", type=int, default=1)
    p.add_argument("--eval_timeout", type=int, default=1200)
    p.add_argument("--frame_idx", type=int, default=0)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--init_population", default=None,
                    help="path to a population JSON built by core/population_init.py; these are evaluated first, counted toward n_samples")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
