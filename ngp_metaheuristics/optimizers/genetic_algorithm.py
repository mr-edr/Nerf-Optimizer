"""
optimizers/genetic_algorithm.py

GA operating in VALUE SPACE (raw physical units), using the shared
core.search_space / core.genome / core.fitness modules.

Selection: elitism (top ~25%) + tournament-free random-parent reproduction,
matching the structure of your earlier v3/v4 scripts but now over the
full combined genome (encoding + network + optimizer together).
"""

import argparse
import csv
import os
import random
import json
from copy import deepcopy

from core.search_space import SEARCH_SPACE, random_vector, sanitize
from core.genome import apply_genome, describe
from core.fitness import evaluate


def mutate(vector, rate=0.3):
    v = list(vector)
    for i, param in enumerate(SEARCH_SPACE):
        if random.random() >= rate:
            continue
        if param.kind == "categorical":
            v[i] = random.choice(param.choices)
        elif param.log_scale:
            import math
            log_val = math.log10(v[i])
            log_val += random.uniform(-0.3, 0.3)
            v[i] = 10 ** log_val
        elif param.kind == "int":
            v[i] = v[i] + random.choice([-1, 1])
        else:
            span = param.hi - param.lo
            v[i] = v[i] + random.uniform(-0.15, 0.15) * span
    return sanitize(v)


def crossover(a, b):
    """Uniform crossover: each gene independently from either parent."""
    child = [a[i] if random.random() < 0.5 else b[i] for i in range(len(a))]
    return sanitize(child)


def run(args):
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.base) as f:
        base_cfg = json.load(f)

    population = [random_vector() for _ in range(args.pop_size)]
    best_fitness = -1e9
    best_vector = None

    log_path = os.path.join(args.out_dir, "ga_log.csv")
    with open(log_path, "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["generation", "individual", "fitness", "ok", "failure_reason",
                          "peak_vram_mb"] + [p.name for p in SEARCH_SPACE])

        for gen in range(1, args.generations + 1):
            print(f"\n=== [GA] Generation {gen}/{args.generations} ===")
            scored = []

            for i, vector in enumerate(population):
                cfg = apply_genome(base_cfg, vector)
                result = evaluate(cfg, args.scene, args.out_dir, args)

                print(f"  gen {gen:02d} ind {i:02d} | psnr={result.psnr:6.3f} | ok={result.ok} | {result.elapsed_s:5.1f}s")

                writer.writerow([gen, i, result.psnr, result.ok, result.failure_reason,
                                  result.peak_vram_mb] + list(vector))
                logfile.flush()

                if not result.ok:
                    print(f"    ^ FAILED ({result.failure_reason}) -- last 20 lines of output:")
                    tail = "\n".join(result.raw_stdout.strip().splitlines()[-20:])
                    print("    " + tail.replace("\n", "\n    "))

                scored.append((result.psnr, vector))

                if result.ok and result.psnr > best_fitness:
                    best_fitness = result.psnr
                    best_vector = vector
                    with open(os.path.join(args.out_dir, "best_ga_config.json"), "w") as bf:
                        json.dump(apply_genome(base_cfg, vector), bf, indent=2)
                    with open(os.path.join(args.out_dir, "best_ga_genome.json"), "w") as bf:
                        json.dump({p.name: v for p, v in zip(SEARCH_SPACE, vector)}, bf, indent=2)

            # --- Selection: keep top 25% as elites ---
            scored.sort(key=lambda x: x[0], reverse=True)
            n_elite = max(1, args.pop_size // 4)
            elites = [deepcopy(v) for _, v in scored[:n_elite]]

            # --- Reproduce next generation ---
            next_pop = elites.copy()
            while len(next_pop) < args.pop_size:
                if len(elites) >= 2 and random.random() < 0.7:
                    a, b = random.sample(elites, 2)
                    child = mutate(crossover(a, b), args.mutation_rate)
                else:
                    parent = random.choice(elites)
                    child = mutate(parent, args.mutation_rate)
                next_pop.append(child)

            population = next_pop

    print("\n=== GA finished ===")
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
    p.add_argument("--mutation_rate", type=float, default=0.3)

    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--eval_width", type=int, default=64)
    p.add_argument("--eval_height", type=int, default=64)
    p.add_argument("--eval_spp", type=int, default=1)
    p.add_argument("--eval_timeout", type=int, default=1200)
    p.add_argument("--frame_idx", type=int, default=0)

    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
