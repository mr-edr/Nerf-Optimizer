"""
optimizers/differential_evolution.py

Classic DE/rand/1/bin, operating in UNIT SPACE ([0,1]^DIM) so that the
same F (differential weight) produces comparable perturbation magnitude
across every dimension -- unlike value space, where a step that's tiny
for learning_rate (1e-4 to 5e-2) would be enormous for n_hidden_layers
(1 to 3).

DE tends to converge faster than vanilla GA on continuous hyperparameter
landscapes with fewer function evaluations, which matters a lot here
since every fitness eval is a real training run. Good comparison point
against your GA baseline.
"""

import argparse
import csv
import json
import os
import random

from core.search_space import SEARCH_SPACE, DIM
from core.genome import from_unit, random_unit_vector, describe, apply_genome, to_unit
from core.fitness import evaluate
from core.population_init import load_population


def run(args):
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.base) as f:
        base_cfg = json.load(f)

    # Population in unit space
    if getattr(args, "init_population", None):
        seed_pop = load_population(args.init_population)  # value space
        population = [to_unit(v) for v in seed_pop]
        while len(population) < args.pop_size:
            population.append(random_unit_vector())
        population = population[:args.pop_size]
    else:
        population = [random_unit_vector() for _ in range(args.pop_size)]
    fitness_cache = [None] * args.pop_size  # (psnr, ok) per individual, filled gen 0

    best_fitness = -1e9
    best_vector = None

    log_path = os.path.join(args.out_dir, "de_log.csv")
    with open(log_path, "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["generation", "individual", "fitness", "ok", "failure_reason",
                          "peak_vram_mb"] + [p.name for p in SEARCH_SPACE])

        def eval_and_log(gen, idx, unit_vec):
            nonlocal best_fitness, best_vector
            value_vec = from_unit(unit_vec)
            cfg = apply_genome(base_cfg, value_vec)
            result = evaluate(cfg, args.scene, args.out_dir, args)

            print(f"  gen {gen:02d} ind {idx:02d} | psnr={result.psnr:6.3f} | ok={result.ok} | {result.elapsed_s:5.1f}s")

            writer.writerow([gen, idx, result.psnr, result.ok, result.failure_reason,
                              result.peak_vram_mb] + list(value_vec))
            logfile.flush()

            if result.ok and result.psnr > best_fitness:
                best_fitness = result.psnr
                best_vector = value_vec
                with open(os.path.join(args.out_dir, "best_de_config.json"), "w") as bf:
                    json.dump(cfg, bf, indent=2)
                with open(os.path.join(args.out_dir, "best_de_genome.json"), "w") as bf:
                    json.dump({p.name: v for p, v in zip(SEARCH_SPACE, value_vec)}, bf, indent=2)

            return result.psnr

        # --- Initial evaluation (generation 0) ---
        print(f"\n=== [DE] Generation 0 (initial population) ===")
        for i, ind in enumerate(population):
            fitness_cache[i] = eval_and_log(0, i, ind)

        # --- Main DE loop ---
        for gen in range(1, args.generations + 1):
            print(f"\n=== [DE] Generation {gen}/{args.generations} ===")
            new_population = list(population)
            new_fitness = list(fitness_cache)

            for i in range(args.pop_size):
                # Pick 3 distinct individuals != i
                candidates = [j for j in range(args.pop_size) if j != i]
                a, b, c = random.sample(candidates, 3)

                # Mutation: donor = a + F * (b - c), clamped to [0,1]
                donor = [
                    min(1.0, max(0.0, population[a][d] + args.F * (population[b][d] - population[c][d])))
                    for d in range(DIM)
                ]

                # Binomial crossover with target vector i
                trial = list(population[i])
                rand_dim = random.randrange(DIM)  # ensure at least one gene from donor
                for d in range(DIM):
                    if d == rand_dim or random.random() < args.crossover_rate:
                        trial[d] = donor[d]

                trial_fitness = eval_and_log(gen, i, trial)

                # Greedy selection: trial replaces target only if it's better
                if trial_fitness > fitness_cache[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness

            population = new_population
            fitness_cache = new_fitness

    print("\n=== DE finished ===")
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
    p.add_argument("--F", type=float, default=0.5, help="differential weight, typically 0.4-0.9")
    p.add_argument("--crossover_rate", type=float, default=0.7, help="CR, typically 0.5-0.9")

    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--eval_width", type=int, default=64)
    p.add_argument("--eval_height", type=int, default=64)
    p.add_argument("--eval_spp", type=int, default=1)
    p.add_argument("--eval_timeout", type=int, default=1200)
    p.add_argument("--frame_idx", type=int, default=0)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--init_population", default=None,
                    help="path to a population JSON built by core/population_init.py; overrides random init")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
