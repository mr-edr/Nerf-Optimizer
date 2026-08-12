"""
optimizers/particle_swarm.py

Standard PSO (inertia weight variant), operating in UNIT SPACE for the
same reason as DE: comparable step sizes across dimensions with very
different physical scales (learning_rate vs n_hidden_layers).

Each particle has:
  - position (unit-space genome)
  - velocity
  - personal best position + fitness
Swarm has a single global best position + fitness.

Velocity update:
  v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
  x = x + v   (clamped to [0,1], velocity zeroed/reflected on clamp)

w (inertia), c1 (cognitive), c2 (social) are the classic tunables.
Since evals here are expensive, keep pop_size and iterations modest just
like GA/DE -- PSO usually needs fewer evaluations than GA to converge on
smooth-ish continuous landscapes, which is one of the things worth
measuring in your comparison.
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

    # Positions + velocities in unit space
    if getattr(args, "init_population", None):
        seed_pop = load_population(args.init_population)  # value space
        positions = [to_unit(v) for v in seed_pop]
        while len(positions) < args.pop_size:
            positions.append(random_unit_vector())
        positions = positions[:args.pop_size]
    else:
        positions = [random_unit_vector() for _ in range(args.pop_size)]
    velocities = [[random.uniform(-0.1, 0.1) for _ in range(DIM)] for _ in range(args.pop_size)]

    personal_best_pos = [list(p) for p in positions]
    personal_best_fit = [-1e9] * args.pop_size

    global_best_pos = None
    global_best_fit = -1e9

    log_path = os.path.join(args.out_dir, "pso_log.csv")
    with open(log_path, "w", newline="") as logfile:
        writer = csv.writer(logfile)
        writer.writerow(["iteration", "particle", "fitness", "ok", "failure_reason",
                          "peak_vram_mb"] + [p.name for p in SEARCH_SPACE])

        def eval_and_log(it, idx, unit_pos):
            nonlocal global_best_pos, global_best_fit
            value_vec = from_unit(unit_pos)
            cfg = apply_genome(base_cfg, value_vec)
            result = evaluate(cfg, args.scene, args.out_dir, args)

            print(f"  iter {it:02d} particle {idx:02d} | psnr={result.psnr:6.3f} | ok={result.ok} | {result.elapsed_s:5.1f}s")

            writer.writerow([it, idx, result.psnr, result.ok, result.failure_reason,
                              result.peak_vram_mb] + list(value_vec))
            logfile.flush()

            if result.ok and result.psnr > personal_best_fit[idx]:
                personal_best_fit[idx] = result.psnr
                personal_best_pos[idx] = list(unit_pos)

            if result.ok and result.psnr > global_best_fit:
                global_best_fit = result.psnr
                global_best_pos = list(unit_pos)
                with open(os.path.join(args.out_dir, "best_pso_config.json"), "w") as bf:
                    json.dump(cfg, bf, indent=2)
                with open(os.path.join(args.out_dir, "best_pso_genome.json"), "w") as bf:
                    json.dump({p.name: v for p, v in zip(SEARCH_SPACE, value_vec)}, bf, indent=2)

            return result.psnr

        # --- Initial evaluation ---
        print(f"\n=== [PSO] Iteration 0 (initial swarm) ===")
        for i, pos in enumerate(positions):
            eval_and_log(0, i, pos)

        if global_best_pos is None:
            # Nothing succeeded on first pass -- fall back to first particle so the
            # loop below has something to converge toward rather than crashing.
            global_best_pos = list(positions[0])

        # --- Main PSO loop ---
        for it in range(1, args.iterations + 1):
            print(f"\n=== [PSO] Iteration {it}/{args.iterations} ===")

            # Linearly decay inertia weight from w_start to w_end over the run --
            # more exploration early, more exploitation/refinement late.
            w = args.w_start + (args.w_end - args.w_start) * (it / args.iterations)

            for i in range(args.pop_size):
                for d in range(DIM):
                    r1, r2 = random.random(), random.random()
                    cognitive = args.c1 * r1 * (personal_best_pos[i][d] - positions[i][d])
                    social = args.c2 * r2 * (global_best_pos[d] - positions[i][d])
                    velocities[i][d] = w * velocities[i][d] + cognitive + social
                    # Clamp velocity magnitude to avoid particles flying off
                    velocities[i][d] = max(-args.v_max, min(args.v_max, velocities[i][d]))

                    new_pos = positions[i][d] + velocities[i][d]
                    if new_pos < 0.0 or new_pos > 1.0:
                        # Reflect off the boundary and damp velocity, rather than
                        # just clamping position (which would zero velocity and
                        # get particles stuck at the edges of the search space).
                        new_pos = max(0.0, min(1.0, new_pos))
                        velocities[i][d] *= -0.5
                    positions[i][d] = new_pos

                eval_and_log(it, i, positions[i])

    print("\n=== PSO finished ===")
    print("Best fitness (PSNR-based):", global_best_fit)
    if global_best_pos:
        print(describe(from_unit(global_best_pos)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ngp_dir", default="../instant-ngp", help="path to instant-ngp checkout root")
    p.add_argument("--scene", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--pop_size", type=int, default=8, help="swarm size")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--w_start", type=float, default=0.9, help="inertia weight, start of run")
    p.add_argument("--w_end", type=float, default=0.4, help="inertia weight, end of run")
    p.add_argument("--c1", type=float, default=1.5, help="cognitive (personal-best) coefficient")
    p.add_argument("--c2", type=float, default=1.5, help="social (global-best) coefficient")
    p.add_argument("--v_max", type=float, default=0.2, help="max velocity magnitude per dim (unit space)")

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
