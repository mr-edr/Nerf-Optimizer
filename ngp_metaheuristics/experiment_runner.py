"""
experiment_runner.py

Top-level entry point for a full comparison run on ONE dataset. For each
algorithm requested:
  1. builds a fresh initial population (mix of Instant-NGP presets +
     random genomes) via core/population_init.py, saved to
     <out_dir>/<algorithm>/init_population.json
  2. runs that algorithm against it, storing its own CSV log and
     best-config files under <out_dir>/<algorithm>/
  3. moves on to the next algorithm

To compare across datasets (fox, chair, ship, ...), run this script once
per dataset with a different --out_dir, e.g.:

    python3 experiment_runner.py --ngp_dir ../instant-ngp \\
        --scene ../instant-ngp/data/nerf/fox \\
        --base ../instant-ngp/configs/nerf/base.json \\
        --out_dir results/fox

    python3 experiment_runner.py --ngp_dir ../instant-ngp \\
        --scene ../instant-ngp/data/nerf/chair \\
        --base ../instant-ngp/configs/nerf/base.json \\
        --out_dir results/chair

Each run is independent -- population is regenerated per dataset (same
mix strategy, but presets are read fresh and random fill uses the given
--seed), so results/fox/ga and results/chair/ga are directly comparable
runs of the same algorithm on different data, not accidentally sharing
state.

At the end, prints a summary table of best PSNR per algorithm for this
dataset, and writes it to <out_dir>/summary.csv.
"""

import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

from core.population_init import generate_and_save

import optimizers.genetic_algorithm as ga_mod
import optimizers.differential_evolution as de_mod
import optimizers.particle_swarm as pso_mod
import optimizers.cma_es as cma_mod
import optimizers.random_search as rs_mod

ALGORITHMS = ["ga", "de", "pso", "cma", "random"]


def make_args(common, out_dir, init_population_path, extra):
    """Builds the SimpleNamespace each optimizer module's run() expects."""
    ns = SimpleNamespace(
        ngp_dir=common.ngp_dir,
        scene=common.scene,
        base=common.base,
        out_dir=out_dir,
        eval_steps=common.eval_steps,
        eval_width=common.eval_width,
        eval_height=common.eval_height,
        eval_spp=common.eval_spp,
        eval_timeout=common.eval_timeout,
        frame_idx=common.frame_idx,
        seed=common.seed,
        init_population=init_population_path,
    )
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def run_one_algorithm(name, common, out_dir):
    algo_dir = os.path.join(out_dir, name)
    os.makedirs(algo_dir, exist_ok=True)

    with open(common.base) as f:
        base_cfg = json.load(f)

    pop_path = os.path.join(algo_dir, "init_population.json")
    generate_and_save(
        base_cfg=base_cfg, ngp_dir=common.ngp_dir, out_path=pop_path,
        pop_size=common.pop_size, preset_fraction=common.preset_fraction, seed=common.seed,
    )

    print(f"\n{'=' * 60}\nRunning {name.upper()} -> {algo_dir}\n{'=' * 60}")
    start = time.time()

    if name == "ga":
        args = make_args(common, algo_dir, pop_path, {"pop_size": common.pop_size,
                          "generations": common.generations, "mutation_rate": common.mutation_rate})
        ga_mod.run(args)
        best_file = os.path.join(algo_dir, "best_ga_genome.json")

    elif name == "de":
        args = make_args(common, algo_dir, pop_path, {"pop_size": common.pop_size,
                          "generations": common.generations, "F": common.de_f, "crossover_rate": common.de_cr})
        de_mod.run(args)
        best_file = os.path.join(algo_dir, "best_de_genome.json")

    elif name == "pso":
        args = make_args(common, algo_dir, pop_path, {"pop_size": common.pop_size,
                          "iterations": common.generations, "w_start": common.pso_w_start,
                          "w_end": common.pso_w_end, "c1": common.pso_c1, "c2": common.pso_c2,
                          "v_max": common.pso_v_max})
        pso_mod.run(args)
        best_file = os.path.join(algo_dir, "best_pso_genome.json")

    elif name == "cma":
        args = make_args(common, algo_dir, pop_path, {"pop_size": common.pop_size,
                          "generations": common.generations})
        cma_mod.run(args)
        best_file = os.path.join(algo_dir, "best_cma_genome.json")

    elif name == "random":
        # Match total evaluation count roughly to DE/PSO (pop_size * (generations+1))
        n_samples = common.pop_size * (common.generations + 1)
        args = make_args(common, algo_dir, pop_path, {"n_samples": n_samples})
        rs_mod.run(args)
        best_file = os.path.join(algo_dir, "best_random_genome.json")

    else:
        raise ValueError(f"Unknown algorithm: {name}")

    elapsed = time.time() - start

    best_psnr = None
    log_path = os.path.join(algo_dir, f"{'random_search' if name == 'random' else name}_log.csv")
    if os.path.exists(log_path):
        with open(log_path) as f:
            reader = csv.DictReader(f)
            oks = [float(row["fitness"]) for row in reader if row.get("ok") == "True"]
        if oks:
            best_psnr = max(oks)

    return {"algorithm": name, "best_psnr": best_psnr, "elapsed_s": round(elapsed, 1),
            "out_dir": algo_dir}


def run(common):
    os.makedirs(common.out_dir, exist_ok=True)
    algorithms = common.algorithms

    results = []
    for name in algorithms:
        result = run_one_algorithm(name, common, common.out_dir)
        results.append(result)
        print(f"  -> {name.upper()} best PSNR: {result['best_psnr']}  ({result['elapsed_s']}s)")

    summary_path = os.path.join(common.out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "best_psnr", "elapsed_s", "out_dir"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n{'=' * 60}\nSUMMARY ({os.path.basename(os.path.normpath(common.out_dir))})\n{'=' * 60}")
    for r in sorted(results, key=lambda r: (r["best_psnr"] is None, -(r["best_psnr"] or 0))):
        print(f"  {r['algorithm']:>8} | best_psnr={r['best_psnr']}  | {r['elapsed_s']}s")
    print(f"\nWritten to {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ngp_dir", default="../instant-ngp", help="path to instant-ngp checkout root")
    p.add_argument("--scene", required=True, help="dataset path, e.g. ../instant-ngp/data/nerf/fox")
    p.add_argument("--base", required=True, help="base Instant-NGP config to start from")
    p.add_argument("--out_dir", required=True, help="results for THIS dataset go here, as out_dir/<algorithm>/")

    p.add_argument("--algorithms", default=",".join(ALGORITHMS),
                    help=f"comma-separated subset of: {','.join(ALGORITHMS)}")

    p.add_argument("--pop_size", type=int, default=8)
    p.add_argument("--generations", type=int, default=8, help="generations/iterations for ga/de/pso/cma")
    p.add_argument("--preset_fraction", type=float, default=0.5,
                    help="target fraction of the initial population seeded from Instant-NGP presets")

    # GA
    p.add_argument("--mutation_rate", type=float, default=0.3)
    # DE
    p.add_argument("--de_f", type=float, default=0.5)
    p.add_argument("--de_cr", type=float, default=0.7)
    # PSO
    p.add_argument("--pso_w_start", type=float, default=0.9)
    p.add_argument("--pso_w_end", type=float, default=0.4)
    p.add_argument("--pso_c1", type=float, default=1.5)
    p.add_argument("--pso_c2", type=float, default=1.5)
    p.add_argument("--pso_v_max", type=float, default=0.2)

    # Shared eval settings
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--eval_width", type=int, default=32)
    p.add_argument("--eval_height", type=int, default=32)
    p.add_argument("--eval_spp", type=int, default=1)
    p.add_argument("--eval_timeout", type=int, default=1200)
    p.add_argument("--frame_idx", type=int, default=0)

    p.add_argument("--seed", type=int, default=1337)

    args = p.parse_args()
    args.algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for a in args.algorithms:
        if a not in ALGORITHMS:
            raise SystemExit(f"Unknown algorithm '{a}' -- choose from {ALGORITHMS}")
    return args


if __name__ == "__main__":
    run(parse_args())
