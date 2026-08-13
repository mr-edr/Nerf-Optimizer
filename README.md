# NGP Metaheuristics

A comparative study of metaheuristic optimization algorithms applied to
hyperparameter tuning for [NVIDIA Instant-NGP](https://github.com/NVlabs/instant-ngp).
Instead of manually tuning encoding/network/optimizer settings by hand,
this project treats hyperparameter search as a black-box optimization
problem and compares how different search strategies perform on it.

This is a from-scratch rebuild (previous attempt used only a GA; this
version compares five algorithms) built and tested on Arch Linux against
an Instant-NGP checkout with a working `pyngp` Python binding.

## Why

Instant-NGP has several hyperparameters that affect rendering quality,
training speed, and GPU memory use in ways that interact non-trivially
(hash grid size vs. resolution vs. learning rate, etc.). Rather than
manual trial-and-error, this project searches for good configurations
automatically and asks a more interesting question along the way:
**does "smart" search actually beat random guessing here, given how
noisy and expensive each evaluation is?**

## Architecture

```
ngp_metaheuristics/
├── core/
│   ├── search_space.py   # single source of truth: which hyperparameters
│   │                     # are tunable, their bounds, type, log-scale flag
│   ├── genome.py         # converts between "value space" (raw units, used
│   │                     # by GA) and "unit space" [0,1]^9 (used by DE/PSO/
│   │                     # CMA-ES so every dimension gets comparable step
│   │                     # sizes); also applies a genome onto a base config
│   ├── eval_pipeline.py  # train -> save snapshot -> render one frame ->
│   │                     # compute PSNR, by calling instant-ngp/scripts/run.py
│   │                     # by path. Never modifies the instant-ngp checkout.
│   └── fitness.py        # shared evaluator every optimizer calls: hard-fails
│                         # on OOM/NaN/crash (fixed penalty, not silently
│                         # continuing), soft penalty for exceeding a VRAM
│                         # budget, times every evaluation
├── optimizers/
│   ├── genetic_algorithm.py       # elitism + uniform crossover + mutation
│   ├── differential_evolution.py  # DE/rand/1/bin
│   ├── particle_swarm.py          # inertia-weight PSO with velocity reflection
│   ├── cma_es.py                  # wraps the `cma` PyPI package
│   └── random_search.py           # no search logic -- pure baseline control
└── analysis/
    └── runs/              # CSV logs + best-found configs per algorithm
```

**Design principle:** the search space, genome encoding, and fitness
function are shared infrastructure. Every optimizer only differs in
*how it picks the next candidates to evaluate* -- which keeps the
comparison fair.

## Search space

| Parameter | Type | Range | Notes |
|---|---|---|---|
| `n_levels` | int | 8–18 | HashGrid encoding levels |
| `n_features_per_level` | categorical | {2, 4, 8} | |
| `log2_hashmap_size` | int | 15–22 | biggest lever for quality/VRAM tradeoff |
| `base_resolution` | int | 4–32 | |
| `per_level_scale` | float | 1.26–2.0 | |
| `n_neurons` | categorical | {16, 32, 64, 128} | fully-fused MLP width |
| `n_hidden_layers` | int | 1–3 | |
| `learning_rate` | float (log) | 1e-4–5e-2 | Adam |
| `l2_reg` | float (log) | 1e-8–1e-4 | Adam |

Scene-correctness parameters (e.g. `aabb_scale`) are deliberately **not**
included -- those should be fixed correctly per-dataset, not evolved,
since a wrong value degrades every candidate equally rather than
representing a real quality/efficiency tradeoff.

## Setup

```bash
# In the same conda env that has a working `pyngp` import:
pip install --break-system-packages numpy Pillow nvidia-ml-py cma
```

`nvidia-ml-py` (pynvml) is optional -- VRAM tracking is skipped
gracefully if it's not installed. `cma` is only required for
`optimizers/cma_es.py`.

## Usage

All optimizers share the same core CLI arguments:

```bash
python3 experiment_runner.py \
  --ngp_dir ../instant-ngp \
  --scene ../instant-ngp/data/nerf/fox \
  --base ../instant-ngp/configs/nerf/base.json \
  --out_dir results/fox \
  --pop_size 8 --generations 8
```


Each run writes a CSV log (`<algorithm>_log.csv`) with per-evaluation
fitness, success/failure status, VRAM peak, and elapsed time, plus
`best_<algorithm>_config.json` / `best_<algorithm>_genome.json` for the
best candidate found.

### Standalone pipeline debugging

To sanity-check the train/render/PSNR pipeline in isolation (with full
live output, unlike the quiet mode used during real optimizer runs):

```bash
python3 -m core.eval_pipeline \
  --ngp_dir ../instant-ngp \
  --scene ../instant-ngp/data/nerf/fox \
  --config ../instant-ngp/configs/nerf/base.json \
  --steps 100 --frame_idx 0 --width 32 --height 32 --spp 1
```

## Results (fox dataset, 100 training steps, 32×32 eval render)

| Algorithm | Evaluations | Best PSNR | Behavior |
|---|---|---|---|
| **CMA-ES** | 64 | **15.19** | Real upward trend across generations -- covariance adaptation visibly working |
| Random search | 72 | 14.58 | No trend (by construction); baseline control |
| Genetic Algorithm | 64 | 14.31 | Found spikes early, then plateaued (premature convergence) |
| Differential Evolution | 72 | 14.17 | Flat and noisy throughout, no clear trend |
| Particle Swarm | 72 | 13.03 | Never escaped the flat plateau -- clear underperformer |

### Key finding

At this evaluation budget, the fitness landscape is mostly flat and
noisy, with sparse, narrow high-PSNR regions. **Only CMA-ES showed a
real, visible improvement trend and clearly beat random search.**
GA, DE, and PSO were statistically indistinguishable from (or worse
than) blind random sampling -- their occasional "good" results look
more like evaluation noise landing on a lucky spike than genuine
search intelligence. PSO in particular underperformed consistently,
which is explainable: its velocity-based, momentum-driven movement
lacks the large exploratory jumps (GA's crossover, DE's differential
vectors, CMA-ES's adaptive step-size expansion) needed to escape a
flat plateau and find sparse good regions.

This suggests the eval budget (100 steps / low render resolution) is
likely too noisy for simple selection/mutation/velocity-based methods
to exploit -- only an algorithm that adapts its search *distribution*
(not just individual candidates) found a consistent edge.

## Known limitations

- `instant-ngp`'s `scripts/run.py` has no `--seed` flag, so repeated
  evaluations of the *identical* config are not reproducible -- some
  amount of noise in every result is unavoidable given the current
  Instant-NGP CLI.
- Results above are from a single dataset (fox) at a single evaluation
  budget; conclusions about "does optimization help" should be
  validated at a full training budget (see below) before generalizing.

## Next steps

- **Full-budget validation**: train the CMA-ES-found config and the
  default config both to convergence (e.g. 10,000 steps, full render
  resolution) and compare final PSNR + VRAM + wall-clock time. This is
  the step that turns "won a noisy 100-step proxy metric" into "is
  actually a better configuration."
- **Budget sensitivity check**: rerun CMA-ES vs. random search at a
  higher eval budget (e.g. 300–400 steps) to see whether CMA-ES's edge
  grows, shrinks, or disappears as per-evaluation noise decreases.
- Possible additions: Simulated Annealing (simple baseline control),
  multi-fidelity/successive-halving scheduling (adaptively allocate
  more steps to promising candidates rather than a fixed budget for
  every evaluation).
