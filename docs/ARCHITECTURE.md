# Architecture

## Overview

Physics simulation library for far-field and near-field electromagnetic scattering by multilayer spheres (Mie theory), plus an inverse problem framework for finding sphere structures that match target scattering profiles.

## Core (`core/`)

### `parameters.py`
Two dataclasses:
- **`ExperimentParameters`** — the main input to all computations. Holds `eps` (permittivities per layer, complex), `r` (radii of layer boundaries), `conducting_core`, `wave_length`. Validates on construction. Derived properties: `k` (wavenumber), `frequency_hz/ghz`.
- **`PlotingParameters`** — display config for matplotlib plots.

### `sphere_difraction.py`
All direct-problem physics. Key functions:
- `calculate_coefficients(params)` → `(D_e, D_m, N)` — Mie scattering coefficients via transfer-matrix method. Builds T-matrices layer by layer, applies inner boundary condition (conducting or dielectric core), stops when series converges.
- `calculate_S(params, M)` → `(S_th, S_ph)` — far-field scattering amplitudes at M evenly spaced angles over [0, 2π]. Symmetry is exploited: only half the angles are computed, then mirrored.
- `calculate_electric_field_far(params)` — far-field E field.
- `calculate_electric_field_close_vectorized(params, limits)` — near-field E on a 2D grid; uses disk cache (`.npz`) for expensive angular-function arrays.
- Helper functions: `psi`, `xi`, `psi_derivative`, `xi_derivative` (Riccati–Bessel functions); `assoc_legendre_derivative` and its vectorized variant.

### `ploting_functions.py`
Matplotlib wrappers for visualizing scattering patterns.

## Inverse Problem (`core/inverse_problem/`)

Framework for finding sphere structures whose scattering matches a target functional.

### `optimization.py` — shared data types
- **`OptimizationTask`** — what to optimize: wavelength(s), angles array, and `functional(S_th, S_ph, angles) → float`. Supports single-frequency and broadband mode. Broadband tasks evaluate the functional once per wavelength and aggregate the results.
- **`SolverConfig`** — controls `n_best` (how many top candidates to return), `aggregation` rule for broadband (`mean`/`max`/`sum`/custom callable), and `progress` flag.
- **`SolverResult`** — output: `best` (list of `(F, ExperimentParameters)` sorted ascending), `n_evaluated`, `n_skipped`, `elapsed_seconds`.

### `solver_base.py`
Abstract `Solver` base class. One method to implement: `run(space, task) → SolverResult`.

### `search_space.py`
- **`DiscreteRange`** / **`ContinuousRange`** — thickness axis types for layer specs.
- **`Layer`** — one coating layer with `thickness` (fixed float / `DiscreteRange` / `ContinuousRange`) and `material` (fixed name / list of names / `None` = all materials).
- **`SearchSpace`** — full discrete parameter space. Defines core + list of layers + material library. Iterates via `iter_candidates()` using `itertools.product`; filters out same-adjacent-material combos and thickness budget violations. Supports `up_to=True` to search 1..N layer counts. `size_estimate()` gives an upper bound before filtering.

### `brute_force_solver.py`
**`BruteForceSolver`** — iterates every candidate from `SearchSpace`, evaluates the functional, keeps top-N. Falls back to a plain percentage counter when tqdm is absent. Single-threaded.

## Tests (`tests/`)

pytest suite covering:
- `test_coefficients.py` — Mie coefficient correctness
- `test_special.py` — Riccati–Bessel helpers
- `test_difraction.py` — scattering amplitude functions
- `test_parameters.py` — `ExperimentParameters` validation
- `test_optimization.py` — `OptimizationTask`, `SolverConfig`, `SolverResult`, `BruteForceSolver`
- `test_search_space.py` — `SearchSpace` iteration, filtering, and size estimation

## Dependencies

- `numpy`, `scipy` — numerics
- `matplotlib` — plotting
- `tqdm` (optional) — progress bars in brute-force search
