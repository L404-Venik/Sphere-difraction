# Architecture

## Overview

Physics simulation library for far-field and near-field electromagnetic scattering by multilayer spheres (Mie theory).

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

## Tests (`tests/`)

pytest suite covering:
- `test_coefficients.py` — Mie coefficient correctness
- `test_special.py` — Riccati–Bessel helpers
- `test_difraction.py` — scattering amplitude functions
- `test_parameters.py` — `ExperimentParameters` validation
- `test_snapshots.py` — end-to-end snapshot tests for 16 diploma example configurations

## Scripts

- **`scripts/generate_snapshots.py`** — re-captures `S_th`/`S_ph` snapshots into `tests/snapshots/*.npz`. Run when results have intentionally changed.
- **`snapshot_configs.py`** (project root) — shared config defining the 16 named `ExperimentParameters` used by both the generator and the test suite.

## Dependencies

- `numpy`, `scipy` — numerics
- `matplotlib` — plotting
- `tqdm` (optional) — progress bars in brute-force search
