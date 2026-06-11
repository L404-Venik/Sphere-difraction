# Architecture

## Overview

Physics simulation library for far-field and near-field electromagnetic scattering by multilayer spheres (Mie theory).

## Core (`core/`)

### `parameters.py`
Three dataclasses, with a deliberate split between the **physical body** (invariant
across wavelength) and the **observation setup** (what/where we probe):
- **`BodyParameters`** — the physical layered sphere. Holds `eps` (permittivities per layer, complex), `r` (radii of layer boundaries), `conducting_core`, optional `label`. Validates on construction. Carries no wavelength.
- **`ObservationParameters`** — the excitation/observation setup. Holds `wavelengths` (scalar accepted, stored as a 1-element array) and `angles` (1D, radians). Derived properties: `k` (wavenumber array), `frequency_hz/ghz` (arrays).
- **`PlotingParameters`** — display config for matplotlib plots.

There is no composite "experiment" type: an experiment is the *action* `calculate_S(body, observation)`, not a struct. This keeps the two lifetimes independent — e.g. the inverse solver sweeps many bodies against one fixed observation.

### `sphere_difraction.py`
All direct-problem physics. Key functions:
- `calculate_coefficients(body, k)` → `(D_e, D_m, N)` — Mie scattering coefficients via transfer-matrix method. Builds T-matrices layer by layer, applies inner boundary condition (conducting or dielectric core), stops when series converges. Takes an explicit scalar `k`.
- `calculate_S(body, observation)` → `(S_th, S_ph)`, each shape `(n_wavelengths, n_angles)` — far-field scattering amplitudes at the given wavelengths and arbitrary angles. The angle axis is fully vectorized (matmul over the series); the wavelength loop is unavoidable (k-dependent), but the angle-dependent Legendre terms are computed once and reused across wavelengths. θ=0 and θ=π are handled by asymptotic formulas via boolean masks (the general term is singular there) — automatic and invisible to the caller. No mirror-symmetry optimization: every angle is computed directly, so `S(2π−θ) = −S(θ)` holds as true physics rather than a copied half.
- `calculate_electric_field_far(body, observation)` — far-field E field, shape `(n_wavelengths, n_angles)`.
- `calculate_electric_field_close_vectorized(body, k, limits)` — near-field E on a 2D grid; uses disk cache (`.npz`) for expensive angular-function arrays.
- Helper functions: `psi`, `xi`, `psi_derivative`, `xi_derivative` (Riccati–Bessel functions); `assoc_legendre_derivative` and its vectorized variant.

### `ploting_functions.py`
Matplotlib wrappers for visualizing scattering patterns.

## Tests (`tests/`)

pytest suite covering:
- `test_coefficients.py` — Mie coefficient correctness
- `test_special.py` — Riccati–Bessel helpers
- `test_difraction.py` — scattering amplitude functions (shape, forward/backward limits, magnitude mirror symmetry, sign-flip parity, optical theorem)
- `test_body_parameters.py` — `BodyParameters` validation
- `test_observation_parameters.py` — `ObservationParameters` validation and derived `k`/frequency arrays
- `test_snapshots.py` — end-to-end snapshot tests for 16 diploma example configurations. Compares only the directly-computed `[0, π]` half of the legacy snapshots (the old second half was an unnegated mirror copy; the refactor computes every angle directly, so its second half is sign-flipped — see `calculate_S`).

## Scripts

- **`scripts/generate_snapshots.py`** — re-captures `S_th`/`S_ph` snapshots into `tests/snapshots/*.npz`. Run when results have intentionally changed.
- **`snapshot_configs.py`** (project root) — shared config defining the 16 named `(name, BodyParameters, wavelength)` cases plus the snapshot angle grid (`SNAPSHOT_ANGLES`, the legacy M=3600 evenly-spaced grid), used by both the generator and the test suite.

## Dependencies

- `numpy`, `scipy` — numerics
- `matplotlib` — plotting
- `tqdm` (optional) — progress bars in brute-force search
