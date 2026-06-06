"""
brute_force_solver.py — Exhaustive search over a discrete parameter space.

Iterates every candidate in a SearchSpace, evaluates the objective
functional, and returns the n_best lowest-scoring structures.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Union

import numpy as np

from .optimization import OptimizationTask, SolverConfig, SolverResult
from .search_space import SearchSpace
from .solver_base import Solver
from ..sphere_difraction import calculate_S

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


class BruteForceSolver(Solver):
    """
    Exhaustive brute-force synthesis solver.

    Evaluates every candidate in the search space and keeps the
    ``config.n_best`` lowest-scoring structures.

    For large spaces this can take minutes — enable ``config.progress``
    to get a progress bar.  The computation is single-threaded; parallelism
    is a future extension.

    Parameters
    ----------
    config : SolverConfig, optional
        Controls n_best, broadband aggregation, and progress reporting.
        Defaults to ``SolverConfig()``.

    Examples
    --------
    >>> solver = BruteForceSolver(SolverConfig(n_best=10, progress=True))
    >>> result = solver.run(space, task)
    >>> best_f, best_params = result.best[0]
    """

    def __init__(self, config: SolverConfig = None):
        self.config = config if config is not None else SolverConfig()

    def run(self, space: SearchSpace, task: OptimizationTask) -> SolverResult:
        """
        Iterate over all candidates in ``space`` and return the best ones.

        Notes
        -----
        S arrays are not stored in the result to keep memory usage low.
        To recover S for a result, call ``calculate_S(params, M=task.M)``.
        The ``wave_length`` field of returned params is set to the first
        (or only) wavelength in the task.
        """
        config = self.config
        wavelengths = task.wavelengths
        M = task.M
        angles = task.angles
        size_hint = space.size_estimate()

        top: list[tuple[float, object]] = []   # (F, ExperimentParameters)
        n_evaluated = 0
        n_skipped = 0
        t_start = time.perf_counter()

        iterator = space.iter_candidates()
        if config.progress:
            if _TQDM_AVAILABLE:
                iterator = tqdm(iterator, total=size_hint, unit="cand", desc="Searching")
            else:
                _print_progress_start(size_hint)

        for params in iterator:
            try:
                f_value = _evaluate(params, wavelengths, M, angles, task.functional, config)
            except Exception:
                n_skipped += 1
                continue

            n_evaluated += 1

            if config.progress and not _TQDM_AVAILABLE:
                _print_progress_tick(n_evaluated, size_hint)

            ref_wl = float(wavelengths[0])
            ref_params = dataclasses.replace(params, wave_length=ref_wl)

            if len(top) < config.n_best:
                top.append((f_value, ref_params))
                top.sort(key=lambda x: x[0])
            elif f_value < top[-1][0]:
                top[-1] = (f_value, ref_params)
                top.sort(key=lambda x: x[0])

        elapsed = time.perf_counter() - t_start

        if config.progress and not _TQDM_AVAILABLE:
            print()

        return SolverResult(
            best=top,
            n_evaluated=n_evaluated,
            n_skipped=n_skipped,
            elapsed_seconds=elapsed,
        )

    def __repr__(self) -> str:
        return f"BruteForceSolver(config={self.config!r})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate(params, wavelengths, M, angles, functional, config):
    if len(wavelengths) == 1:
        p = dataclasses.replace(params, wave_length=float(wavelengths[0]))
        S_th, S_ph = calculate_S(p, M=M)
        return float(functional(S_th, S_ph, angles))

    per_wl = np.empty(len(wavelengths), dtype=np.float64)
    for i, wl in enumerate(wavelengths):
        p = dataclasses.replace(params, wave_length=float(wl))
        S_th, S_ph = calculate_S(p, M=M)
        per_wl[i] = float(functional(S_th, S_ph, angles))

    return config._aggregate(per_wl)


# --- Fallback progress (no tqdm) ---

_progress_last_pct = -1

def _print_progress_start(total: int) -> None:
    global _progress_last_pct
    _progress_last_pct = -1
    print(f"Searching ~{total:,} candidates ", end="", flush=True)


def _print_progress_tick(n: int, total: int) -> None:
    global _progress_last_pct
    if total <= 0:
        return
    pct = int(100 * n / total)
    if pct >= _progress_last_pct + 10:
        print(f"{pct}%.. ", end="", flush=True)
        _progress_last_pct = pct
