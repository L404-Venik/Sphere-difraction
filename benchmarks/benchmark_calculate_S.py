"""Quick benchmark for core.calculate_S.

Measures runtime for different numbers of layers and angular sample counts (M).

Usage (from project root):
    python -m benchmarks.benchmark_calculate_S

Options are available via --Ms, --layers, --repeats and --out.
Defaults are small so the script runs quickly as a smoke test.
"""
from __future__ import annotations

import argparse
import csv
import time
from statistics import mean, stdev
from typing import List

import numpy as np

from core.Parameters import ExperimentParameters
from core.sphere_difraction import calculate_S


def build_params_for_num_layers(num_layers: int) -> ExperimentParameters:
    """Create a simple ExperimentParameters with `num_layers` layers.

    Radii are increasing values and epsilons vary between layers; exterior medium is 1.0.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be >= 1")

    # radii: evenly spaced between 0.02 and 0.2 (meters)
    r = np.linspace(0.02, 0.2, num=num_layers)

    # eps: len(r) + 1 entries (innermost -> exterior). Use simple varying permittivities.
    eps_inner = [2.5 + 0.1j * (i % 2) for i in range(num_layers)]
    eps = np.array(eps_inner + [1.0], dtype=np.complex128)

    return ExperimentParameters(eps=eps, r=r, wave_length=0.5, conducting_core=False)


def run_benchmark(Ms: List[int], layers_list: List[int], repeats: int, out: str | None):
    rows = []
    header = ["num_layers", "M", "trial", "seconds"]

    for num_layers in layers_list:
        params = build_params_for_num_layers(num_layers)
        for M in Ms:
            # Warm-up once (import-time caches, special functions)
            _ = calculate_S(params, M=M)

            times = []
            for trial in range(1, repeats + 1):
                t0 = time.perf_counter()
                _ = calculate_S(params, M=M)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                times.append(elapsed)
                rows.append((num_layers, M, trial, elapsed))

            # Print a small summary per configuration
            if repeats >= 2:
                avg = mean(times)
                sd = stdev(times) if len(times) > 1 else 0.0
                print(f"layers={num_layers:2d}  M={M:5d}  trials={repeats}  mean={avg:.4f}s  stdev={sd:.4f}s")
            else:
                print(f"layers={num_layers:2d}  M={M:5d}  trial_time={times[0]:.4f}s")

    if out:
        with open(out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in rows:
                writer.writerow(r)
        print(f"Saved results to {out}")


def parse_list_of_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark calculate_S for different Ms and layer counts")
    parser.add_argument("--Ms", type=str, default="360, 720, 1800, 3600",
                        help="Comma-separated list of M values (angular samples). Default: 360, 720, 1800, 3600")
    parser.add_argument("--layers", type=str, default="1,2,5",
                        help="Comma-separated list of numbers of layers to test. Default: 1,2,5")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per configuration (default: 3)")
    parser.add_argument("--out", type=str, default=None, help="CSV output path to save results")

    args = parser.parse_args()
    Ms = parse_list_of_ints(args.Ms)
    layers_list = parse_list_of_ints(args.layers)

    print("Starting benchmark: calculate_S")
    print(f"Ms: {Ms}")
    print(f"layers: {layers_list}")
    print(f"repeats: {args.repeats}")

    run_benchmark(Ms, layers_list, args.repeats, args.out)


if __name__ == "__main__":
    main()
