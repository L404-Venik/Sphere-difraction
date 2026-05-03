"""Plot benchmark CSV produced by `benchmark_calculate_S`.

Reads CSV with columns: num_layers, M, trial, seconds
Aggregates mean and std per (num_layers, M) and draws errorbar plots.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_csv(path: str) -> List[Tuple[int, int, int, float]]:
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            num_layers = int(r[0])
            M = int(r[1])
            trial = int(r[2])
            seconds = float(r[3])
            rows.append((num_layers, M, trial, seconds))
    return rows


def aggregate(rows: List[Tuple[int, int, int, float]]):
    # key: (num_layers, M) -> list of seconds
    data: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for num_layers, M, trial, seconds in rows:
        data[(num_layers, M)].append(seconds)

    # convert to structure per num_layers: dict M -> (mean, std)
    per_layer = {}
    for (num_layers, M), vals in data.items():
        mu = mean(vals)
        sigma = stdev(vals) if len(vals) > 1 else 0.0
        per_layer.setdefault(num_layers, {})[M] = (mu, sigma)

    return per_layer


def plot_per_layer(per_layer, out: str | None = None, show: bool = True):
    plt.figure(figsize=(8, 5))
    for num_layers, mm in sorted(per_layer.items()):
        Ms = sorted(mm.keys())
        mus = [mm[M][0] for M in Ms]
        sigs = [mm[M][1] for M in Ms]
        plt.errorbar(Ms, mus, yerr=sigs, marker='o', label=f"layers={num_layers}")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('M (angular samples)')
    plt.ylabel('Time (s)')
    plt.title('calculate_S runtime vs M')
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()

    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved plot to {out}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot benchmark CSV')
    parser.add_argument('--in', dest='infile', required=True, help='CSV file produced by benchmark_calculate_S')
    parser.add_argument('--out', dest='out', default=None, help='Optional output PNG file')
    parser.add_argument('--no-show', dest='no_show', action='store_true', help='Do not call plt.show()')

    args = parser.parse_args()
    rows = read_csv(args.infile)
    per_layer = aggregate(rows)
    plot_per_layer(per_layer, out=args.out, show=not args.no_show)


if __name__ == '__main__':
    main()
