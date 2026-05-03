"""Run benchmark and plot results in one command.

This script calls benchmark_calculate_S to produce a CSV then calls plot_benchmarks to render it.
"""
from __future__ import annotations

import argparse
import tempfile
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description='Run benchmark and plot in one step')
    parser.add_argument('--Ms', default='360,720,1800,3600')
    parser.add_argument('--layers', default='1,2,5')
    parser.add_argument('--repeats', default='3')
    parser.add_argument('--out', default=None, help='Optional PNG output path')

    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(prefix='bench_', suffix='.csv', delete=False) as tmp:
        csv_path = tmp.name

    try:
        cmd_bench = [
            'python', '-m', 'benchmarks.benchmark_calculate_S',
            '--Ms', args.Ms,
            '--layers', args.layers,
            '--repeats', args.repeats,
            '--out', csv_path,
        ]
        print('Running benchmark...')
        subprocess.check_call(cmd_bench)

        cmd_plot = [
            'python', '-m', 'benchmarks.plot_benchmarks',
            '--in', csv_path,
        ]
        if args.out:
            cmd_plot += ['--out', args.out]

        print('Plotting results...')
        subprocess.check_call(cmd_plot)

    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


if __name__ == '__main__':
    main()
