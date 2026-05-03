# Benchmarks — calculate_S

This folder contains a small benchmark and plotting pipeline for the core far-field routine `calculate_S`.

Files
- `benchmark_calculate_S.py` — CLI script that measures runtime of `calculate_S` for a set of M values and layer counts. Produces an optional CSV.
- `plot_benchmarks.py` — Reads the CSV output and draws a log-log errorbar plot of runtime vs M for each layer count.
- `run_and_plot.py` — Convenience wrapper that runs the benchmark and immediately plots results. It uses a temporary CSV file under the hood.

Quick one-line usage (PowerShell)

Run benchmark and show plot (default settings):

```powershell
python -m benchmarks.run_and_plot
```

Run with custom M values, layers and repeats, and save plot to `out.png`:

```powershell
python -m benchmarks.run_and_plot --Ms "360,720,1800,3600" --layers "1,2,5" --repeats 3 --out out.png
```

Or run only the benchmark and save results to CSV:

```powershell
python -m benchmarks.benchmark_calculate_S --Ms "360,720" --layers "1,2" --repeats 3 --out results.csv
```

Then plot the CSV separately (if you prefer to run plotting later):

```powershell
python -m benchmarks.plot_benchmarks --in results.csv --out results.png
```

Notes
- The benchmark script uses small, deterministic test `ExperimentParameters`. Adjust `benchmarks/benchmark_calculate_S.py` to test specific experimental presets.
- Timings depend a lot on CPU and Python/NumPy/SciPy builds — run multiple repeats for stable statistics.
- The plotting script uses log-log axes to visualize scaling with M.
