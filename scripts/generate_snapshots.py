#!/usr/bin/env python3
"""
Capture known-good S_th / S_ph snapshots for the known example configurations.

This overwrites existing snapshots unconditionally — only run when the current
output is known to be correct.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from core.sphere_difraction import calculate_S
from snapshot_configs import known_cases

SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "tests", "snapshots")


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    cases = known_cases()
    for name, experiment in cases:
        S_th, S_ph = calculate_S(experiment)
        path = os.path.join(SNAPSHOT_DIR, f"{name}.npz")
        np.savez(path, S_th=S_th, S_ph=S_ph)
        print(f"  saved {name}.npz")
    print(f"\nDone — {len(cases)} snapshots written to {SNAPSHOT_DIR}")


if __name__ == "__main__":
    main()
