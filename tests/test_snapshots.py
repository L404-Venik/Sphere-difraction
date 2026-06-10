import os
import numpy as np
import pytest
from core.sphere_difraction import calculate_S
from snapshot_configs import known_cases

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

_cases = known_cases()


@pytest.mark.parametrize("name,experiment", _cases, ids=[c[0] for c in _cases])
def test_S_matches_snapshot(name, experiment):
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{name}.npz")
    if not os.path.exists(snapshot_path):
        pytest.skip(f"snapshot missing — run scripts/generate_snapshots.py")

    ref = np.load(snapshot_path)
    S_th, S_ph = calculate_S(experiment)

    np.testing.assert_allclose(S_th, ref["S_th"], rtol=1e-10)
    np.testing.assert_allclose(S_ph, ref["S_ph"], rtol=1e-10)
