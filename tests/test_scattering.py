# test_scattering.py
import numpy as np
import pytest
from core.Parameters import ExperimentParameters
from core.sphere_difraction import calculate_S

def test_optical_theorem():
    """
    Optical theorem: Im(S_th(0)) = Im(S_ph(0)).
    Both components of the forward amplitude must be equal.
    """
    params = ExperimentParameters(
        eps=[2.0 + 0.5j, 1.0], r=[0.2], wave_length=1.0
    )
    S_th, S_ph = calculate_S(params, M=2)
    # Forward direction is index 0
    assert S_th[0].imag == pytest.approx(S_ph[0].imag, rel=1e-8)
    assert S_th[1].imag == pytest.approx(S_ph[1].imag, rel=1e-8)

def test_S_has_correct_length():
    for M in [360, 361]:  # test both even and odd
        params = ExperimentParameters(eps=[1.5, 1.0], r=[0.1], wave_length=0.5)
        S_th, S_ph = calculate_S(params, M=M)
        assert len(S_th) == M
        assert len(S_ph) == M