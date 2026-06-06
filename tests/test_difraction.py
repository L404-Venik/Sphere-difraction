# test_scattering.py
import numpy as np
import pytest
from core.parameters import (ExperimentParameters)
from core.sphere_difraction import (calculate_S)

@pytest.fixture
def simple_params():
    return ExperimentParameters(
        eps=[2.0 + 0.5j, 1.0], r=[0.2], wave_length=1.0
    )

# ── Length ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("M", [360, 361, 3, 4, 1])
def test_output_length(simple_params, M):
    S_th, S_ph = calculate_S(simple_params, M=M)
    assert len(S_th) == M
    assert len(S_ph) == M

# ── Forward scattering (theta = 0) ────────────────────────────────────────────

@pytest.mark.parametrize("M", [360, 361])
def test_forward_scattering_sph_is_negated(simple_params, M):
    """S_ph[0] = -S_th[0] at theta=0 (asymptotic formula)."""
    S_th, S_ph = calculate_S(simple_params, M=M)
    assert S_ph[0] == pytest.approx(-S_th[0], rel=1e-12)

# ── Backscattering (theta = pi, even M only) ──────────────────────────────────

@pytest.mark.parametrize("M", [360, 100, 4])
def test_backscattering_components_equal_for_even_M(simple_params, M):
    """S_th[M//2] == S_ph[M//2] at theta=pi."""
    S_th, S_ph = calculate_S(simple_params, M=M)
    assert S_th[M // 2] == pytest.approx(S_ph[M // 2], rel=1e-12)

@pytest.mark.parametrize("M", [361, 101, 3])
def test_no_exact_pi_sample_for_odd_M(simple_params, M):
    """For odd M, no sample falls exactly on pi, so S_th != S_ph everywhere."""
    S_th, S_ph = calculate_S(simple_params, M=M)
    # The backscatter equality S_th==S_ph holds *only* at pi.
    # For odd M the closest samples straddle pi, so none should satisfy it exactly.
    mid = M // 2
    assert S_th[mid] != pytest.approx(S_ph[mid], rel=1e-6)

# ── Mirror symmetry ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("M", [360, 361])
def test_mirror_symmetry_S_th(simple_params, M):
    """S_th[m] == S_th[M - m] for all interior indices."""
    S_th, _ = calculate_S(simple_params, M=M)
    indices = np.arange(1, M // 2)          # interior, excluding 0 and pi
    np.testing.assert_allclose(
        S_th[indices], S_th[M - indices],
        rtol=1e-12,
        err_msg="S_th mirror symmetry failed"
    )

@pytest.mark.parametrize("M", [360, 361])
def test_mirror_symmetry_S_ph(simple_params, M):
    """S_ph[m] == S_ph[M - m] for all interior indices."""
    _, S_ph = calculate_S(simple_params, M=M)
    indices = np.arange(1, M // 2)
    np.testing.assert_allclose(
        S_ph[indices], S_ph[M - indices],
        rtol=1e-12,
        err_msg="S_ph mirror symmetry failed"
    )

# ── Optical theorem ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("M", [360, 361])
def test_optical_theorem(simple_params, M):
    """Im(S_th[0]) == Im(S_ph[0]) in magnitude (both = -Im(S_th[0]))."""
    S_th, S_ph = calculate_S(simple_params, M=M)
    assert abs(S_th[0].imag) == pytest.approx(abs(S_ph[0].imag), rel=1e-12)

# ── Sampling consistency ──────────────────────────────────────────────────────

def test_even_odd_M_agree_at_shared_angles(simple_params):
    """
    Even and odd M should produce identical values at angles they share.
    M=360 and M=361 both sample theta=0, and their interior angles
    differ only in density — but theta=0 is the same point.
    """
    S_th_even, S_ph_even = calculate_S(simple_params, M=360)
    S_th_odd,  S_ph_odd  = calculate_S(simple_params, M=361)
    # theta=0 is always index 0
    assert S_th_even[0] == pytest.approx(S_th_odd[0], rel=1e-12)
    assert S_ph_even[0] == pytest.approx(S_ph_odd[0], rel=1e-12)