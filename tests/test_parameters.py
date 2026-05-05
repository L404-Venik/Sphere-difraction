# test_parameters.py
import numpy as np
import re
import pytest
from core.parameters import ExperimentParameters
import scipy.constants as const
c = const.speed_of_light


# ── Constructor / derived properties ─────────────────────────────────────────

def test_valid_single_layer():
    p = ExperimentParameters(eps=[1.5, 1.0], r=[0.1], wave_length=0.5)
    assert p.k == pytest.approx(2 * np.pi / 0.5)

def test_frequency_consistency():
    p = ExperimentParameters(eps=[1.0, 1.0], r=[0.1], wave_length=0.03)
    assert p.frequency_hz  == pytest.approx(c / 0.03)
    assert p.frequency_ghz == pytest.approx(p.frequency_hz / 1e9)

def test_complex_eps_accepted():
    """Lossy materials have a complex permittivity — must not raise."""
    p = ExperimentParameters(eps=[2.5 + 0.3j, 1.0], r=[0.1], wave_length=1.0)
    assert p.eps[0].imag == pytest.approx(0.3)

def test_inputs_are_converted_to_arrays():
    """Lists and tuples must be silently converted to ndarrays."""
    p = ExperimentParameters(eps=(2.0, 1.0), r=(0.1,), wave_length=1.0)
    assert isinstance(p.eps, np.ndarray)
    assert isinstance(p.r,   np.ndarray)

def test_eps_dtype_is_complex128():
    p = ExperimentParameters(eps=[1.5, 1.0], r=[0.1], wave_length=1.0)
    assert p.eps.dtype == np.complex128

def test_r_dtype_is_float64():
    p = ExperimentParameters(eps=[1.5, 1.0], r=[0.1], wave_length=1.0)
    assert p.r.dtype == np.float64

# ── Validation: r ─────────────────────────────────────────────────────────────

def test_2d_r_raises():
    with pytest.raises(ValueError, match="r must be a 1D"):
        ExperimentParameters(eps=[1.5, 1.0, 1.0], r=[[0.1, 0.2]])

def test_empty_r_raises():
    """len(r) == 0 should be caught by the r[0] > 0 guard."""
    with pytest.raises(ValueError, match=re.escape("r[0] must be positive.")):
        ExperimentParameters(eps=[1.0], r=[])

def test_r0_zero_raises():
    with pytest.raises(ValueError, match=re.escape("r[0] must be positive.")):
        ExperimentParameters(eps=[1.5, 1.0], r=[0.0])

def test_r0_negative_raises():
    with pytest.raises(ValueError, match=re.escape("r[0] must be positive.")):
        ExperimentParameters(eps=[1.5, 1.0], r=[-0.1])

def test_inner_radius_negative_raises():
    """r[0] is valid but a later radius is negative."""
    with pytest.raises(ValueError, match="All radii must be non-negative."):
        ExperimentParameters(eps=[1.5, 1.0, 1.0], r=[0.1, -0.05])

def test_unsorted_radii_raises():
    with pytest.raises(ValueError, match=re.escape("Radii must be in non-decreasing order")):
        ExperimentParameters(eps=[1.5, 1.0, 1.0], r=[0.3, 0.1])

def test_equal_radii_accepted():
    """Duplicate radii represent a zero-thickness layer — allowed."""
    ExperimentParameters(eps=[2.0, 1.5, 1.0], r=[0.1, 0.1], wave_length=1.0)

# ── Validation: eps ───────────────────────────────────────────────────────────

def test_2d_eps_raises():
    with pytest.raises(ValueError, match="eps must be a 1D"):
        ExperimentParameters(eps=[[1.5, 1.0]], r=[0.1])

def test_too_few_permittivities_raises():
    with pytest.raises(ValueError, match=re.escape("len(eps) must be at least len(r) + 1")):
        ExperimentParameters(eps=[1.5], r=[0.1, 0.2])

# ── Validation: wave_length ───────────────────────────────────────────────────

def test_zero_wavelength_raises():
    with pytest.raises(ValueError, match=re.escape("wave_length must be > 0")):
        ExperimentParameters(eps=[1.0, 1.0], r=[0.1], wave_length=0.0)

def test_negative_wavelength_raises():
    with pytest.raises(ValueError, match=re.escape("wave_length must be > 0")):
        ExperimentParameters(eps=[1.0, 1.0], r=[0.1], wave_length=-1.0)
   
# ── to_dict ───────────────────────────────────────────────────────────────────

def test_to_dict_round_trip():
    p = ExperimentParameters(
        eps=[2.0 + 0.1j, 1.0], r=[0.2],
        wave_length=0.5, conducting_core=True, label="test"
    )
    d = p.to_dict()
    p2 = ExperimentParameters(**d)
    np.testing.assert_array_almost_equal(p2.eps, p.eps)
    np.testing.assert_array_almost_equal(p2.r,   p.r)
    assert p2.conducting_core == p.conducting_core
    assert p2.wave_length     == p.wave_length
    assert p2.label           == p.label

def test_to_dict_contains_expected_keys():
    p = ExperimentParameters(eps=[1.5, 1.0], r=[0.1], wave_length=1.0)
    assert set(p.to_dict().keys()) == {"eps", "r", "conducting_core", "wave_length", "label"}


# ── __repr__ ──────────────────────────────────────────────────────────────────

def test_repr_smoke():
    """repr must not raise and should mention key fields."""
    p = ExperimentParameters(eps=[1.5, 1.0], r=[0.1], wave_length=0.03, label="x")
    r = repr(p)
    assert "x" in r
    assert "GHz" in r