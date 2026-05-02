# test_parameters.py
import numpy as np
import re
import pytest
from core.Parameters import ExperimentParameters

def test_valid_single_layer():
    p = ExperimentParameters(eps=[1.5, 1.0], r=[0.1], wave_length=0.5)
    assert p.k == pytest.approx(2 * np.pi / 0.5)

def test_too_few_permittivities_raises():
    with pytest.raises(ValueError, match=re.escape("len(eps) must be at least len(r) + 1 (one permittivity per layer plus exterior).")):
        ExperimentParameters(eps=[1.5], r=[0.1, 0.2])

def test_negative_radius_raises():
    with pytest.raises(ValueError, match=re.escape("r[0] must be positive.")):
        ExperimentParameters(eps=[1.5, 1.0], r=[-0.1])

def test_unsorted_radii_raises():
    with pytest.raises(ValueError, match=re.escape("Radii must be in non-decreasing order (innermost -> outermost).")):
        ExperimentParameters(eps=[1.5, 1.0, 1.0], r=[0.3, 0.1])

def test_zero_wavelength_raises():
    with pytest.raises(ValueError, match=re.escape("wave_length must be > 0 (meters).")):
        ExperimentParameters(eps=[1.0, 1.0], r=[0.1], wave_length=0.0)

def test_frequency_consistency():
    import scipy.constants as const
    p = ExperimentParameters(eps=[1.0, 1.0], r=[0.1], wave_length=0.03)
    assert p.frequency_hz == pytest.approx(const.speed_of_light / 0.03)
    assert p.frequency_ghz == pytest.approx(p.frequency_hz / 1e9)