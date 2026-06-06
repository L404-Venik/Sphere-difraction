from dataclasses import dataclass, field
from typing import Sequence, Optional, Tuple
import numpy as np
import scipy.constants as const
c = const.speed_of_light

@dataclass
class ExperimentParameters:
    """
    Holds parameters describing a layered spherical experiment.

    Fields:
      eps: array-like of relative permittivities (complex allowed). Length = num_layers + 1
      r: array-like of radii for layer boundaries (meters). Length = num_layers
      conducting_core: whether core is conducting (bool)
      wave_length: wavelength in meters (float)
      label: optional human-readable label

    Derived properties:
      frequency_hz, frequency_ghz: wave frequency
      k: wave number (rad / m)
    """
    eps: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex128))
    r: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    conducting_core: bool = False
    wave_length: float = 1.0  # meters
    label: Optional[str] = None

    def __post_init__(self):
        # Normalize inputs
        self.eps = np.asarray(self.eps, dtype=np.complex128)
        self.r = np.asarray(self.r, dtype=np.float64)

        # Basic validation with informative exceptions
        if self.r.ndim != 1:
            raise ValueError("r must be a 1D sequence of radii (meters).")
        if self.eps.ndim != 1:
            raise ValueError("eps must be a 1D sequence of permittivities (can be complex).")
        if len(self.eps) < len(self.r) + 1:
            raise ValueError("len(eps) must be at least len(r) + 1")
        if len(self.r) == 0 or self.r[0] <= 0:
            raise ValueError("r[0] must be positive.")
        if np.any(self.r < 0):
            raise ValueError("All radii must be non-negative.")
        if np.any(np.diff(self.r) < 0):
            raise ValueError("Radii must be in non-decreasing order (innermost -> outermost).")
        if self.wave_length <= 0:
            raise ValueError("wave_length must be > 0 (meters).")

    @property
    def frequency_hz(self) -> float:
        return c / self.wave_length

    @property
    def frequency_ghz(self) -> float:
        return self.frequency_hz / 1e9

    @property
    def k(self) -> float:
        """Wave number in radians per meter."""
        return 2.0 * np.pi / self.wave_length

    def create_label(self, create_from: Optional[str] = None) -> str:
        if self.label:
            return self.label
        if create_from == 'r':
            return " ".join(f"r{i}={float(val):g}" for i, val in enumerate(self.r))
        if create_from == 'eps':
            # skip core permittivity if conducting_core True (match your previous intention)
            start = 1 if self.conducting_core else 0
            eps_list = [f"ε{i}={self.eps[i]}" for i in range(start, len(self.eps)-1)]
            return " ".join(eps_list) if eps_list else "-"
        return "-"

    def to_dict(self) -> dict:
        return {
            "eps": self.eps.tolist(),
            "r": self.r.tolist(),
            "conducting_core": bool(self.conducting_core),
            "wave_length": float(self.wave_length),
            "label": self.label
        }

    def __repr__(self) -> str:
        return (f"ExperimentParameters(label={self.label!r}, num_layers={len(self.r)}, "
                f"wave_length={self.wave_length} m, freq={self.frequency_ghz:.3f} GHz)")


@dataclass
class PlotingParameters:
    title: str
    legend_title: str
    font_size: int
    show_ticklabels: bool

    experiment_description: str
    description_pos: tuple

    angle_limits: tuple
    
    experiments: ExperimentParameters
    polarization: str

    def __init__(self, Experiments, 
                 Title = None, 
                 LegendTitle = None, 
                 ExpDescr = None, 
                 DescriptionPos = [0.5, 0.1], 
                 FontSize = 8,
                 ShowTicklabels = False,
                 Polarization = None,
                 AngleLimits = None):
        
        self.experiments = Experiments
        self.font_size = FontSize

        assert len(DescriptionPos) == 2, 'this should be x and y of the experiment_descriprion'
        self.description_pos = DescriptionPos

        self.title = Title
        self.legend_title = LegendTitle

        self.experiment_description = ExpDescr

        self.show_ticklabels = ShowTicklabels
        
        if Polarization is not None:
            assert Polarization == 'VV' or Polarization == 'HH', 'polarization could be only VV or HH'
        self.polarization = Polarization
        
        self.angle_limits = AngleLimits