from dataclasses import dataclass
import numpy as np
import scipy.constants as const
c = const.speed_of_light

@dataclass
class ExperimentParameters:
    eps: np.array
    r: np.array
    conducting_core: bool
    wave_length: float
    wave_frequency: float   # GHz
    k: float                # wave number

    _label: str

    def __init__(self, Eps, R, ConductingCore, WaveLength, Label = None):
        self.eps = Eps
        self.r = R
        self.conducting_core = ConductingCore
        self.wave_length = WaveLength

        self.k = 2 * np.pi / WaveLength
        self.wave_frequency = c / (WaveLength * 1e9)

        self._label = Label

    def create_label(self, label = None, CreateFrom = None ):
        self._label = label

        if CreateFrom == 'r':
            self._label = ''
            for i in range(len(self.r)):
                self._label  += f"r{i}={self.r[i]} "
        elif CreateFrom == 'eps':
            self._label = ''
            if self.conducting_core:
                for i in range(1,len(self.eps)-1):
                    self._label  += f"ε{i}={self.eps[i]} "
            else:
                for i in range(len(self.eps)-1):
                    self._label  += f"ε{i}={self.eps[i]} "

            if self._label == '':
                self._label = '-'
        else:
            assert False, 'can create automaticaly only from \'r\' or \'eps\''

    def get_label(self):
        return self._label
    
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