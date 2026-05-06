from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from core import ExperimentParameters
from .cache import ResultCache
from .computation import ComputationManager, ComputationResult


class AppController(QObject):
    computation_started = pyqtSignal()
    computation_finished = pyqtSignal(object)
    computation_failed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cache = ResultCache()
        self._computation_manager = ComputationManager()
        self._current_params = ExperimentParameters(
            eps=[1.0, 1.0],
            r=[1.0],
            conducting_core=True,
            wave_length=0.5,
            label="Default",
        )

        self._computation_manager.started.connect(self.computation_started)
        self._computation_manager.finished.connect(self._on_worker_finished)
        self._computation_manager.failed.connect(self.computation_failed)

    @property
    def current_parameters(self) -> ExperimentParameters:
        return self._current_params

    def request_compute(self, M: int = 3600) -> None:
        cached = self._cache.get(self._current_params, M)
        if cached is not None:
            self.computation_finished.emit(cached)
            return

        self.computation_started.emit()
        self._computation_manager.request_compute(self._current_params, M=M)

    def shutdown(self) -> None:
        self._computation_manager.shutdown()

    def _cache_result(self, result: ComputationResult) -> None:
        self._cache.put(result.params, result.M, result)

    def _is_current_result(self, params: ExperimentParameters, M: int) -> bool:
        return (
            tuple(params.eps.tolist()),
            tuple(params.r.tolist()),
            bool(params.conducting_core),
            float(params.wave_length),
            int(M),
        ) == (
            tuple(self._current_params.eps.tolist()),
            tuple(self._current_params.r.tolist()),
            bool(self._current_params.conducting_core),
            float(self._current_params.wave_length),
            int(M),
        )

    @pyqtSlot(float)
    def set_wave_length(self, wave_length: float) -> None:
        self._current_params = ExperimentParameters(
            eps=self._current_params.eps,
            r=self._current_params.r,
            conducting_core=self._current_params.conducting_core,
            wave_length=wave_length,
            label=self._current_params.label,
        )
        self.request_compute()

    @pyqtSlot(bool)
    def set_conducting_core(self, conducting: bool) -> None:
        self._current_params = ExperimentParameters(
            eps=self._current_params.eps,
            r=self._current_params.r,
            conducting_core=conducting,
            wave_length=self._current_params.wave_length,
            label=self._current_params.label,
        )
        self.request_compute()

    def set_experiment_parameters(self, params: ExperimentParameters) -> None:
        self._current_params = params
        self.request_compute()

    def _on_worker_finished(self, params: ExperimentParameters, result: object, M: int) -> None:
        computation_result = ComputationResult(params=params, M=M, S_th=result[0], S_ph=result[1])
        self._cache_result(computation_result)
        if self._is_current_result(params, M):
            self.computation_finished.emit(computation_result)
