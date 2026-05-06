from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread

from core import ExperimentParameters, calculate_S


@dataclass
class ComputationResult:
    params: ExperimentParameters
    M: int
    S_th: object
    S_ph: object


class Worker(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(ExperimentParameters, object, int)
    failed = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pending: tuple[ExperimentParameters, int] | None = None
        self._busy = False

    @pyqtSlot(object, int)
    def enqueue(self, params: ExperimentParameters, M: int) -> None:
        self._pending = (params, M)
        if self._busy:
            return
        self._process()

    @pyqtSlot()
    def cancel_pending(self) -> None:
        self._pending = None

    def _process(self) -> None:
        self._busy = True
        try:
            while self._pending is not None:
                params, M = self._pending
                self._pending = None
                self.started.emit()
                try:
                    S_th, S_ph = calculate_S(params, M=M)
                except Exception as exc:
                    self.failed.emit(str(exc))
                    continue
                self.finished.emit(params, (S_th, S_ph), M)
        finally:
            self._busy = False


class ComputationManager(QObject):
    started = pyqtSignal()
    finished = pyqtSignal(ExperimentParameters, object, int)
    failed = pyqtSignal(str)
    compute_requested = pyqtSignal(object, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._worker = Worker()
        self._worker.moveToThread(self._thread)

        self.compute_requested.connect(self._worker.enqueue)
        self._worker.started.connect(self.started)
        self._worker.finished.connect(self.finished)
        self._worker.failed.connect(self.failed)

        self._thread.started.connect(lambda: None)
        self._thread.start()

    def request_compute(self, params: ExperimentParameters, M: int = 3600) -> None:
        if M <= 0:
            raise ValueError("M must be a positive integer")
        self.compute_requested.emit(params, int(M))

    def cancel_pending(self) -> None:
        self._worker.cancel_pending()

    def shutdown(self) -> None:
        if self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
