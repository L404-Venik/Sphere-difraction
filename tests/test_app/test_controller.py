from core import ExperimentParameters
from app.application.controller import AppController


def test_controller_set_wave_length_requests_compute(monkeypatch):
    controller = AppController()
    captured = []

    def fake_request_compute(params, M=3600):
        captured.append((float(params.wave_length), M))

    monkeypatch.setattr(controller._computation_manager, "request_compute", fake_request_compute)
    controller.set_wave_length(0.75)
    controller.shutdown()

    assert captured == [(0.75, 3600)]
