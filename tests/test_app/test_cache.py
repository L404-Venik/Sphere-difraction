from core import ExperimentParameters
from app.application.cache import ResultCache
from app.application.computation import ComputationResult


def test_result_cache_stores_and_retrieves_by_parameters_and_fidelity():
    params = ExperimentParameters(eps=[1.0, 1.0], r=[0.1], wave_length=0.5)
    cache = ResultCache()
    assert cache.get(params, 3600) is None

    result = ComputationResult(params=params, M=3600, S_th=[1, 2, 3], S_ph=[4, 5, 6])
    cache.put(params, 3600, result)

    cached = cache.get(params, 3600)
    assert cached is result
    assert cache.get(params, 1800) is None
