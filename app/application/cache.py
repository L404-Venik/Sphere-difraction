from __future__ import annotations

from typing import Any, Dict, Tuple

from core import ExperimentParameters
from .computation import ComputationResult

CacheKey = Tuple[Tuple[Any, ...], Tuple[float, ...], bool, float, int]


def make_cache_key(params: ExperimentParameters, M: int) -> CacheKey:
    return (
        tuple(params.eps.tolist()),
        tuple(params.r.tolist()),
        bool(params.conducting_core),
        float(params.wave_length),
        int(M),
    )


class ResultCache:
    def __init__(self) -> None:
        self._cache: Dict[CacheKey, ComputationResult] = {}

    def get(self, params: ExperimentParameters, M: int) -> ComputationResult | None:
        return self._cache.get(make_cache_key(params, M))

    def put(self, params: ExperimentParameters, M: int, result: ComputationResult) -> None:
        self._cache[make_cache_key(params, M)] = result

    def clear(self) -> None:
        self._cache.clear()
