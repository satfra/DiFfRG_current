"""Oscillation-origin diagnostics for finite-volume HDF5 output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


MapReader = Callable[[str, object], tuple[np.ndarray, np.ndarray]]

RESIDUAL_MAP_LABELS = {
    "cell_advection_contribution": "advection",
    "cell_diffusion_contribution": "diffusion",
    "cell_source_contribution": "source",
    "cell_mass_contribution": "mass",
    "cell_total_residual": "total",
}


@dataclass(frozen=True)
class ConvexityMargin:
    value: float
    sigma: float
    side: str
    slope: float


@dataclass(frozen=True)
class CurvatureIndicator:
    value: float
    sigma: float


@dataclass(frozen=True)
class ResidualAttribution:
    name: str
    value: float
    sigma: float


@dataclass(frozen=True)
class OscillationTracePoint:
    number: int
    time: float
    convexity: ConvexityMargin
    curvature: CurvatureIndicator
    slope_jump: float
    slope_jump_sigma: float
    dominant_residual: ResidualAttribution | None


def finite_values(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def min_convexity_margin(lambda_uv: float, time: float, minus: tuple[np.ndarray, np.ndarray],
                         plus: tuple[np.ndarray, np.ndarray]) -> ConvexityMargin:
    k = lambda_uv * np.exp(-time)
    k2 = k * k
    candidates = []
    for side, (x_raw, slope_raw) in (("minus", minus), ("plus", plus)):
        x, slope = finite_values(x_raw, slope_raw)
        if slope.size == 0:
            continue
        margin = k2 + slope
        index = int(np.argmin(margin))
        candidates.append(ConvexityMargin(float(margin[index]), float(x[index]), side, float(slope[index])))
    if not candidates:
        return ConvexityMargin(float("nan"), float("nan"), "none", float("nan"))
    return min(candidates, key=lambda item: item.value)


def cell_curvature_indicator(cell_x_raw: np.ndarray, cell_u_raw: np.ndarray) -> CurvatureIndicator:
    cell_x, cell_u = finite_values(cell_x_raw, cell_u_raw)
    if cell_u.size < 3:
        return CurvatureIndicator(float("nan"), float("nan"))
    dx = float(np.median(np.diff(cell_x)))
    second_difference = (cell_u[2:] - 2.0 * cell_u[1:-1] + cell_u[:-2]) / (dx * dx)
    index = int(np.argmax(np.abs(second_difference)))
    return CurvatureIndicator(float(abs(second_difference[index])), float(cell_x[index + 1]))


def face_slope_jump(minus: tuple[np.ndarray, np.ndarray], plus: tuple[np.ndarray, np.ndarray]) -> tuple[float, float]:
    x_minus, y_minus = finite_values(*minus)
    x_plus, y_plus = finite_values(*plus)
    if y_minus.size == 0 or y_plus.size == 0:
        return float("nan"), float("nan")
    count = min(y_minus.size, y_plus.size)
    jump = np.abs(y_plus[:count] - y_minus[:count])
    index = int(np.argmax(jump))
    return float(jump[index]), float(x_minus[index])


def dominant_residual_near_sigma(residuals: dict[str, tuple[np.ndarray, np.ndarray]], sigma: float) -> ResidualAttribution | None:
    best: ResidualAttribution | None = None
    for name, (x_raw, values_raw) in residuals.items():
        x, values = finite_values(x_raw, values_raw)
        if values.size == 0 or not np.isfinite(sigma):
            continue
        index = int(np.argmin(np.abs(x - sigma)))
        candidate = ResidualAttribution(name=name, value=float(values[index]), sigma=float(x[index]))
        if best is None or abs(candidate.value) > abs(best.value):
            best = candidate
    return best


def trace_series(series: object, lambda_uv: float, read_map: MapReader) -> OscillationTracePoint:
    minus = read_map("face_du_dx_minus", series)
    plus = read_map("face_du_dx_plus", series)
    convexity = min_convexity_margin(lambda_uv, series.time, minus, plus)

    cell_x, cell_u = read_map("cell_u", series)
    curvature = cell_curvature_indicator(cell_x, cell_u)
    slope_jump, slope_jump_sigma = face_slope_jump(minus, plus)

    residuals = {}
    for name in RESIDUAL_MAP_LABELS:
        try:
            residuals[name] = read_map(name, series)
        except KeyError:
            continue
    dominant = dominant_residual_near_sigma(residuals, convexity.sigma)

    return OscillationTracePoint(number=series.number, time=series.time, convexity=convexity, curvature=curvature,
                                 slope_jump=slope_jump, slope_jump_sigma=slope_jump_sigma,
                                 dominant_residual=dominant)


def select_trace_rows(trace: list[OscillationTracePoint], convexity_margin: float, trace_count: int) -> list[OscillationTracePoint]:
    if not trace:
        return []
    dangerous = [point for point in trace if np.isfinite(point.convexity.value) and point.convexity.value <= convexity_margin]
    if dangerous:
        return dangerous[:trace_count]
    return sorted(trace, key=lambda point: point.convexity.value)[:trace_count]
