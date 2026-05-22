import numpy as np

from diffrg_fv_diagnostics.oscillation_trace import (
    cell_curvature_indicator,
    dominant_residual_near_sigma,
    min_convexity_margin,
    select_trace_rows,
    trace_series,
)


class DummySeries:
    def __init__(self, number: int, time: float):
        self.number = number
        self.time = time


def test_min_convexity_margin_detects_face_and_side():
    x = np.array([0.021, 0.022, 0.023])
    minus = (x, np.array([-0.10, -0.20, -0.35]))
    plus = (x, np.array([-0.15, -0.30, -0.25]))

    margin = min_convexity_margin(0.6, 0.0, minus, plus)

    assert margin.side == "minus"
    assert margin.sigma == 0.023
    np.testing.assert_allclose(margin.value, 0.01)
    np.testing.assert_allclose(margin.slope, -0.35)


def test_cell_curvature_indicator_finds_largest_second_difference():
    x = np.array([0.0, 0.1, 0.2, 0.3])
    u = np.array([0.0, 0.0, 1.0, 0.0])

    curvature = cell_curvature_indicator(x, u)

    assert curvature.sigma == 0.2
    np.testing.assert_allclose(curvature.value, 200.0)


def test_dominant_residual_near_sigma_uses_largest_local_magnitude():
    x = np.array([0.0225, 0.0235])
    residuals = {
        "cell_advection_contribution": (x, np.array([0.1, 0.2])),
        "cell_diffusion_contribution": (x, np.array([0.3, -0.7])),
        "cell_source_contribution": (x, np.array([0.1, 0.1])),
    }

    dominant = dominant_residual_near_sigma(residuals, 0.023)

    assert dominant is not None
    assert dominant.name == "cell_diffusion_contribution"
    assert dominant.sigma == 0.0225
    np.testing.assert_allclose(dominant.value, 0.3)


def test_trace_series_combines_convexity_curvature_jump_and_residuals():
    x_face = np.array([0.022, 0.023, 0.024])
    x_cell = np.array([0.021, 0.022, 0.023, 0.024])
    maps = {
        "face_du_dx_minus": (x_face, np.array([-0.10, -0.35, -0.20])),
        "face_du_dx_plus": (x_face, np.array([-0.11, -0.20, -0.21])),
        "cell_u": (x_cell, np.array([0.0, 0.0, 1.0, 0.0])),
        "cell_advection_contribution": (x_cell, np.array([0.0, 0.1, 0.0, 0.0])),
        "cell_diffusion_contribution": (x_cell, np.array([0.0, 0.0, -0.8, 0.0])),
    }

    point = trace_series(DummySeries(7, 0.0), 0.6, lambda name, _series: maps[name])

    assert point.number == 7
    assert point.convexity.side == "minus"
    assert point.convexity.sigma == 0.023
    np.testing.assert_allclose(point.slope_jump, 0.15)
    assert point.dominant_residual is not None
    assert point.dominant_residual.name == "cell_diffusion_contribution"


def test_select_trace_rows_prefers_first_margin_crossings():
    points = [
        trace_series(DummySeries(0, 0.0), 0.6, lambda name, _series: {
            "face_du_dx_minus": (np.array([0.0]), np.array([-0.1])),
            "face_du_dx_plus": (np.array([0.0]), np.array([-0.1])),
            "cell_u": (np.array([0.0, 1.0, 2.0]), np.zeros(3)),
        }[name]),
        trace_series(DummySeries(1, 0.0), 0.6, lambda name, _series: {
            "face_du_dx_minus": (np.array([0.0]), np.array([-0.31])),
            "face_du_dx_plus": (np.array([0.0]), np.array([-0.1])),
            "cell_u": (np.array([0.0, 1.0, 2.0]), np.zeros(3)),
        }[name]),
    ]

    selected = select_trace_rows(points, convexity_margin=0.1, trace_count=1)

    assert [point.number for point in selected] == [1]
