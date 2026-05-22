import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from diffrg_fv_diagnostics.dashboard import (
    CurveSpec,
    MapSpec,
    autoscale_y,
    available_panels,
    cell_interval_edges,
    choose_series,
    metadata_advection_offset,
    parse_x_range,
    resolve_advection_offset,
    Series,
    warning_line_data,
    warning_line_label,
    warning_line_value,
)


def test_parse_x_range_accepts_colon_and_comma():
    assert parse_x_range("0.015:0.035") == (0.015, 0.035)
    assert parse_x_range("0.015,0.035") == (0.015, 0.035)


def test_parse_x_range_rejects_degenerate_range():
    try:
        parse_x_range("1:1")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("degenerate x range was accepted")


def test_choose_series_by_nearest_unique_times():
    series = [Series(number=0, time=0.0), Series(number=1, time=0.5), Series(number=2, time=1.0)]
    assert choose_series(series, max_slices=None, series_numbers=None, times=[0.49, 0.51]) == [
        Series(number=1, time=0.5)
    ]


def test_warning_line_value_uses_running_cutoff_scale():
    assert warning_line_value(0.6, Series(number=0, time=0.0)) == -0.36
    np.testing.assert_allclose(warning_line_value(0.6, Series(number=1, time=np.log(2.0))), -0.09)


def test_warning_line_data_uses_sigma_bound_for_cell_state_panel():
    x, y = warning_line_data("Cell state and reconstruction", 0.6, Series(number=0, time=0.0), (0.0, 0.5))

    np.testing.assert_allclose(x, [0.0, 0.5])
    np.testing.assert_allclose(y, [0.0, -0.18])
    assert warning_line_label("Cell state and reconstruction") == r"$-k^2\sigma$"


def test_warning_line_data_applies_advection_offset_to_cell_state_panel():
    x, y = warning_line_data("Cell state and reconstruction", 0.6, Series(number=0, time=0.0), (0.0, 0.5),
                             advection_offset=0.001695)

    np.testing.assert_allclose(x, [0.0, 0.5])
    np.testing.assert_allclose(y, [-0.001695, -0.181695])
    assert warning_line_label("Cell state and reconstruction", 0.001695) == r"$-k^2\sigma-c_\sigma$"


def test_warning_line_data_keeps_horizontal_bound_for_face_slopes():
    x, y = warning_line_data("Face slopes", 0.6, Series(number=0, time=0.0), (0.0, 0.5))

    np.testing.assert_allclose(x, [0.0, 0.5])
    np.testing.assert_allclose(y, [-0.36, -0.36])
    assert warning_line_label("Face slopes") == r"$-k^2$"


def test_warning_line_data_does_not_apply_advection_offset_to_face_slopes():
    x, y = warning_line_data("Face slopes", 0.6, Series(number=0, time=0.0), (0.0, 0.5),
                             advection_offset=0.001695)

    np.testing.assert_allclose(x, [0.0, 0.5])
    np.testing.assert_allclose(y, [-0.36, -0.36])


def test_metadata_advection_offset_reads_hdf5_configuration(tmp_path):
    path = tmp_path / "diagnostic.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["configuration_json"] = '{"physical":{"Lambda":0.6,"cSigma":0.001695}}'

    with h5py.File(path, "r") as h5:
        assert metadata_advection_offset([h5]) == 0.001695
        assert resolve_advection_offset([h5], None) == 0.001695
        assert resolve_advection_offset([h5], 0.2) == 0.2


def test_autoscale_y_ignores_warning_lines():
    figure, axis = plt.subplots()
    try:
        axis.plot([0.015, 0.035], [1.0e-8, 2.0e-8])
        (warning_line,) = axis.plot([0.015, 0.035], [-1.0, -1.0])
        warning_line.set_gid("warning-line")

        autoscale_y(axis, (0.015, 0.035))

        lower, upper = axis.get_ylim()
        assert lower > -1.0e-6
        assert upper < 1.0e-6
    finally:
        plt.close(figure)


def test_autoscale_y_includes_visible_line_segment_boundary_values():
    figure, axis = plt.subplots()
    try:
        axis.plot([0.0, 1.0], [-100.0, 100.0])

        autoscale_y(axis, (0.49, 0.51))

        lower, upper = axis.get_ylim()
        assert lower < -2.0
        assert upper > 2.0
    finally:
        plt.close(figure)


def test_cell_interval_edges_prefers_face_coordinates():
    left, right = cell_interval_edges(np.array([0.5, 1.5]), np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(left, [0.0, 1.0])
    np.testing.assert_allclose(right, [1.0, 2.0])


def test_cell_interval_edges_infers_missing_boundaries():
    left, right = cell_interval_edges(np.array([0.5, 1.5, 2.5]), None)
    np.testing.assert_allclose(left, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(right, [1.0, 2.0, 3.0])


def test_available_panels_synthesizes_cell_reconstruction_panel_for_legacy_maps():
    maps = [
        MapSpec(file_index=0, file_path=None, name="cell_u"),
        MapSpec(file_index=0, file_path=None, name="cell_du_dx"),
        MapSpec(file_index=0, file_path=None, name="face_u_minus"),
        MapSpec(file_index=0, file_path=None, name="face_advection_flux"),
    ]

    panels = available_panels(maps)

    assert panels[0].title == "Cell state and reconstruction"
    assert panels[0].curves == (
        CurveSpec("cell_u_constant", r"$u_i$", linewidth=1.15),
        CurveSpec("cell_u_reconstruction", r"$\tilde{u}_i(\sigma)$", style="--"),
    )


def test_available_panels_adds_standalone_face_diffusion_flux_panel():
    panels = available_panels([
        MapSpec(file_index=0, file_path=None, name="face_diffusion_flux"),
    ])

    assert any(panel.title == "Face diffusion flux" for panel in panels)
