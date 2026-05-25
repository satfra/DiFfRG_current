import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from diffrg_fv_diagnostics.dashboard import (
    CurveSpec,
    MapSpec,
    autoscale_y,
    available_panels,
    cell_boundary_coordinates,
    cell_interval_edges,
    choose_series,
    face_slope_margin_tick,
    format_margin_tick,
    metadata_advection_offset,
    parse_x_range,
    read_curve_data,
    resolve_advection_offset,
    Series,
    set_cell_boundary_grid,
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


def test_face_slope_margin_tick_formats_offset_from_convexity_boundary():
    series = Series(number=0, time=0.0)

    assert format_margin_tick(0.0) == "0"
    assert face_slope_margin_tick(-0.36, 0.6, series) == "0"
    assert face_slope_margin_tick(-0.36 + 1.25e-7, 0.6, series) == "1.3e-07"


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


def test_autoscale_y_can_include_warning_lines():
    figure, axis = plt.subplots()
    try:
        axis.plot([0.015, 0.035], [1.0e-8, 2.0e-8])
        (warning_line,) = axis.plot([0.015, 0.035], [-1.0, -1.0])
        warning_line.set_gid("warning-line")

        autoscale_y(axis, (0.015, 0.035), include_warning_lines=True)

        lower, upper = axis.get_ylim()
        assert lower < -0.9
        assert upper > 1.0e-8
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


def write_map(h5: h5py.File, name: str, coordinates: list[float]) -> None:
    group = h5.create_group(f"maps/{name}/0")
    group.create_dataset("coordinates", data=np.asarray(coordinates, dtype=float))
    group.create_dataset("data", data=np.zeros(len(coordinates), dtype=float))


def write_valued_map(h5: h5py.File, name: str, coordinates: list[float], values: list[float]) -> None:
    group = h5.create_group(f"maps/{name}/0")
    group.create_dataset("coordinates", data=np.asarray(coordinates, dtype=float))
    group.create_dataset("data", data=np.asarray(values, dtype=float))


def test_derived_mass_reconstructions_are_cellwise_constant(tmp_path):
    path = tmp_path / "diagnostic.h5"
    with h5py.File(path, "w") as h5:
        write_valued_map(h5, "cell_u", [1.0, 3.0], [1.0, 3.0])
        write_valued_map(h5, "cell_du_dx", [1.0, 3.0], [0.1, 0.2])
        write_valued_map(h5, "face_u_minus", [0.0, 2.0, 4.0], [0.0, 0.0, 0.0])

    with h5py.File(path, "r") as h5:
        specs = {
            "cell_u": MapSpec(file_index=0, file_path=path, name="cell_u"),
            "cell_du_dx": MapSpec(file_index=0, file_path=path, name="cell_du_dx"),
            "face_u_minus": MapSpec(file_index=0, file_path=path, name="face_u_minus"),
        }
        x_pion, y_pion = read_curve_data([h5], specs, "cell_m2_pion_constant", Series(number=0, time=0.0), 0,
                                         advection_offset=1.0)
        x_sigma, y_sigma = read_curve_data([h5], specs, "cell_m2_sigma_constant", Series(number=0, time=0.0), 0)

    np.testing.assert_allclose(x_pion, [0.0, 2.0, 2.0, 4.0])
    np.testing.assert_allclose(y_pion, [2.0, 2.0, 4.0 / 3.0, 4.0 / 3.0])
    np.testing.assert_allclose(x_sigma, [0.0, 2.0, 2.0, 4.0])
    np.testing.assert_allclose(y_sigma, [0.1, 0.1, 0.2, 0.2])


def test_cell_boundary_coordinates_prefers_face_coordinates(tmp_path):
    path = tmp_path / "diagnostic.h5"
    with h5py.File(path, "w") as h5:
        write_map(h5, "cell_u", [0.5, 1.5])
        write_map(h5, "face_u_minus", [0.0, 1.0, 2.0])

    with h5py.File(path, "r") as h5:
        specs = {
            "cell_u": MapSpec(file_index=0, file_path=path, name="cell_u"),
            "face_u_minus": MapSpec(file_index=0, file_path=path, name="face_u_minus"),
        }

        boundaries = cell_boundary_coordinates([h5], specs, Series(number=0, time=0.0))

    np.testing.assert_allclose(boundaries, [0.0, 1.0, 2.0])


def test_cell_boundary_coordinates_infers_residual_only_boundaries(tmp_path):
    path = tmp_path / "diagnostic.h5"
    with h5py.File(path, "w") as h5:
        write_map(h5, "cell_total_residual", [0.5, 1.5, 2.5])

    with h5py.File(path, "r") as h5:
        specs = {
            "cell_total_residual": MapSpec(file_index=0, file_path=path, name="cell_total_residual"),
        }

        boundaries = cell_boundary_coordinates([h5], specs, Series(number=0, time=0.0))

    np.testing.assert_allclose(boundaries, [0.0, 1.0, 2.0, 3.0])


def test_cell_boundary_coordinates_uses_face_only_coordinates(tmp_path):
    path = tmp_path / "diagnostic.h5"
    with h5py.File(path, "w") as h5:
        write_map(h5, "face_diffusion_flux", [0.0, 1.0, 2.0])

    with h5py.File(path, "r") as h5:
        specs = {
            "face_diffusion_flux": MapSpec(file_index=0, file_path=path, name="face_diffusion_flux"),
        }

        boundaries = cell_boundary_coordinates([h5], specs, Series(number=0, time=0.0))

    np.testing.assert_allclose(boundaries, [0.0, 1.0, 2.0])


def test_set_cell_boundary_grid_uses_axis_spanning_collection():
    figure, axis = plt.subplots()
    try:
        collection = set_cell_boundary_grid(axis, np.array([0.0, 1.0, 2.0]))

        assert collection.get_gid() == "cell-boundary-grid"
        assert collection in axis.collections
        assert len(collection.get_segments()) == 3

        set_cell_boundary_grid(axis, np.array([0.5, 1.5]), collection)

        segments = collection.get_segments()
        assert len(segments) == 2
        np.testing.assert_allclose(segments[0], [[0.5, 0.0], [0.5, 1.0]])
    finally:
        plt.close(figure)


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
