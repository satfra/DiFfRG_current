#!/usr/bin/env python3
"""Plot finite-volume diagnostics from DiFfRG HDF5 files.

Example:
  diffrg-fv-dashboard /tmp/run_fv_reconstruction_diagnostics.h5
  diffrg-fv-dashboard /tmp/run_fv_residual_contribution_diagnostics.h5

Optional examples:
  diffrg-fv-dashboard diagnostic.h5 --max-slices 10
  diffrg-fv-dashboard diagnostic.h5 --series 0,25,50,100,172
  diffrg-fv-dashboard diagnostic.h5 --component 0 --output diagnostic.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from contextlib import ExitStack
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import RangeSlider, Slider
import numpy as np

from diffrg_fv_diagnostics.oscillation_trace import (
    OscillationTracePoint,
    RESIDUAL_MAP_LABELS,
    select_trace_rows,
    trace_series,
)


RECONSTRUCTION_MAP_NAMES = (
    "cell_u",
    "cell_du_dx",
    "cell_u_constant",
    "cell_u_reconstruction",
    "face_u_minus",
    "face_u_plus",
    "face_du_dx_minus",
    "face_du_dx_plus",
    "face_advection_flux",
    "face_diffusion_flux",
    "face_total_flux",
)

RESIDUAL_CONTRIBUTION_MAP_NAMES = (
    "cell_advection_contribution",
    "cell_diffusion_contribution",
    "cell_source_contribution",
    "cell_mass_contribution",
    "cell_total_residual",
    "cell_total_minus_residual",
)

MAP_NAMES = RECONSTRUCTION_MAP_NAMES + RESIDUAL_CONTRIBUTION_MAP_NAMES
DEFAULT_LAMBDA_UV = 0.6
WARNING_LINE_PANELS = {"Cell state and reconstruction", "Face slopes"}
CELL_STATE_PANEL_TITLE = "Cell state and reconstruction"
CELL_COORDINATE_MAP_NAMES = (
    "cell_u",
    "cell_du_dx",
) + RESIDUAL_CONTRIBUTION_MAP_NAMES
FACE_COORDINATE_MAP_NAMES = (
    "face_u_minus",
    "face_u_plus",
    "face_du_dx_minus",
    "face_du_dx_plus",
    "face_advection_flux",
    "face_diffusion_flux",
    "face_total_flux",
)


@dataclass(frozen=True)
class Series:
    number: int
    time: float


@dataclass(frozen=True)
class MapSpec:
    file_index: int
    file_path: Path
    name: str


@dataclass(frozen=True)
class CurveSpec:
    map_name: str
    label: str
    style: str = "-"
    linewidth: float = 1.35
    alpha: float = 1.0


@dataclass(frozen=True)
class PanelSpec:
    title: str
    ylabel: str
    curves: tuple[CurveSpec, ...]


DASHBOARD_PANELS = (
    PanelSpec(
        title=r"Cell state and reconstruction",
        ylabel=r"$u(\sigma)$",
        curves=(
            CurveSpec("cell_u_constant", r"$u_i$", linewidth=1.15),
            CurveSpec("cell_u_reconstruction", r"$\tilde{u}_i(\sigma)$", style="--"),
        ),
    ),
    PanelSpec(
        title=r"Face slopes",
        ylabel=r"$\partial_\sigma u^\pm$",
        curves=(
            CurveSpec("face_du_dx_minus", r"$\partial_\sigma u^-$"),
            CurveSpec("face_du_dx_plus", r"$\partial_\sigma u^+$", style="--"),
        ),
    ),
    PanelSpec(
        title=r"Face fluxes",
        ylabel=r"$F$",
        curves=(
            CurveSpec("face_advection_flux", r"$H$"),
            CurveSpec("face_diffusion_flux", r"$D$", style="--"),
            CurveSpec("face_total_flux", r"$H-D$", style=":"),
        ),
    ),
    PanelSpec(
        title=r"Face diffusion flux",
        ylabel=r"$D$",
        curves=(
            CurveSpec("face_diffusion_flux", r"$D$"),
        ),
    ),
    PanelSpec(
        title=r"Residual flux contributions",
        ylabel=r"$R$",
        curves=(
            CurveSpec("cell_advection_contribution", r"$R_{\mathrm{adv}}$"),
            CurveSpec("cell_diffusion_contribution", r"$R_{\mathrm{diff}}$", style="--"),
        ),
    ),
    PanelSpec(
        title=r"Local residual contributions",
        ylabel=r"$R$",
        curves=(
            CurveSpec("cell_source_contribution", r"$R_{\mathrm{src}}$"),
            CurveSpec("cell_mass_contribution", r"$R_{\dot{u}}$", style="--"),
        ),
    ),
    PanelSpec(
        title=r"Residual check",
        ylabel=r"$R$",
        curves=(
            CurveSpec("cell_total_residual", r"$R_{\mathrm{tot}}$"),
            CurveSpec("cell_total_minus_residual", r"$R_{\mathrm{tot}}-R$", style="--"),
        ),
    ),
)


def parse_csv_ints(value: str) -> list[int]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(int(item))
    return result


def parse_csv_floats(value: str) -> list[float]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(float(item))
    return result


def parse_x_range(value: str) -> tuple[float, float]:
    if ":" in value:
        lower, upper = value.split(":", maxsplit=1)
    elif "," in value:
        lower, upper = value.split(",", maxsplit=1)
    else:
        raise argparse.ArgumentTypeError("Expected x range as lower:upper or lower,upper.")
    lower_value = float(lower)
    upper_value = float(upper)
    if not lower_value < upper_value:
        raise argparse.ArgumentTypeError("Expected lower x bound to be smaller than upper x bound.")
    return lower_value, upper_value


def cutoff_scale(lambda_uv: float, series: Series) -> float:
    return lambda_uv * np.exp(-series.time)


def warning_line_value(lambda_uv: float, series: Series) -> float:
    k = cutoff_scale(lambda_uv, series)
    return -(k * k)


def warning_line_label(panel_title: str, advection_offset: float = 0.0) -> str:
    if panel_title == CELL_STATE_PANEL_TITLE:
        if advection_offset != 0.0:
            return r"$-k^2\sigma-c_\sigma$"
        return r"$-k^2\sigma$"
    return r"$-k^2$"


def warning_line_data(panel_title: str, lambda_uv: float, series: Series,
                      x_range: tuple[float, float], advection_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_range, dtype=float)
    y_value = warning_line_value(lambda_uv, series)
    if panel_title == CELL_STATE_PANEL_TITLE:
        return x, y_value * x - advection_offset
    return x, np.full_like(x, y_value, dtype=float)


def series_for_map(h5: h5py.File, map_name: str) -> list[Series]:
    group = h5[f"maps/{map_name}"]
    series = []
    for key in group.keys():
        if not key.isdigit():
            continue
        sub_group = group[key]
        series.append(Series(number=int(key), time=float(sub_group.attrs["time"])))
    return sorted(series, key=lambda item: item.number)


def choose_series(all_series: list[Series], max_slices: int | None, series_numbers: list[int] | None,
                  times: list[float] | None) -> list[Series]:
    if series_numbers is not None:
        by_number = {item.number: item for item in all_series}
        return [by_number[number] for number in series_numbers if number in by_number]

    if times is not None:
        selected = []
        for time in times:
            selected.append(min(all_series, key=lambda item: abs(item.time - time)))
        unique = {}
        for item in selected:
            unique[item.number] = item
        return list(sorted(unique.values(), key=lambda item: item.number))

    if max_slices is None or max_slices >= len(all_series):
        return all_series

    indices = np.linspace(0, len(all_series) - 1, max_slices, dtype=int)
    return [all_series[int(index)] for index in indices]


def read_coordinates(series_group: h5py.Group) -> np.ndarray:
    coordinates = np.asarray(series_group["coordinates"])
    if coordinates.dtype.fields:
        first_field = next(iter(coordinates.dtype.fields))
        coordinates = coordinates[first_field]
    if coordinates.ndim == 2 and coordinates.shape[1] == 1:
        coordinates = coordinates[:, 0]
    return np.asarray(coordinates, dtype=float).reshape(-1)


def read_component(series_group: h5py.Group, component: int) -> np.ndarray:
    data = np.asarray(series_group["data"])
    if data.dtype.fields:
        field_names = list(data.dtype.fields)
        field_name = f"component {component}"
        if field_name not in data.dtype.fields:
            if component >= len(field_names):
                raise IndexError(f"component {component} is not available; fields are {field_names}")
            field_name = field_names[component]
        data = data[field_name]
    elif data.ndim > 1:
        data = data[:, component]
    return np.asarray(data, dtype=float).reshape(-1)


def cell_interval_edges(cell_x: np.ndarray, face_x: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if face_x is not None and len(face_x) == len(cell_x) + 1:
        return face_x[:-1], face_x[1:]

    if len(cell_x) == 1:
        width = 1.0
        return cell_x - 0.5 * width, cell_x + 0.5 * width

    edges = np.empty(len(cell_x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (cell_x[:-1] + cell_x[1:])
    edges[0] = cell_x[0] - 0.5 * (cell_x[1] - cell_x[0])
    edges[-1] = cell_x[-1] + 0.5 * (cell_x[-1] - cell_x[-2])
    return edges[:-1], edges[1:]


def read_map_coordinates(h5_files: list[h5py.File], spec_by_name: dict[str, MapSpec], map_name: str,
                         series: Series) -> np.ndarray:
    spec = spec_by_name[map_name]
    series_group = h5_files[spec.file_index][f"maps/{spec.name}/{series.number}"]
    return read_coordinates(series_group)


def cell_boundary_coordinates(h5_files: list[h5py.File], spec_by_name: dict[str, MapSpec],
                              series: Series) -> np.ndarray:
    face_x = None
    for map_name in FACE_COORDINATE_MAP_NAMES:
        if map_name in spec_by_name:
            face_x = read_map_coordinates(h5_files, spec_by_name, map_name, series)
            break

    cell_x = None
    for map_name in CELL_COORDINATE_MAP_NAMES:
        if map_name in spec_by_name:
            cell_x = read_map_coordinates(h5_files, spec_by_name, map_name, series)
            break
    if cell_x is None or len(cell_x) == 0:
        if face_x is None:
            return np.empty(0, dtype=float)
        return face_x[np.isfinite(face_x)]

    if face_x is not None and len(face_x) != len(cell_x) + 1:
        face_x = None

    left_x, right_x = cell_interval_edges(cell_x, face_x)
    boundaries = np.concatenate((left_x[:1], right_x))
    return boundaries[np.isfinite(boundaries)]


def cell_boundary_segments(boundaries: np.ndarray) -> list[list[tuple[float, float]]]:
    return [[(float(boundary), 0.0), (float(boundary), 1.0)] for boundary in boundaries]


def set_cell_boundary_grid(axis: plt.Axes, boundaries: np.ndarray,
                           collection: LineCollection | None = None) -> LineCollection:
    if collection is None:
        collection = LineCollection([], transform=axis.get_xaxis_transform(), colors="0.65",
                                    linestyles=":", linewidths=0.6, alpha=0.45, zorder=0.5)
        collection.set_gid("cell-boundary-grid")
        axis.add_collection(collection, autolim=False)
    collection.set_segments(cell_boundary_segments(boundaries))
    return collection


def read_curve_data(h5_files: list[h5py.File], spec_by_name: dict[str, MapSpec], map_name: str, series: Series,
                    component: int) -> tuple[np.ndarray, np.ndarray]:
    if map_name in spec_by_name:
        spec = spec_by_name[map_name]
        series_group = h5_files[spec.file_index][f"maps/{spec.name}/{series.number}"]
        return read_coordinates(series_group), read_component(series_group, component)

    if map_name not in {"cell_u_constant", "cell_u_reconstruction"}:
        raise KeyError(f"Unsupported diagnostic map {map_name!r}.")

    cell_spec = spec_by_name["cell_u"]
    slope_spec = spec_by_name["cell_du_dx"]
    cell_group = h5_files[cell_spec.file_index][f"maps/{cell_spec.name}/{series.number}"]
    slope_group = h5_files[slope_spec.file_index][f"maps/{slope_spec.name}/{series.number}"]
    cell_x = read_coordinates(cell_group)
    cell_u = read_component(cell_group, component)
    cell_du_dx = read_component(slope_group, component)

    face_x = None
    face_spec = spec_by_name.get("face_u_minus") or spec_by_name.get("face_u_plus")
    if face_spec is not None:
        face_group = h5_files[face_spec.file_index][f"maps/{face_spec.name}/{series.number}"]
        face_x = read_coordinates(face_group)
    left_x, right_x = cell_interval_edges(cell_x, face_x)

    x = np.empty(2 * len(cell_x), dtype=float)
    y = np.empty(2 * len(cell_x), dtype=float)
    x[0::2] = left_x
    x[1::2] = right_x
    if map_name == "cell_u_constant":
        y[0::2] = cell_u
        y[1::2] = cell_u
    else:
        y[0::2] = cell_u + cell_du_dx * (left_x - cell_x)
        y[1::2] = cell_u + cell_du_dx * (right_x - cell_x)
    return x, y


def format_label(series: Series) -> str:
    return rf"$t={series.time:.6g}$  #{series.number}"


def diagnostic_title(h5_paths: list[Path], map_names: list[str]) -> str:
    if map_names and all(name in RESIDUAL_CONTRIBUTION_MAP_NAMES for name in map_names):
        return f"FV residual contribution diagnostics: {h5_paths[0].name}"
    elif map_names and all(name in RECONSTRUCTION_MAP_NAMES for name in map_names):
        return f"FV reconstruction diagnostics: {h5_paths[0].name}"
    return "FV diagnostics: " + ", ".join(path.name for path in h5_paths)


def autoscale_y(axis: plt.Axes, xlim: tuple[float, float]) -> None:
    visible_values = []
    lower, upper = xlim
    for line in axis.lines:
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        if line.get_gid() in {"warning-line", "diagnostic-marker"}:
            continue
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size == 0:
            continue

        mask = (x >= lower) & (x <= upper)
        if np.any(mask):
            visible_values.append(y[mask])
        if x.size < 2:
            continue

        edge_values = []
        for boundary in (lower, upper):
            x0 = x[:-1]
            x1 = x[1:]
            y0 = y[:-1]
            y1 = y[1:]
            spans_boundary = ((x0 <= boundary) & (boundary <= x1)) | ((x1 <= boundary) & (boundary <= x0))
            nonvertical = x0 != x1
            segment_mask = spans_boundary & nonvertical
            if np.any(segment_mask):
                fraction = (boundary - x0[segment_mask]) / (x1[segment_mask] - x0[segment_mask])
                edge_values.append(y0[segment_mask] + fraction * (y1[segment_mask] - y0[segment_mask]))
        if edge_values:
            visible_values.append(np.concatenate(edge_values))

    if not visible_values:
        axis.relim()
        axis.autoscale_view(scalex=False, scaley=True)
        return

    values = np.concatenate(visible_values)
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if y_min == y_max:
        margin = max(1e-12, abs(y_min) * 0.05)
    else:
        margin = 0.05 * (y_max - y_min)
    axis.set_ylim(y_min - margin, y_max + margin)


def apply_x_range(axes: Iterable[plt.Axes], xlim: tuple[float, float]) -> None:
    for axis in axes:
        axis.set_xlim(*xlim)
        autoscale_y(axis, xlim)


def available_maps(h5_files: list[h5py.File], h5_paths: list[Path]) -> list[MapSpec]:
    result = []
    seen = set()
    for file_index, h5 in enumerate(h5_files):
        for name in MAP_NAMES:
            if f"maps/{name}" not in h5:
                continue
            key = (file_index, name)
            if key in seen:
                continue
            seen.add(key)
            result.append(MapSpec(file_index=file_index, file_path=h5_paths[file_index], name=name))
    return result


def configuration_from_hdf5(h5_files: Iterable[h5py.File]) -> dict:
    for h5 in h5_files:
        value = h5.attrs.get("configuration_json")
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode()
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def metadata_advection_offset(h5_files: Iterable[h5py.File]) -> float | None:
    configuration = configuration_from_hdf5(h5_files)
    physical = configuration.get("physical")
    if not isinstance(physical, dict) or "cSigma" not in physical:
        return None
    return float(physical["cSigma"])


def resolve_advection_offset(h5_files: Iterable[h5py.File], cli_value: float | None) -> float:
    if cli_value is not None:
        return cli_value
    return metadata_advection_offset(h5_files) or 0.0


def common_series(h5_files: list[h5py.File], map_specs: Iterable[MapSpec]) -> list[Series]:
    common_numbers: set[int] | None = None
    by_number: dict[int, Series] = {}
    for spec in map_specs:
        series = series_for_map(h5_files[spec.file_index], spec.name)
        numbers = {item.number for item in series}
        common_numbers = numbers if common_numbers is None else common_numbers & numbers
        for item in series:
            by_number.setdefault(item.number, item)

    if common_numbers is None:
        return []
    return [by_number[number] for number in sorted(common_numbers)]


def sibling_diagnostic_paths(paths: Iterable[Path]) -> list[Path]:
    result = []
    seen = set()
    replacements = {
        "_fv_reconstruction_diagnostics.h5": "_fv_residual_contribution_diagnostics.h5",
        "_fv_residual_contribution_diagnostics.h5": "_fv_reconstruction_diagnostics.h5",
    }
    for path in paths:
        for candidate in (path,):
            if candidate not in seen:
                result.append(candidate)
                seen.add(candidate)
        for suffix, sibling_suffix in replacements.items():
            if path.name.endswith(suffix):
                sibling = path.with_name(path.name.removesuffix(suffix) + sibling_suffix)
                if sibling.exists() and sibling not in seen:
                    result.append(sibling)
                    seen.add(sibling)
    return result


def available_panels(map_specs: Iterable[MapSpec]) -> list[PanelSpec]:
    map_names = {spec.name for spec in map_specs}
    if {"cell_u", "cell_du_dx"} <= map_names and ({"face_u_minus", "face_u_plus"} & map_names):
        map_names = map_names | {"cell_u_constant", "cell_u_reconstruction"}
    panels = []
    for panel in DASHBOARD_PANELS:
        curves = tuple(curve for curve in panel.curves if curve.map_name in map_names)
        if curves:
            panels.append(PanelSpec(title=panel.title, ylabel=panel.ylabel, curves=curves))
    return panels


def print_oscillation_trace(trace: list[OscillationTracePoint], convexity_margin: float, trace_count: int) -> None:
    rows = select_trace_rows(trace, convexity_margin, trace_count)
    if not rows:
        print("[FV oscillation trace] no trace rows available")
        return

    print("[FV oscillation trace] series,time,critical_margin,critical_kind,sigma,side,mode_value,"
          "diffusion_margin,advection_margin,advection_singular_count,max_abs_d2u,sigma_d2u,"
          "slope_jump,sigma_jump,dominant_residual,residual_sigma,residual_value")
    for point in rows:
        residual_name = "unavailable"
        residual_sigma = float("nan")
        residual_value = float("nan")
        if point.dominant_residual is not None:
            residual_name = RESIDUAL_MAP_LABELS.get(point.dominant_residual.name, point.dominant_residual.name)
            residual_sigma = point.dominant_residual.sigma
            residual_value = point.dominant_residual.value
        print(f"[FV oscillation trace] {point.number},{point.time:.16e},{point.critical_margin.value:.16e},"
              f"{point.critical_margin.kind},{point.critical_margin.sigma:.16e},{point.critical_margin.side},"
              f"{point.critical_margin.slope:.16e},{point.diffusion_margin.value:.16e},"
              f"{point.advection_margin.value:.16e},{point.advection_margin.singular_count},"
              f"{point.curvature.value:.16e},{point.curvature.sigma:.16e},{point.slope_jump:.16e},"
              f"{point.slope_jump_sigma:.16e},{residual_name},{residual_sigma:.16e},{residual_value:.16e}")


def plot_oscillation_trace(axis_margin: plt.Axes, axis_curvature: plt.Axes, trace: list[OscillationTracePoint],
                           convexity_margin: float) -> None:
    times = np.asarray([point.time for point in trace], dtype=float)
    margins = np.asarray([point.critical_margin.value for point in trace], dtype=float)
    diffusion_margins = np.asarray([point.diffusion_margin.value for point in trace], dtype=float)
    advection_margins = np.asarray([point.advection_margin.value for point in trace], dtype=float)
    curvatures = np.asarray([point.curvature.value for point in trace], dtype=float)
    slope_jumps = np.asarray([point.slope_jump for point in trace], dtype=float)

    axis_margin.plot(times, margins, linewidth=1.35, label="critical")
    axis_margin.plot(times, diffusion_margins, "--", linewidth=1.0, label=r"$k^2+m_\sigma^2$")
    axis_margin.plot(times, advection_margins, ":", linewidth=1.0, label=r"$k^2+m_\pi^2$")
    axis_margin.axhline(convexity_margin, color="red", linestyle="--", linewidth=1.1, alpha=0.85,
                        label="margin")
    axis_margin.axhline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7, label="zero")
    axis_margin.set_title("Convexity margin history")
    axis_margin.set_xlabel(r"$t$")
    axis_margin.set_ylabel("margin")
    axis_margin.grid(True, alpha=0.25)
    axis_margin.legend(loc="best", fontsize="small", frameon=False)

    axis_curvature.plot(times, curvatures, linewidth=1.35, label=r"$\max |\Delta_\sigma^2 u|$")
    axis_curvature.plot(times, slope_jumps, "--", linewidth=1.15, label=r"$\max |u_\sigma^+-u_\sigma^-|$")
    axis_curvature.set_title("Oscillation indicator history")
    axis_curvature.set_xlabel(r"$t$")
    axis_curvature.set_ylabel("indicator")
    axis_curvature.grid(True, alpha=0.25)
    axis_curvature.legend(loc="best", fontsize="small", frameon=False)


def plot_maps(h5_paths: list[Path], selected_series: Iterable[Series], component: int, output: Path | None,
              xlim: tuple[float, float] | None, lambda_uv: float, oscillation_trace: bool,
              convexity_margin: float, trace_count: int, advection_offset_cli: float | None) -> None:
    selected_series = list(selected_series)
    if not selected_series:
        raise RuntimeError("No matching time slices were selected.")

    with ExitStack() as stack:
        h5_files = [stack.enter_context(h5py.File(path, "r")) for path in h5_paths]
        map_specs = available_maps(h5_files, h5_paths)
        if not map_specs:
            raise RuntimeError("No supported diagnostic maps were found in the HDF5 file.")
        map_names = [spec.name for spec in map_specs]
        spec_by_name = {spec.name: spec for spec in map_specs}
        advection_offset = resolve_advection_offset(h5_files, advection_offset_cli)
        panels = available_panels(map_specs)
        if not panels:
            raise RuntimeError("No supported dashboard panels can be built from the HDF5 file.")

        trace_points: list[OscillationTracePoint] = []
        trace_by_number: dict[int, OscillationTracePoint] = {}
        if oscillation_trace:
            required_trace_maps = {"cell_u", "face_u_minus", "face_u_plus", "face_du_dx_minus", "face_du_dx_plus"}
            missing_trace_maps = sorted(required_trace_maps - set(spec_by_name))
            if missing_trace_maps:
                raise RuntimeError("Oscillation trace requires missing maps: " + ", ".join(missing_trace_maps))

            all_trace_series = common_series(h5_files, map_specs)

            def read_trace_map(map_name: str, series: Series) -> tuple[np.ndarray, np.ndarray]:
                if map_name not in spec_by_name:
                    raise KeyError(map_name)
                return read_curve_data(h5_files, spec_by_name, map_name, series, component)

            trace_points = [
                trace_series(series, lambda_uv, read_trace_map, advection_offset) for series in all_trace_series
            ]
            trace_by_number = {point.number: point for point in trace_points}
            print_oscillation_trace(trace_points, convexity_margin, trace_count)

        n_columns = 2
        n_trace_panels = 2 if trace_points else 0
        n_rows = int(np.ceil((len(panels) + n_trace_panels) / n_columns))
        fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 3.2 * n_rows + 1.0),
                                 constrained_layout=output is not None)
        axes_flat = np.asarray(axes).reshape(-1)
        data_axes = axes_flat[:len(panels)]
        global_x_min = np.inf
        global_x_max = -np.inf
        time_lines: list[tuple[plt.Axes, str, object]] = []
        warning_lines: list[tuple[plt.Axes, str, tuple[float, float], object]] = []
        diagnostic_markers: list[tuple[plt.Axes, object]] = []
        boundary_grids: list[tuple[plt.Axes, LineCollection]] = []
        interactive = output is None

        for axis, panel in zip(data_axes, panels):
            series_to_plot = selected_series[:1] if interactive else selected_series
            axis_x_min = np.inf
            axis_x_max = -np.inf
            boundaries = cell_boundary_coordinates(h5_files, spec_by_name, series_to_plot[0])
            boundary_grid = set_cell_boundary_grid(axis, boundaries)
            if interactive:
                boundary_grids.append((axis, boundary_grid))
            for curve in panel.curves:
                for series_index, series in enumerate(series_to_plot):
                    x, y = read_curve_data(h5_files, spec_by_name, curve.map_name, series, component)
                    label = curve.label if interactive or series_index == 0 else "_nolegend_"
                    (line,) = axis.plot(x, y, curve.style, linewidth=curve.linewidth, alpha=curve.alpha,
                                        label=label)
                    if interactive:
                        time_lines.append((axis, curve.map_name, line))
                    axis_x_min = min(axis_x_min, float(np.min(x)))
                    axis_x_max = max(axis_x_max, float(np.max(x)))
                    global_x_min = min(global_x_min, float(np.min(x)))
                    global_x_max = max(global_x_max, float(np.max(x)))

            if panel.title in WARNING_LINE_PANELS and np.isfinite(axis_x_min) and np.isfinite(axis_x_max):
                x_range = (axis_x_min, axis_x_max)
                for series_index, series in enumerate(series_to_plot):
                    label = warning_line_label(panel.title, advection_offset) if interactive or series_index == 0 else "_nolegend_"
                    x_warning, y_warning = warning_line_data(panel.title, lambda_uv, series, x_range, advection_offset)
                    (warning_line,) = axis.plot(x_warning, y_warning, color="red", linestyle="--", linewidth=1.25,
                                                alpha=0.9, label=label)
                    warning_line.set_gid("warning-line")
                    if interactive:
                        warning_lines.append((axis, panel.title, x_range, warning_line))

            if trace_by_number:
                for series_index, series in enumerate(series_to_plot):
                    point = trace_by_number.get(series.number)
                    if point is None or not np.isfinite(point.critical_margin.sigma):
                        continue
                    label = r"trace $\sigma$" if interactive or series_index == 0 else "_nolegend_"
                    marker = axis.axvline(point.critical_margin.sigma, color="black", linestyle=":", linewidth=1.0,
                                          alpha=0.5, label=label)
                    marker.set_gid("diagnostic-marker")
                    if interactive:
                        diagnostic_markers.append((axis, marker))

            axis.set_title(panel.title)
            axis.set_xlabel(r"$\sigma$")
            axis.set_ylabel(panel.ylabel)
            axis.grid(True, axis="y", alpha=0.25)
            axis.legend(loc="best", fontsize="small", frameon=False)

        if trace_points:
            plot_oscillation_trace(axes_flat[len(panels)], axes_flat[len(panels) + 1], trace_points, convexity_margin)

        for axis in axes_flat[len(panels) + n_trace_panels:]:
            axis.set_visible(False)

        title = fig.suptitle(f"{diagnostic_title(h5_paths, map_names)}, component {component}")

        if xlim is not None:
            apply_x_range(data_axes, xlim)

        if output is not None:
            fig.savefig(output, dpi=180)
        else:
            fig.subplots_adjust(bottom=0.18, hspace=0.42, wspace=0.28)
            initial_xlim = xlim if xlim is not None else (global_x_min, global_x_max)
            apply_x_range(data_axes, initial_xlim)
            sigma_axis = fig.add_axes([0.12, 0.035, 0.76, 0.03])
            sigma_slider = RangeSlider(sigma_axis, r"$\sigma$", global_x_min, global_x_max, valinit=initial_xlim,
                                       valfmt="%.5g")

            def update_x(bounds: tuple[float, float]) -> None:
                apply_x_range(data_axes, bounds)
                fig.canvas.draw_idle()

            sigma_slider.on_changed(update_x)

            time_axis = fig.add_axes([0.12, 0.085, 0.76, 0.03])
            series_slider = Slider(time_axis, "series", 0, len(selected_series) - 1, valinit=0, valstep=1)

            def update_time(index_value: float) -> None:
                index = int(index_value)
                series = selected_series[index]
                for axis, map_name, line in time_lines:
                    x, y = read_curve_data(h5_files, spec_by_name, map_name, series, component)
                    line.set_data(x, y)
                    autoscale_y(axis, axis.get_xlim())
                for axis, panel_title, x_range, line in warning_lines:
                    x_warning, y_warning = warning_line_data(panel_title, lambda_uv, series, x_range, advection_offset)
                    line.set_data(x_warning, y_warning)
                    autoscale_y(axis, axis.get_xlim())
                for axis, marker in diagnostic_markers:
                    point = trace_by_number.get(series.number)
                    if point is not None and np.isfinite(point.critical_margin.sigma):
                        marker.set_xdata([point.critical_margin.sigma, point.critical_margin.sigma])
                    autoscale_y(axis, axis.get_xlim())
                for axis, boundary_grid in boundary_grids:
                    boundaries = cell_boundary_coordinates(h5_files, spec_by_name, series)
                    set_cell_boundary_grid(axis, boundaries, boundary_grid)
                title.set_text(f"{diagnostic_title(h5_paths, map_names)}, component {component} - "
                               f"{format_label(series)}")
                fig.canvas.draw_idle()

            update_time(0)
            series_slider.on_changed(update_time)
            plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_file", type=Path, nargs="+",
                        help="Path(s) to reconstruction and/or residual-contribution diagnostics HDF5 files. "
                             "A sibling diagnostic file is auto-added when present.")
    parser.add_argument("--component", type=int, default=0, help="Component index to plot.")
    parser.add_argument("--max-slices", type=int, default=6,
                        help="Number of evenly spaced time slices to plot when --series/--times is omitted.")
    parser.add_argument("--series", type=parse_csv_ints,
                        help="Comma-separated HDF5 series numbers, e.g. 0,25,50,100.")
    parser.add_argument("--times", type=parse_csv_floats,
                        help="Comma-separated times; nearest available series are plotted.")
    parser.add_argument("--xlim", type=parse_x_range,
                        help="Initial/static x-axis range as lower:upper, e.g. 0.015:0.035.")
    parser.add_argument("--lambda-uv", type=float, default=DEFAULT_LAMBDA_UV,
                        help=f"UV cutoff Lambda used for -k^2 and -k^2 sigma warning lines. "
                             f"Defaults to {DEFAULT_LAMBDA_UV}.")
    parser.add_argument("--oscillation-trace", action="store_true",
                        help="Add oscillator-origin trace tables and history panels.")
    parser.add_argument("--convexity-margin", type=float, default=0.1,
                        help="Warning threshold used when selecting oscillation trace rows. Defaults to 0.1.")
    parser.add_argument("--advection-offset", type=float,
                        help="Explicit-breaking offset c_sigma for the advection/pion convexity bound. "
                             "Defaults to /physical/cSigma metadata when available, otherwise 0.")
    parser.add_argument("--trace-count", type=int, default=12,
                        help="Maximum number of oscillation trace rows to print. Defaults to 12.")
    parser.add_argument("--output", type=Path, help="Optional image path. If omitted, opens an interactive window.")
    args = parser.parse_args()

    if args.series is not None and args.times is not None:
        raise SystemExit("Use either --series or --times, not both.")

    h5_paths = sibling_diagnostic_paths(args.hdf5_file)
    with ExitStack() as stack:
        h5_files = [stack.enter_context(h5py.File(path, "r")) for path in h5_paths]
        map_specs = available_maps(h5_files, h5_paths)
        if not map_specs:
            raise SystemExit("No supported diagnostic maps were found in the HDF5 file.")
        all_series = common_series(h5_files, map_specs)
    if args.output is None and args.series is None and args.times is None:
        selected_series = all_series
    else:
        selected_series = choose_series(all_series, args.max_slices, args.series, args.times)

    plot_maps(h5_paths, selected_series, args.component, args.output, args.xlim, args.lambda_uv,
              args.oscillation_trace, args.convexity_margin, args.trace_count, args.advection_offset)


if __name__ == "__main__":
    main()
