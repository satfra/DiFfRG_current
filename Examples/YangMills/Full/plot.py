#!/usr/bin/env python3
"""Port of plot.jl to Python.

Reads the Yang-Mills output HDF5 file and produces three figures:

  1. The gluon mass parameter m2A over the flow (with a zoom inset).
  2. The gluon (Z_A^-1) and ghost (Z_c) dressings over momentum.
  3. The running couplings g_A3, g_A4, g_Acbc over momentum.

The HDF5 layout has changed since the original Julia script: the three-point
vertices (ZA3, ZAcbc, ZA4tadpole) are now angle-resolved 3-D grids over
(|p1|, |p2|, cos(p1,p2)) on a coarse 8-point momentum grid, while ZA4 has been
replaced by the 4-gluon symmetric-point dressing ZA4SP (1-D). For the coupling
figure we evaluate the three-point vertices at the symmetric point (the diagonal
|p1| = |p2| = p, at the angle closest to cos = -1/2) and interpolate the 96-point
ZA / Zc dressings onto the coarse vertex grid.
"""

import shutil

import h5py
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Matplotlib style (mirrors the rcParams block in plot.jl).
# usetex is only enabled when a LaTeX installation is actually available, so the
# script also runs on machines without TeX. The labels below use \mathrm / \bar,
# which render identically under mathtext and usetex.
# ----------------------------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = shutil.which("latex") is not None
plt.rcParams["font.size"] = 11
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["xtick.minor.width"] = 0.6
plt.rcParams["ytick.minor.width"] = 0.6

# The main HDF5 file contains the FE data, all correlation functions and scalar objects.
main_file = "./build/output.h5"


def reload_data():
    """Load scalars and maps from the HDF5 file.

    Returns:
        scalars: dict of 1-D arrays, truncated to the length of the time axis.
        maps:    nested dict {name -> {step_index -> {"coordinates", "data"}}},
                 with the step indices sorted numerically.
    """
    with h5py.File(main_file, "r") as file:
        print(list(file.keys()))

        sg = file["scalars"]
        print(list(sg.keys()))
        n_time = sg["time"].shape[0]
        scalars = {s: sg[s][()][:n_time] for s in sg.keys()}

        mg = file["maps"]
        maps = {}
        for name in mg.keys():
            steps = sorted(mg[name].keys(), key=lambda x: int(x))
            maps[name] = {
                s: {
                    "coordinates": mg[name][s]["coordinates"][()],
                    "data": mg[name][s]["data"][()],
                }
                for s in steps
            }
    return scalars, maps


def last_step(maps, name):
    """Return the map entry for the final (largest-index) flow step."""
    steps = sorted(maps[name].keys(), key=lambda x: int(x))
    return maps[name][steps[-1]]


def coord_component(coords, component=0):
    """Extract one component of a structured coordinate array as a flat float array."""
    return coords[f"component {component}"]


scalars, maps = reload_data()

# ----------------------------------------------------------------------------
# Figure 1: m2A over the flow.
# ----------------------------------------------------------------------------
m2A = scalars["m2A"]
k = scalars["k"]

fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
ax.plot(k, m2A, label=r"$m_A^2$")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$m_A^2$")
ax.set_xscale("log")

# Inset to zoom in at small k.
inset_ax = fig.add_axes([0.3, 0.3, 0.35, 0.35])
inset_ax.plot(k, m2A)
inset_ax.set_xlim(0.0001, 1.0)
inset_ax.set_ylim(-0.04999, 0.05)
inset_ax.set_xscale("log")
fig
fig.savefig("m2A.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------------
# Figure 2: gluon and ghost dressings.
# ----------------------------------------------------------------------------
ZA = last_step(maps, "ZA")
dat_ZA = ZA["data"]
pGeV_ZA = coord_component(ZA["coordinates"], 0)

Zc = last_step(maps, "Zc")
dat_Zc = Zc["data"]
pGeV_Zc = coord_component(Zc["coordinates"], 0)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=150)

ax1.plot(pGeV_ZA, dat_ZA**-1, label=r"$Z_A$", color=plt.cm.tab10(3))
ax1.set_xlabel(r"$p\ \mathrm{[GeV]}$")
ax1.set_ylabel(r"$Z_A^{-1}$")
ax1.set_xscale("log")

ax2.plot(pGeV_Zc, dat_Zc, label=r"$Z_c$", color=plt.cm.tab10(0))
ax2.set_xlabel(r"$p\ \mathrm{[GeV]}$")
ax2.set_ylabel(r"$Z_c$")
ax2.set_xscale("log")
ax2.set_yscale("log")

fig2.tight_layout()
fig2.savefig("dressings.pdf")


# ----------------------------------------------------------------------------
# Figure 3: running couplings at the symmetric point.
#
# The three-point vertices live on a 3-D grid (|p1|, |p2|, cos) of shape
# (np, np, n_angles). The symmetric point is the diagonal |p1| = |p2| = p at the
# angle closest to cos = -1/2. ZA / Zc (96-point grid) are interpolated onto the
# coarse vertex momentum grid; ZA4SP is the 4-gluon symmetric point (1-D, 96).
# ----------------------------------------------------------------------------
def symmetric_point_slice(maps, name):
    """Diagonal |p1|=|p2|=p slice of a 3-point vertex at cos nearest -1/2.

    Returns (p, values) with p the symmetric-point momenta.
    """
    entry = last_step(maps, name)
    coords = entry["coordinates"]  # structured array, shape (np, np, n_angles)
    data = entry["data"].reshape(coords.shape)  # C order: index (i*np + j)*na + l

    cos_axis = coord_component(coords, 2)[0, 0, :]
    l = int(np.argmin(np.abs(cos_axis + 0.5)))

    p1 = coord_component(coords, 0)  # |p1| varies along axis 0
    n = data.shape[0]
    diag = np.arange(n)
    p = p1[diag, diag, l]
    vals = data[diag, diag, l]
    return p, vals


# Symmetric-point momenta and vertex dressings.
p3, dat_ZA3 = symmetric_point_slice(maps, "ZA3")
p_cbc, dat_ZAcbc = symmetric_point_slice(maps, "ZAcbc")


# Interpolate the propagator dressings onto the vertex momentum grid. We
# interpolate linearly in the dressing against log(p); ZA goes strongly negative
# at small momentum (the gluon mass), so a log-log interpolation is not valid.
def interp_log(p_target, p_src, y_src):
    return np.interp(np.log(p_target), np.log(p_src), y_src)


ZA_on_p3 = interp_log(p3, pGeV_ZA, dat_ZA)
ZA_on_cbc = interp_log(p_cbc, pGeV_ZA, dat_ZA)
Zc_on_cbc = interp_log(p_cbc, pGeV_Zc, dat_Zc)

# 4-gluon coupling from the symmetric-point dressing ZA4SP (1-D, 96-grid).
ZA4SP = last_step(maps, "ZA4SP")
dat_ZA4SP = ZA4SP["data"]
pGeV_ZA4 = coord_component(ZA4SP["coordinates"], 0)

gA3 = (dat_ZA3**2 / ZA_on_p3**3) / (4 * np.pi)
gA4 = (dat_ZA4SP / dat_ZA**2) / (4 * np.pi)
gAcbc = (dat_ZAcbc**2 / (ZA_on_cbc * Zc_on_cbc**2)) / (4 * np.pi)

fig3, ax3 = plt.subplots(figsize=(4, 3), dpi=300)
ax3.plot(p3, gA3, label=r"$g_{A3}$", color=plt.cm.tab10(3), marker="o", ms=3)
ax3.plot(pGeV_ZA4, gA4, label=r"$g_{A4}$", color=plt.cm.tab10(1))
ax3.plot(
    p_cbc, gAcbc, label=r"$g_{A\bar{c}c}$", color=plt.cm.tab10(0), marker="o", ms=3
)
ax3.set_xlabel(r"$p\ \mathrm{[GeV]}$")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.legend(frameon=False)
ax3.set_ylim(5e-5, 6)
ax3.set_xlim(1e-4, 100)
fig3.tight_layout()
fig3.savefig("couplings.pdf")

plt.show()
