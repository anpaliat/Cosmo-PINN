"""Figures for a Cosmo-PINN run.

    python plot.py results/pantheon_plus
    python plot.py results/pantheon_plus --format pdf

Reads run_results.npz and, when present, bayes_bands.npz and hmc_chain.npz
from the run directory and writes the figures into <run>/figures/.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR = "#1f4e79"
COLOR_MAP = "#c0392b"
COLOR_DATA = "black"

LABELS = {
    "w_DE": r"$w_{\rm DE}(z)$",
    "H": r"$H(z)\,[{\rm km\,s^{-1}\,Mpc^{-1}}]$",
    "Omega_DE": r"$\Omega_{\rm DE}(z)$",
    "rho_DE": r"$\rho_{\rm DE}(z)/\rho_{\rm DE,0}$",
    "q": r"$q(z)$",
}


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 12, "axes.labelsize": 13,
        "axes.titlesize": 13, "legend.fontsize": 10, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "savefig.bbox": "tight",
    })


def _save(fig, out_dir: str, name: str, formats: Sequence[str]):
    for ext in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def _load(path: str):
    return np.load(path, allow_pickle=False) if os.path.exists(path) else None


def _field(ax, key, run, bands, normalise=False):
    z = run["z"]
    curve = run[key] / (run[key][0] if normalise else 1.0)
    if bands is not None:
        zb = bands["z"]
        norm = bands[f"{key}_samples"][:, :1] if normalise else 1.0
        if normalise:
            s = bands[f"{key}_samples"] / norm
            lo95, lo68, med, hi68, hi95 = np.percentile(s, [2.5, 16, 50, 84, 97.5], axis=0)
        else:
            lo95, lo68, med, hi68, hi95 = (bands[f"{key}_{t}"] for t in
                                           ("lo95", "lo68", "median", "hi68", "hi95"))
        ax.fill_between(zb, lo95, hi95, color=COLOR, alpha=0.18, lw=0, label="95% CI")
        ax.fill_between(zb, lo68, hi68, color=COLOR, alpha=0.38, lw=0, label="68% CI")
        ax.plot(zb, med, color=COLOR, lw=1.8, label="posterior median")
        ax.plot(z, curve, color=COLOR_MAP, lw=1.2, ls="--", label="MAP")
    else:
        ax.plot(z, curve, color=COLOR, lw=1.8, label="MAP")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(LABELS[key])
    ax.set_xlim(0, z[-1])


def plot_wde(run, bands, out_dir, formats):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _field(ax, "w_DE", run, bands)
    ax.axhline(-1.0, color="gray", ls=":", lw=1.0, label=r"$\Lambda$CDM")
    ax.set_title(f"{run['wde_label']}  |  SN: {run['sn_dataset']}")
    ax.legend(frameon=False)
    _save(fig, out_dir, "fig_wde", formats)


def plot_background(run, bands, out_dir, formats):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _field(axes[0, 0], "H", run, bands)
    axes[0, 0].errorbar(run["z_data"], run["H_data"], yerr=run["sigma_data"], fmt="o", ms=3,
                        color=COLOR_DATA, capsize=2, elinewidth=0.8, label="CC")
    _field(axes[0, 1], "rho_DE", run, bands, normalise=True)
    axes[0, 1].axhline(1.0, color="gray", ls=":", lw=1.0)
    _field(axes[1, 0], "Omega_DE", run, bands)
    _field(axes[1, 1], "q", run, bands)
    axes[1, 1].axhline(0.0, color="gray", ls=":", lw=1.0)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig_background", formats)


def plot_training(run, out_dir, formats):
    ep = run["hist_epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for key, lab in (("pde", "PDE"), ("data", "CC"), ("bao", "BAO"), ("sn", "SN")):
        if np.any(run[f"hist_{key}"] > 0):
            axes[0].semilogy(ep, run[f"hist_{key}"], lw=1.4, label=lab)
    axes[0].set_ylabel("loss term")
    for key, lab in (("cc_chi2_dof", "CC"), ("bao_chi2_dof", "BAO"), ("sn_chi2_dof", "SN")):
        if np.any(run[f"hist_{key}"] > 0):
            axes[1].plot(ep, run[f"hist_{key}"], lw=1.4, label=lab)
    axes[1].axhline(1.0, color="gray", ls=":", lw=1.0)
    axes[1].set_ylabel(r"$\chi^2/N$")
    axes[1].set_yscale("log")
    ax3 = axes[2]
    ax3.plot(ep, run["hist_H0"], color=COLOR, lw=1.4, label=r"$H_0$")
    ax3.set_ylabel(r"$H_0$")
    ax4 = ax3.twinx()
    ax4.plot(ep, run["hist_Omega_m0"], color=COLOR_MAP, lw=1.4, label=r"$\Omega_{m0}$")
    ax4.set_ylabel(r"$\Omega_{m0}$")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig_training", formats)


def _hpd_levels(H: np.ndarray, probs=(0.95, 0.68)):
    flat = np.sort(H.ravel())[::-1]
    csum = np.cumsum(flat)
    csum = csum / csum[-1]
    return sorted(flat[np.searchsorted(csum, p)] for p in probs)


def plot_corner(samples: np.ndarray, labels: Sequence[str], out_dir, formats,
                truths: Optional[Sequence[float]] = None, bins: int = 40):
    from scipy.ndimage import gaussian_filter

    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2.6 * d, 2.6 * d), squeeze=False)
    ranges = []
    for k in range(d):
        lo, hi = np.percentile(samples[:, k], [0.5, 99.5])
        pad = 0.05 * (hi - lo + 1e-12)
        ranges.append((lo - pad, hi + pad))

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                x = samples[:, i]
                ax.hist(x, bins=bins, range=ranges[i], histtype="stepfilled", color=COLOR, alpha=0.35)
                ax.hist(x, bins=bins, range=ranges[i], histtype="step", color=COLOR, lw=1.2)
                lo, med, hi = np.percentile(x, [16, 50, 84])
                for v, ls in ((lo, "--"), (med, "-"), (hi, "--")):
                    ax.axvline(v, color=COLOR, lw=0.9, ls=ls)
                if truths is not None:
                    ax.axvline(truths[i], color=COLOR_MAP, lw=0.9)
                ax.set_yticks([])
                ax.set_title(f"{labels[i]} = ${med:.3g}^{{+{hi - med:.2g}}}_{{-{med - lo:.2g}}}$",
                             fontsize=10)
            else:
                H, xe, ye = np.histogram2d(samples[:, j], samples[:, i], bins=bins,
                                           range=[ranges[j], ranges[i]])
                H = gaussian_filter(H, sigma=1.0)
                levels = _hpd_levels(H)
                xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
                ax.contourf(xc, yc, H.T, levels=[levels[0], levels[1], H.max() + 1],
                            colors=["#bcd6f0", "#5b9bd5"], alpha=0.9)
                ax.contour(xc, yc, H.T, levels=levels, colors=COLOR, linewidths=0.8)
                ax.set_xlim(ranges[j])
                ax.set_ylim(ranges[i])
            if i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(labels[j])
            if j > 0:
                ax.set_yticklabels([])
            elif i > 0:
                ax.set_ylabel(labels[i])
    fig.tight_layout()
    _save(fig, out_dir, "fig_corner", formats)


def plot_traces(traces: np.ndarray, labels: Sequence[str], out_dir, formats):
    n_c, _, d = traces.shape
    fig, axes = plt.subplots(d, 1, figsize=(9, 1.8 * d), sharex=True, squeeze=False)
    for k in range(d):
        for c in range(n_c):
            axes[k, 0].plot(traces[c, :, k], lw=0.6, alpha=0.8, label=f"chain {c}")
        axes[k, 0].set_ylabel(labels[k])
    axes[-1, 0].set_xlabel("iteration")
    axes[0, 0].legend(frameon=False, ncol=n_c, fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig_traces", formats)


def make_figures(run_dir: str, formats: Sequence[str] = ("png", "pdf")) -> str:
    """Write every available figure for the run in run_dir; return the figure directory."""
    _style()
    run = _load(os.path.join(run_dir, "run_results.npz"))
    if run is None:
        raise FileNotFoundError(f"No run_results.npz in {run_dir}")
    bands = _load(os.path.join(run_dir, "bayes_bands.npz"))
    chain = _load(os.path.join(run_dir, "hmc_chain.npz"))
    out_dir = os.path.join(run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    plot_wde(run, bands, out_dir, formats)
    plot_background(run, bands, out_dir, formats)
    plot_training(run, out_dir, formats)

    if chain is not None and "scalar_traces" in chain.files:
        names = [r"$\Omega_{m0}$", r"$H_0$", r"$r_{\rm drag}$"]
        traces = chain["scalar_traces"]
        keep = [k for k in range(traces.shape[-1]) if np.std(traces[..., k]) > 1e-12]
        if keep:
            plot_traces(traces[..., keep], [names[k] for k in keep], out_dir, formats)
        if len(keep) >= 2:
            flat = traces.reshape(-1, traces.shape[-1])[:, keep]
            plot_corner(flat, [names[k] for k in keep], out_dir, formats,
                        truths=chain["prior_mu"][keep])

    print(f"Figures written to {out_dir}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="directory containing run_results.npz")
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="both")
    args = parser.parse_args()
    make_figures(args.run_dir, ("png", "pdf") if args.format == "both" else (args.format,))
