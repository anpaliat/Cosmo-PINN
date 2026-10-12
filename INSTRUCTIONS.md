# Cosmo-PINN: detailed instructions

Complete documentation of the code: the model, how to change the physics, installation,
configuration, pipeline, outputs and a reference for every module. For a short overview
see [README.md](README.md); for the data see [DATA.md](DATA.md).

If you use this code, please cite [arXiv:2605.30139](https://arxiv.org/abs/2605.30139).

---

## Contents

1. [Model](#1-model)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Repository layout](#4-repository-layout)
5. [Configuration reference](#5-configuration-reference)
6. [Pipeline](#6-pipeline)
7. [Outputs](#7-outputs)
8. [Code reference](#8-code-reference)
9. [Data](#9-data)
10. [Notes on reproducibility](#10-notes-on-reproducibility)

---

## 1. Model

The reconstruction uses the normalised redshift $x = z/z_{\max} \in [0,1]$ and a flat
universe with pressureless matter (baryons plus dark matter) and dark energy:

$$
E^2(z) = \frac{H^2(z)}{H_0^2} = (\Omega_{b0} + \Omega_{m0})(1+z)^3 + \tilde\rho_{\rm DE}(z),
\qquad \Omega_{\rm DE0} = 1 - \Omega_{m0} - \Omega_{b0}.
$$

Two functions are learned:

* **The dark-energy density.** A tanh MLP with Fourier input features returns $N_m(x)$, and
  $\tilde\rho_{\rm DE}(z) = \Omega_{\rm DE0}\, e^{x N_m(x)}$. This is positive by construction
  and satisfies $\tilde\rho_{\rm DE}(0) = \Omega_{\rm DE0}$ exactly.
* **The equation of state**, from a second tanh network:

  $$ w_{\rm DE}(x) = b + \sum_j a_j\, \phi_j(x), $$

  where $\phi_j(x)$ are the learned features of its last hidden layer and $(a_j, b)$ its
  output layer. The output layer is initialised at the ΛCDM point $a_j = 0$, $b = -1$, so
  training starts from $w_{\rm DE} = -1$.

The two functions are tied together by the dark-energy continuity equation, which is imposed
as the PINN residual:

$$
R(z) = -(1+z)\frac{d\tilde\rho_{\rm DE}}{dz} + 3\,[1 + w_{\rm DE}(z)]\,\tilde\rho_{\rm DE}(z) = 0 .
$$

$H_0$, $\Omega_{m0}$ and $r_{\rm drag}$ are trainable scalars with Gaussian priors.
$\Omega_{m0}$ is passed through a sigmoid; $H_0$ and $r_{\rm drag}$ through a softplus.
The physical baryon density $\omega_b = \Omega_b h^2$ is held fixed (BBN), so
$\Omega_{b0} = \omega_b / h^2$ follows the trainable $H_0$.

**Quintessence scenario.** Setting `lambda_wde_bound > 0` switches to the bounded
parametrisation $w_{\rm DE} = -1 + 2\,\sigma(s(x))$, where $s(x)$ is the network
output. This guarantees $-1 < w_{\rm DE} < 1$ during training and for every posterior
sample.

**Data:**

* **Cosmic chronometers (CC):** the 15 Moresco et al. (2012, 2015, 2016) $H(z)$
  measurements, with the full Moresco et al. (2020) covariance.
* **DESI DR2 BAO:** $D_V/r_d$, $D_M/r_d$ and $D_H/r_d$ with the full 13×13 covariance.
* **One SNIa compilation per run:** Pantheon+, Pantheon+&SH0ES, Union3 or DES-Dovekie. The
  likelihood is analytically marginalised over the absolute magnitude $M$.

### Changing the physics

The physics is written in two short files. Everything else (network, training, HMC, plots)
uses them without modification.

**1. The Hubble function: [`cosmopinn/physics/hubble.py`](cosmopinn/physics/hubble.py).**
`background(z, x, N_m, w_DE, Omega_m0, Omega_b0, H0)` builds $H(z)$ and all the background
fields from the network outputs and the cosmological parameters:

```python
Omega_DE0 = 1 - Omega_m0 - Omega_b0
rho_DE    = Omega_DE0 * exp(x * N_m)
E2        = (Omega_b0 + Omega_m0) * (1+z)**3 + rho_DE
H         = H0 * sqrt(E2)
```

It returns the dict of fields (`rho_DE`, `w_DE`, `H`, `h`, `Omega_DE`, `Omega_m`,
`Omega_b`) used by the likelihoods and by the residual. Any field added to this dict is
automatically available in `residual.py`, together with its derivative. `deceleration(f)` in
the same file gives $q(z)$ from the component equations of state; update it together with
`background` when you add a component. For example, radiation is added with:

```python
Omega_r0 = 9.0e-5
Omega_r = Omega_r0 * (1.0 + z) ** 4
Omega_DE0 = jnp.clip(1.0 - Omega_m0 - Omega_b0 - Omega_r0, 1e-6, 1.0 - 1e-6)
E2 = jnp.clip(Omega_b + Omega_m + Omega_r + rho_DE, 1e-12, None)
# also return "Omega_r": Omega_r, and in deceleration() add Omega_r to E2 and to the numerator
```

**2. The differential equation: [`cosmopinn/physics/residual.py`](cosmopinn/physics/residual.py).**
Both functions there receive the redshifts `z`, the dict `f` of fields from `background`,
and a dict `df` of their derivatives $d/dz$. They return an array that vanishes when the
physics holds:

```python
def pde_residual(z, f, df):
    return -(1.0 + z) * df["rho_DE"] + 3.0 * (1.0 + f["w_DE"]) * f["rho_DE"]
```

Editing `pde_residual` changes the training loss. `pde_residual_normalised` (by default
`pde_residual / (3 rho_DE)`) is used by the HMC likelihood and by the resampling, and
follows automatically. For example, an interaction $Q = \delta\,h\,\tilde\rho_{\rm DE}$
between dark energy and dark matter becomes

```python
def pde_residual(z, f, df):
    delta = 0.1
    Q = delta * f["h"] * f["rho_DE"]
    return -(1.0 + z) * df["rho_DE"] + 3.0 * (1.0 + f["w_DE"]) * f["rho_DE"] + Q
```

For a consistent interacting model, the matter density in `background` (`hubble.py`) must be
modified as well.

---

## 2. Installation

Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

The `data/` folder contains mock data. Download the real datasets as described in
[DATA.md](DATA.md) before running the analysis.

For GPU support, install the matching JAX build first (for example
`pip install "jax[cuda12]"`); see the JAX installation guide. All computations run in
float64 except the Adam phase, which by default runs in float32 for speed.

---

## 3. Quick start

```bash
python main.py                         # train + HMC + figures with config.yaml
python main.py my_run.yaml             # another configuration
python main.py my_run.yaml --no-plot   # skip figures
python plot.py results/pantheon_plus   # (re)make figures for a finished run
```

To reproduce the different runs of the paper, copy `config.yaml` and change:

| What to change | Key |
|---|---|
| SNIa compilation | `supernovae.dataset`: `pantheon_plus`, `union3`, `des_dovekie`, `pantheon_plus_sh0es`, `none` |
| Output directory | `output.save_dir` |
| Size of the $w_{\rm DE}$ network | `wde_representation.hidden_layers` |
| Quintessence (bounded $w_{\rm DE}$) | `loss_weights.lambda_wde_bound: 10.0` |
| Sensitivity of the PDE likelihood | `bayesian.pde_sigma` |

---

## 4. Repository layout

```
Cosmo-PINN/
├── main.py                 entry point: training, HMC, figures
├── plot.py                 all figures of a run
├── config.yaml             main configuration (paper setup)
├── README.md               short overview
├── INSTRUCTIONS.md         this file
├── DATA.md                 how to download the real data
├── requirements.txt
├── cosmopinn/
│   ├── __init__.py         enables float64 in JAX
│   ├── config.py           defaults, YAML loader, run summary
│   ├── physics/            the physics: edit these two files to change the model
│   │   ├── hubble.py       Hubble function H(z) and background fields
│   │   └── residual.py     differential equation imposed by the PINN
│   ├── training/           step 1: the PINN and its training
│   │   ├── model.py        neural networks, w_DE representation, fields and derivatives
│   │   ├── loss.py         training loss
│   │   ├── grids.py        collocation / integration grids, RAD resampling
│   │   └── trainer.py      Adam + GradNorm + RAD, L-BFGS, export
│   ├── data/               likelihoods
│   │   ├── cc.py           cosmic chronometers and Moresco covariance
│   │   ├── bao.py          BAO likelihood, distance integrals
│   │   └── sn.py           SNIa catalogues and M-marginalised likelihood
│   └── bayes/              step 2: posterior sampling
│       ├── packer.py       sampled parameters, theta <-> model
│       ├── posterior.py    log-posterior, posterior bands
│       ├── sampler.py      BlackJAX NUTS, R-hat, ESS
│       └── driver.py       orchestration and outputs
└── data/
    ├── cc/                 HzTable_MM_BC03.dat, data_MM20.dat
    ├── bao/                desi_gaussian_bao_ALL_GCcomb_{mean,cov}.txt
    └── sn/                 PantheonPlus/, Union3/, DES-Dovekie/
```

---

## 5. Configuration reference

Every key is optional; missing keys take the defaults in `cosmopinn/config.py`.
Relative data paths are resolved from the repository root.
`save_dir` is resolved from the working directory.

### `cosmology`
| key | meaning |
|---|---|
| `H0` | initial $H_0$ [km/s/Mpc] |
| `OMEGA_M0` | initial total matter density $\Omega_{m0}$ |
| `R_DRAG` | sound horizon at the drag epoch [Mpc] |
| `Z_MAX` | maximum redshift of the reconstruction (must cover the data) |
| `omega_b_h2` | physical baryon density $\Omega_b h^2$, held fixed |

### `priors`
One block each for `omega_m0`, `H0` and `r_drag`:
| key | meaning |
|---|---|
| `trainable` | train (and sample) the scalar, or keep it fixed |
| `prior_mu`, `prior_sigma` | Gaussian prior, used in training and in the HMC posterior |
| `lambda_prior` | weight of the prior term in the training loss |

### `wde_representation`, `network`
| key | meaning |
|---|---|
| `wde_representation.hidden_layers`, `fourier_freqs` | network for $w_{\rm DE}(x)$ |
| `network.hidden_layers`, `network.fourier_freqs` | network for $N_m(x)$ |

### `training`
| key | meaning |
|---|---|
| `adam_lr`, `adam_lr_min`, `adam_epochs`, `scheduler` | Adam with cosine (or step) decay |
| `max_grad_norm` | global gradient clipping |
| `lbfgs_epochs`, `lbfgs_max_iter` | L-BFGS: outer iterations × inner steps |
| `lbfgs_early_stop` | stop L-BFGS on a relative plateau |
| `n_collocation` | number of PDE collocation points |
| `n_bao_grid` | points of the grid used for distance integrals |
| `log_every` | logging interval (epochs) |
| `adam_dtype`, `dtype` | precision of the Adam phase / of L-BFGS and the outputs |
| `seed` | random seed for the network initialisation and RAD resampling |
| `resample_every`, `resample_start` | RAD resampling interval and first epoch |

### `loss_weights`
| key | term |
|---|---|
| `lambda_pde` | continuity residual $\langle R^2\rangle$ |
| `lambda_data`, `lambda_bao`, `lambda_sn` | CC, BAO and SN $\chi^2/N$ (BAO and SN soft-capped) |
| `lambda_smooth`, `lambda_dwde` | $\langle (w_{\rm DE}'')^2\rangle$, $\langle (w_{\rm DE}')^2\rangle$ in $x$ |
| `lambda_smooth_h`, `lambda_dh` | curvature of $h = H/H_0$; deviation of $dh/dx$ from its running average |
| `lambda_omega_smooth` | curvature of $\tilde\rho_{\rm DE}$ |
| `lambda_*_max` | maximum (instead of mean) of the three curvature terms |
| `lambda_bounds` | penalty keeping $0 \le \Omega_{\rm DE}(z) \le 1$ |
| `lambda_wde_bound` | > 0 selects the bounded quintessence parametrisation |

### `robust`
`soft_cap_c_sn` and `soft_cap_c_bao`: the soft cap $\rho(L) = c^2 \ln(1 + L/c^2)$ applied to
the SN and BAO $\chi^2/N$ during training.

### `balancing`
`mode: gradnorm` rescales the PDE, CC, BAO and SN weights every `gradnorm_every` epochs
towards equal gradient norms, with exponent `gradnorm_alpha`. Weights are clipped to
[`gradnorm_lam_lo`, `gradnorm_lam_hi`] and frozen after `gradnorm_freeze_after`. Any other
`mode` keeps the weights fixed.

### `cosmic_chronometers`
| key | meaning |
|---|---|
| `use_cc` | include CC in the training loss |
| `data_path` | H(z) table; the first three columns are z, H, σ (comma- or whitespace-separated) |
| `use_cov` | use the Moresco et al. (2020) covariance (otherwise diagonal) |
| `sys_path` | percentage systematics table `data_MM20.dat` |
| `cov_components` | any of `imf`, `slib`, `sps`, `spsooo` (default as in the CCcovariance notebooks) |

### `bao`
| key | meaning |
|---|---|
| `data_path` | three columns: z, value, quantity (`DV`, `DM` or `DH`, optionally with a suffix such as `_over_rs`, each divided by $r_{\rm drag}$) |
| `cov_path` | covariance matrix of the measurements, in the same order |

### `supernovae`
`dataset`: `pantheon_plus`, `pantheon_plus_sh0es`, `union3`, `des_dovekie` or `none`.

### `output`
`save_dir`: directory for all outputs of the run.

### `bayesian`
| key | meaning |
|---|---|
| `use_hmc` | run Phase 3 |
| `scope` | sampled parameters. `scalars`: $\Omega_{m0}, H_0, r_d$ only. `wnet`: scalars + output layer of the $w_{\rm DE}$ network. `wde`: as `wnet`, plus the last layer of the $\tilde\rho_{\rm DE}$ network (paper). `full`: all weights |
| `num_warmup`, `num_samples`, `num_chains` | NUTS settings |
| `vectorise_chains` | run the chains in parallel with `vmap` |
| `target_accept`, `max_tree_depth`, `dense_mass`, `init_step_size` | NUTS / window-adaptation settings |
| `jitter`, `seed` | over-dispersed chain initialisation around the MAP |
| `use_cc`, `use_bao`, `use_sn` | likelihood components |
| `use_pde`, `pde_sigma`, `pde_n_colloc` | Gaussian likelihood on $R/(3\tilde\rho_{\rm DE})$. `pde_sigma` is the allowed pointwise error on $w_{\rm DE}$ |
| `wnet_prior_sigma` | ΛCDM-centred prior $(a_j, b) \sim \mathcal N((0, -1), \sigma^2)$ on the output layer of the $w_{\rm DE}$ network. A value $\le 0$ uses the MAP-centred ridge instead |
| `weight_prior_sigma` | Gaussian prior around the MAP on the remaining sampled network weights |
| `band_n_z`, `thin`, `save_chain`, `dtype`, `log_every` | output and precision settings |

---

## 6. Pipeline

1. **Phase 1 – Adam.**
   * Minimises the loss of `loss.compute_loss`.
   * The PDE, CC, BAO and SN weights are balanced with GradNorm.
   * The collocation points are redistributed according to the continuity residual (RAD) every
     `resample_every` epochs after `resample_start`.
   * Blocks of epochs run inside a single `jax.lax.scan`.
2. **Phase 2 – L-BFGS.** Polishes the solution in float64 with the final weights. The result
   is the MAP network (`model.eqx`).
3. **Phase 3 – NUTS.**
   * Samples the posterior of the chosen `scope` around the MAP. The likelihood is
     CC + BAO + SN + PDE, as true $\chi^2$ values without soft caps.
   * The priors are the Gaussian priors on the scalars (with their Jacobians), the ΛCDM-centred
     prior on the output layer of the
     $w_{\rm DE}$ network, and a MAP-centred ridge on the remaining weights.
   * Writes bands of $H$, $w_{\rm DE}$, $\Omega_{\rm DE}$, $\tilde\rho_{\rm DE}$ and $q$, the
     convergence diagnostics (R-hat, bulk ESS, divergences).
4. **Figures.** `plot.make_figures` runs automatically unless `--no-plot` is given.

---

## 7. Outputs

All outputs are written to `save_dir`.

| file | content |
|---|---|
| `model.eqx` | MAP network (Equinox serialisation) |
| `run_results.npz` | MAP fields on 1000 redshifts (`z, H, w_DE, rho_DE, Omega_DE, Omega_m, Omega_b, q, q_autodiff`), the CC data used, learned scalars, and the training history (`hist_*`) |
| `bayes_bands.npz` | posterior samples and median/68%/95% bands of `H, w_DE, Omega_DE, rho_DE, q` on `band_n_z` redshifts, plus the scalar samples |
| `hmc_chain.npz` | raw chains, log-densities, R-hat, ESS, step sizes, divergences, scalar traces, masks and priors of the $w_{\rm DE}$ output layer |
| `hmc_summary.txt` | readable summary of diagnostics and scalar constraints |
| `figures/` | `fig_wde`, `fig_background`, `fig_training`, `fig_corner`, `fig_traces` (PNG and PDF) |

To load a trained model:

```python
import equinox as eqx, jax
import cosmopinn
from cosmopinn.config import load_cfg
from cosmopinn.training.model import WdePhysics, compute_physical_quantities

cfg = load_cfg("config.yaml")
skeleton = WdePhysics().build_model(cfg, jax.random.PRNGKey(0))
model = eqx.tree_deserialise_leaves("results/pantheon_plus/model.eqx", skeleton)
fields = compute_physical_quantities(model, jax.numpy.linspace(0, 2, 50).reshape(-1, 1), cfg)
```

---

## 8. Code reference

### `cosmopinn/config.py`
* `Cfg` – all settings and their defaults. `update_derived()` recomputes $\Omega_b$ and
  logit($\Omega_{m0}$) and resolves the data paths.
* `load_cfg(path)` – reads a YAML file into a `Cfg`.
* `print_summary(cfg)`, `wde_label(cfg)` – run summary and $w_{\rm DE}$ label.
* `as_jnp(spec)` – dtype string → `jnp` dtype.

### `cosmopinn/physics/hubble.py`
* `background(z, x, N_m, w_DE, Omega_m0, Omega_b0, H0)` – $H(z)$, $E^2(z)$ and the dict of
  background fields.
* `deceleration(f)` – $q = [\tfrac12(\Omega_b+\Omega_m) + \tfrac12\tilde\rho_{\rm DE}(1+3w_{\rm DE})]/E^2$.

### `cosmopinn/physics/residual.py`
* `pde_residual(z, f, df)` – the equation imposed in training, $R(z)$ above.
* `pde_residual_normalised(z, f, df)` – the same in units of $w_{\rm DE}$, $R/(3\tilde\rho_{\rm DE})$,
  used by the HMC likelihood (`bayesian.pde_sigma`) and by RAD resampling.

### `cosmopinn/training/model.py`
* `WdePINN(cfg, key)` – Equinox model:
  * `layers`: the $N_m$ MLP.
  * `w_layers`: the $w_{\rm DE}$ network.
  * Raw scalars. `Omega_m0()`, `H0()` and `R_drag()` return the transformed values.
  * `forward_scalar(x)` returns $(N_m, w_{\rm DE})$; `w_features(x)` returns the learned
    features of the $w_{\rm DE}$ network.
* `fields_single(model, consts, z)` – evaluates the network at one redshift and passes
  its outputs to `hubble.background`.
* `compute_physical_quantities(model, z, cfg)` – batched version for `(N, 1)` inputs.
* `fields_and_derivatives(model, consts, z)` – all fields and their $d/dz$ (forward-mode `jax.jvp`).
* `normalised_residual_at(model, consts, z)` – evaluates `pde_residual_normalised` (see
  `residual.py`); used by the HMC likelihood and by RAD.
* `deceleration_autodiff(...)` – $q = (1+z)H'/H - 1$.
* `WdePhysics` – interface used by the loss, trainer and sampler: `build_model`,
  `compute_physical_quantities`, `prior_loss`.

### `cosmopinn/training/loss.py`
* `compute_loss(physics, model, z_colloc, z_data, H_data, sigma_data, z_bao, sn_data, cfg, lambdas, dh_dx_ema)`
  returns a dict with the total loss, every individual term, the raw $\chi^2/N$ values and
  the current scalars.
  * First derivatives come from forward-mode `jax.jvp`.
  * Second derivatives come from `jacfwd`, on at most 500 points spread evenly over the
    sorted collocation set.
* `soft_cap(L, c)` – robust cap of a $\chi^2/N$.

### `cosmopinn/training/grids.py`
* `collocation_points`, `integration_grid` (uniform, refined around the BAO redshifts),
  `resample_collocation` (RAD).

### `cosmopinn/training/trainer.py`
* `Trainer(physics, cfg).run()` – Phases 1–3. It saves `model.eqx` and `run_results.npz`,
  then calls `bayes.driver.run_bayesian_phase`.

### `cosmopinn/data`
* `cc.load_cc_data(cfg)` – returns $(z, H, \sigma)$. With the covariance on,
  $\sigma = \sqrt{{\rm diag}\,C}$.
* `cc.build_cc_covariance(z, H, sigma, sys_path, components)` – implements
  $C_{ij} = \sigma_i^2\delta_{ij} + \sum_X H_i\eta_X(z_i)\,H_j\eta_X(z_j)$, applied to the
  Moresco 2012/2015/2016 measurements (`moresco_mask`), following
  [gitlab.com/mmoresco/CCcovariance](https://gitlab.com/mmoresco/CCcovariance).
* `cc.cc_whitened_residuals(H_model, H_data, sigma, cfg)` – returns
  $r_w = L^{-1}(H_{\rm model}-H_{\rm obs})$ with $C = LL^T$, so $\chi^2 = \sum r_w^2$.
* `bao.load_bao_data(cfg)` – reads the BAO measurements and covariance (cached).
  `bao.bao_redshifts(cfg)` returns their redshifts.
* `bao.compute_bao_chi2_total(H_grid, z_grid, r_drag, cfg)` – BAO $\chi^2$.
  `cumulative_comoving_distance` and `interp` are the differentiable distance integral and
  interpolation.
* `sn.build_sn_data(cfg)` – loads the selected catalogue, cuts it to $0 < z < z_{\max}$ and
  removes the calibrators for Pantheon+. Factorises the covariance once.
* `sn.compute_sn_chi2_total(...)` – computes $\chi^2 = A - B^2/C$ with
  $A = \Delta^T C^{-1}\Delta$, $B = \mathbf 1^T C^{-1}\Delta$ and $C = \mathbf 1^T C^{-1}\mathbf 1$.
  For Pantheon+&SH0ES, the calibrators use the Cepheid distances.

### `cosmopinn/bayes`
* `packer.make_packer(model, cfg, scope)` – flattens the sampled parameters into
  `theta`. It holds the masks, the ΛCDM point and the prior widths of the output layer
  of the $w_{\rm DE}$ network. `reconstruct(theta)` rebuilds the model.
* `posterior.make_log_posterior(...)` – returns `logdensity(theta)` for NUTS.
* `posterior.propagate_samples_to_bands(...)` – computes the field bands.
* `sampler.run_nuts_chains(...)` – runs BlackJAX NUTS with window adaptation (dense or
  diagonal mass), with the chains vmapped or sequential. `gelman_rubin` and
  `ess_per_parameter` give the diagnostics (bulk ESS from ArviZ).
* `driver.run_bayesian_phase(...)` – Phase 3 and its output files.

### `plot.py`
`make_figures(run_dir)` writes:
* `fig_wde` – $w_{\rm DE}(z)$.
* `fig_background` – $H(z)$ with the CC data, $\tilde\rho_{\rm DE}/\tilde\rho_{\rm DE,0}$,
  $\Omega_{\rm DE}$ and $q$.
* `fig_training` – loss terms, $\chi^2/N$ and the scalars against epoch.
* `fig_corner` and `fig_traces` – the cosmological scalars.

The posterior bands are used when `bayes_bands.npz` exists; otherwise the MAP curves are
shown.

---

## 9. Data

The repository ships **mock data** in the formats of the real datasets, so the code runs
out of the box. To use the real data, follow [DATA.md](DATA.md), which downloads the
public files (Cobaya `bao_data` and `sn_data`, Moresco `CCcovariance`) into the same paths.

| file | content (real dataset) |
|---|---|
| `data/cc/HzTable_MM_BC03.dat` | Moresco et al. cosmic-chronometer $H(z)$ measurements (BC03) |
| `data/cc/data_MM20.dat` | CC systematics in percent (Moresco et al. 2020) |
| `data/bao/desi_gaussian_bao_ALL_GCcomb_mean.txt`, `..._cov.txt` | DESI DR2 BAO: $D_V$, $D_M$, $D_H$ over $r_{\rm drag}$ and their covariance |
| `data/sn/PantheonPlus/` | Pantheon+ / Pantheon+&SH0ES distance moduli and STAT+SYS covariance |
| `data/sn/Union3/` | Union3 binned distance moduli and covariance |
| `data/sn/DES-Dovekie/` | DES-SN 5-year sample (Dovekie recalibration) and inverse covariance |

Please cite the original papers of every dataset you use.

---

## 10. Notes on reproducibility

* Runs are seeded (`training.seed`, `bayesian.seed`). However, parallel floating-point
  reductions on CPU and GPU are not bit-reproducible, especially with the float32 Adam phase.
  Two identical runs therefore agree statistically, not digit by digit.
* The effective SN weight in the paper runs is `lambda_sn: 1.0`.
