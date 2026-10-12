"""Physics residual imposed by the PINN.

This is the only place where the differential equation is written. To change the physics,
edit the two functions below; training, HMC sampling and collocation resampling all use them.

Both functions receive, at an array of redshifts z:
    f   dict of fields:      rho_DE, w_DE, H, h, Omega_DE, Omega_m, Omega_b
    df  dict of d(field)/dz: same keys
and must return an array of the same shape as z that vanishes when the physics holds.
"""


def pde_residual(z, f, df):
    """Dark-energy continuity equation, d rho_DE / d ln a + 3 (1 + w_DE) rho_DE = 0.

    Training loss term: lambda_pde * mean(R^2).
    """
    return -(1.0 + z) * df["rho_DE"] + 3.0 * (1.0 + f["w_DE"]) * f["rho_DE"]


def pde_residual_normalised(z, f, df):
    """The residual divided by 3 rho_DE, i.e. in units of w_DE.

    Used in the HMC likelihood, where bayesian.pde_sigma is the allowed error on this
    quantity, and to place the collocation points during resampling.
    """
    return pde_residual(z, f, df) / (3.0 * f["rho_DE"])
