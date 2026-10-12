"""Background cosmology: the Hubble function H(z) and the fields built from it.

This is the only place where the expansion history is written. To change the background
(for example to add radiation, curvature or an interaction with matter), edit `background`.
Every field it returns is available, with its derivative d/dz, in `residual.py`.
"""

import jax.numpy as jnp


def background(z, x, N_m, w_DE, Omega_m0, Omega_b0, H0):
    """Fields at redshift z.

    z         redshift; x = z / Z_MAX
    N_m       network output that modulates the dark-energy density
    w_DE      dark-energy equation of state (network output)
    Omega_m0  total matter density today (trainable)
    Omega_b0  baryon density today (from the fixed omega_b h^2 and H0)
    H0        Hubble constant [km/s/Mpc] (trainable)

    Flat universe, pressureless matter, and
    rho_DE(z) = Omega_DE0 exp(x N_m)  so that rho_DE(0) = Omega_DE0 = 1 - Omega_m0 - Omega_b0,
    E^2(z) = (Omega_b0 + Omega_m0) (1+z)^3 + rho_DE(z),   H(z) = H0 E(z).
    """
    Omega_DE0 = jnp.clip(1.0 - Omega_m0 - Omega_b0, 1e-6, 1.0 - 1e-6)
    Omega_b = Omega_b0 * (1.0 + z) ** 3
    Omega_m = Omega_m0 * (1.0 + z) ** 3

    """Definition of the dark energy density rho_DE(z) = Omega_DE0 exp(x N_m)"""
    rho_DE = Omega_DE0 * jnp.exp(x * N_m)

    E2 = jnp.clip(Omega_b + Omega_m + rho_DE, 1e-12, None)
    H = H0 * jnp.sqrt(E2)
    return {
        "rho_DE": rho_DE,
        "Omega_DE": rho_DE / E2,
        "Omega_m": Omega_m,
        "Omega_b": Omega_b,
        "H": H,
        "w_DE": w_DE,
        "h": H / H0,
    }


def deceleration(f):
    """q(z) from the component equations of state: matter w = 0, dark energy w_DE.

    q = [ (Omega_b + Omega_m)/2 + rho_DE (1 + 3 w_DE)/2 ] / E^2.
    Update it together with `background` if new components are added.
    """
    E2 = f["Omega_b"] + f["Omega_m"] + f["rho_DE"]
    return (0.5 * (f["Omega_b"] + f["Omega_m"]) + 0.5 * f["rho_DE"] * (1.0 + 3.0 * f["w_DE"])) / E2
