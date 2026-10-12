# Cosmo-PINN

**Cosmo-PINN** is a physics-informed neural network (PINN) that reconstructs the
dark-energy equation of state $w_{\rm DE}(z)$ directly from cosmological data.

A PINN is a neural network trained to fit the data and, at the same time, to satisfy a
differential equation. Here two functions are learned:
* the dark-energy density $\tilde\rho_{\rm DE}(z)$, from a neural network;
* the equation of state $w_{\rm DE}(z)$, from a second neural network.

They are tied together by the dark-energy continuity equation

$$
-(1+z)\frac{d\tilde\rho_{\rm DE}}{dz} + 3\,[1 + w_{\rm DE}(z)]\,\tilde\rho_{\rm DE}(z) = 0 ,
$$

and they determine the expansion history
$H^2(z) = H_0^2\,[\Omega_{m0}(1+z)^3 + \tilde\rho_{\rm DE}(z)]$ (the matter term includes
baryons). The network is trained on cosmic chronometers, DESI DR2 BAO and type Ia
supernovae (Pantheon+, Union3 or DES-Dovekie). $H_0$, $\Omega_{m0}$ and $r_{\rm drag}$ are
fitted together with the network. Training (Adam, then L-BFGS) is followed by Hamiltonian
Monte Carlo (NUTS), which gives credible bands on $w_{\rm DE}(z)$, $H(z)$ and the other
reconstructed quantities.

## Quick start

```bash
pip install -r requirements.txt
python main.py                          # train, sample and plot with config.yaml
python plot.py results/pantheon_plus    # remake the figures of a finished run
```

The `data/` folder contains **mock data**, so the code runs out of the box. To run the
real analysis, download the public datasets with the commands in [DATA.md](DATA.md).

## Changing the physics

The physics is written in two short files:
* [`cosmopinn/physics/hubble.py`](cosmopinn/physics/hubble.py) – the Hubble function $H(z)$ and the background fields;
* [`cosmopinn/physics/residual.py`](cosmopinn/physics/residual.py) – the differential equation imposed by the PINN.

Training, sampling and plotting use them without further changes.

## Documentation

* [INSTRUCTIONS.md](INSTRUCTIONS.md) – the model, how to change the physics, every
  configuration option, the pipeline, the outputs, and a reference for each module.
* [DATA.md](DATA.md) – how to download the data.

## Citation

If you use this code, please cite:

> Cosmo-PINN, [arXiv:2605.30139](https://arxiv.org/abs/2605.30139)

Please also cite the papers of the datasets you use (see [DATA.md](DATA.md)).
