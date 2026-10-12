"""Cosmo-PINN: train the w_DE(z) PINN (Adam + L-BFGS), sample the posterior (NUTS), plot.

    python main.py                  # uses config.yaml
    python main.py my_config.yaml
    python main.py my_config.yaml --no-plot
"""

import argparse
import os

import cosmopinn  # noqa: F401  (enables float64)
from cosmopinn.config import load_cfg, print_summary
from cosmopinn.training.model import WdePhysics
from cosmopinn.training.trainer import Trainer

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default=os.path.join(HERE, "config.yaml"))
    parser.add_argument("--no-plot", action="store_true", help="skip the figures")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    print_summary(cfg)
    Trainer(WdePhysics(), cfg).run()

    if not args.no_plot:
        import plot
        plot.make_figures(cfg.SAVE_DIR)


if __name__ == "__main__":
    main()
