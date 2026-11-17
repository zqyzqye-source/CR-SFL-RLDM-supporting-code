"""Runnable examples for the recovery and MSE interfaces using synthetic data."""

import argparse
from pathlib import Path
import numpy as np

if __package__:
    from .data_recovery import em_recover_asym
    from .mse_visualization import ensemble_mse, plot_mse
else:
    from data_recovery import em_recover_asym
    from mse_visualization import ensemble_mse, plot_mse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/section_b"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed = np.array([-0.5, -0.1, 0.2, 1.5])
    predicted = np.zeros_like(observed)
    recovered = em_recover_asym(observed, predicted, 1.5, 0.5, 0.1)
    np.savetxt(args.output_dir / "recovery_example.csv",
               np.column_stack([observed, predicted, recovered]), delimiter=",",
               header="observed,predicted,recovered", comments="")
    rng = np.random.RandomState(2024)
    scale = 0.5 * np.exp(-np.arange(2000) / 400) + 0.05
    errors = rng.normal(size=(8, len(scale))) * scale
    mse = ensemble_mse(errors)
    np.savez(args.output_dir / "synthetic_errors.npz", errors=errors, mse=mse)
    plot_mse({"Synthetic error example": mse}, args.output_dir / "mse_example.pdf")
    print(f"Auxiliary examples saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
