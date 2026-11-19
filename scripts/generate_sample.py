"""Regenerate the synthetic demonstration CSV; never runs a CEC benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 20261119
DEMO_FINAL_FE = 1000


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    grid = np.arange(100, DEMO_FINAL_FE + 1, 100)
    rows = []
    for algorithm, floor, rate in (("A", 10, 0.003), ("B", 6, 0.0035)):
        for seed in range(1, 6):
            amplitude = rng.uniform(180, 220)
            trajectory = floor + amplitude * np.exp(-rate * grid) + rng.uniform(0, 2, len(grid))
            trajectory = np.minimum.accumulate(trajectory)
            for fe, fitness in zip(grid, trajectory, strict=True):
                rows.append((algorithm, "F1", seed, int(fe), round(float(fitness), 8)))
    output = Path(__file__).resolve().parents[1] / "data" / "sample_experiments.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["algorithm", "function", "seed", "fe", "best_fitness"]).to_csv(output, index=False)
    print(f"Generated simulated demo data: {output}; demo FE budget={DEMO_FINAL_FE}")


if __name__ == "__main__":
    main()
