"""Deterministic statistics for minimization experiments; no model calls."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("algorithm", "function", "seed", "fe", "best_fitness")


class AnalysisError(ValueError):
    """An experiment does not satisfy the declared data contract."""


def analyze_dataframe(
    frame: pd.DataFrame, function_name: str, *, monotonic_atol: float = 1e-10
) -> dict[str, Any]:
    """Validate all rows, then summarize one function using final FE per seed.

    FE grids and seed sets must agree within each function. At least two
    seeds per algorithm are required to define sample standard deviation.
    Input row order is irrelevant. The caller's frame is never modified.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise AnalysisError(f"Missing columns: {', '.join(missing)}")
    if not np.isfinite(monotonic_atol) or monotonic_atol < 0:
        raise AnalysisError("monotonic_atol must be finite and non-negative")
    df = frame.loc[:, REQUIRED_COLUMNS].copy()
    if df.empty:
        raise AnalysisError("CSV contains no experiment rows")
    if df.isna().any().any():
        raise AnalysisError("Missing values are not allowed")
    for column in ("algorithm", "function"):
        if not df[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise AnalysisError(f"{column} must contain non-empty strings")
        if df[column].str.contains(r"[|\r\n`]").any():
            raise AnalysisError(f"{column} contains unsupported Markdown delimiters")
    for column in ("seed", "fe", "best_fitness"):
        if df[column].map(lambda x: isinstance(x, (bool, np.bool_))).any():
            raise AnalysisError(f"{column} must be numeric, not boolean")
        try:
            values = pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError(f"{column} must be numeric") from exc
        if not np.isfinite(values).all():
            raise AnalysisError(f"{column} must contain finite numbers")
        if column in ("seed", "fe"):
            if ((values <= 0) | (values != np.floor(values)) | (values > 2**53 - 1)).any():
                raise AnalysisError(f"{column} must contain positive, exactly representable integers")
            df[column] = values.astype(np.int64)
        else:
            df[column] = values
    keys = ["algorithm", "function", "seed", "fe"]
    if df.duplicated(keys).any():
        raise AnalysisError("Duplicate algorithm/function/seed/FE records")
    df = df.sort_values(keys).reset_index(drop=True)
    for func, function_df in df.groupby("function", sort=True):
        reference_grid = None
        reference_seeds = None
        for algorithm, algorithm_df in function_df.groupby("algorithm", sort=True):
            seeds = tuple(sorted(algorithm_df["seed"].unique()))
            if len(seeds) < 2:
                raise AnalysisError(f"{func}/{algorithm}: at least two seeds required for sample std")
            if reference_seeds is None:
                reference_seeds = seeds
            elif seeds != reference_seeds:
                raise AnalysisError(f"{func}: seed sets differ between algorithms")
            for seed, trajectory in algorithm_df.groupby("seed", sort=True):
                grid = tuple(trajectory["fe"])
                if reference_grid is None:
                    reference_grid = grid
                elif grid != reference_grid:
                    raise AnalysisError(f"{func}/{algorithm}/{seed}: inconsistent FE grid")
                if (np.diff(trajectory["best_fitness"].to_numpy()) > monotonic_atol).any():
                    raise AnalysisError(f"{func}/{algorithm}/{seed}: best-so-far fitness increases")
    selected = df[df["function"] == function_name]
    if selected.empty:
        raise AnalysisError(f"Function {function_name!r} not found")
    if selected["algorithm"].nunique() < 2:
        raise AnalysisError("Comparison requires at least two algorithms")
    algorithms = {}
    curves = {}
    for name, group in selected.groupby("algorithm", sort=True):
        finals = group.groupby("seed", sort=True).tail(1)
        fitness = finals["best_fitness"]
        algorithms[str(name)] = {
            "n_seeds": len(finals),
            "final_by_seed": [
                {"seed": int(row.seed), "fe": int(row.fe), "fitness": float(row.best_fitness)}
                for row in finals.itertuples()
            ],
            "mean": float(fitness.mean()),
            "sample_std": float(fitness.std(ddof=1)),
            "median": float(fitness.median()),
            "best": float(fitness.min()),
            "worst": float(fitness.max()),
        }
        curves[str(name)] = [
            {"fe": int(fe), "mean_fitness": float(value)}
            for fe, value in group.groupby("fe", sort=True)["best_fitness"].mean().items()
        ]
    # Reject numerical overflow in computed statistics instead of emitting NaN/Infinity.
    statistics = [v[key] for v in algorithms.values() for key in ("mean", "sample_std", "median", "best", "worst")]
    statistics += [p["mean_fitness"] for curve in curves.values() for p in curve]
    if not np.isfinite(statistics).all():
        raise AnalysisError("Computed statistics overflowed; check fitness scale")
    ranking = sorted(algorithms, key=lambda name: (algorithms[name]["mean"], name))
    best_mean = algorithms[ranking[0]]["mean"]
    return {
        "function": function_name,
        "objective": "minimize",
        "fe_grid": sorted(int(x) for x in selected["fe"].unique()),
        "monotonic_atol": float(monotonic_atol),
        "algorithms": algorithms,
        "mean_convergence": curves,
        "comparison": {
            "ranking_by_final_mean": ranking,
            "lowest_mean_algorithms": [a for a in ranking if algorithms[a]["mean"] == best_mean],
            "significance_tested": False,
        },
    }


def evidence_markdown(result: dict[str, Any]) -> str:
    """Numeric evidence tables only; the LLM must write its own interpretation.

    12 significant digits are displayed. Full precision remains in the JSON
    result. These exact tables let the CLI verify the report's numeric core.
    """
    lines = [
        f"Source: `{result['source']['csv_path']}`",
        f"SHA256: `{result['source']['sha256']}`",
        f"Function: `{result['function']}`; objective: minimize; std: ddof=1.",
        "",
        "| Algorithm | Seeds | Mean | Sample Std | Median | Best | Worst |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, stats in result["algorithms"].items():
        values = " | ".join(format(stats[k], ".12g") for k in ("mean", "sample_std", "median", "best", "worst"))
        lines.append(f"| {name} | {stats['n_seeds']} | {values} |")
    lines += ["", "| Algorithm | FE | Mean fitness |", "| --- | --- | --- |"]
    for name, curve in result["mean_convergence"].items():
        for point in curve:
            lines.append(f"| {name} | {point['fe']} | {point['mean_fitness']:.12g} |")
    return "\n".join(lines)


def analyze_csv(csv_path: Path | str, function_name: str, *, monotonic_atol: float = 1e-10) -> dict[str, Any]:
    """Read a CSV once, hash those exact bytes, and return JSON-compatible evidence."""
    path = Path(csv_path)
    try:
        raw = path.read_bytes()
        frame = pd.read_csv(BytesIO(raw), dtype={"algorithm": str, "function": str})
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise AnalysisError(f"Cannot read experiment CSV: {type(exc).__name__}") from exc
    result = analyze_dataframe(frame, function_name, monotonic_atol=monotonic_atol)
    result["source"] = {
        "csv_path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data_kind": "simulated_demo",
        "note": "Phase 1 inputs are simulated demonstration data, not formal CEC2013 experiments.",
    }
    result["evidence_markdown"] = evidence_markdown(result)
    return result
