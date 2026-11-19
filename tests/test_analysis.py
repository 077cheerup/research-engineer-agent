import json
import math

import numpy as np
import pandas as pd
import pytest

from research_agent.analysis import AnalysisError, analyze_csv, analyze_dataframe


@pytest.fixture
def frame():
    return pd.DataFrame(
        [(a, "F1", s, fe, value)
         for a, finals in (("A", (2.0, 4.0)), ("B", (1.0, 3.0)))
         for s, final in enumerate(finals, 1)
         for fe, value in ((100, final + 10), (200, final))],
        columns=["algorithm", "function", "seed", "fe", "best_fitness"],
    )


def test_final_seed_statistics_and_sample_std(frame):
    original = frame.copy(deep=True)
    result = analyze_dataframe(frame.sample(frac=1, random_state=7), "F1")
    a = result["algorithms"]["A"]
    assert a["mean"] == 3.0  # All-FE mean would incorrectly be 8.
    assert a["sample_std"] == pytest.approx(math.sqrt(2))
    assert a["sample_std"] != np.std([2, 4], ddof=0)
    assert (a["median"], a["best"], a["worst"]) == (3, 2, 4)
    assert a["final_by_seed"] == [{"seed": 1, "fe": 200, "fitness": 2.0}, {"seed": 2, "fe": 200, "fitness": 4.0}]
    assert result["mean_convergence"]["A"] == [{"fe": 100, "mean_fitness": 13.0}, {"fe": 200, "mean_fitness": 3.0}]
    assert result["comparison"]["ranking_by_final_mean"] == ["B", "A"]
    assert result["comparison"]["significance_tested"] is False
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("column", ["algorithm", "function", "seed", "fe", "best_fitness"])
def test_missing_columns(frame, column):
    with pytest.raises(AnalysisError, match="Missing columns"):
        analyze_dataframe(frame.drop(columns=column), "F1")


@pytest.mark.parametrize("column,value", [
    ("algorithm", None), ("function", ""), ("seed", np.nan),
    ("fe", 0), ("fe", -1), ("fe", 1.5), ("fe", np.inf),
    ("fe", "bad"), ("seed", "bad"), ("best_fitness", np.nan),
    ("best_fitness", np.inf), ("best_fitness", -np.inf), ("best_fitness", "bad"),
])
def test_invalid_values(frame, column, value):
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(AnalysisError):
        analyze_dataframe(frame, "F1")


def test_duplicates(frame):
    with pytest.raises(AnalysisError, match="Duplicate"):
        analyze_dataframe(pd.concat([frame, frame.iloc[:1]]), "F1")


def test_grid_mismatch(frame):
    frame.loc[0, "fe"] = 99
    with pytest.raises(AnalysisError, match="FE grid"):
        analyze_dataframe(frame, "F1")


def test_missing_fe_point(frame):
    with pytest.raises(AnalysisError, match="FE grid"):
        analyze_dataframe(frame.iloc[1:], "F1")


def test_monotonicity_and_tolerance(frame):
    frame.loc[1, "best_fitness"] = frame.loc[0, "best_fitness"] + 1e-9
    with pytest.raises(AnalysisError, match="best-so-far"):
        analyze_dataframe(frame, "F1")
    analyze_dataframe(frame, "F1", monotonic_atol=1e-8)


def test_insufficient_seeds(frame):
    with pytest.raises(AnalysisError, match="at least two seeds"):
        analyze_dataframe(frame[frame.seed == 1], "F1")


def test_seed_set_mismatch(frame):
    frame.loc[(frame.algorithm == "B") & (frame.seed == 2), "seed"] = 3
    with pytest.raises(AnalysisError, match="seed sets"):
        analyze_dataframe(frame, "F1")


def test_missing_function_and_empty(frame):
    with pytest.raises(AnalysisError, match="not found"):
        analyze_dataframe(frame, "F2")
    with pytest.raises(AnalysisError, match="no experiment"):
        analyze_dataframe(frame.iloc[:0], "F1")


def test_tied_means(frame):
    frame.loc[frame.algorithm == "B", "best_fitness"] += 1
    result = analyze_dataframe(frame, "F1")
    assert result["comparison"]["lowest_mean_algorithms"] == ["A", "B"]


def test_csv_json_and_provenance(frame, tmp_path):
    import hashlib

    path = tmp_path / "demo.csv"
    frame.to_csv(path, index=False)
    result = analyze_csv(path, "F1")
    assert result["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "| A | 2 | 3 | 1.41421356237 | 3 | 2 | 4 |" in result["evidence_markdown"]
    json.dumps(result, allow_nan=False)
