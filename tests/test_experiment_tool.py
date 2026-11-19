import json
from pathlib import Path

import pytest

from research_agent.tools.experiment import make_analyze_experiment


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data").mkdir()
    source = Path(__file__).resolve().parents[1] / "data" / "sample_experiments.csv"
    (tmp_path / "data" / "demo.csv").write_bytes(source.read_bytes())
    return tmp_path


def test_tool_schema_and_json(workspace, monkeypatch):
    monkeypatch.chdir(workspace.parent)
    tool = make_analyze_experiment(workspace)
    assert tool.name == "analyze_experiment"
    assert set(tool.args) == {"csv_path", "function_name"}
    result = tool.invoke({"csv_path": "data/demo.csv", "function_name": "F1"})
    assert result["algorithms"]["A"]["n_seeds"] == 5
    assert result["source"]["csv_path"] == "data/demo.csv"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("path", ["../outside.csv", "/etc/passwd", "data/../../outside.csv", "data/missing.csv", "data/file.txt", "reports/report.csv"])
def test_unsafe_or_invalid_paths(workspace, path):
    result = make_analyze_experiment(workspace).invoke({"csv_path": path, "function_name": "F1"})
    assert isinstance(result, str)
    assert "CSV" in result


def test_outside_csv_and_symlink(workspace):
    outside = workspace.parent / "outside.csv"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "data" / "link.csv").symlink_to(outside)
    tool = make_analyze_experiment(workspace)
    for path in (str(outside), "data/link.csv"):
        assert "inside" in tool.invoke({"csv_path": path, "function_name": "F1"})


def test_data_directory_symlink_rejected(workspace):
    (workspace / "data").rename(workspace / "elsewhere")
    (workspace / "data").symlink_to(workspace / "elsewhere", target_is_directory=True)
    result = make_analyze_experiment(workspace).invoke({"csv_path": "data/demo.csv", "function_name": "F1"})
    assert "inside" in result


@pytest.mark.parametrize("content,message", [
    ("algorithm,function\nA,F1\n", "Missing columns"),
    ("algorithm,function,seed,fe,best_fitness\nA,F1,1,100,\n", "Missing values"),
])
def test_tool_reports_validation_errors(workspace, content, message):
    (workspace / "data" / "bad.csv").write_text(content, encoding="utf-8")
    result = make_analyze_experiment(workspace).invoke({"csv_path": "data/bad.csv", "function_name": "F1"})
    assert message in result
