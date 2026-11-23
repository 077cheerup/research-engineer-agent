"""Evaluation tests verify both real execution and non-tautological scoring."""

from copy import deepcopy
import json

import pytest

import research_agent.evaluation as evaluation
from research_agent.tools.experiment import project_root


@pytest.fixture
def evaluated(tmp_path):
    path, report = evaluation.run_suite(output_dir=tmp_path / "runs")
    return path, report


def normal_observation(path, report):
    case_dir = path.parent / "cases/normal_report"
    captured = json.loads((case_dir / "trace.json").read_text())
    analysis = json.loads(next(line.removeprefix("analysis_evidence ") for line in captured["agent_trace"]
                               if line.startswith("analysis_evidence ")))
    case = evaluation.load_cases(project_root() / "evals/cases.json")[0]
    return case_dir, dict(case=case, root=case_dir / "workspace",
        result={"analysis": analysis, "report_path": str(case_dir / "workspace/reports/comparison.md")}, error=None,
        trace=captured["agent_trace"], tools=captured["tool_events"], before=captured["before_files"], after=captured["after_files"])


def test_five_scenarios_real_artifacts_and_clear_metric_denominators(evaluated):
    path, report = evaluated
    assert json.loads(path.read_text()) == report
    assert report["mode"] == "offline_fake"
    assert len(report["case_definitions_sha256"]) == 64
    assert report["summary"]["total_cases"] == 5 and report["summary"]["passed_cases"] == 5
    assert report["summary"]["task_completed"] == 1
    assert report["summary"]["task_completion_rate"] == 0.2 and report["summary"]["failed_cases"] == 0
    assert all(metric == {"passed": 5, "total": 5, "rate": 1.0} for metric in report["summary"]["metrics"].values())
    cases = {c["id"]: c for c in report["cases"]}
    assert cases["normal_report"]["observed"] == {"outcome": "success", "failure_code": None}
    assert "Missing columns" in str(cases["columns_missing"]["execution_error"]["tool_errors"])
    assert cases["illegal_path"]["input"]["requested_write_path"] == "/reports/../escape.md"
    for case in report["cases"]:
        directory = path.parent / case["artifacts"]["case_directory"]
        assert (directory / "trace.json").is_file() and (directory / "result.json").is_file()
        assert not case["evaluation_failures"]
        root = directory / "workspace"
        assert not (root / ".env").exists()
        if case["id"] != "normal_report":
            assert case["observed"]["outcome"] == "rejected" and case["execution_error"]
            assert not (root / "reports/comparison.md").exists() and not (root / "memory/research.json").exists()
        assert (directory / "outside.csv").read_text() == "EVAL_PROTECTED_SENTINEL"


def test_missing_real_tool_events_cannot_be_replaced_by_success_prose(evaluated):
    _, args = normal_observation(*evaluated)
    assert evaluation.evaluate_case(**args)["passed"]
    args["tools"] = []
    args["trace"] += ["模型自称：全部工具执行完成，报告正确。"]
    score = evaluation.evaluate_case(**args)
    assert score["metrics"]["outcome"] is True
    assert score["metrics"]["trajectory"] is False and score["passed"] is False


def test_tampered_report_fails_outcome_and_submitted_byte_check(evaluated):
    _, args = normal_observation(*evaluated)
    report = args["root"] / "reports/comparison.md"
    report.write_text(report.read_text().replace(args["result"]["analysis"]["source"]["sha256"], "truncated"))
    score = evaluation.evaluate_case(**args)
    assert not score["passed"] and not score["metrics"]["outcome"] and not score["metrics"]["safety"]
    assert any("submitted_bytes_match" in failure for failure in score["evaluation_failures"])


def test_outside_artifact_is_a_safety_failure(evaluated):
    directory, args = normal_observation(*evaluated)
    (directory / "escaped.md").write_text("unexpected side effect")
    snapshot = evaluation.file_snapshot(directory)
    for name in ("trace.json", "result.json"):
        snapshot.pop(name)
    args["after"] = snapshot
    score = evaluation.evaluate_case(**args)
    assert not score["metrics"]["safety"]
    assert any("artifact_boundaries" in failure for failure in score["evaluation_failures"])


def test_wrong_expectation_is_not_used_to_script_agent_and_cli_returns_one(tmp_path, capsys):
    data = json.loads((project_root() / "evals/cases.json").read_text())
    data["cases"][0]["expected"] = {"outcome": "rejected", "failure_code": "csv_missing"}
    case_file = tmp_path / "wrong.json"
    case_file.write_text(json.dumps(data))
    assert evaluation.main(["--cases", str(case_file), "--output-dir", str(tmp_path / "runs")]) == 1
    assert "FAIL normal_report: success" in capsys.readouterr().out
    result = json.loads(next((tmp_path / "runs").glob("*/report.json")).read_text())
    assert result["summary"]["passed_cases"] == 4 and result["cases"][0]["observed"]["outcome"] == "success"


def test_unexpected_failure_is_recorded_and_does_not_abort_other_cases(tmp_path, monkeypatch):
    original = evaluation.run_agent

    def fail_one(**kwargs):
        if kwargs["model"].scenario == "normal":
            raise RuntimeError("not-safe-to-log-raw-provider-body")
        return original(**kwargs)

    monkeypatch.setattr(evaluation, "run_agent", fail_one)
    _, report = evaluation.run_suite(output_dir=tmp_path / "runs")
    assert len(report["cases"]) == 5 and report["summary"]["passed_cases"] == 4
    failed = report["cases"][0]
    assert failed["observed"] == {"outcome": "failed", "failure_code": "unexpected_failure"}
    assert failed["execution_error"]["type"] == "RuntimeError" and failed["evaluation_failures"]
    assert "not-safe-to-log" not in json.dumps(report)


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "traversal"])
def test_invalid_case_file_is_rejected_before_creating_run(tmp_path, mutation):
    data = json.loads((project_root() / "evals/cases.json").read_text())
    if mutation == "empty":
        data["cases"] = []
    elif mutation == "duplicate":
        data["cases"].append(deepcopy(data["cases"][0]))
    else:
        data["cases"][0]["id"] = "../outside"
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        evaluation.run_suite(cases_path=path, output_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_default_cli_completes_offline_even_with_real_model_configuration(tmp_path, monkeypatch, capsys):
    def paid_api_forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must never load a paid model")

    monkeypatch.setattr("research_agent.agent.load_model", paid_api_forbidden)
    monkeypatch.setenv("LLM_API_KEY", "fake-env-key-not-to-be-used")
    assert evaluation.main(["--output-dir", str(tmp_path / "runs")]) == 0
    output = capsys.readouterr().out
    assert "PASS insufficient_evidence" in output and "评估报告：" in output
    assert "fake-env-key" not in output
