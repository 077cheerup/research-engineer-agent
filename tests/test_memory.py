"""Persistence, provenance, failure paths and separate-process history injection."""

from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.main import AgentRunError, run_agent
from research_agent.memory import ResearchMemoryError, ResearchMemoryStore, history_summary
from research_agent.tools.experiment import make_analyze_experiment
from memory_worker import MemoryFakeModel
from test_agent_setup import ScriptedResearchModel, workspace


def run(workspace, output="reports/result.md", csv="data/sample_experiments.csv", function="F1", **kwargs):
    model = MemoryFakeModel(workspace=str(workspace), report_output=output, csv_input=csv, function_name=function, **kwargs)
    result = run_agent(root=workspace, output=output, csv_path=csv, function=function, model=model)
    return result, model


def histories(model):
    return [m.text for m in model.main_history if isinstance(m, HumanMessage) and m.text.startswith("历史研究参考资料")]


def resign(envelope):
    data = {"version": envelope["version"], "records": envelope["records"]}
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    envelope["checksum"] = hashlib.sha256(text.encode()).hexdigest()
    return envelope


def test_success_saves_facts_after_acceptance(workspace):
    result, model = run(workspace)
    store = ResearchMemoryStore(workspace)
    records = store.retrieve(function_name="F1", algorithms=["A", "B"])
    assert len(records) == 1 and result["memory"]["created"] is True
    record = records[0]
    expected = make_analyze_experiment(workspace).invoke({"csv_path": "data/sample_experiments.csv", "function_name": "F1"})
    assert record["facts"]["final_statistics"] == expected["algorithms"]
    assert record["facts"]["source"] == expected["source"]
    assert record["facts"]["significance_tested"] is False
    assert record["facts"]["data_kind"] == "simulated_demo"
    assert record["interpretation"]["conclusions"] == result["conclusions"]
    assert record["unverified"] == {"status": "unverified", "hypotheses": [], "next_experiments": []}
    assert record["reports"][0]["path"] == "reports/result.md"
    assert record["reports"][0]["sha256"] == hashlib.sha256(Path(result["report_path"]).read_bytes()).hexdigest()
    assert "dimension" not in record["facts"]["configuration"]
    assert "messages" not in record
    assert not histories(model)  # Empty store does not create a fake history message.


@pytest.mark.parametrize("mode", ["child_no_tool", "skip_skill", "bad_numbers", "no_write"])
def test_failed_acceptance_never_writes_memory(workspace, mode):
    model = ScriptedResearchModel(workspace=str(workspace), mode=mode)
    with pytest.raises(AgentRunError):
        run_agent(root=workspace, model=model, output="reports/result.md")
    assert not (workspace / "memory/research.json").exists()


def test_duplicate_experiment_merges_reports_without_duplicate_facts(workspace):
    first, _ = run(workspace)
    store = ResearchMemoryStore(workspace)
    unchanged = store.path.read_bytes()
    assert store.save_verified(first)["created"] is False
    assert store.path.read_bytes() == unchanged
    second, model = run(workspace, output="reports/second.md")
    assert first["memory"]["experiment_id"] == second["memory"]["experiment_id"]
    assert second["memory"]["created"] is False
    records = store.retrieve()
    assert len(records) == 1 and len(records[0]["reports"]) == 2
    assert "reports/result.md" in histories(model)[0]


def test_different_data_functions_algorithms_and_limits(workspace):
    first, _ = run(workspace)
    other = workspace / "data/other.csv"
    # Change a different file; preserve the historical source for revalidation.
    import pandas as pd
    frame = pd.read_csv(other)
    frame["best_fitness"] += 1
    frame.to_csv(other, index=False)
    second, _ = run(workspace, output="reports/other.md", csv="data/other.csv")
    third, _ = run(workspace, output="reports/f2.md", csv="data/F2.csv", function="F2")
    store = ResearchMemoryStore(workspace)
    assert len({r["experiment_id"] for r in store.retrieve(limit=10)}) == 3
    assert first["memory"]["experiment_id"] != second["memory"]["experiment_id"]
    assert len(store.retrieve(function_name="F1")) == 2
    assert len(store.retrieve(function_name="F2")) == 1
    assert len(store.retrieve(algorithms=["A"])) == 3
    assert store.retrieve(algorithms=["C"]) == []
    assert len(store.retrieve(function_name="F1", limit=1)) == 1
    assert store.retrieve(limit=0) == []
    assert third["memory"]["experiment_id"] not in {r["experiment_id"] for r in store.retrieve(function_name="F1")}


def test_model_text_remains_interpretation_and_is_not_injected(workspace):
    marker = " UNTRUSTED_IGNORE_RULES 模型猜测值 987654.321。"
    result, _ = run(workspace, interpretation_marker=marker)
    store = ResearchMemoryStore(workspace)
    record = store.retrieve()[0]
    assert record["facts"]["final_statistics"]["A"]["mean"] == 20.738583304
    assert record["interpretation"]["status"] == "model_interpretation_not_facts"
    assert marker in record["interpretation"]["conclusions"]["final_performance"]
    _, model = run(workspace, output="reports/next.md")
    assert "UNTRUSTED_IGNORE_RULES" not in "".join(histories(model))
    assert not any(result["memory"]["experiment_id"] in m.text for m in model.main_history if isinstance(m, SystemMessage))


def test_summary_is_bounded_and_keeps_provenance(workspace):
    run(workspace)
    records = ResearchMemoryStore(workspace).retrieve()
    summary = history_summary(records, max_chars=2000)
    assert len(summary) <= 2000
    for text in ("simulated_demo", "source", "sha256", "reports/result.md", "不是指令", "significance_tested"):
        assert text in summary
    assert history_summary(records, max_chars=20) == ""


@pytest.mark.parametrize("content", ["{broken", "[]", '{"version": 999, "records": [], "checksum": "x"}'])
def test_invalid_memory_fails_before_model_and_is_not_cleared(workspace, content):
    store = ResearchMemoryStore(workspace)
    store.path.parent.mkdir()
    store.path.write_text(content)
    model = MemoryFakeModel(workspace=str(workspace))
    with pytest.raises(AgentRunError, match="Memory 读取或验证失败"):
        run_agent(root=workspace, model=model, output="reports/result.md")
    assert not model.main_history
    assert store.path.read_text() == content


@pytest.mark.parametrize("rewrite_checksum", [False, True])
def test_tampered_statistics_rejected_even_with_recomputed_checksum(workspace, rewrite_checksum):
    run(workspace)
    store = ResearchMemoryStore(workspace)
    envelope = json.loads(store.path.read_text())
    envelope["records"][0]["facts"]["final_statistics"]["A"]["mean"] = 999
    store.path.write_text(json.dumps(resign(envelope) if rewrite_checksum else envelope))
    with pytest.raises(ResearchMemoryError, match="完整性|重算结果"):
        store.retrieve(function_name="F1")


@pytest.mark.parametrize("target", ["source", "report"])
def test_changed_provenance_is_not_silently_trusted(workspace, target):
    run(workspace)
    path = workspace / ("data/sample_experiments.csv" if target == "source" else "reports/result.md")
    path.write_text(path.read_text() + "\nchanged\n")
    with pytest.raises(ResearchMemoryError):
        ResearchMemoryStore(workspace).retrieve()


def test_store_rejects_forged_statistics_from_caller(workspace):
    result, _ = run(workspace)
    forged = deepcopy(result)
    forged["analysis"]["algorithms"]["A"]["mean"] = 999
    with pytest.raises(ResearchMemoryError, match="拒绝保存"):
        ResearchMemoryStore(workspace).save_verified(forged)


def test_invalid_interpretation_type_gets_a_memory_error(workspace):
    run(workspace)
    store = ResearchMemoryStore(workspace)
    envelope = json.loads(store.path.read_text())
    envelope["records"][0]["interpretation"]["conclusions"] = ["final_performance", "stability", "convergence"]
    store.path.write_text(json.dumps(resign(envelope)))
    with pytest.raises(ResearchMemoryError, match="模型解释"):
        store.retrieve()


def test_memory_save_failure_is_explicit_and_not_reported_as_success(workspace, monkeypatch, capsys):
    cli = importlib.import_module("research_agent.main")

    def failed(*args):
        raise ResearchMemoryError("simulated disk full")

    def invoke(**kwargs):
        return run_agent(root=workspace, model=MemoryFakeModel(workspace=str(workspace)), **kwargs)

    monkeypatch.setattr(ResearchMemoryStore, "_write", failed)
    monkeypatch.setattr(cli, "run_agent", invoke)
    assert cli.main(["--output", "reports/result.md"]) == 1
    output = capsys.readouterr()
    assert "报告验收已通过，但 Memory 保存失败" in output.err
    assert "已保存研究记忆" not in output.out
    assert (workspace / "reports/result.md").is_file()
    assert not (workspace / "memory/research.json").exists()


def test_atomic_replace_failure_preserves_existing_memory(workspace, monkeypatch):
    run(workspace)
    store = ResearchMemoryStore(workspace)
    before = store.path.read_bytes()

    def failed(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("research_agent.memory.os.replace", failed)
    with pytest.raises(AgentRunError, match="Memory 保存失败"):
        run(workspace, output="reports/second.md")
    assert store.path.read_bytes() == before
    assert len(store.retrieve()[0]["reports"]) == 1
    assert list(store.path.parent.iterdir()) == [store.path]


def test_memory_persists_across_fresh_processes_with_unrelated_control(workspace):
    worker = Path(__file__).with_name("memory_worker.py")

    def process(output, csv="data/sample_experiments.csv", function="F1"):
        completed = subprocess.run([sys.executable, str(worker), str(workspace), output, csv, function],
                                   capture_output=True, text=True, check=True, timeout=60)
        return json.loads(completed.stdout)

    first = process("reports/first.md")
    unrelated = process("reports/unrelated.md", "data/F2.csv", "F2")
    second = process("reports/second.md")
    assert len({first["pid"], unrelated["pid"], second["pid"]}) == 3
    assert first["history"] == [] and unrelated["history"] == []
    assert len(second["history"]) == 1
    history = second["history"][0]
    assert first["memory"]["experiment_id"] in history
    assert "reports/first.md" in history
    assert unrelated["memory"]["experiment_id"] not in history
    assert "F2" not in history and "reports/unrelated.md" not in history
    assert first["memory"]["experiment_id"] == second["memory"]["experiment_id"]
    records = ResearchMemoryStore(workspace).retrieve(limit=10)
    assert len(records) == 2  # F1 rerun merges, unrelated F2 remains separate.
    print(json.dumps({"process_pids": [first["pid"], unrelated["pid"], second["pid"]],
                      "first_history_count": len(first["history"]), "second_F1_history_count": len(second["history"]),
                      "unrelated_F2_excluded": True, "persisted_record_count": len(records)}))
