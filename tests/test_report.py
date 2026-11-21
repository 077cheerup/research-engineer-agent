"""Trusted evidence is assembled before the real built-in write_file call."""

import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from research_agent.main import AgentRunError, run_agent, verify_report_evidence
from research_agent.memory import ResearchMemoryStore
from research_agent.report import EVIDENCE_START, EVIDENCE_END, ReportAssemblyError, assemble_report
from research_agent.tools.experiment import make_analyze_experiment
from memory_worker import MemoryFakeModel
from test_agent_setup import workspace


class DraftReportModel(MemoryFakeModel):
    truncated_sha: bool = False
    historical_evidence: str = ""
    draft: str = ""

    def main_reply(self, messages):
        message = super().main_reply(messages)
        for call in message.tool_calls:
            if call["name"] != "write_file":
                continue
            payload = json.loads(next(m.text for m in reversed(messages) if isinstance(m, ToolMessage)))
            inner = self.historical_evidence
            if self.truncated_sha:
                analysis = payload["analysis"]
                inner = analysis["evidence_markdown"].replace(
                    analysis["source"]["sha256"], "30ace40ae9579f727d64202d5c3b")
            # Main's prose and ordering must survive deterministic assembly.
            prose = "# Main 组织的研究报告\n\n模拟数据，非正式 CEC2013。\n\n"
            for title, key in [("最终性能", "final_performance"), ("稳定性", "stability"), ("收敛趋势", "convergence")]:
                prose += f"## {title}\n\n{payload['conclusions'][key]}\n\n"
            self.draft = (prose + "## 可信数值依据\n\n" + EVIDENCE_START + "\n" + inner + "\n" + EVIDENCE_END
                          + "\n\n## 局限\n\n未进行显著性检验，仅描述当前样本。\n")
            call["args"]["content"] = self.draft
        return message


def check_submission(model, result):
    saved = Path(result["report_path"]).read_text()
    calls = [c for m in model.main_history if isinstance(m, AIMessage) for c in m.tool_calls if c["name"] == "write_file"]
    assert len(calls) == 1 and calls[0]["args"]["content"] == saved
    assert saved != model.draft
    assert any(isinstance(m, ToolMessage) and m.name == "write_file" and m.status == "success" for m in model.main_history)
    assert "# Main 组织的研究报告" in saved and "## 局限" in saved
    assert result["analysis"]["evidence_markdown"] in saved
    verify_report_evidence(saved, result["analysis"])
    return saved


def test_llm_truncated_sha_is_replaced_before_actual_write(workspace):
    model = DraftReportModel(workspace=str(workspace), truncated_sha=True)
    result = run_agent(root=workspace, model=model, output="reports/result.md")
    saved = check_submission(model, result)
    assert "30ace40ae9579f727d64202d5c3b" in model.draft
    assert "30ace40ae9579f727d64202d5c3b" not in saved
    assert saved.count(result["analysis"]["source"]["sha256"]) == 1
    # The validator itself still rejects the original truncated draft.
    with pytest.raises(AgentRunError, match="来源信息"):
        verify_report_evidence(model.draft, result["analysis"])


def test_injected_history_cannot_supply_current_report_evidence(workspace):
    first = run_agent(root=workspace, output="reports/first.md", model=MemoryFakeModel(
        workspace=str(workspace), report_output="reports/first.md"))
    current = workspace / "data/other.csv"
    frame = pd.read_csv(current)
    frame["best_fitness"] += 1
    frame.to_csv(current, index=False)
    model = DraftReportModel(workspace=str(workspace), report_output="reports/second.md", csv_input="data/other.csv",
                             historical_evidence=first["analysis"]["evidence_markdown"])
    trace = []
    result = run_agent(root=workspace, output="reports/second.md", csv_path="data/other.csv", model=model, trace=trace.append)
    assert any(e.startswith("memory_retrieved matched=1 injected_chars=") and not e.endswith("=0") for e in trace)
    history = next(m.text for m in model.main_history if isinstance(m, HumanMessage) and m.text.startswith("历史研究参考资料"))
    assert first["analysis"]["source"]["sha256"] in history
    expected = make_analyze_experiment(workspace).invoke({"csv_path": "data/other.csv", "function_name": "F1"})
    assert result["analysis"] == expected
    saved = check_submission(model, result)
    assert first["analysis"]["source"]["sha256"] not in saved
    records = ResearchMemoryStore(workspace).retrieve()
    assert len(records) == 2
    assert any(r["facts"]["source"] == expected["source"] for r in records)


def test_empty_evidence_region_generates_complete_verified_report(workspace):
    model = DraftReportModel(workspace=str(workspace))
    result = run_agent(root=workspace, model=model, output="reports/result.md")
    saved = check_submission(model, result)
    assert model.draft.split(EVIDENCE_START)[1].split(EVIDENCE_END)[0].strip() == ""
    for title in ("## 最终性能", "## 稳定性", "## 收敛趋势", "## 可信数值依据", "## 局限"):
        assert title in saved
    assert result["analysis"] == make_analyze_experiment(workspace).invoke({
        "csv_path": "data/sample_experiments.csv", "function_name": "F1"})
    assert len(ResearchMemoryStore(workspace).retrieve()) == 1
    assert "reports/result.md" in result["summary"]


@pytest.mark.parametrize("draft", ["无标记", EVIDENCE_END + EVIDENCE_START,
                                  EVIDENCE_START + EVIDENCE_START + EVIDENCE_END])
def test_missing_repeated_or_reversed_regions_are_explicit_errors(draft):
    with pytest.raises(ReportAssemblyError):
        assemble_report(draft, {"evidence_markdown": "可信证据"})


def test_invalid_region_never_reaches_writer_or_memory(workspace, monkeypatch):
    class InvalidDraftModel(DraftReportModel):
        def main_reply(self, messages):
            message = super().main_reply(messages)
            for call in message.tool_calls:
                if call["name"] == "write_file":
                    call["args"]["content"] = call["args"]["content"].replace(EVIDENCE_END, "")
            return message

    def unexpected_write(*args):
        raise AssertionError("An invalid draft must not reach the filesystem")

    monkeypatch.setattr("research_agent.agent.ReportFilesystemBackend.write", unexpected_write)
    with pytest.raises(AgentRunError, match="报告可信证据组装失败"):
        run_agent(root=workspace, model=InvalidDraftModel(workspace=str(workspace)), output="reports/result.md")
    assert not (workspace / "reports/result.md").exists()
    assert not (workspace / "memory/research.json").exists()
