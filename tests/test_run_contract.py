"""Fixed task facts survive lossy summaries and misleading model replies."""

import importlib
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field

from research_agent.agent import ReportFilesystemBackend
from research_agent.main import AgentRunError, run_agent
from research_agent.memory import ResearchMemoryStore
from test_agent_setup import workspace
from memory_worker import MemoryFakeModel
from test_summarization import HarnessModel, RecordingModel, is_summary


def runtime_facts(messages):
    text = messages[0].text.split("本轮程序事实（优先于历史摘要，目标路径固定）：", 1)[1]
    return json.JSONDecoder().raw_decode(text)[0]


class ContractModel(HarnessModel):
    facts_seen: list = Field(default_factory=list)
    main_toolsets: list = Field(default_factory=list)

    def _generate(self, messages, **kwargs):
        if "analyze_experiment" not in kwargs.get("offline_tools", []):
            self.main_toolsets.append(kwargs.get("offline_tools", []))
        return super()._generate(messages, **kwargs)

    def main_reply(self, messages):
        # Unlike the earlier scripted model, discover the actual target from
        # what this model call sees. report_output is deliberately left at its
        # unrelated default; it cannot provide the correct test output path.
        facts = runtime_facts(messages)
        self.facts_seen.append(facts)
        message = super().main_reply(messages)
        for call in message.tool_calls:
            if call["name"] == "write_file":
                call["args"]["file_path"] = facts["report_path"]
        if not message.tool_calls:
            message.content = "本次模拟分析完成，报告另存为 reports/rogue_v2.md。"
        return message


def execute(workspace, model, **kwargs):
    return run_agent(root=workspace, model=model, summary_model=RecordingModel(),
                     summary_trigger_messages=7, summary_keep_messages=2,
                     output="reports/locked_job.md", max_steps=60, **kwargs)


@pytest.mark.parametrize("history", [False, True])
def test_current_target_and_saved_fact_survive_summary_without_fake_cached_target(workspace, history):
    if history:
        run_agent(root=workspace, model=MemoryFakeModel(workspace=str(workspace)), output="reports/result.md")
    model = ContractModel(workspace=str(workspace))
    result = execute(workspace, model)
    assert model.report_output == "reports/result.md"  # Not the requested target.
    assert all(f["report_path"] == "/reports/locked_job.md" for f in model.facts_seen)
    paired = list(zip(model.main_requests, model.facts_seen, model.main_toolsets, strict=True))
    assert any(any(is_summary(m) for m in request) and facts["analysis_verified"]
               and not facts["report_write_succeeded"] for request, facts, _ in paired)
    assert any(any(is_summary(m) for m in request) and facts["report_write_succeeded"]
               and not any(isinstance(m, ToolMessage) and m.tool_call_id == "write-report" for m in request)
               for request, facts, _ in paired)
    assert all(set(tools) == {"write_todos"} for _, facts, tools in paired if facts["report_write_succeeded"])
    assert result["report_path"] == str(workspace / "reports/locked_job.md")
    assert "reports/locked_job.md" in result["summary"] and result["report_path"] in result["summary"]
    assert "rogue_v2.md" not in result["summary"] and "rogue_v2.md" in result["model_summary"]
    records = ResearchMemoryStore(workspace).retrieve()
    assert len(records) == 1 and len(records[0]["reports"]) == (2 if history else 1)


def test_hallucinated_duplicate_and_renamed_writes_cannot_create_extra_artifacts(workspace):
    class IgnoringModel(ContractModel):
        def main_reply(self, messages):
            latest = next((m.tool_call_id for m in reversed(messages) if isinstance(m, ToolMessage)), None)
            if latest in {"repeat-write", "rename-write"}:
                # These extra tool IDs aren't part of HarnessModel's script.
                self.main_history = list(messages)
                self.main_requests.append(list(messages))
                facts = runtime_facts(messages)
                self.facts_seen.append(facts)
                if latest == "repeat-write":
                    return self.call("write_file", {"file_path": facts["report_path"].replace(".md", "_v2.md"),
                                                     "content": "renamed"}, "rename-write")
                return self.call("write_todos", {"todos": [{"content": "本轮报告已保存", "status": "completed"}]}, "plan-finish")
            message = super().main_reply(messages)
            if latest == "write-report":
                return self.call("write_file", {"file_path": self.facts_seen[-1]["report_path"],
                                                 "content": "duplicate"}, "repeat-write")
            return message

    model = IgnoringModel(workspace=str(workspace))
    trace = []
    result = execute(workspace, model, trace=trace.append)
    assert sum(e.startswith("tool_result write_file ") and "status=success" in e for e in trace) == 1
    assert any("tool_result write_file" in e and "status=error" in e for e in trace)
    assert not (workspace / "reports/locked_job_v2.md").exists()
    assert "duplicate" not in (workspace / "reports/locked_job.md").read_text()
    assert result["memory"] and "rogue_v2.md" not in result["summary"]


def test_backend_locks_target_even_before_first_write(workspace):
    backend = ReportFilesystemBackend(workspace, output="reports/locked_job.md")
    assert backend.write("/reports/other.md", "wrong").error
    assert not (workspace / "reports/other.md").exists()
    assert backend.write("/reports/locked_job.md", "first").error is None
    assert backend.write("/reports/locked_job.md", "second").error
    assert backend.write("/reports/locked_job_v2.md", "third").error
    assert (workspace / "reports/locked_job.md").read_text() == "first"


def test_wrong_target_is_not_accepted_or_saved_to_memory(workspace):
    class WrongTarget(ContractModel):
        def main_reply(self, messages):
            message = super().main_reply(messages)
            for call in message.tool_calls:
                if call["name"] == "write_file":
                    call["args"]["file_path"] = "/reports/other.md"
            return message

    model = WrongTarget(workspace=str(workspace))
    with pytest.raises(AgentRunError, match="输出路径固定"):
        execute(workspace, model)
    assert not any(f["report_write_succeeded"] for f in model.facts_seen)
    assert not (workspace / "reports/other.md").exists() and not (workspace / "memory/research.json").exists()


def test_deterministic_location_does_not_bypass_file_content_verification(workspace, monkeypatch):
    original = ReportFilesystemBackend.write

    def tamper(self, path, content):
        return original(self, path, content + "\nchanged behind the submitted arguments")

    monkeypatch.setattr(ReportFilesystemBackend, "write", tamper)
    with pytest.raises(AgentRunError, match="磁盘报告与实际 write_file 调用内容不一致"):
        execute(workspace, ContractModel(workspace=str(workspace)))
    assert not (workspace / "memory/research.json").exists()


def test_cli_prints_only_verified_location_not_model_claim(workspace, monkeypatch, capsys):
    cli = importlib.import_module("research_agent.main")
    monkeypatch.setattr(cli, "run_agent", lambda **kwargs: execute(workspace, ContractModel(workspace=str(workspace))))
    assert cli.main(["--output", "reports/locked_job.md"]) == 0
    output = capsys.readouterr().out
    assert "rogue_v2.md" not in output
    assert str(workspace / "reports/locked_job.md") in output and "已核验报告" in output
