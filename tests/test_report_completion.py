"""A bounded reminder repairs omission, never failed writes or evidence."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from research_agent.agent import ReportFilesystemBackend
from research_agent.main import AgentRunError, run_agent
from research_agent.memory import ResearchMemoryStore
from memory_worker import MemoryFakeModel
from test_agent_setup import workspace


class EarlyStopModel(MemoryFakeModel):
    stops_remaining: int = 1
    stop_count: int = 0
    stop_reason: str = "stop"
    malformed_call: bool = False
    stop_text: str = "已有历史报告，分析已完成。"

    def main_reply(self, messages):
        message = super().main_reply(messages)
        if self.stops_remaining and any(c["name"] == "write_file" for c in message.tool_calls):
            self.stops_remaining -= 1
            self.stop_count += 1
            return AIMessage(
                content=self.stop_text, response_metadata={"finish_reason": self.stop_reason},
                invalid_tool_calls=([{"name": "write_file", "args": "{bad JSON", "id": "bad", "error": "invalid"}]
                                    if self.malformed_call else []),
            )
        return message


def reminders(trace):
    return [event for event in trace if event.startswith("report_completion_reminder ")]


@pytest.mark.parametrize("with_history", [False, True])
def test_reminder_continues_real_main_graph_and_saves_new_report(workspace, with_history):
    first = None
    if with_history:
        first = run_agent(root=workspace, output="reports/first.md", model=MemoryFakeModel(
            workspace=str(workspace), report_output="reports/first.md"))
    model = EarlyStopModel(workspace=str(workspace), report_output="reports/second.md")
    trace = []
    result = run_agent(root=workspace, output="reports/second.md", model=model, trace=trace.append)
    assert reminders(trace) == ["report_completion_reminder attempt=1"]
    assert model.stop_count == 1
    assert (workspace / "reports/second.md").is_file()
    assert len(model.child_requests) == 3  # No repeat of Skill load or analysis.
    assert sum(event.startswith("tool_call task ") for event in trace) == 1
    assert sum(event.startswith("tool_call write_file ") for event in trace) == 1
    reminder = [m.text for m in model.main_history if isinstance(m, HumanMessage) and m.text.startswith("本次任务尚未完成")]
    assert len(reminder) == 1 and '/reports/second.md' in reminder[0]
    assert bool([m for m in model.main_history if isinstance(m, HumanMessage) and m.text.startswith("历史研究参考资料")]) == with_history
    records = ResearchMemoryStore(workspace).retrieve()
    assert len(records) == 1
    assert len(records[0]["reports"]) == (2 if with_history else 1)
    if first:
        assert result["memory"]["created"] is False
        assert result["memory"]["experiment_id"] == first["memory"]["experiment_id"]
    assert any(event.startswith("main_reply ") and "已有历史报告" in event for event in trace)


def test_persistent_omission_stops_after_one_reminder_and_preserves_memory(workspace):
    run_agent(root=workspace, model=MemoryFakeModel(workspace=str(workspace)), output="reports/result.md")
    memory = (workspace / "memory/research.json").read_bytes()
    model = EarlyStopModel(workspace=str(workspace), report_output="reports/second.md", stops_remaining=100)
    trace = []
    with pytest.raises(AgentRunError, match="Main 已结束回答.*finish_reason=stop"):
        run_agent(root=workspace, model=model, output="reports/second.md", trace=trace.append)
    assert model.stop_count == 2
    assert len(reminders(trace)) == 1
    assert not (workspace / "reports/second.md").exists()
    assert (workspace / "memory/research.json").read_bytes() == memory
    assert not any(event.startswith("memory_saved ") for event in trace)


@pytest.mark.parametrize("settings", [{"stop_reason": "length"}, {"stop_reason": "content_filter"}, {"malformed_call": True}])
def test_protocol_or_provider_stop_is_diagnosed_without_reminder(workspace, settings):
    model = EarlyStopModel(workspace=str(workspace), **settings)
    trace = []
    with pytest.raises(AgentRunError, match="write_file.*Main 已结束回答"):
        run_agent(root=workspace, model=model, output="reports/result.md", trace=trace.append)
    assert not reminders(trace)
    reply = json.loads(next(e.removeprefix("main_reply ") for e in trace if e.startswith("main_reply ")))
    assert reply["finish_reason"] == model.stop_reason
    assert reply["invalid_tool_calls"] == int(model.malformed_call)
    assert not (workspace / "memory/research.json").exists()


def test_failed_write_is_not_retried(workspace, monkeypatch):
    from deepagents.backends.protocol import WriteResult

    writes = []

    def denied(*args):
        writes.append(args)
        return WriteResult(error="disk denied")

    monkeypatch.setattr(ReportFilesystemBackend, "write", denied)
    trace = []
    with pytest.raises(AgentRunError, match="disk denied"):
        run_agent(root=workspace, model=MemoryFakeModel(workspace=str(workspace)), output="reports/result.md", trace=trace.append)
    assert len(writes) == 1 and not reminders(trace)
    assert not (workspace / "memory/research.json").exists()


def test_reminder_does_not_reset_step_budget(workspace):
    trace = []
    with pytest.raises(AgentRunError, match="步数上限"):
        run_agent(root=workspace, model=EarlyStopModel(workspace=str(workspace)), output="reports/result.md",
                  max_steps=12, trace=trace.append)
    assert reminders(trace) == ["report_completion_reminder attempt=1"]
    assert not (workspace / "memory/research.json").exists()


@pytest.mark.parametrize("mode", ["bad_numbers", "parallel_write"])
def test_invalid_report_or_order_is_not_repaired(workspace, mode):
    trace = []
    with pytest.raises(AgentRunError):
        run_agent(root=workspace, model=MemoryFakeModel(workspace=str(workspace), mode=mode),
                  output="reports/result.md", trace=trace.append)
    assert not reminders(trace)
    assert not (workspace / "memory/research.json").exists()


def test_cli_still_fails_when_reminder_is_ignored(workspace, monkeypatch, capsys):
    import importlib

    cli = importlib.import_module("research_agent.main")
    model = EarlyStopModel(workspace=str(workspace), stops_remaining=100)

    def invoke(**kwargs):
        return run_agent(root=workspace, model=model, **kwargs)

    monkeypatch.setattr(cli, "run_agent", invoke)
    assert cli.main(["--output", "reports/result.md", "--trace"]) == 1
    captured = capsys.readouterr()
    assert "Main 已结束回答" in captured.err
    assert "已核验报告" not in captured.out and "已保存研究记忆" not in captured.out
    assert model.stop_count == 2
    assert not (workspace / "memory/research.json").exists()
