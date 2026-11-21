"""Ordinary tool submission replaces model-authored final JSON text."""

import json

import pytest

from research_agent.main import AgentRunError, main, run_agent
from research_agent.subagent import submit_analysis
from test_agent_setup import ScriptedResearchModel, workspace


def test_real_submission_executes_and_ends_child_without_parsing_prose(workspace, monkeypatch):
    submitted = []
    original = submit_analysis.func

    def observe(**kwargs):
        submitted.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(submit_analysis, "func", observe)

    class AnnotatedModel(ScriptedResearchModel):
        def _generate(self, *args, **kwargs):
            result = super()._generate(*args, **kwargs)
            message = result.generations[0].message
            if any(c["name"] == "submit_analysis" for c in message.tool_calls):
                message.content = "分析完成，下面是结论。这里的说明不需要是 JSON。"
            return result

    model = AnnotatedModel(workspace=str(workspace))
    result = run_agent(root=workspace, model=model, output="reports/result.md")
    assert len(submitted) == 1
    assert submitted[0]["source_sha256"] == result["analysis"]["source"]["sha256"]
    assert submitted[0]["conclusions"] == result["conclusions"]
    # read -> analyze -> submit. return_direct avoids a fourth model turn.
    assert len(model.child_requests) == 3
    assert (workspace / "memory/research.json").is_file()
    assert "submit_analysis" not in model.bound_names


def test_submission_cannot_replace_analysis_or_skill_execution(workspace):
    class PrematureModel(ScriptedResearchModel):
        def child_reply(self, messages):
            from langchain_core.messages import AIMessage

            return AIMessage(content=json.dumps({
                "source_sha256": "0" * 64,
                "conclusions": {key: "自称已完成分析" for key in ("final_performance", "stability", "convergence")},
                "significance_tested": False,
            }))

    with pytest.raises(AgentRunError, match="未真正成功执行"):
        run_agent(root=workspace, model=PrematureModel(workspace=str(workspace)), output="reports/result.md")
    assert not (workspace / "reports/result.md").exists()
    assert not (workspace / "memory/research.json").exists()


@pytest.mark.parametrize("conclusions", [{}, {"final_performance": "", "stability": "结论", "convergence": "结论"}])
def test_incomplete_submitted_conclusions_still_fail(workspace, conclusions):
    class IncompleteModel(ScriptedResearchModel):
        def child_reply(self, messages):
            message = super().child_reply(messages)
            if not message.tool_calls:
                reply = json.loads(message.text)
                reply["conclusions"] = conclusions
                message.content = json.dumps(reply)
            return message

    with pytest.raises(AgentRunError, match="缺少最终性能"):
        run_agent(root=workspace, model=IncompleteModel(workspace=str(workspace)), output="reports/result.md")
    assert not (workspace / "memory/research.json").exists()


def test_plain_summary_failure_never_prints_verified_or_saves_memory(workspace, monkeypatch, capsys):
    import importlib

    cli = importlib.import_module("research_agent.main")

    def invoke(**kwargs):
        return run_agent(root=workspace, model=ScriptedResearchModel(workspace=str(workspace), mode="child_summary"), **kwargs)

    monkeypatch.setattr(cli, "run_agent", invoke)
    assert main(["--output", "reports/result.md"]) == 1
    captured = capsys.readouterr()
    assert "submit_analysis" in captured.err
    assert "已核验报告" not in captured.out
    assert "已保存研究记忆" not in captured.out
    assert not (workspace / "memory/research.json").exists()
