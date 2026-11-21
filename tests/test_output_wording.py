"""Wording is advisory; technical evidence and Memory facts remain verified."""

import importlib
import json

import pytest

from research_agent.main import run_agent
from research_agent.memory import ResearchMemoryStore
from test_agent_setup import ScriptedResearchModel, workspace


class WordingModel(ScriptedResearchModel):
    def child_reply(self, messages):
        message = super().child_reply(messages)
        if self.mode == "child_wording" and not message.tool_calls:
            reply = json.loads(message.text)
            reply["conclusions"]["convergence"] = "B 全程领先。"
            message.content = json.dumps(reply, ensure_ascii=False)
        return message

    def main_reply(self, messages):
        message = super().main_reply(messages)
        if message.tool_calls:
            for call in message.tool_calls:
                if call["name"] == "write_file" and self.mode == "report_wording":
                    call["args"]["content"] += "\nB 显著更稳定，下降速度始终更快。"
        elif self.mode == "final_wording":
            message.content += "\nB 显著更优（更低）。"
        return message


@pytest.mark.parametrize("mode", ["child_wording", "report_wording", "final_wording"])
def test_language_does_not_block_technical_acceptance(workspace, mode):
    model = WordingModel(workspace=str(workspace), mode=mode)
    trace = []
    result = run_agent(root=workspace, model=model, output="reports/result.md", trace=trace.append)
    assert any(e.startswith("skill_load_verified ") for e in trace)
    assert result["analysis"]["comparison"]["significance_tested"] is False
    assert result["analysis"]["algorithms"]["A"]["mean"] == 20.738583304
    record = ResearchMemoryStore(workspace).retrieve(function_name="F1")[0]
    assert record["facts"]["significance_tested"] is False
    assert record["interpretation"]["status"] == "model_interpretation_not_facts"
    if mode == "child_wording":
        assert record["interpretation"]["conclusions"]["convergence"] == "B 全程领先。"


def test_cli_can_complete_without_language_rule_blocking(workspace, monkeypatch, capsys):
    cli = importlib.import_module("research_agent.main")

    def invoke(**kwargs):
        return run_agent(root=workspace, model=WordingModel(workspace=str(workspace), mode="final_wording"), **kwargs)

    monkeypatch.setattr(cli, "run_agent", invoke)
    assert cli.main(["--output", "reports/result.md"]) == 0
    output = capsys.readouterr()
    assert "已核验报告" in output.out and "已保存研究记忆" in output.out
    assert "统计语言核验失败" not in output.err
