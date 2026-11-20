"""Targeted language regressions; real graph/tools with an offline Fake Model."""

import importlib
import json

import pytest

from research_agent.main import AgentRunError, run_agent
from research_agent.statistical_language import StatisticalLanguageError, verify_statistical_language
from test_agent_setup import ScriptedResearchModel, workspace


@pytest.mark.parametrize("text", [
    "B 显著更优（更低）。",
    "B 在统计意义上显著优于 A。",
    "B 的均值**显著低于** A。",
    "B 显著更稳定。",
    "A 和 B 存在统计学显著差异。",
    "B 统计上显著更加稳定。",
    "未进行任何统计显著性检验，但 B 显著更优。",
    "不能宣称显著更优；然而 B 显著更稳定。",
    "B 的下降速度始终更快。",
    "B 在每个阶段的表现均领先于 A。",
    "B 全程领先。",
    "B 在所有随机种子下都更优。",
    "B 在真实 CEC2013 实验中都更优。",
    "B 在不同随机种子下的表现更加一致和稳定。",
])
def test_reject_unsupported_positive_assertions(text):
    with pytest.raises(StatisticalLanguageError, match="核验失败"):
        verify_statistical_language(text, significance_tested=False, context="测试：")


@pytest.mark.parametrize("text", [
    "未进行任何统计显著性检验。",
    "不能据此宣称 B 显著更优。",
    "较小的标准差不等于统计上显著更稳定。",
    "没有证据证明 B 在真实 CEC2013 实验中更优。",
    "B 在当前样本中的最终平均 fitness 较低。",
    "B 的样本标准差较小。",
    "B 在全部记录的 FE 采样点上的平均 fitness 均低于 A。",
    "在本次观测样本中，B 的均值和中位数较低；这是描述性比较。",
    "在所有已观测的随机种子中，B 的最终值更低。",
    "不能由某 FE 的水平推断 B 的下降速度更快。",
    "图例颜色变化显著，便于辨认。",  # '显著' alone is not prohibited.
])
def test_allow_descriptions_and_correct_negative_statements(text):
    verify_statistical_language(text, significance_tested=False, context="测试：")


class LanguageRegressionModel(ScriptedResearchModel):
    """Alter exactly one output surface, leaving tool evidence untouched."""

    def child_reply(self, messages):
        message = super().child_reply(messages)
        if self.mode == "unsafe_child" and not message.tool_calls:
            payload = json.loads(message.text)
            payload["conclusions"]["final_performance"] = "B 显著更优。未进行统计显著性检验。"
            message.content = json.dumps(payload, ensure_ascii=False)
        return message

    def main_reply(self, messages):
        message = super().main_reply(messages)
        if message.tool_calls:
            for call in message.tool_calls:
                if call["name"] == "write_file" and self.mode == "unsafe_report":
                    call["args"]["content"] += "\nB 显著更稳定。未进行统计显著性检验。"
        elif self.mode == "unsafe_final":
            message.content += "\nB 显著更优（更低）。未进行统计显著性检验。"
        if self.mode == "descriptive":
            description = "本次模拟样本中 B 的最终平均 fitness 较低，样本标准差较小。未进行统计显著性检验。"
            if message.tool_calls:
                for call in message.tool_calls:
                    if call["name"] == "write_file":
                        call["args"]["content"] += "\n" + description
            else:
                message.content += "\n" + description
        return message


@pytest.mark.parametrize("mode,context", [
    ("unsafe_child", "子 Agent conclusions.final_performance"),
    ("unsafe_report", "Markdown 报告"),
    ("unsafe_final", "Main 最终回答"),
])
def test_checks_each_output_surface_after_trusted_analysis(workspace, mode, context):
    model = LanguageRegressionModel(workspace=str(workspace), mode=mode)
    trace = []
    with pytest.raises(AgentRunError, match=context):
        run_agent(root=workspace, model=model, output="reports/result.md", trace=trace.append)
    if mode == "unsafe_child":
        assert not (workspace / "reports/result.md").exists()
    else:
        assert any(e.startswith("skill_load_verified ") for e in trace)
        assert any("tool_result write_file" in e and "status=success" in e for e in trace)


def test_descriptive_report_and_final_answer_still_pass(workspace):
    model = LanguageRegressionModel(workspace=str(workspace), mode="descriptive")
    result = run_agent(root=workspace, model=model, output="reports/result.md")
    saved = (workspace / "reports/result.md").read_text()
    assert "模拟样本" in saved and "模拟样本" in result["summary"]
    assert "未进行统计显著性检验" in saved
    assert result["analysis"]["comparison"]["significance_tested"] is False
    assert result["analysis"]["algorithms"]["A"]["mean"] == 20.738583304


@pytest.mark.parametrize("mode", ["unsafe_report", "unsafe_final"])
def test_cli_never_prints_verified_success_on_language_failure(workspace, monkeypatch, capsys, mode):
    cli = importlib.import_module("research_agent.main")
    model = LanguageRegressionModel(workspace=str(workspace), mode=mode)

    def invoke(**kwargs):
        return run_agent(root=workspace, model=model, **kwargs)

    monkeypatch.setattr(cli, "run_agent", invoke)
    assert cli.main(["--output", "reports/result.md", "--trace"]) == 1
    output = capsys.readouterr()
    assert "已核验报告" not in output.out
    assert "统计语言核验失败" in output.err
