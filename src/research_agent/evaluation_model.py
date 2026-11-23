"""Offline scripted policy for evaluation, not a research-capable LLM."""

import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from research_agent.report import EVIDENCE_START, EVIDENCE_END


class EvaluationModel(BaseChatModel):
    """Drive real tools; scenario controls behavior, never expected scores."""

    csv_input: str
    scenario: str = "normal"
    requested_write_path: str | None = None

    @property
    def _llm_type(self):
        return "offline-evaluation-script"

    def bind_tools(self, tools, **kwargs):
        return self.bind(evaluation_tools=[t.name if hasattr(t, "name") else t["name"] for t in tools])

    @staticmethod
    def call(name, args, call_id):
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])

    def analyst(self, messages):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if not results:
            path = re.search(r"/skills/experiment-analysis/SKILL\.md", messages[0].text)
            if path is None:
                return AIMessage(content="未发现分析 Skill，无法继续。")
            return self.call("read_file", {"file_path": path.group(), "limit": 1000}, "eval-skill")
        analysis = next((m for m in results if m.name == "analyze_experiment"), None)
        if analysis is None:
            return self.call("analyze_experiment", {"csv_path": self.csv_input, "function_name": "F1"}, "eval-analysis")
        if analysis.status != "success":
            return AIMessage(content="分析工具执行失败，停止提交。")
        if self.scenario == "insufficient_evidence":
            return AIMessage(content="算法 B 更好。只有总结，不提交结构化来源引用。")
        data = json.loads(analysis.text)
        return self.call("submit_analysis", {
            "source_sha256": data["source"]["sha256"], "significance_tested": False,
            "conclusions": {
                "final_performance": "当前模拟样本的最终性能比较以工具 Mean、Median 等为依据。",
                "stability": "样本标准差描述当前 seed 结果的离散程度，未进行显著性检验。",
                "convergence": "平均曲线仅描述已记录 FE 采样点，不推断未观测区间的速度。",
            },
        }, "eval-submit")

    def main_reply(self, messages):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        statuses = ["in_progress", "pending"]
        if not results:
            name, call_id = "write_todos", "eval-plan-start"
        else:
            latest = results[-1].tool_call_id
            if latest == "eval-plan-start":
                return self.call("task", {"subagent_type": "experiment-analyst", "description":
                    f'csv_path={json.dumps(self.csv_input)}, function_name="F1"；分析模拟实验，不写文件。'}, "eval-task")
            if latest == "eval-task":
                statuses, name, call_id = ["completed", "in_progress"], "write_todos", "eval-plan-ready"
            elif latest == "eval-plan-ready":
                text = messages[0].text.split("本轮程序事实（优先于历史摘要，目标路径固定）：", 1)[1]
                facts, _ = json.JSONDecoder().raw_decode(text)
                return self.call("write_file", {"file_path": self.requested_write_path or facts["report_path"], "content":
                    "# 离线评估报告\n\n模拟数据；未做统计显著性检验。\n\n"
                    "## 分析说明\n\n最终性能、离散程度和采样点曲线依据本次工具结果。\n\n"
                    + EVIDENCE_START + "\n" + EVIDENCE_END}, "eval-write")
            elif latest == "eval-write":
                statuses = ["completed", "completed" if results[-1].status == "success" else "in_progress"]
                name, call_id = "write_todos", "eval-plan-finish"
            else:
                return AIMessage(content="本次模拟数据分析流程结束。")
        return self.call(name, {"todos": [{"content": label, "status": status} for label, status in
                         zip(["取得可信分析", "生成并保存报告"], statuses, strict=True)]}, call_id)

    def _generate(self, messages, **kwargs):
        message = self.analyst(messages) if "analyze_experiment" in kwargs.get("evaluation_tools", []) else self.main_reply(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])
