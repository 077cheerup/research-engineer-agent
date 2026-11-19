"""CLI and observable acceptance checks for the analyze -> write loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APIStatusError, APITimeoutError

from research_agent.agent import ConfigurationError, build_agent, report_path
from research_agent.tools.experiment import project_root, resolve_csv_path


class AgentRunError(RuntimeError):
    """The graph ran without satisfying the observable research contract."""


def verify_run(
    messages: list, *, root: Path, csv: Path, function: str, destination: Path
) -> dict[str, Any]:
    """Require real ToolMessages, correct ordering, evidence, and disk content.

    Natural-language reasoning is still reviewed by the human; these checks
    verify the exact numeric tables, not every possible claim in the prose.
    """
    calls = {}
    evidence = None
    written_content = None
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls[call["id"]] = (call, evidence)
        if not isinstance(message, ToolMessage) or message.status != "success":
            continue
        call, preceding_evidence = calls.get(message.tool_call_id, ({}, None))
        if message.name == "analyze_experiment" and call.get("name") == message.name:
            args = call["args"]
            if args.get("function_name") != function:
                continue
            if resolve_csv_path(args["csv_path"], root) != csv:
                continue
            try:
                parsed = json.loads(message.content)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict) and "evidence_markdown" in parsed:
                evidence = parsed
        if message.name == "write_file" and call.get("name") == message.name and preceding_evidence:
            args = call["args"]
            if report_path(args["file_path"], root) == destination:
                written_content = args["content"]
                evidence = preceding_evidence
    if evidence is None:
        raise AgentRunError("Agent 未成功调用指定数据的 analyze_experiment；请检查服务的 Tool Calling 支持")
    if written_content is None:
        raise AgentRunError("未观察到分析成功之后的 write_file 成功调用；报告未验证保存")
    if not destination.is_file():
        raise AgentRunError("write_file 返回后预期报告文件不存在")
    saved = destination.read_text(encoding="utf-8")
    if saved != written_content:
        raise AgentRunError("磁盘报告与实际 write_file 调用内容不一致")
    if evidence["evidence_markdown"] not in saved:
        raise AgentRunError("报告数值证据与 Python Tool 结果不一致（需原样保留 evidence_markdown）")
    if "模拟数据" not in saved:
        raise AgentRunError("报告缺少模拟数据声明")
    final = messages[-1] if messages else None
    if not isinstance(final, AIMessage) or final.tool_calls or not final.text.strip():
        raise AgentRunError("缺少 Agent 最终回答")
    relative = destination.relative_to(root).as_posix()
    if relative not in final.text or str(destination) not in final.text:
        raise AgentRunError("Agent 最终回答未包含实际报告路径")
    return {"summary": final.text, "report_path": str(destination), "analysis": evidence}


def run_agent(
    *, csv_path: str = "data/sample_experiments.csv", function: str = "F1",
    output: str = "reports/F1_comparison.md", root: Path | str | None = None,
    model: BaseChatModel | None = None, max_steps: int = 24,
    trace: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Invoke a bounded real graph; no deterministic substitute for the agent."""
    workspace = Path(root).resolve() if root is not None else project_root()
    csv = resolve_csv_path(csv_path, workspace)
    destination = report_path(output, workspace)
    if destination.exists():
        raise AgentRunError("目标报告已存在，拒绝覆盖；请通过 --output 指定新文件名")
    if max_steps < 1:
        raise ValueError("max_steps 必须为正整数")
    graph = build_agent(workspace, model=model)
    request = (
        f"分析算法 A 和算法 B 在 CEC2013 {function} 上的模拟实验结果，比较最终性能、稳定性和收敛趋势，"
        f"生成中文 Markdown 报告。CSV 参数：{json.dumps(csv.relative_to(workspace).as_posix())}；"
        f"function_name 参数：{json.dumps(function)}。先分析，拿到结果后再写入 "
        f"/{destination.relative_to(workspace).as_posix()}。实际主机绝对路径为 {destination}。"
        "请在最终回答包含以上相对路径和绝对路径，并给出摘要。"
    )
    seen = set()
    messages = []
    try:
        for state in graph.stream(
            {"messages": [{"role": "user", "content": request}]},
            config={"recursion_limit": max_steps}, stream_mode="values",
        ):
            messages = state["messages"]
            if trace:
                for message in messages:
                    if message.id in seen:
                        continue
                    seen.add(message.id)
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            trace(f"tool_call {call['name']} id={call['id']}")
                    elif isinstance(message, ToolMessage):
                        trace(f"tool_result {message.name} id={message.tool_call_id} status={message.status}")
                        if message.name == "analyze_experiment" and message.status == "success":
                            result = json.loads(message.content)
                            trace("analysis_evidence " + json.dumps(result, ensure_ascii=False, allow_nan=False))
    except GraphRecursionError as exc:
        raise AgentRunError("达到运行步数上限；可能存在重复 Tool Calling，报告未验收") from exc
    except (APIConnectionError, APITimeoutError) as exc:
        raise AgentRunError(f"模型服务连接或超时失败 ({type(exc).__name__})；请检查 LLM_BASE_URL 和网络") from exc
    except APIStatusError as exc:
        raise AgentRunError(
            f"模型服务返回 HTTP {exc.status_code}；检查 LLM_* 配置及 OpenAI Chat Completions "
            "的 tools/tool_calls 支持。未退化为伪 Agent，报告未验收。"
        ) from exc
    except NotImplementedError as exc:
        raise AgentRunError("模型未实现 Tool Calling/bind_tools；请选择支持工具调用的服务") from exc
    return verify_run(messages, root=workspace, csv=csv, function=function, destination=destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mini Research Engineer Agent — Phase 1")
    parser.add_argument("--csv", default="data/sample_experiments.csv")
    parser.add_argument("--function", default="F1")
    parser.add_argument("--output", default="reports/F1_comparison.md")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--trace", action="store_true", help="显示工具调用及统计结果，不输出模型配置或密钥")
    args = parser.parse_args(argv)
    try:
        result = run_agent(
            csv_path=args.csv, function=args.function, output=args.output,
            max_steps=args.max_steps, trace=print if args.trace else None,
        )
    except (ConfigurationError, AgentRunError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Some compatible providers return malformed responses. Do not print
        # their raw bodies, which may echo credentials or request headers.
        print(f"错误：Agent 执行异常 ({type(exc).__name__})；请检查服务的 tools/tool_calls 协议兼容性，报告未验收", file=sys.stderr)
        return 1
    print(result["summary"])
    print(f"已核验报告：{result['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
