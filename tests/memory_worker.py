"""Offline subprocess worker; no model API, no process-local memory reuse."""

import json
import os
import sys

from langchain_core.messages import HumanMessage

from research_agent.main import run_agent
from test_agent_setup import ScriptedResearchModel


class MemoryFakeModel(ScriptedResearchModel):
    csv_input: str = "data/sample_experiments.csv"
    function_name: str = "F1"
    report_output: str = "reports/result.md"
    interpretation_marker: str = ""

    def child_reply(self, messages):
        message = super().child_reply(messages)
        for call in message.tool_calls:
            if call["name"] == "analyze_experiment":
                call["args"].update(csv_path=self.csv_input, function_name=self.function_name)
        if not message.tool_calls and self.interpretation_marker:
            reply = json.loads(message.text)
            reply["conclusions"]["final_performance"] += self.interpretation_marker
            message.content = json.dumps(reply, ensure_ascii=False)
        return message

    def main_reply(self, messages):
        message = super().main_reply(messages)
        for call in message.tool_calls:
            if call["name"] == "task":
                call["args"]["description"] = json.dumps({"csv_path": self.csv_input, "function_name": self.function_name})
            if call["name"] == "write_file":
                call["args"]["file_path"] = "/" + self.report_output
        if not message.tool_calls:
            message.content = message.text.replace("reports/result.md", self.report_output)
        return message


def main() -> None:
    import httpx
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden in the offline subprocess demo")

    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    httpx.Client.send = forbidden
    httpx.AsyncClient.send = forbidden
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    root, output, csv, function = sys.argv[1:]
    model = MemoryFakeModel(workspace=root, report_output=output, csv_input=csv, function_name=function)
    result = run_agent(root=root, output=output, csv_path=csv, function=function, model=model)
    history = [m.text for m in model.main_history if isinstance(m, HumanMessage) and m.text.startswith("历史研究参考资料")]
    print(json.dumps({"pid": os.getpid(), "history": history, "memory": result["memory"], "report_path": result["report_path"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
