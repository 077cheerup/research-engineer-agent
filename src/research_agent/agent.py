"""Construct the real DeepAgents graph and its narrowly scoped filesystem."""

from __future__ import annotations

import os
import re
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import ReadResult, WriteResult
from deepagents.middleware.filesystem import FilesystemMiddleware
from dotenv import dotenv_values
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from research_agent.prompts import SYSTEM_PROMPT
from research_agent.tools.experiment import make_analyze_experiment, project_root


class ConfigurationError(ValueError):
    """Missing model configuration; values and credentials are never logged."""


def report_path(output: str, root: Path) -> Path:
    """Use one simple, stable virtual namespace: /reports/<safe-name>.md."""
    value = output.removeprefix("/")
    if not re.fullmatch(r"reports/[A-Za-z0-9][A-Za-z0-9_.-]*\.md", value):
        raise ValueError("报告路径必须是 reports/<安全文件名>.md（字母、数字、_、-、.）")
    root = root.resolve()
    candidate = root / value
    if (root / "reports").resolve() != root / "reports" or candidate.resolve() != candidate:
        raise ValueError("报告路径不能经过符号链接")
    return candidate


class ReportFilesystemBackend(FilesystemBackend):
    """Keep the built-in write_file, but limit writes and never overwrite.

    DeepAgents 0.7.40's base write() truncates existing files. Exclusive
    creation implements the task's stronger no-overwrite requirement.
    No third-party sources are modified.
    """

    def __init__(self, root: Path):
        self.workspace = root.resolve()
        super().__init__(root_dir=self.workspace, virtual_mode=True)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        # The CSV tool owns reads. This also covers dotfiles: the framework's
        # /** permission glob does not match .env in version 0.7.40.
        return ReadResult(error="permission denied: use analyze_experiment to read data/*.csv")

    def write(self, file_path: str, content: str) -> WriteResult:
        try:
            destination = report_path(file_path, self.workspace)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
            return WriteResult(path="/" + destination.relative_to(self.workspace).as_posix())
        except FileExistsError:
            return WriteResult(error="报告已存在，拒绝覆盖；请指定新的 reports/*.md 文件名")
        except (OSError, ValueError) as exc:
            # Do not relay arbitrary system/provider details to the model.
            return WriteResult(error=f"报告写入失败 ({type(exc).__name__})；仅允许新建 reports/*.md")


def load_model(root: Path) -> ChatOpenAI:
    """Read project .env, with explicit LLM_* environment overrides."""
    file_config = dotenv_values(root / ".env", interpolate=False)
    names = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
    values = {name: os.environ.get(name, file_config.get(name) or "").strip() for name in names}
    missing = [name for name in names if not values[name]]
    if missing:
        raise ConfigurationError("缺少模型配置：" + ", ".join(missing) + "；请设置项目 .env")
    return ChatOpenAI(
        model=values["LLM_MODEL"],
        api_key=values["LLM_API_KEY"],
        base_url=values["LLM_BASE_URL"],
        use_responses_api=False,
        max_retries=0,
        timeout=60,
    )


def build_agent(root: Path | str | None = None, *, model: BaseChatModel | None = None):
    """Use the real create_deep_agent; injectable models enable offline tests."""
    workspace = Path(root).resolve() if root is not None else project_root()
    backend = ReportFilesystemBackend(workspace)
    return create_deep_agent(
        model=model if model is not None else load_model(workspace),
        tools=[make_analyze_experiment(workspace)],
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        # 0.7.40 requires read_file in this list; the backend denies every read.
        middleware=[FilesystemMiddleware(backend=backend, tools=["read_file", "write_file"])],
        name="mini-research-engineer",
    )
