"""Official compaction with a private, append-only text-history backend."""

from pathlib import Path
import re

from deepagents.backends.protocol import BackendProtocol, EditResult, FileDownloadResponse, ReadResult, WriteResult
from deepagents.middleware.summarization import SummarizationMiddleware, create_summarization_middleware
from langchain_core.language_models.chat_models import BaseChatModel


class HistoryArchiveError(RuntimeError):
    """History cannot be archived safely; the CLI must not accept the run."""


class HistoryArchiveBackend(BackendProtocol):
    """Internal backend only, never bound to Main's filesystem tools.

    Native summarization downloads the existing session file, then creates it
    or appends through edit(). No general read, arbitrary write, or overwrite.
    """

    def __init__(self, root: Path):
        self.archive_root = root.resolve() / ".research_history"

    def archive_path(self, file_path: str) -> Path:
        if not re.fullmatch(r"/conversation_history/session_[0-9a-f]{32}\.md", file_path):
            raise HistoryArchiveError("only conversation_history/session_<uuid>.md is allowed")
        path = self.archive_root / file_path.lstrip("/")
        if path.resolve() != path:
            raise HistoryArchiveError("history archive paths cannot pass through symlinks")
        return path

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return ReadResult(error="permission denied: history is internal, not a model-readable filesystem")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        results = []
        for name in paths:
            try:
                results.append(FileDownloadResponse(path=name, content=self.archive_path(name).read_bytes()))
            except FileNotFoundError:
                results.append(FileDownloadResponse(path=name, error="file_not_found"))
            except (HistoryArchiveError, OSError):
                results.append(FileDownloadResponse(path=name, error="permission_denied"))
        return results

    def write(self, file_path: str, content: str) -> WriteResult:
        try:
            path = self.archive_path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(content)
            return WriteResult(path=file_path)
        except (HistoryArchiveError, OSError) as exc:
            return WriteResult(error=f"history archive create failed ({type(exc).__name__})")

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        try:
            path = self.archive_path(file_path)
            if replace_all or not old_string or not new_string.startswith(old_string) or len(new_string) <= len(old_string):
                return EditResult(error="history archive only accepts appends")
            with path.open("r+", encoding="utf-8", newline="") as handle:
                if handle.read() != old_string:
                    return EditResult(error="history archive changed; append refused")
                handle.seek(0, 2)
                handle.write(new_string[len(old_string):])
            return EditResult(path=file_path, occurrences=1)
        except (HistoryArchiveError, OSError) as exc:
            return EditResult(error=f"history archive append failed ({type(exc).__name__})")


def make_summarization(
    root: Path, model: BaseChatModel, *, summary_model: BaseChatModel | None = None,
    trigger_messages: int | None = None, keep_messages: int = 6,
):
    """Replace the built-in middleware by name, without a custom summary policy."""
    backend = HistoryArchiveBackend(root)
    if trigger_messages is None:
        return create_summarization_middleware(model, backend, summarizer=summary_model)
    if keep_messages < 1 or trigger_messages <= keep_messages:
        raise ValueError("摘要触发消息数必须大于保留消息数，保留消息数至少为 1")
    return SummarizationMiddleware(
        model, backend=backend, trigger=("messages", trigger_messages), keep=("messages", keep_messages),
        summarizer=summary_model, trim_tokens_to_summarize=None,
    )
