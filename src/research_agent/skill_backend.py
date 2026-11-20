"""The analyst's filesystem exposes one read-only Skill, never project data."""

import hashlib
from pathlib import Path

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import BackendProtocol, FileDownloadResponse, LsResult, ReadResult, WriteResult

SKILL_NAME = "experiment-analysis"
SKILL_PATH = "/skills/experiment-analysis/SKILL.md"


class SkillReadBackend(BackendProtocol):
    """Public BackendProtocol with a single-file allowlist.

    SkillsMiddleware uses ls/download_files for discovery. Only a complete
    read() can produce a load receipt; discovery is deliberately not a load.
    No write-capable filesystem backend is exposed to the analyst's tools.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.path = self.root / SKILL_PATH.lstrip("/")
        self._filesystem = FilesystemBackend(root_dir=self.root, virtual_mode=True)
        self.loaded: list[dict[str, str]] = []

    def _allowed(self, path: str) -> bool:
        return path == SKILL_PATH and self.path.resolve() == self.path

    def ls(self, path: str) -> LsResult:
        if path.rstrip("/") != "/skills":
            return LsResult(error="permission denied: only /skills/ discovery is allowed")
        if not self._allowed(SKILL_PATH) or not self.path.parent.is_dir():
            return LsResult(error="experiment-analysis Skill directory is missing or is a symlink")
        return LsResult(entries=[{"path": "/skills/experiment-analysis", "is_dir": True}])

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        # Discovery may read the file privately, but does not disclose its body
        # to the model or count as an explicit read_file execution.
        responses = []
        for path in paths:
            if not self._allowed(path):
                responses.append(FileDownloadResponse(path=path, error="permission_denied"))
            else:
                responses.extend(self._filesystem.download_files([path]))
        return responses

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        if not self._allowed(file_path):
            return ReadResult(error="permission denied: only experiment-analysis/SKILL.md may be read")
        result = self._filesystem.read(file_path, offset=offset, limit=limit)
        if (not result.error and result.start_line == 1 and result.next_offset is None
                and result.total_lines == result.end_line and not result.no_lines_requested):
            self.loaded.append({"name": SKILL_NAME, "path": SKILL_PATH, "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()})
        return result

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error="permission denied: analyst Skill backend is read-only")
