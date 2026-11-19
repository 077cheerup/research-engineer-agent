"""Project-scoped LangChain tool. No LLM and no report generation."""

from pathlib import Path

from langchain_core.tools import ToolException, tool

from research_agent.analysis import AnalysisError, analyze_csv, evidence_markdown


def project_root() -> Path:
    """Editable/source checkout root; independent of the shell working directory."""
    return Path(__file__).resolve().parents[3]


def resolve_csv_path(csv_path: str, root: Path) -> Path:
    """Accept project-relative data/*.csv or a host path inside that directory."""
    root = root.resolve()
    data_dir = root / "data"
    path = Path(csv_path)
    if ".." in path.parts or "~" in path.parts:
        raise AnalysisError("CSV path traversal is forbidden; use data/<name>.csv")
    candidate = (path if path.is_absolute() else root / path).resolve()
    if not candidate.is_relative_to(data_dir) or data_dir.resolve() != data_dir:
        raise AnalysisError("CSV must be inside this project's data/ directory")
    if candidate.suffix.lower() != ".csv" or not candidate.is_file():
        raise AnalysisError("CSV must be an existing regular .csv file under data/")
    return candidate


def make_analyze_experiment(root: Path | str | None = None):
    """Bind an explicit workspace root without relying on global cwd/state."""
    workspace = Path(root).resolve() if root is not None else project_root()

    @tool
    def analyze_experiment(csv_path: str, function_name: str) -> dict:
        """Analyze simulated minimization CSV data under project data/.

        Args:
            csv_path: Project-relative CSV path, e.g. data/sample_experiments.csv.
            function_name: Exact function label, e.g. F1.

        Returns:
            JSON-compatible source hash, final-by-seed statistics (sample std
            ddof=1), mean convergence curves, descriptive ranking, and numeric
            evidence tables to copy verbatim. No significance test is performed.
        """
        try:
            path = resolve_csv_path(csv_path, workspace)
            result = analyze_csv(path, function_name)
            result["source"]["csv_path"] = path.relative_to(workspace).as_posix()
            result["evidence_markdown"] = evidence_markdown(result)
            return result
        except AnalysisError as exc:
            raise ToolException(str(exc)) from exc

    analyze_experiment.handle_tool_error = True
    return analyze_experiment


analyze_experiment = make_analyze_experiment()
