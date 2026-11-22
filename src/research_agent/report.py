"""Assemble trusted evidence inside an otherwise model-authored report."""

EVIDENCE_START = "<!-- BEGIN_ANALYSIS_EVIDENCE -->"
EVIDENCE_END = "<!-- END_ANALYSIS_EVIDENCE -->"


class ReportAssemblyError(ValueError):
    """The draft does not identify one replaceable evidence region."""


def assemble_report(draft: str, analysis: dict) -> str:
    """Replace the entire evidence region, including any copied/truncated data.

    The caller supplies analysis from verified current-run execution, never
    model arguments or historical Memory. Prose outside this region stays
    untouched. Validation of the resulting file remains a separate step.
    """
    if draft.count(EVIDENCE_START) != 1 or draft.count(EVIDENCE_END) != 1:
        raise ReportAssemblyError("报告草稿必须包含且只包含一组 BEGIN_ANALYSIS_EVIDENCE / END_ANALYSIS_EVIDENCE 标记")
    before, remainder = draft.split(EVIDENCE_START)
    if EVIDENCE_END not in remainder:
        raise ReportAssemblyError("报告可信证据区域的开始和结束标记顺序错误")
    _, after = remainder.split(EVIDENCE_END)
    return before + EVIDENCE_START + "\n\n" + analysis["evidence_markdown"] + "\n\n" + EVIDENCE_END + after


def verified_summary(conclusions: dict, relative_path: str, absolute_path: str) -> str:
    """Present verified-run model conclusions with program-owned location data."""
    return (
        "模拟数据分析摘要（未进行统计显著性检验）：\n\n"
        f"- 最终性能：{conclusions['final_performance']}\n"
        f"- 稳定性：{conclusions['stability']}\n"
        f"- 收敛趋势：{conclusions['convergence']}\n\n"
        f"报告相对路径：{relative_path}\n实际保存路径：{absolute_path}"
    )
