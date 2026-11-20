"""Narrow Chinese assertion checks, not a general statistical semantics judge."""

import re


class StatisticalLanguageError(ValueError):
    """A common unsupported assertion was found in an output surface."""


# Split contrasting assertions so an earlier disclaimer cannot excuse a later
# claim: '未做检验，但 B 显著更优' must still fail. Markdown emphasis is cosmetic.
_CLAUSES = re.compile(r"[。！？!?；;，,\n]+|但是|然而|不过|但|却")
_NEGATION = re.compile(r"不能|不可|不应|不得|无法|不足以|不等于|不意味着|不代表|并非|不是|没有证据|未能证明|尚未证实|不$")
_OBSERVED = re.compile(r"当前|本次|样本|已观测|已记录|记录的|采样点")

_SIGNIFICANCE = re.compile(
    r"显著(?:性地|地)?(?:更(?:加)?(?:优|好|稳定|低|高)|优于|低于|高于)"
    r"|(?:存在|具有|达到|获得)(?:统计(?:学)?(?:意义上|上)?的?)?显著(?:性)?(?:差异|优势)"
    r"|差异(?:具有|达到)统计(?:学)?意义"
)
_UNSUPPORTED = (
    ("不能由采样点水平推断下降速度", re.compile(r"下降速度(?:始终|总是|一直)?更快|下降(?:始终)?更快|收敛速度(?:始终)?更快"), False),
    ("不能推广到未观测区间", re.compile(r"全程领先|(?:每个|所有|各个)阶段.{0,15}(?:领先|更优)|差距(?:随.{0,10})?持续扩大"), False),
    ("随机种子结论需要限定观测样本", re.compile(r"(?:所有|任意)随机种子.{0,18}(?:更优|优于|更好|领先)"), True),
    ("模拟样本不能证明真实 CEC2013 优势", re.compile(r"(?:真实|正式)CEC2013.{0,18}(?:更优|优于|更好|领先)", re.I), False),
    ("稳定性比较需要限定当前样本", re.compile(r"不同随机种子下.{0,18}(?:更加|更)(?:一致|稳定)"), True),
    ("总体优劣判断需要限定当前样本", re.compile(r"最终性能和稳定性.{0,12}(?:均优于|都优于)"), True),
)


def verify_statistical_language(text: str, *, significance_tested: bool, context: str) -> None:
    """Reject selected positive assertions; allow local negation and sample scope.

    Covers common Chinese formulations observed in this project. Negation is
    local to a punctuation/contrast-delimited clause. This deliberately does
    not establish truth of arbitrary prose, infer numbers, or parse full grammar.
    """
    for clause in _CLAUSES.split(text):
        compact = re.sub(r"[*_`\s]", "", clause)
        rules = list(_UNSUPPORTED)
        if not significance_tested:
            rules.insert(0, ("未做显著性检验，不能断言显著优势", _SIGNIFICANCE, False))
        for reason, pattern, needs_scope in rules:
            for match in pattern.finditer(compact):
                if _NEGATION.search(compact[:match.start()]):
                    continue
                if needs_scope and _OBSERVED.search(compact):
                    continue
                raise StatisticalLanguageError(f"{context}统计语言核验失败：{reason}；发现“{match.group()}”")
