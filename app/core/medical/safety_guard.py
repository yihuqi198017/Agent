# -*- coding: utf-8 -*-
"""输入脱敏与输出合规审查。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_PHONE_RE = re.compile(r"1\d{10}")
_ID_RE = re.compile(r"\b\d{17}[\dXx]\b")

_FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(你(这)?就是|可以确诊为|明确诊断为)[^。\n]{0,30}", "[已移除诊断性表述]"),
    (r"\b\d+(\.\d+)?\s*(mg|ml|毫克|片|粒)\b[^。\n]{0,24}", "[已移除处方剂量建议]"),
    (r"(百分之百|保证治愈|一定会好)", "[已移除绝对化承诺]"),
)


@dataclass
class OutputAudit:
    """输出审查结果。"""

    text: str
    blocked: bool
    violations: list[str] = field(default_factory=list)


def sanitize_input_text(text: str) -> tuple[str, list[str]]:
    """对输入做轻量脱敏。"""
    masked = text
    redactions: list[str] = []

    if _PHONE_RE.search(masked):
        masked = _PHONE_RE.sub("[PHONE]", masked)
        redactions.append("phone")

    if _ID_RE.search(masked):
        masked = _ID_RE.sub("[ID]", masked)
        redactions.append("id_card")

    return masked, redactions


def sanitize_output_text(text: str) -> OutputAudit:
    """对模型输出做二次审查并替换不合规片段。"""
    out = text
    violations: list[str] = []

    for pattern, replacement in _FORBIDDEN_PATTERNS:
        if re.search(pattern, out, flags=re.IGNORECASE):
            violations.append(pattern)
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    return OutputAudit(text=out, blocked=bool(violations), violations=violations)
