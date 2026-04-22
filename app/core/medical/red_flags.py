# -*- coding: utf-8 -*-
"""高危症状红线规则。"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedFlagRule:
    code: str
    label: str
    severity: str
    pattern: str


@dataclass(frozen=True)
class RedFlagHit:
    code: str
    label: str
    severity: str
    matched_text: str


_RULES: tuple[RedFlagRule, ...] = (
    RedFlagRule("emg_chest_pain", "胸痛/胸闷", "critical", r"(胸痛|胸闷|压榨样胸痛)"),
    RedFlagRule("emg_dyspnea", "呼吸困难", "critical", r"(呼吸困难|喘不上气|气促|窒息感)"),
    RedFlagRule("emg_consciousness", "意识障碍", "critical", r"(意识不清|昏迷|叫不醒|神志不清)"),
    RedFlagRule("emg_convulsion", "抽搐", "critical", r"(抽搐|惊厥)"),
    RedFlagRule("emg_bleeding", "严重出血", "critical", r"(大出血|喷射性出血|止不住血|呕血|便血)"),
    RedFlagRule("urg_high_fever", "高热", "high", r"(高烧|发热|体温\s*3[89](\.\d)?)"),
    RedFlagRule("urg_stroke", "疑似卒中", "critical", r"(口角歪斜|单侧肢体无力|言语不清|突发偏瘫)"),
)


def detect_red_flags(text: str) -> list[RedFlagHit]:
    """在输入文本中检测红线命中。"""
    hits: list[RedFlagHit] = []
    if not text.strip():
        return hits

    for rule in _RULES:
        m = re.search(rule.pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        hits.append(
            RedFlagHit(
                code=rule.code,
                label=rule.label,
                severity=rule.severity,
                matched_text=m.group(0),
            )
        )
    return hits
