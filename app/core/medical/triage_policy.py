# -*- coding: utf-8 -*-
"""分诊策略与协议版本。"""

from __future__ import annotations

from app.models.schemas import TriageLevel

PROTOCOL_VERSION = "triage-cn-v1.0.0"
PROTOCOL_UPDATED_AT = "2026-04-17"

DEFAULT_DISCLAIMER = (
    "本助手仅提供健康科普信息与就医建议，不构成医疗诊断、处方或替代医生面诊。"
    "若症状加重或持续不缓解，请尽快到正规医疗机构就诊。"
)


def infer_triage_level(*, risk_severities: list[str], age: int | None, temperature: float | None, symptom_text: str) -> TriageLevel:
    """根据红线、年龄、体温与症状文本推断分诊级别。"""
    if any(s in {"critical", "high"} for s in risk_severities):
        return TriageLevel.EMERGENCY

    if temperature is not None and temperature >= 39.5:
        return TriageLevel.URGENT

    if age is not None and age <= 1 and temperature is not None and temperature >= 38.0:
        return TriageLevel.URGENT

    lower = symptom_text.lower()
    urgent_terms = ("持续高热", "剧烈", "反复呕吐", "脱水", "持续胸闷", "严重头痛")
    if any(t in lower for t in urgent_terms):
        return TriageLevel.URGENT

    mild_terms = ("轻微", "偶发", "鼻塞", "打喷嚏", "轻咳", "轻度不适")
    if any(t in lower for t in mild_terms):
        return TriageLevel.SELF_CARE

    return TriageLevel.ROUTINE


def should_suggest_handoff(level: TriageLevel) -> bool:
    """是否建议转人工/医生。"""
    return level in {TriageLevel.EMERGENCY, TriageLevel.URGENT}
