# -*- coding: utf-8 -*-
"""医疗分诊提示词与模板。"""

from __future__ import annotations

from app.models.schemas import TriageChatRequest, TriageLevel

MEDICAL_TRIAGE_SYSTEM_PROMPT = """你是中文医疗分诊助手（严格科普模式）。
必须遵守：
1. 不做疾病诊断，不开药，不给剂量建议。
2. 只能根据提供的上下文与常识给出健康教育与就医建议。
3. 给出结论时尽量引用来源编号，例如 [1][2]。
4. 若存在高风险信号，优先给出急诊/就医建议，不继续病因推断。
5. 输出需简洁、结构化、可执行。
"""


def build_triage_query(req: TriageChatRequest, sanitized_symptom: str) -> str:
    """把结构化字段拼成检索/生成查询。"""
    parts = [
        f"症状描述：{sanitized_symptom}",
        f"年龄：{req.age if req.age is not None else '未提供'}",
        f"性别：{req.sex or '未提供'}",
        f"持续时间：{req.duration or '未提供'}",
        f"体温：{req.temperature if req.temperature is not None else '未提供'}",
        f"妊娠状态：{req.pregnancy_status or '未提供'}",
        f"既往史：{'；'.join(req.history) if req.history else '未提供'}",
        f"当前用药：{'；'.join(req.current_medications) if req.current_medications else '未提供'}",
        f"过敏史：{'；'.join(req.allergies) if req.allergies else '未提供'}",
        "请给出分诊建议、居家观察要点、何时立即就医。",
    ]
    return "\n".join(parts)


def build_emergency_advice(level: TriageLevel, hit_labels: list[str]) -> str:
    """高危命中时使用固定模板。"""
    reasons = "、".join(hit_labels) if hit_labels else "高风险症状"
    if level == TriageLevel.EMERGENCY:
        return (
            f"检测到{reasons}相关高风险信号。建议立即前往最近急诊或呼叫120，"
            "在等待救援期间避免剧烈活动，保持呼吸道通畅，并尽快联系家属陪同。"
        )
    return (
        f"检测到{reasons}相关风险信号。建议尽快到线下医疗机构就诊，"
        "若出现加重（如呼吸困难、意识异常、持续高热）请立即急诊处理。"
    )
