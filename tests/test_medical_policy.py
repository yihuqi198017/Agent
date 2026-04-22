from app.core.medical.triage_policy import infer_triage_level
from app.models.schemas import TriageLevel


def test_infer_triage_level_emergency_with_critical_flag() -> None:
    level = infer_triage_level(
        risk_severities=["critical"],
        age=35,
        temperature=37.2,
        symptom_text="胸痛",
    )
    assert level == TriageLevel.EMERGENCY


def test_infer_triage_level_self_care_for_mild_text() -> None:
    level = infer_triage_level(
        risk_severities=[],
        age=25,
        temperature=36.8,
        symptom_text="轻微鼻塞，偶发打喷嚏",
    )
    assert level == TriageLevel.SELF_CARE
