from app.core.medical.red_flags import detect_red_flags


def test_detect_red_flags_hits_critical_terms() -> None:
    text = "患者突发胸痛并呼吸困难，伴有出冷汗"
    hits = detect_red_flags(text)
    codes = {h.code for h in hits}
    assert "emg_chest_pain" in codes
    assert "emg_dyspnea" in codes
