from app.core.medical.safety_guard import sanitize_input_text, sanitize_output_text


def test_sanitize_input_text_masks_phone_and_id() -> None:
    masked, redactions = sanitize_input_text("电话 13812345678, 身份证 110101199001011234")
    assert "[PHONE]" in masked
    assert "[ID]" in masked
    assert "phone" in redactions
    assert "id_card" in redactions


def test_sanitize_output_text_blocks_diagnosis_and_dosage() -> None:
    text = "你这是肺炎，每次吃500mg，一天三次，保证治愈。"
    audit = sanitize_output_text(text)
    assert audit.blocked is True
    assert "[已移除诊断性表述]" in audit.text or "[已移除处方剂量建议]" in audit.text
