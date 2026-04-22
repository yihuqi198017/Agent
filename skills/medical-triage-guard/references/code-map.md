# Medical Triage Code Map

## Core Files

- `app/core/medical/triage_service.py`
  - Orchestrates request sanitization, red-flag detection, triage-level inference, retrieval, generation, and response assembly.
  - Appends `DEFAULT_DISCLAIMER` to final advice when missing.
- `app/core/medical/triage_policy.py`
  - Defines protocol version metadata.
  - Implements `infer_triage_level` and `should_suggest_handoff`.
- `app/core/medical/red_flags.py`
  - Defines high-risk symptom regex rules and hit extraction.
- `app/core/medical/safety_guard.py`
  - Applies input masking for phone/ID.
  - Applies output filtering for diagnosis-like claims, dosage, and absolute promises.
- `app/core/medical/prompt_templates.py`
  - Defines triage generation prompts and fallback advice templates.

## API Surface

- `app/api/routes/triage.py`
  - Exposes triage endpoints and protocol version endpoint.

## Test Anchors

- `tests/test_medical_policy.py`
- `tests/test_medical_red_flags.py`
- `tests/test_medical_safety_guard.py`

Use these tests first after triage changes. Add new focused tests if behavior contracts change.
