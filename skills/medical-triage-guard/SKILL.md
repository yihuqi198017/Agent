---
name: medical-triage-guard
description: Maintain and extend the medical triage pipeline in this repository with safety-first behavior. Use when changing triage policy, red-flag rules, safety guard filters, triage prompts, triage service orchestration, or `/api/v1/triage/*` behavior; and when validating that disclaimer, handoff, and safety tests still pass.
---

# Medical Triage Guard

## Overview

Implement triage-related code changes without introducing unsafe guidance or contract regressions.
Prefer rule and policy updates over prompt-only fixes, and verify every change with focused medical tests.

## Workflow

1. Classify the request before editing:
- Red-flag detection or severity mapping: edit `app/core/medical/red_flags.py`.
- Triage level or handoff policy: edit `app/core/medical/triage_policy.py`.
- Input/output compliance filtering: edit `app/core/medical/safety_guard.py`.
- End-to-end triage behavior, retrieval, fallback, response assembly: edit `app/core/medical/triage_service.py`.
- Prompt behavior for generated advice: edit `app/core/medical/prompt_templates.py`.

2. Keep these invariants unchanged unless the user explicitly asks to change them:
- `TriageChatResponse` must include `triage_level`, `advice`, `risk_flags`, `suggest_handoff`, `disclaimer`, and `trace_id`.
- `DEFAULT_DISCLAIMER` must be present in final advice output.
- Emergency and urgent levels should still set `suggest_handoff=True`.
- Unsafe output phrases should still be filtered by `sanitize_output_text`.

3. Apply the smallest effective change:
- Update the rule/policy module first.
- Update service integration only if required.
- Add or update focused tests in `tests/test_medical_*.py` for behavior changes.

4. Validate locally:
- Run `python skills/medical-triage-guard/scripts/run_triage_checks.py`.
- If the change impacts broader behavior, run `python -m pytest tests -q`.

5. Report results with three items:
- What changed.
- Which safety invariants were preserved or intentionally changed.
- Which tests were run and their outcomes.

## Repository References

Load only what is needed:
- `references/code-map.md`: triage module responsibilities and coupling.
- `references/safety-checklist.md`: pre-merge safety checklist.

## Scripts

Run deterministic checks:
- `scripts/run_triage_checks.py`: runs focused medical tests quickly (`test_medical_policy`, `test_medical_red_flags`, `test_medical_safety_guard`).

## Boundaries

Do not provide diagnosis, prescriptions, or dosage instructions.
Escalate or hand off for emergency and urgent patterns.
