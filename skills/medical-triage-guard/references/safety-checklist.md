# Safety Checklist

Run this checklist before finishing any triage-related change.

1. Confirm no direct diagnosis language is introduced in templates or fallbacks.
2. Confirm no medication dosage or prescription instruction is introduced.
3. Confirm emergency and urgent patterns still trigger handoff (`suggest_handoff=True`).
4. Confirm `DEFAULT_DISCLAIMER` is still included in returned advice.
5. Confirm input redaction still masks phone and ID-like patterns.
6. Confirm focused tests pass:
   - `tests/test_medical_policy.py`
   - `tests/test_medical_red_flags.py`
   - `tests/test_medical_safety_guard.py`
7. If policy/rule behavior changes, add or update tests in `tests/test_medical_*.py`.
