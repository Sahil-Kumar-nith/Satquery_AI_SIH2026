# Phase 6 Hotfix — VQA crash + grounding UX

## Bug 1 (P0, real bug) — VQA crash: `operator torchvision::nms does not exist`
Root cause: `torch==2.4.1` was pinned in `backend/requirements.txt`, `torchvision` was not pinned at all.
Whatever torchvision version pip resolved doesn't match torch 2.4.1's compiled ops -> ABI mismatch at import time inside `transformers`.
Fix: pinned `torchvision==0.19.1` (the matched build for torch 2.4.1 per pytorch.org's compatibility matrix).

**Action required on your machine:**
```
pip install -r backend/requirements.txt --force-reinstall --no-deps torch torchvision
```
(force-reinstall only those two, so you don't churn every dependency)

## Issue 2 (not a bug) — grounding "blocked" on a Before-slot image
This was `validate_inputs()` working correctly: grounding requires `role=single`, the image was uploaded
with `role=before`. PS explicitly requires input/compatibility checking, so the check itself is correct
and should NOT be removed to make the system feel more permissive.

Fixed the UX instead, not the validation:
- `frontend/src/app/page.tsx`: added an inline hint under the upload row explaining Optical/SAR vs Before/After.
- `backend/agent/router.py`: `validate_inputs()` now appends an actionable hint to role-mismatch issues
  (e.g. "re-upload it under the Optical or SAR slot instead of Before/After"), so the execution trace
  itself teaches the correct action instead of just stating the failure.

No change to `TASK_REGISTRY`, no change to what's allowed — only to how clearly the system explains it.
