# Async TTS Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an async HTTP TTS service for ViZipVoice with clone-audio upload, Docker packaging, and Modal deployment.

**Architecture:** Add a small `tts_service` package for schemas, app factory, and local in-process jobs. Add `modal_app.py` to expose the same API shape on Modal using `Function.spawn()` for async GPU work and `FunctionCall.get(timeout=0)` for polling.

**Tech Stack:** Python, FastAPI, Pydantic, pytest, uvicorn, Modal, existing `zipvoice.vizipvoice.ViZipVoiceTTS`.

---

### Task 1: API Schema Tests

**Files:**
- Create: `tests/tts_service/test_models.py`
- Create: `tts_service/models.py`

- [ ] Write failing tests for default generation, output validation, and required text/reference transcript.
- [ ] Run `pytest tests/tts_service/test_models.py -q` and verify tests fail because `tts_service` does not exist.
- [ ] Implement Pydantic models with defaults matching `ViZipVoiceTTS.synthesize()`.
- [ ] Re-run `pytest tests/tts_service/test_models.py -q` and verify pass.

### Task 2: Local Async App Tests

**Files:**
- Create: `tests/tts_service/test_app.py`
- Create: `tts_service/app.py`
- Create: `tts_service/local_runner.py`
- Create: `tts_service/synthesizer.py`

- [ ] Write failing tests that submit multipart form data with fake audio, poll status, and download fake WAV bytes.
- [ ] Run `pytest tests/tts_service/test_app.py -q` and verify fail because app factory does not exist.
- [ ] Implement app factory and local runner with `queued`, `running`, `succeeded`, `failed` states.
- [ ] Re-run app tests and verify pass.

### Task 3: Modal Entrypoint

**Files:**
- Create: `modal_app.py`
- Modify: `tts_service/app.py`

- [ ] Add import/compile test coverage or verification for Modal app module.
- [ ] Implement Modal image from Dockerfile, GPU worker function, ASGI function, and Modal-backed runner adapter.
- [ ] Verify `python -m compileall tts_service modal_app.py`.

### Task 4: Packaging

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-service.txt`
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `tts_service/__init__.py`

- [ ] Add service dependencies without disturbing model dependencies.
- [ ] Add Dockerfile that installs repo, exposes port `8000`, and runs `uvicorn tts_service.app:create_app --factory`.
- [ ] Verify compile and document deploy commands.

