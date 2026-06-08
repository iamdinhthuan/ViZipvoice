# Async TTS Service Design

## Goal

Build a TTS-only service for ViZipVoice that accepts a target text, an uploaded clone reference audio, and the transcript for that reference audio. The service returns immediately with a job id, lets clients poll job state, and exposes the generated audio when the job succeeds.

## Scope

The service does not manage preset voices or a reusable voice library. The caller owns any voice management outside this service and uploads the clone audio for each generation request.

## API Contract

`POST /v1/tts/jobs` accepts `multipart/form-data`:

- `reference_audio`: required file upload.
- `request`: required JSON string containing `text`, `reference_text`, optional generation settings, optional postprocess settings, optional output settings, optional callback URL, and optional client request id.

`GET /v1/tts/jobs/{job_id}` returns job state: `queued`, `running`, `succeeded`, `failed`, `canceled`, or `expired`.

`GET /v1/tts/jobs/{job_id}/audio` returns `audio/wav` when the job has succeeded.

## Architecture

The reusable package lives in `tts_service/`. It contains Pydantic schemas, FastAPI route wiring, and a local runner used by Docker/dev tests. Modal deployment lives in `modal_app.py` and uses Modal's job queue pattern: the web function accepts uploads, spawns a GPU worker function, uses the Modal function call id as `job_id`, and polls the call result for status/audio.

## Validation

Validation should reject missing text, missing reference text, empty uploaded audio, unsupported output formats, and out-of-range generation/postprocess settings. Defaults should mirror the existing `ViZipVoiceTTS.synthesize()` wrapper.

## Deployment

`Dockerfile` runs the local FastAPI app with `uvicorn`. `modal_app.py` builds a Modal image from the Dockerfile via `modal.Image.from_dockerfile(...)` and exposes the ASGI app via `@modal.asgi_app()`. Model path/configuration remains environment-driven because the user has already prepared model setup.

