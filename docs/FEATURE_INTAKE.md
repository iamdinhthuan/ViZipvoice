# Feature Intake

Use this lightweight intake before non-trivial implementation.

## Classification

- Tiny: localized edit, no behavior contract change.
- Normal: new endpoint, service behavior, packaging, or deployment path.
- High-risk: auth/security boundary, persistent data migration, payment, destructive behavior, or broad architecture change.

## Current Request

- Bead: `ViZipvoice-82p`
- Classification: Normal
- Scope: asynchronous TTS service that accepts uploaded clone audio and prompt transcript per request, returns a job id, exposes status polling, and serves generated WAV output.
- Out of scope: voice preset management, voice library storage, model training, checkpoint selection UI, and third-party coordination systems.
- Affected contracts: HTTP multipart create-job endpoint, job status response schema, audio download endpoint, Modal deployment entrypoint, Docker runtime.
- Expected proof: focused pytest coverage for request validation and async job lifecycle, import checks for Modal entrypoint, and Docker/Modal commands documented.

