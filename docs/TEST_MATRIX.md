# Test Matrix

| Area | Proof | Command |
| --- | --- | --- |
| API schema validation | Required multipart fields and JSON options parse into stable Pydantic models | `pytest tests/tts_service -q` |
| Local async lifecycle | Submit job, poll completion, download WAV bytes using a fake synthesizer | `pytest tests/tts_service -q` |
| Modal deploy import | Modal app module imports without executing model inference | `python -m compileall tts_service modal_app.py` |
| Docker runtime metadata | Dockerfile and startup command point to FastAPI app | `python -m compileall tts_service` |

