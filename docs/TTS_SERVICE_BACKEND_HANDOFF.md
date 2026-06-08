# ViZipVoice Async TTS Service - Backend Handoff

Tai lieu nay mo ta service TTS async da duoc tach rieng cho ViZipVoice. Doi backend co the dung file nay de nam scope, API contract, cach deploy, cach tich hop, va cac gioi han hien tai cua service.

## 1. Tom tat

ViZipVoice TTS Service la service chuyen dung de sinh audio TTS tieng Viet tu mot doan audio clone/reference voice va transcript cua audio do.

Service nay khong quan ly preset voice, khong quan ly danh sach speaker, va khong luu cau hinh preset. Preset audio neu co se duoc he thong khac quan ly. Backend khi goi service nay phai upload audio clone/reference theo tung request.

Flow tich hop chinh:

1. Backend nhan yeu cau TTS tu product/client.
2. Backend lay hoac nhan `reference_audio` tu he thong preset/audio rieng.
3. Backend POST job moi len TTS service bang multipart form.
4. TTS service tra ve `202 Accepted` kem `job_id`.
5. Backend luu `job_id` va poll status.
6. Khi job `succeeded`, backend download file WAV tu `audio_url`.
7. Backend tu quyet dinh viec upload audio ket qua len storage/CDN rieng, neu san pham can URL lau dai.

Production deployment hien tai dung Modal:

- Web API: FastAPI ASGI app.
- Worker inference: Modal function async.
- GPU: NVIDIA L4.
- Cold start/scale-to-zero: bat mac dinh de tranh treo GPU idle.

Deployment URL hien tai:

```text
https://tni-tech-lab--vizipvoice-tts-fastapi-app.modal.run
```

## 2. Scope cua service

Service nay lam:

- Nhan text can doc.
- Nhan audio clone/reference voice theo tung job.
- Nhan transcript dung voi audio clone/reference.
- Chay ViZipVoice inference.
- Tra ve trang thai job async.
- Tra ve audio WAV khi job thanh cong.

Service nay khong lam:

- Khong quan ly preset voice.
- Khong tao/sua/xoa voice profile.
- Khong luu metadata speaker.
- Khong lam auth/user permission.
- Khong luu output lau dai nhu S3/R2/CDN.
- Khong thuc hien idempotency/dedup job theo `client_request_id`.
- Chua thuc thi callback HTTP du `callback_url` da co trong schema.

Neu backend can cac phan tren, nen lam o service/backend khac va coi TTS service nay la inference worker async.

## 3. Thanh phan code

| File | Vai tro |
| --- | --- |
| `tts_service/models.py` | Pydantic schema cho request, response, status, options, metrics |
| `tts_service/app.py` | FastAPI app va HTTP endpoints |
| `tts_service/local_runner.py` | Local in-process async runner, dung cho Docker/local smoke test |
| `tts_service/modal_runner.py` | Modal async runner, spawn worker function va doc ket qua qua `FunctionCall` |
| `tts_service/synthesizer.py` | Bridge tu API payload sang `zipvoice.vizipvoice.ViZipVoiceTTS` |
| `modal_app.py` | Modal app, web function, GPU worker function, env/volume/secret wiring |
| `Dockerfile` | Runtime image cho local Docker va Modal worker image |
| `requirements-service.txt` | Web/API dependencies |
| `requirements-tts-runtime.txt` | Runtime inference dependencies toi thieu hon full training/dev requirements |

## 4. API overview

Base URL production:

```text
https://tni-tech-lab--vizipvoice-tts-fastapi-app.modal.run
```

Endpoints:

| Method | Path | Muc dich |
| --- | --- | --- |
| `GET` | `/health` | Health check CPU-only |
| `POST` | `/v1/tts/jobs` | Tao job TTS async |
| `GET` | `/v1/tts/jobs/{job_id}` | Poll trang thai job |
| `GET` | `/v1/tts/jobs/{job_id}/audio` | Download WAV sau khi job thanh cong |

FastAPI OpenAPI UI khi service chay:

```text
/docs
/openapi.json
```

## 5. Tao job TTS

### Endpoint

```http
POST /v1/tts/jobs
Content-Type: multipart/form-data
```

### Multipart fields

| Field | Type | Required | Mo ta |
| --- | --- | --- | --- |
| `reference_audio` | File | Yes | Audio clone/reference voice cho job nay |
| `request` | JSON string | Yes | JSON payload theo schema ben duoi |

`reference_audio` khong duoc rong. API hien tai chua enforce max file size va chua validate extension o HTTP layer. File hop le phu thuoc vao audio loader/model runtime. Nen dung WAV/MP3 sach, it nhieu, mot nguoi noi, khoang 3-10 giay.

### Request JSON schema

```json
{
  "text": "Noi dung tieng Viet can doc.",
  "reference_text": "Transcript dung voi noi dung trong reference_audio.",
  "generation": {
    "num_steps": 16,
    "guidance_scale": 1.0,
    "speed": 1.0,
    "t_shift": 0.5,
    "seed": 666,
    "max_duration_seconds": 100,
    "normalize_vietnamese": true,
    "split_sentences": true,
    "remove_long_silence": false
  },
  "postprocess": {
    "crossfade_ms": 80,
    "silence_ms": 180,
    "fade_in_ms": 20,
    "fade_out_ms": 80
  },
  "output": {
    "format": "wav"
  },
  "callback_url": null,
  "client_request_id": "optional-id-from-caller"
}
```

### Required top-level fields

| Field | Type | Required | Validation | Mo ta |
| --- | --- | --- | --- | --- |
| `text` | string | Yes | Non-blank | Text can sinh audio |
| `reference_text` | string | Yes | Non-blank | Transcript chinh xac cua `reference_audio` |
| `generation` | object | No | Co defaults | Tham so inference |
| `postprocess` | object | No | Co defaults | Tham so ghep/fade audio |
| `output` | object | No | Co defaults | Dinh dang output |
| `callback_url` | URL/null | No | URL hop le | Reserved, hien tai chua goi callback |
| `client_request_id` | string/null | No | Khong dedupe | ID cua caller, service chi echo lai |

### `generation` fields

| Field | Type | Default | Range | Mo ta |
| --- | --- | --- | --- | --- |
| `num_steps` | integer | `16` | `4..64` | So diffusion/flow steps. Cao hon co the cham hon, co the on dinh hon |
| `guidance_scale` | number | `1.0` | `0.0..5.0` | Guidance strength |
| `speed` | number | `1.0` | `0.5..1.5` | Toc do doc |
| `t_shift` | number | `0.5` | `0.1..1.0` | Tham so sampling cua model |
| `seed` | integer/null | `666` | `>= -1` | Seed inference. `-1` tuy model wrapper co the hieu la random |
| `max_duration_seconds` | number | `100` | `1..300` | Gioi han duration inference |
| `normalize_vietnamese` | boolean | `true` | - | Chuan hoa text tieng Viet bang wrapper |
| `split_sentences` | boolean | `true` | - | Tach text dai thanh nhieu segment |
| `remove_long_silence` | boolean | `false` | - | Xoa silence dai sau inference |

### `postprocess` fields

| Field | Type | Default | Range | Mo ta |
| --- | --- | --- | --- | --- |
| `crossfade_ms` | integer | `80` | `0..300` | Crossfade giua cac segment |
| `silence_ms` | integer | `180` | `0..800` | Silence chen giua segment |
| `fade_in_ms` | integer | `20` | `0..500` | Fade in dau audio |
| `fade_out_ms` | integer | `80` | `0..500` | Fade out cuoi audio |

### `output` fields

| Field | Type | Default | Supported | Mo ta |
| --- | --- | --- | --- | --- |
| `format` | string | `wav` | `wav` only | Service hien chi tra WAV |

### Minimal request

```bash
curl -X POST "https://tni-tech-lab--vizipvoice-tts-fastapi-app.modal.run/v1/tts/jobs" \
  -F "reference_audio=@prompt.wav" \
  -F 'request={"text":"Xin chao, day la noi dung can doc.","reference_text":"Day la transcript cua audio mau."}'
```

### Full request

```bash
curl -X POST "https://tni-tech-lab--vizipvoice-tts-fastapi-app.modal.run/v1/tts/jobs" \
  -F "reference_audio=@prompt.wav" \
  -F 'request={
    "text":"ViZipVoice dang sinh giong noi tieng Viet theo yeu cau.",
    "reference_text":"Xin chao, day la doan audio mau cua toi.",
    "generation":{
      "num_steps":16,
      "guidance_scale":1.0,
      "speed":1.0,
      "t_shift":0.5,
      "seed":666,
      "max_duration_seconds":100,
      "normalize_vietnamese":true,
      "split_sentences":true,
      "remove_long_silence":false
    },
    "postprocess":{
      "crossfade_ms":80,
      "silence_ms":180,
      "fade_in_ms":20,
      "fade_out_ms":80
    },
    "output":{"format":"wav"},
    "client_request_id":"order-123"
  }'
```

### Success response

`POST /v1/tts/jobs` tra `202 Accepted`.

```json
{
  "job_id": "fc-xxxxxxxxxxxxxxxx",
  "status": "queued",
  "status_url": "/v1/tts/jobs/fc-xxxxxxxxxxxxxxxx",
  "audio_url": null,
  "metrics": null,
  "error": null,
  "client_request_id": "order-123"
}
```

Trong Modal deployment, `job_id` la `object_id` cua Modal `FunctionCall`. Backend nen coi no la opaque string, khong parse format.

## 6. Poll job status

### Endpoint

```http
GET /v1/tts/jobs/{job_id}
```

### Status values

| Status | Y nghia | Backend nen lam gi |
| --- | --- | --- |
| `queued` | Job da duoc submit | Poll lai sau delay |
| `running` | Worker dang xu ly hoac Modal call chua co ket qua | Poll lai sau delay |
| `succeeded` | Audio da sinh xong | Goi `audio_url` de download |
| `failed` | Job loi | Hien/log `error`, retry co kiem soat neu phu hop |
| `canceled` | Job bi terminate/cancel | Coi nhu terminal failure |
| `expired` | Ket qua Modal da expire | Can submit job moi neu van can audio |

### Running response

```json
{
  "job_id": "fc-xxxxxxxxxxxxxxxx",
  "status": "running",
  "status_url": "/v1/tts/jobs/fc-xxxxxxxxxxxxxxxx",
  "audio_url": null,
  "metrics": null,
  "error": null,
  "client_request_id": null
}
```

### Succeeded response

```json
{
  "job_id": "fc-xxxxxxxxxxxxxxxx",
  "status": "succeeded",
  "status_url": "/v1/tts/jobs/fc-xxxxxxxxxxxxxxxx",
  "audio_url": "/v1/tts/jobs/fc-xxxxxxxxxxxxxxxx/audio",
  "metrics": {
    "elapsed_seconds": 12.42,
    "wav_seconds": 7.85,
    "rtf": 1.58,
    "segments": 2,
    "sample_rate": 24000
  },
  "error": null,
  "client_request_id": "order-123"
}
```

### Metrics

| Field | Type | Mo ta |
| --- | --- | --- |
| `elapsed_seconds` | number | Thoi gian inference tinh bang giay |
| `wav_seconds` | number | Duration cua output WAV |
| `rtf` | number | Real-time factor. `elapsed_seconds / wav_seconds` |
| `segments` | integer | So segment sau khi tach/ghep |
| `sample_rate` | integer | Sample rate output, mac dinh `24000` |

## 7. Download audio

### Endpoint

```http
GET /v1/tts/jobs/{job_id}/audio
```

Response khi thanh cong:

```http
200 OK
Content-Type: audio/wav
```

Vi du:

```bash
curl -L "https://tni-tech-lab--vizipvoice-tts-fastapi-app.modal.run/v1/tts/jobs/fc-xxxxxxxxxxxxxxxx/audio" \
  -o output.wav
```

Neu job chua `succeeded`, endpoint tra `409 Conflict`.

Luu y quan trong: service hien tai khong dam bao luu audio lau dai. Voi Modal runner, audio bytes nam trong ket qua cua async function call. Backend nen download ngay khi job succeeded va upload sang object storage/CDN cua he thong neu can phuc vu lau dai.

## 8. Error handling

### HTTP status codes

| Code | Truong hop |
| --- | --- |
| `202` | Job duoc accept |
| `200` | Health/status/audio thanh cong |
| `404` | Khong tim thay `job_id` hoac output bi mat |
| `409` | Lay audio khi job chua succeeded |
| `422` | Payload sai, JSON sai, field blank, audio rong, option vuot range |
| `500` | Loi runtime khong duoc handle o web layer |

### Validation examples

`request` khong phai JSON hop le:

```json
{
  "detail": "request must be valid JSON"
}
```

`text` rong:

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["text"],
      "msg": "Value error, must not be blank",
      "input": ""
    }
  ]
}
```

`reference_audio` rong:

```json
{
  "detail": "reference_audio must not be empty"
}
```

## 9. Backend integration recommendations

### Suggested polling policy

Nen poll co backoff nhe de tranh spam web function:

- Lan 1: sau 2 giay.
- Lan 2-5: moi 3-5 giay.
- Sau do: moi 8-10 giay.
- Timeout phia backend tuy SLA san pham, vi du 10-15 phut.

Worker timeout mac dinh tren Modal la 900 giay.

### Suggested backend state model

Backend co the luu bang rieng:

| Field | Mo ta |
| --- | --- |
| `id` | ID request noi bo cua backend |
| `tts_job_id` | `job_id` tra ve tu TTS service |
| `client_request_id` | ID correlation tu client/product |
| `status` | Status mirror tu TTS service |
| `text` | Text can doc, neu backend duoc phep luu |
| `reference_audio_id` | ID audio preset/source o service khac |
| `output_audio_url` | URL sau khi backend upload WAV len object storage |
| `metrics` | Metrics khi succeeded |
| `error` | Error message khi failed/expired |
| `created_at` | Thoi diem tao |
| `updated_at` | Thoi diem update |

### Idempotency

`client_request_id` hien tai chi duoc echo lai. Service khong dedupe job. Neu backend retry `POST /v1/tts/jobs`, co the tao nhieu job inference va ton GPU.

Neu can idempotency, backend nen:

- Tao request record truoc khi submit TTS.
- Luu `client_request_id`/hash payload o database backend.
- Neu retry cung request, tra lai job da co thay vi submit job moi.

### Storage

Service tra file WAV truc tiep. De product dung on dinh, backend nen:

1. Poll den `succeeded`.
2. Download `audio_url`.
3. Upload WAV len storage rieng.
4. Luu URL stable vao database product.
5. Khong phu thuoc lau dai vao `audio_url` cua TTS service.

### Security

Hien service chua co auth. Neu public URL duoc goi truc tiep, bat ky ai biet URL co the submit job va lam ton GPU.

Khuyen nghi truoc production public:

- Dat API gateway/backend noi bo dung truoc TTS service.
- Khong expose Modal URL truc tiep cho client.
- Them auth header hoac secret token neu can goi cross-service.
- Gioi han max file size cho `reference_audio`.
- Gioi han max text length phia backend.
- Rate limit theo user/tenant.
- Log request id, khong log full audio/text nhay cam neu khong can.

## 10. Docker runtime

Build image:

```bash
docker build -t vizipvoice-tts .
```

Run local:

```bash
docker run --rm -p 8000:8000 \
  -e VIZIPVOICE_MODEL_DIR=/models/ViZipvoice \
  -v "$PWD/models/ViZipvoice:/models/ViZipvoice:ro" \
  vizipvoice-tts
```

Health check:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

Docker image duoc thiet ke de khong cai full dev/training stack. No dung:

- `requirements-service.txt` cho FastAPI/Pydantic/multipart.
- `requirements-tts-runtime.txt` cho inference runtime.
- Chi copy cac folder can cho runtime: `zipvoice`, `tts_service`, `README.md`, `LICENSE`.
- `.dockerignore` bo `docs`, `tests`, `assets`, `egs`, model/audio/cache/result de context gon hon.

Image van lon vi PyTorch/CUDA runtime va audio dependencies. Lan smoke test local gan nhat image khoang 3.33 GB.

## 11. Modal deployment

File deploy:

```text
modal_app.py
```

Deploy:

```bash
modal deploy modal_app.py
```

Hien tai deploy thanh cong tai:

```text
https://tni-tech-lab--vizipvoice-tts-fastapi-app.modal.run
```

### Cold-start GPU config

Worker inference Modal function:

```python
@app.function(
    image=worker_image,
    gpu=MODAL_GPU,
    timeout=WORKER_TIMEOUT_SECONDS,
    startup_timeout=WORKER_STARTUP_TIMEOUT_SECONDS,
    min_containers=0,
    buffer_containers=0,
    max_containers=WORKER_MAX_CONTAINERS,
    scaledown_window=WORKER_SCALEDOWN_WINDOW_SECONDS,
)
```

Defaults:

| Config | Default | Ly do |
| --- | --- | --- |
| `VIZIPVOICE_MODAL_GPU` | `L4` | Dung NVIDIA L4 theo yeu cau |
| `VIZIPVOICE_MODAL_MAX_CONTAINERS` | `1` | Khong scale nhieu GPU ngoai y muon |
| `VIZIPVOICE_MODAL_SCALEDOWN_WINDOW` | `5` | Tat worker nhanh sau khi idle |
| `min_containers` | `0` | Khong giu GPU warm |
| `buffer_containers` | `0` | Khong pre-warm GPU |
| `VIZIPVOICE_MODAL_WORKER_TIMEOUT` | `900` | Gioi han job 15 phut |
| `VIZIPVOICE_MODAL_STARTUP_TIMEOUT` | `900` | Cho model cold start |

Web API Modal function cung scale-to-zero:

| Config | Default |
| --- | --- |
| `min_containers` | `0` |
| `buffer_containers` | `0` |
| `VIZIPVOICE_MODAL_WEB_SCALEDOWN_WINDOW` | `10` |
| `modal.concurrent(max_inputs=20)` | 20 request dong thoi cho web API |

### Modal volume

Neu model da nam trong Modal Volume:

```bash
export VIZIPVOICE_MODAL_VOLUME=vizipvoice-models
export VIZIPVOICE_MODAL_VOLUME_MOUNT=/models
export VIZIPVOICE_MODEL_DIR=/models/ViZipvoice
modal deploy modal_app.py
```

`modal_app.py` dung:

```python
modal.Volume.from_name(volume_name, create_if_missing=False)
```

Nghia la deploy se fail neu volume chua ton tai. Cach nay tranh viec vo tinh tao volume rong roi worker khong thay model.

### Modal secret

Neu can Hugging Face token hoac env secret:

```bash
export VIZIPVOICE_MODAL_SECRET=vizipvoice-secrets
modal deploy modal_app.py
```

Secret nay se duoc attach vao worker function.

## 12. Environment variables

### Model/runtime env

| Env | Default | Mo ta |
| --- | --- | --- |
| `VIZIPVOICE_REPO_ID` | Wrapper default | Hugging Face repo id |
| `VIZIPVOICE_REVISION` | `None` | HF revision/branch/commit |
| `VIZIPVOICE_MODEL_DIR` | `None` | Local/model volume path. Neu set thi load tu thu muc nay |
| `VIZIPVOICE_CHECKPOINT_NAME` | `latest` | Checkpoint file hoac latest |
| `VIZIPVOICE_VOCODER_PATH` | `None` | Custom vocoder path neu co |
| `VIZIPVOICE_DEVICE` | Auto | `cuda`, `cpu`, `mps`, tuy wrapper ho tro |
| `VIZIPVOICE_FP16` | `1` | `0` de tat FP16 |
| `VIZIPVOICE_NUM_THREADS` | `1` | So CPU threads |
| `HF_HOME` | Modal/default env | Hugging Face cache dir |
| `HF_HUB_CACHE` | Modal/default env | Hugging Face hub cache dir |

### Local service env

| Env | Default | Mo ta |
| --- | --- | --- |
| `TTS_JOB_DIR` | `/tmp/vizipvoice_tts_jobs` | Thu muc local runner ghi input/output |

### Modal deploy env

| Env | Default | Mo ta |
| --- | --- | --- |
| `VIZIPVOICE_MODAL_APP_NAME` | `vizipvoice-tts` | Ten Modal app |
| `VIZIPVOICE_MODAL_GPU` | `L4` | GPU type |
| `VIZIPVOICE_MODAL_WORKER_TIMEOUT` | `900` | Worker timeout seconds |
| `VIZIPVOICE_MODAL_STARTUP_TIMEOUT` | `900` | Worker startup timeout seconds |
| `VIZIPVOICE_MODAL_SCALEDOWN_WINDOW` | `5` | Worker scaledown window seconds |
| `VIZIPVOICE_MODAL_MAX_CONTAINERS` | `1` | Max worker containers |
| `VIZIPVOICE_MODAL_WEB_SCALEDOWN_WINDOW` | `10` | Web function scaledown window seconds |
| `VIZIPVOICE_MODAL_VOLUME` | unset | Modal volume name |
| `VIZIPVOICE_MODAL_VOLUME_MOUNT` | `/models` | Mount path cho volume |
| `VIZIPVOICE_MODAL_SECRET` | unset | Modal secret name |

## 13. Local runner vs Modal runner

### Local runner

Dung khi chay Docker/local:

- Luu job trong memory process.
- Ghi file vao `TTS_JOB_DIR`.
- Dung `ThreadPoolExecutor(max_workers=1)`.
- Khong ben vung neu restart process.
- Phu hop smoke test/local dev, khong phai production queue.

### Modal runner

Dung tren Modal production:

- `POST /v1/tts/jobs` goi `synthesize_tts_job.spawn(...)`.
- Web API tra `job_id` ngay.
- `GET /v1/tts/jobs/{job_id}` dung `modal.FunctionCall.from_id(job_id).get(timeout=0)`.
- Neu chua co result, service map thanh `running`.
- Neu function result co san, service tra `succeeded` va metrics.
- `GET /audio` doc `audio_bytes` tu Modal function result.

Luu y: vi `audio_bytes` nam trong function result, output co the expire theo chinh sach Modal. Neu expire, status co the thanh `expired` va backend can submit lai neu chua luu output.

## 14. Quality guidelines cho input

De voice clone tot hon:

- Audio reference nen 3-10 giay.
- Chi mot nguoi noi.
- Giong ro, it noise, it echo.
- Khong nhac nen.
- Transcript phai khop gan nhu tuyet doi voi audio.
- Text can doc nen co dau cau ro rang.
- Voi text dai, de `split_sentences=true`.

Nhung dau vao de gay loi/kem chat luong:

- Transcript sai noi dung audio.
- Audio qua ngan hoac qua dai.
- Nhieu speaker trong cung file.
- Nhac nen, echo phong manh, tieng on lon.
- Text co nhieu ky tu ngoai vocab/model.

## 15. Observability hien tai

Service response tra metrics khi succeeded, nhung hien chua co he thong logging/trace rieng.

Backend nen log:

- `client_request_id`
- `job_id`
- status transition
- request created time
- succeeded/failed time
- `metrics.elapsed_seconds`
- `metrics.wav_seconds`
- `metrics.rtf`
- `error`

Khong nen log raw audio. Viec log full text tuy thuoc policy rieng cua san pham.

## 16. Known limitations

Danh sach nay quan trong cho backend khi tich hop:

- Chua co auth trong TTS service.
- Chua co rate limit.
- Chua co max upload size o FastAPI layer.
- Chua co max text length rieng ngoai `max_duration_seconds`.
- `callback_url` la reserved field, chua duoc worker goi.
- `client_request_id` khong tao idempotency.
- Output chi ho tro WAV.
- Service khong upload output len object storage.
- Local runner khong persistent.
- Modal output co the expire neu backend khong download kip.
- `max_containers=1` giup kiem soat cost nhung lam throughput production bi gioi han.
- Cold start giup tiet kiem tien GPU idle nhung request dau sau idle se cham hon do can start container/load model.

## 17. Suggested production hardening backlog

Neu chuan bi dua vao production traffic lon, nen them:

1. Auth cross-service, vi du Bearer token noi bo.
2. Max upload size va MIME allowlist.
3. Max text length theo product plan.
4. Idempotency theo `client_request_id`.
5. Callback worker khi job terminal, hoac backend queue consumer rieng.
6. Upload output vao S3/R2/GCS tu worker hoac backend.
7. Structured logs va request correlation id.
8. Metrics dashboard cho queue time, inference time, fail rate, GPU cold starts.
9. Concurrency policy: giu `max_containers=1` de tiet kiem, hoac tang neu can throughput.
10. Retry policy cho loi tam thoi, tranh retry vo han gay ton GPU.

## 18. Verification da chay

Cac lenh da chay khi implement:

```bash
python -m pytest tests/tts_service -q
python -m compileall tts_service modal_app.py
docker build -t vizipvoice-tts:codex .
docker run --rm -d -p 8001:8000 --name vizipvoice-tts-codex-smoke vizipvoice-tts:codex
curl -fsS http://localhost:8001/health
modal deploy modal_app.py
curl -fsS https://tni-tech-lab--vizipvoice-tts-fastapi-app.modal.run/health
```

Ket qua gan nhat:

```text
10 passed
Docker /health: {"status":"ok"}
Modal /health: {"status":"ok"}
```

Khong submit job inference that len Modal trong smoke test cuoi de tranh kich hoat GPU L4 va phat sinh chi phi khong can thiet.

## 19. Example backend pseudo-code

```python
import json
import time
import requests


BASE_URL = "https://tni-tech-lab--vizipvoice-tts-fastapi-app.modal.run"


def create_tts_job(reference_audio_path: str, text: str, reference_text: str, request_id: str) -> str:
    payload = {
        "text": text,
        "reference_text": reference_text,
        "client_request_id": request_id,
    }
    with open(reference_audio_path, "rb") as audio_file:
        response = requests.post(
            f"{BASE_URL}/v1/tts/jobs",
            files={"reference_audio": audio_file},
            data={"request": json.dumps(payload)},
            timeout=30,
        )
    response.raise_for_status()
    return response.json()["job_id"]


def wait_for_tts(job_id: str, timeout_seconds: int = 900) -> dict:
    deadline = time.time() + timeout_seconds
    delay = 2
    while time.time() < deadline:
        response = requests.get(f"{BASE_URL}/v1/tts/jobs/{job_id}", timeout=10)
        response.raise_for_status()
        job = response.json()
        if job["status"] == "succeeded":
            return job
        if job["status"] in {"failed", "canceled", "expired"}:
            raise RuntimeError(job.get("error") or f"TTS job ended with {job['status']}")
        time.sleep(delay)
        delay = min(delay + 2, 10)
    raise TimeoutError(f"TTS job timed out: {job_id}")


def download_tts_audio(job_id: str) -> bytes:
    response = requests.get(f"{BASE_URL}/v1/tts/jobs/{job_id}/audio", timeout=60)
    response.raise_for_status()
    return response.content
```

## 20. Owner handoff notes

Nhung diem backend team can confirm truoc khi noi vao production:

- Modal URL se duoc goi truc tiep tu backend noi bo hay di qua API gateway.
- Ai chiu trach nhiem luu output WAV lau dai.
- `reference_audio` se lay tu preset service nao va backend co dam bao transcript dung khong.
- Gioi han text/audio theo product tier.
- Co can idempotency theo `client_request_id` khong.
- Co can callback hay backend poll la du.
- Co chap nhan cold start latency de tiet kiem GPU idle cost khong.
