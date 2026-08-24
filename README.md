# Voice Attribute Inference API

A small FastAPI service for estimating caller gender and age bracket from a short audio sample. It is designed as an assessment-quality baseline: the service contract, audio handling, quality signaling, observability, containerization, and tests are production-shaped, while the inference implementation is intentionally transparent and replaceable.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`ffmpeg` must be installed locally. On macOS: `brew install ffmpeg`.

## Run with Docker

```bash
docker compose up --build
```

The API is available at `http://localhost:8000`; interactive OpenAPI docs are at `/docs`.

## Smoke test

Use any short WAV, MP3, M4A, OGG, or other ffmpeg-readable voice recording. The request is decoded in memory and is not persisted.

```bash
curl -X POST http://localhost:8000/analyze \
  -F 'file=@sample.wav' \
  -F 'contact_id=8d3f7f90-2d4a-4fd4-a7ce-d4d5f6d0b9de'
```

Raw audio bodies are also accepted:

```bash
curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: audio/wav' \
  --data-binary '@sample.wav'
```

## Design write-up

The service uses FastAPI for a small asynchronous HTTP surface and ffmpeg for broad codec support. Each request is read into memory, decoded to mono 16 kHz float PCM through a subprocess pipe, analyzed, and released; no audio path or persistent store exists. A NumPy feature pass computes RMS energy, zero-crossing rate, spectral centroid, and dominant frequency. These features feed a deliberately conservative heuristic baseline that returns `unknown` for insufficient audio and lowers confidence for degraded audio. This keeps the behavior explainable and makes the inference module easy to replace with a validated SpeechBrain or pyannote model later. Confidence values are model scores, not calibrated population probabilities, and the README should be updated with validation results before production use.

For 1,000 concurrent calls, I would separate decoding and inference into a bounded worker pool, keep a warm model process per CPU/GPU worker, apply request and queue timeouts, and expose queue depth, latency percentiles, quality rates, and model errors to metrics. An API gateway could enforce authentication, quotas, and payload limits while an autoscaled stateless inference tier handles traffic. Audio would remain ephemeral throughout, with encrypted transport and strict access logging that excludes payloads.

## API response

```json
{
  "contact_id": "uuid",
  "gender": {"prediction": "male", "confidence": 0.87},
  "age_bracket": {"prediction": "31-45", "confidence": 0.63},
  "processing_ms": 142,
  "audio_quality": "good"
}
```

`audio_quality` is `good`, `degraded`, or `insufficient`. Invalid media returns HTTP 422; payloads over 25 MB return HTTP 413. `GET /health` returns a liveness response.

## Testing

```bash
.venv/bin/pip install pytest httpx
.venv/bin/pytest -q
```

The tests generate a tiny WAV fixture in memory, so no audio sample is committed. For a meaningful evaluation, use a consented public voice dataset such as Mozilla Common Voice and report accuracy, macro-F1, unknown rate, and confidence calibration separately by quality condition.

## Limitations and next steps

This heuristic is a baseline, not a clinically or scientifically validated demographic classifier. Voice characteristics do not reliably establish identity, gender, or age, and performance can vary by language, accent, recording device, and environment. A production version should use consented, representative data, evaluate fairness and calibration, document intended use, and prefer `unknown` when confidence is not adequate. The next engineering step would be a versioned model adapter with offline evaluation and a streaming WebSocket endpoint that emits quality updates before final inference.
