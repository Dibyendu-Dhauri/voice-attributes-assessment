from __future__ import annotations

import logging
import time
from uuid import UUID, uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .audio import AudioDecodeError, assess_quality, decode_audio
from .inference import infer_attributes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voice-attributes")
app = FastAPI(title="Voice Attribute Inference API", version="1.0.0")


class AttributeResponse(BaseModel):
    prediction: str
    confidence: float = Field(ge=0, le=1)


class AnalysisResponse(BaseModel):
    contact_id: UUID
    gender: AttributeResponse
    age_bracket: AttributeResponse
    processing_ms: int = Field(ge=0)
    audio_quality: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    file: UploadFile | None = File(default=None),
    contact_id: UUID | None = None,
) -> AnalysisResponse:
    started = time.perf_counter()
    payload = await file.read() if file is not None else await request.body()
    if len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio file exceeds 25 MB limit")
    try:
        signal = decode_audio(payload, file.filename if file is not None else None)
    except AudioDecodeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        if file is not None:
            await file.close()
        del payload

    quality = assess_quality(signal)
    prediction = infer_attributes(signal, quality)
    processing_ms = round((time.perf_counter() - started) * 1000)
    logger.info("audio analyzed quality=%s duration=%.2fs processing_ms=%d", quality, signal.duration_seconds, processing_ms)
    return AnalysisResponse(
        contact_id=contact_id or uuid4(),
        gender=AttributeResponse(
            prediction=prediction.gender.prediction,
            confidence=prediction.gender.confidence,
        ),
        age_bracket=AttributeResponse(
            prediction=prediction.age_bracket.prediction,
            confidence=prediction.age_bracket.confidence,
        ),
        processing_ms=processing_ms,
        audio_quality=quality,
    )
