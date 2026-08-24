from io import BytesIO
from wave import Wave_write, open as wave_open

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def wav_bytes(amplitude: int = 12000, duration: float = 0.5) -> bytes:
    output = BytesIO()
    with wave_open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        samples = (amplitude.to_bytes(2, "little", signed=True) for _ in range(int(16000 * duration)))
        wav.writeframes(b"".join(samples))
    return output.getvalue()


def test_analyze_returns_contract_and_quality() -> None:
    response = client.post("/analyze", files={"file": ("sample.wav", wav_bytes(), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"contact_id", "gender", "age_bracket", "processing_ms", "audio_quality"}
    assert body["audio_quality"] in {"good", "degraded", "insufficient"}
    assert 0 <= body["gender"]["confidence"] <= 1


def test_silent_audio_is_marked_insufficient() -> None:
    response = client.post("/analyze", files={"file": ("silence.wav", wav_bytes(0), "audio/wav")})
    assert response.status_code == 200
    assert response.json()["audio_quality"] == "insufficient"
    assert response.json()["gender"]["prediction"] == "unknown"


def test_invalid_audio_is_rejected() -> None:
    response = client.post("/analyze", files={"file": ("bad.bin", b"not audio", "application/octet-stream")})
    assert response.status_code == 422


def test_raw_audio_body_is_accepted() -> None:
    response = client.post("/analyze", content=wav_bytes(), headers={"content-type": "audio/wav"})
    assert response.status_code == 200
