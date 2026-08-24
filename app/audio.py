from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np


class AudioDecodeError(ValueError):
    """Raised when ffmpeg cannot turn an upload into PCM audio."""


@dataclass(frozen=True)
class AudioSignal:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return self.samples.size / self.sample_rate


def decode_audio(payload: bytes, filename: str | None = None) -> AudioSignal:
    if not payload:
        raise AudioDecodeError("audio payload is empty")
    try:
        process = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1",
            ],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise AudioDecodeError("audio decoder is unavailable or timed out") from error
    if process.returncode != 0 or not process.stdout:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"unsupported or invalid audio: {detail[:160]}")

    samples = np.frombuffer(process.stdout, dtype=np.float32).copy()
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(samples), initial=0.0))
    if peak > 1.0:
        samples /= peak
    return AudioSignal(samples=samples, sample_rate=16000)


def assess_quality(signal: AudioSignal) -> str:
    samples = signal.samples
    if signal.duration_seconds < 0.25 or samples.size == 0:
        return "insufficient"
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms < 0.006:
        return "insufficient"
    clipped = float(np.mean(np.abs(samples) >= 0.995))
    zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(samples)))))
    if clipped > 0.02 or zero_crossings > 0.35 or rms < 0.015:
        return "degraded"
    return "good"
