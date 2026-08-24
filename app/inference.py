from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import AudioSignal


@dataclass(frozen=True)
class Prediction:
    prediction: str
    confidence: float


@dataclass(frozen=True)
class AttributePrediction:
    gender: Prediction
    age_bracket: Prediction


def _spectral_features(signal: AudioSignal) -> tuple[float, float, float]:
    samples = signal.samples
    window = min(samples.size, signal.sample_rate * 5)
    segment = samples[:window]
    frequencies = np.fft.rfftfreq(segment.size, 1 / signal.sample_rate)
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
    total = float(np.sum(spectrum)) or 1.0
    centroid = float(np.sum(frequencies * spectrum) / total)
    dominant = float(frequencies[int(np.argmax(spectrum))])
    rms = float(np.sqrt(np.mean(segment * segment)))
    return centroid, dominant, rms


def infer_attributes(signal: AudioSignal, quality: str) -> AttributePrediction:
    if quality == "insufficient":
        unknown = Prediction("unknown", 0.0)
        return AttributePrediction(unknown, unknown)

    centroid, dominant, rms = _spectral_features(signal)
    # Transparent baseline: pitch and spectral balance are weak proxies, never
    # a substitute for a validated demographic model.
    pitch_proxy = max(70.0, min(320.0, dominant or centroid / 4))
    gender_score = max(0.0, min(1.0, (pitch_proxy - 135.0) / 90.0))
    if 0.35 < gender_score < 0.65:
        gender = Prediction("unknown", round(0.5 + abs(gender_score - 0.5), 2))
    elif gender_score >= 0.5:
        gender = Prediction("female", round(0.55 + 0.4 * gender_score, 2))
    else:
        gender = Prediction("male", round(0.95 - 0.4 * gender_score, 2))

    # Higher dominant frequency and spectral centroid generally correlate with
    # a brighter voice; this intentionally stays conservative around boundaries.
    brightness = (centroid - 500.0) / 1800.0
    age_score = max(0.0, min(1.0, 0.65 - brightness))
    age_index = min(3, int(age_score * 4))
    age_brackets = ("18-30", "31-45", "46-60", "60+")
    distance = abs(age_score * 4 - (age_index + 0.5))
    age_confidence = round(max(0.5, min(0.85, 0.85 - distance * 0.25)) * (0.8 if quality == "degraded" else 1.0), 2)
    age = Prediction(age_brackets[age_index], age_confidence)
    if quality == "degraded":
        gender = Prediction(gender.prediction, round(max(0.5, gender.confidence * 0.8), 2))
    return AttributePrediction(gender, age)
