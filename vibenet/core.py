import contextlib
import os
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from os import PathLike
from typing import Any, BinaryIO, Protocol, Sequence, Union

import audioread
import librosa
import numpy as np
import soundfile as sf
import soxr
from numpy.lib.stride_tricks import sliding_window_view
from pydub import AudioSegment
from scipy.fft import rfft
from scipy.signal import get_window

from vibenet import LIKELIHOODS, labels

SAMPLE_RATE = 16000 # This is the sample rate used by the backend model

AudioInput = Union[
    str,
    Sequence[str],
    PathLike[Any],
    Sequence[PathLike[Any]],
    np.ndarray,
    Sequence[np.ndarray]
]

@contextlib.contextmanager
def _suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


@dataclass
class InferenceResult:
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    speechiness: float
    valence: float
    
    @classmethod
    def from_logits(cls, logits: Sequence[float]) -> "InferenceResult":
        values = {}
        
        for i,lbl in enumerate(labels):
            if lbl in LIKELIHOODS:
                values[lbl] = float(1/(1+np.exp(-logits[i])))
            else:
                values[lbl] = float(np.clip(logits[i], 0, 1))
                
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class Model(Protocol):
    def predict(
        self,
        inputs: AudioInput,
        sr: int | None = None
    ) -> list[InferenceResult]:
        """Run feature inference on audio

        Args:
            inputs: One or many audio sources
                - str / PathLike: file path
                - BinaryIO: open file handle
                - np.ndarray: raw waveform
            sr: Required if `inputs` are raw waveforms
        """
        ...
        

def load_audio(path, target_sr=16000):
    try:
        with _suppress_output():
            y, sr = sf.read(path, always_2d=False)
    except Exception:
        with _suppress_output():
            y, sr = librosa.load(path, sr=target_sr, mono=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = soxr.resample(y, sr, target_sr)
    return y
    

def create_batch(inputs, sr: int | None):
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
        
    out = []
    
    for item in inputs:
        if isinstance(item, np.ndarray):
            if sr is None:
                raise ValueError("When passing raw waveforms, their sample rate (sr) must be provided.")
            if sr != SAMPLE_RATE:
                item = soxr.resample(item, sr, SAMPLE_RATE)
                
            out.append(item)
        else:
            waveform = load_audio(item)
            out.append(waveform)
    
    return out


@lru_cache(maxsize=32)
def _cached_window(win_length: int, win_type: str = "hann") -> np.ndarray:
    return get_window(win_type, win_length, fftbins=True).astype(np.float32)


@lru_cache(maxsize=32)
def _cached_mel(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).astype(np.float32)


def _power_to_db_fast(S: np.ndarray, top_db: float = 80.0, amin: float = 1e-10, ref: float = 1.0) -> np.ndarray:
    S_db = 10.0 * np.log10(np.maximum(amin, S)) - 10.0 * np.log10(np.maximum(amin, ref))
    if top_db is not None:
        S_db = np.maximum(S_db, S_db.max() - top_db)
    return S_db.astype(np.float32)


def extract_mel(
    y: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int = 640,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = 8000.0,
    window_type: str = "hann",
    center: bool = False,
) -> np.ndarray:
    if center:
        pad = (win_length // 2)
        y = np.pad(y, (pad, pad), mode="reflect")

    frames = sliding_window_view(y, win_length)[::hop_length]
    window = _cached_window(win_length, window_type)
    frames = frames * window[None, :]

    spec = rfft(frames, n=n_fft, axis=-1, workers=-1)
    S_pow = (spec.real**2 + spec.imag**2).astype(np.float32)

    mel_fb = _cached_mel(sr, n_fft, n_mels, fmin, fmax)
    mel = mel_fb @ S_pow.T
    mel = mel.astype(np.float32)

    mel_db = _power_to_db_fast(mel, top_db=80.0, ref=1.0)
    return mel_db