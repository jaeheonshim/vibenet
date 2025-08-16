from typing import Any, Protocol, Sequence, Union, BinaryIO
from dataclasses import dataclass
from os import PathLike
import numpy as np
from pydub import AudioSegment
import soxr
import librosa
from vibenet import labels,  LIKELIHOODS
from functools import lru_cache
from scipy.signal import get_window
from scipy.fft import rfft
from numpy.lib.stride_tricks import sliding_window_view

SAMPLE_RATE = 16000 # This is the sample rate used by the backend model

AudioInput = Union[
    str,
    Sequence[str],
    PathLike[Any],
    Sequence[PathLike[Any]],
    np.ndarray,
    Sequence[np.ndarray]
]

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
        
def _load_with_pydub(inp: str | PathLike[str]) -> np.ndarray:
    audio = AudioSegment.from_file(inp)
    audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    arr = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    return arr
    

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
            waveform = _load_with_pydub(item)
            out.append(waveform)
    
    return out


@lru_cache(maxsize=32)
def _cached_window(win_length: int, win_type: str = "hann") -> np.ndarray:
    return get_window(win_type, win_length, fftbins=True).astype(np.float32)


@lru_cache(maxsize=32)
def _cached_mel(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    import librosa
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