from os import PathLike
from typing import Any, BinaryIO, Sequence
import numpy as np
from numpy import ndarray
from vibenet import labels
from vibenet.core import InferenceResult, Model, create_batch, extract_mel
import onnxruntime as ort
import importlib.resources as resources

class EfficientNetModel(Model):
    def __init__(self):
        with resources.path("vibenet.artifacts", "efficientnet_model.onnx") as model_path:
            self.ort_sess = ort.InferenceSession(str(model_path))
        
    def predict(self, inputs: str | Sequence[str] | PathLike[Any] | Sequence[PathLike[Any]] | BinaryIO | Sequence[BinaryIO] | ndarray | Sequence[ndarray], sr: int | None = None) -> list[InferenceResult]:
        batch = create_batch(inputs, sr)
        print("Created batch")
        
        results = []
        
        for wf in batch:
            mel = extract_mel(wf, 16000)
            print("Extracted mel")
            outputs = self.ort_sess.run(None, {'x': mel[np.newaxis, :]})
            results.append(InferenceResult.from_logits([outputs[0][0][i].item() for i, _ in enumerate(labels)])) # type: ignore[index]
            
        return results