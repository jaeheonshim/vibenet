import importlib.resources as resources
from os import PathLike
from typing import Any, BinaryIO, Sequence

import numpy as np
import onnxruntime as ort
from numpy import ndarray

from vibenet import labels
from vibenet.core import InferenceResult, Model, create_batch, extract_mel


class EfficientNetModel(Model):
    def __init__(self):
        with resources.path("vibenet.artifacts", "efficientnet_model.onnx") as model_path:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
                    
            self.ort_sess = ort.InferenceSession(str(model_path))
            
    def predict(
        self,
        inputs: str | Sequence[str] | PathLike[Any] | Sequence[PathLike[Any]] | BinaryIO | Sequence[BinaryIO] | ndarray | Sequence[ndarray],
        sr: int | None = None,
    ) -> list[InferenceResult]:
        batch = create_batch(inputs, sr=sr)
        
        results = []
        
        for wf in batch:
            mel = extract_mel(wf, 16000)
            outputs = self.ort_sess.run(None, {'x': mel[np.newaxis, :]})
            results.append(InferenceResult.from_logits([outputs[0][0][i].item() for i, _ in enumerate(labels)])) # type: ignore[index]
        return results