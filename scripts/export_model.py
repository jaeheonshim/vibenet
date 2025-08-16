# Script for exporting model to ONNX

import torch
from vibenet.models.student import EfficientNetRegressor
from vibenet import labels

model = EfficientNetRegressor()
checkpoint = torch.load("checkpoints/efficientnet_best.pt")
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval()

input_tensor = torch.rand((1, 128, 1024)) # [B, n_mels, T]

torch.onnx.export(
    model,
    (input_tensor,),
    "model.onnx",
    input_names=["x"],
    output_names=["out"],
    dynamic_axes={
        "x": {0: "batch", 2: "time"},
        "out": {0: "batch"}
    },
    opset_version=20,
    export_params=True,
    external_data=False,
    dynamo=True
)