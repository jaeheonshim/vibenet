from pathlib import Path

import numpy as np
import pytest
from vibenet import load_model
from vibenet.core import load_audio, SAMPLE_RATE

def _get_lossless_test_audio_file(ext: str) -> str:
	"""
	Gets the test_lossless_audio file with the specified extension.
	"""
	fixtures = Path("tests/data_fixtures/audio")
	file_path = fixtures / f"test_lossless_audio{ext}"
	if not file_path.exists():
		pytest.fail(f"Required test file not found: {file_path}")
	return str(file_path)


@pytest.mark.parametrize(
	"audio_format",
	[
		".wav",
		".m4a",
		".flac",
	],
)
def test_lossless_audio_loading_produces_valid_waveform_data(audio_format):
	"""Test that audio loading produces valid waveform data for supported formats."""
	path = _get_lossless_test_audio_file(audio_format)
	y = load_audio(path, target_sr=SAMPLE_RATE)
	assert y.ndim in (1, 2)
	assert y.dtype == np.float32
	assert np.isfinite(y).all()
	assert y.size > 0
	# Not a constant signal
	assert np.std(y) > 0
	# Reasonable amplitude after scaling
	assert float(np.max(np.abs(y))) <= 1.2


def test_predict_is_deterministic_per_lossless_file():
	"""
	Asserts the model prediction on the .flac file is identical to the prediction on other lossless formats.
	Predictions will be similar but not identical for non-lossless formats.
	"""
	model = load_model()
	flac_path = _get_lossless_test_audio_file(".flac")
	flac_prediction = model.predict(flac_path)[0].to_dict()
	for ext in [".wav", ".m4a"]:
		path = _get_lossless_test_audio_file(ext)
		other_prediction = model.predict(path)[0].to_dict()
		for k in flac_prediction:
			assert abs(flac_prediction[k] - other_prediction[k]) == 0, f"Prediction mismatch for {ext} file at key '{k}': flac={flac_prediction[k]}, {ext}={other_prediction[k]}"
