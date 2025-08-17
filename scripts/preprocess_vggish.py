import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
from torchvggish import vggish_input
from tqdm import tqdm

from vibenet.utils import load

OUTPUT_DIR = 'data/preprocessed/vggish'
MAX_WIDTH = 500_000 # 31.25 sec
CHUNK_SIZE = 500
NUM_WORKERS = 12

os.makedirs(f'{OUTPUT_DIR}/data', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels', exist_ok=True)

df = load('data/fma_metadata/echonest.csv')
track_ids = df.index.to_list()
audio_features = df['echonest', 'audio_features']
feature_cols = [c for c in audio_features.columns if c != 'tempo']
labels_df = audio_features[feature_cols]

def process_track(track_id):
    try:
        path = f'data/fma_large/{str(track_id // 1000).zfill(3)}/{str(track_id).zfill(6)}.mp3'
        if not os.path.exists(path):
            return None

        y, sr = librosa.load(path, sr=16000)
        y = y[:MAX_WIDTH]
        y = np.pad(y, (0, MAX_WIDTH - len(y)))
        y = vggish_input.waveform_to_examples(y, sample_rate=16000, return_tensor=False)

        label = labels_df.loc[track_id].to_numpy(dtype=float)

        return y, label
    except:
        return None

batch_data = []
batch_labels = []
batch_index = 0
processed = 0

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    for z in tqdm(executor.map(process_track, track_ids), total=len(track_ids)):
        if z is None:
            print(f'Warning: Track Skipped')
            continue
        
        mel, label = z

        batch_data.append(mel)
        batch_labels.append(label)
        processed += 1

        if processed % CHUNK_SIZE == 0:
            np.save(f'{OUTPUT_DIR}/data/{batch_index}.npy', np.array(batch_data, dtype=np.float32))
            np.save(f'{OUTPUT_DIR}/labels/{batch_index}.npy', np.array(batch_labels, dtype=np.float32))
            batch_data = []
            batch_labels = []
            batch_index += 1

if batch_data:
    np.save(f'{OUTPUT_DIR}/data/{batch_index}.npy', np.array(batch_data, dtype=np.float32))
    np.save(f'{OUTPUT_DIR}/labels/{batch_index}.npy', np.array(batch_labels, dtype=np.float32))