import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
from torchvggish import vggish_input
from tqdm import tqdm

from vibenet.utils import load

OUTPUT_DIR = 'data/preprocessed/waveforms_val'
MAX_WIDTH = 32000 * 31
OVERLAP = MAX_WIDTH // 2
CHUNK_SIZE = 1000
NUM_WORKERS = 8

os.makedirs(f'{OUTPUT_DIR}/data', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels', exist_ok=True)

df = pd.read_csv('test.csv', index_col=0, header=[0, 1, 2])
track_ids = df.index.to_list()
audio_features = df['echonest', 'audio_features']
feature_cols = [c for c in audio_features.columns if c != 'tempo']
labels_df = audio_features[feature_cols]

def process_track(track_id):
    try:
        path = f'data/fma_full_echonest/{str(track_id // 1000).zfill(3)}/{str(track_id).zfill(6)}.mp3'
        if not os.path.exists(path):
            return None

        y, _ = librosa.load(path, sr=32000)
        chunks = []
        for i in range(0, len(y), MAX_WIDTH - OVERLAP):
            chunk = y[i : i + MAX_WIDTH]
            if len(chunk) == MAX_WIDTH:
                chunks.append(chunk)
        label = labels_df.loc[track_id].to_numpy(dtype=float)

        return chunks, label
    except:
        return None

batch_data = []
batch_labels = []
batch_index = 0
processed = 0
sample_count = 0

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    with tqdm(executor.map(process_track, track_ids), total=len(track_ids)) as pbar:
        for z in pbar:
            if z is None:
                print('Warning: Track Skipped', file=sys.stderr)
                continue
            
            waveforms, label = z

            for w in waveforms:
                batch_data.append(w)
                batch_labels.append(label)
                processed += 1
                if processed % CHUNK_SIZE == 0:
                    np.save(f'{OUTPUT_DIR}/data/{batch_index}.npy', np.array(batch_data, dtype=np.float32))
                    np.save(f'{OUTPUT_DIR}/labels/{batch_index}.npy', np.array(batch_labels, dtype=np.float32))
                    batch_data = []
                    batch_labels = []
                    batch_index += 1

            sample_count += len(waveforms)

            pbar.set_postfix({'num_samples': sample_count})

if batch_data:
    np.save(f'{OUTPUT_DIR}/data/{batch_index}.npy', np.array(batch_data, dtype=np.float32))
    np.save(f'{OUTPUT_DIR}/labels/{batch_index}.npy', np.array(batch_labels, dtype=np.float32))