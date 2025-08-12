import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from torchvggish import vggish_input
from concurrent.futures import ProcessPoolExecutor, as_completed

from vibenet.utils import load
from vibenet.models.teacher import PANNsMLP

CSV_FILE = 'data/distillation/test.csv'
OUTPUT_DIR = 'data/preprocessed/waveforms_distill_test'
MAX_WIDTH = 32000 * 31
OVERLAP = MAX_WIDTH // 2
CHUNK_SIZE = 1000
BATCH_SIZE = 250
NUM_WORKERS = 8

os.makedirs(f'{OUTPUT_DIR}/data', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/labels', exist_ok=True)

device = 'cuda'

model = PANNsMLP()
model = model.to(device)
data = torch.load('checkpoints/pretrained_PANN.pt')
model.load_state_dict(data['state_dict'])

def eval(batch):
    model.eval()

    with torch.inference_mode():
        x = torch.tensor(batch).to(device).float()  # [B, T]
        pred = model(x)

    pred['acousticness'] = F.sigmoid(pred['acousticness'])
    pred['instrumentalness'] = F.sigmoid(pred['instrumentalness'])
    pred['liveness'] = F.sigmoid(pred['liveness'])
    pred['speechiness'] = torch.clamp(pred['speechiness'], 0, 1)
    pred['danceability'] = torch.clamp(pred['danceability'], 0, 1)
    pred['energy'] = torch.clamp(pred['energy'], 0, 1)
    pred['valence'] = torch.clamp(pred['valence'], 0, 1)

    return pred

df = pd.read_csv(CSV_FILE, index_col=0)
track_ids = df.index.to_list()

def get_32khz_waveform(track_id):
    try:
        path = f'data/fma_large/{str(track_id // 1000).zfill(3)}/{str(track_id).zfill(6)}.mp3'
        if not os.path.exists(path):
            print(f'Warning: Track {track_id} not found', file=sys.stderr)
            return None

        y, _ = librosa.load(path, sr=32000)
        
        if len(y) < MAX_WIDTH:
            y = np.pad(y, (0, MAX_WIDTH - len(y)))
        else:
            y = y[:MAX_WIDTH]

        return y
    except Exception as e:
        print(f'Error processing track {track_id}: {e}', file=sys.stderr)
        return None

batch_data = []
batch_labels = []
batch_index = 0

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    with tqdm(range(0, len(track_ids), BATCH_SIZE)) as pbar:
        for i in pbar:
            waveforms_32khz = []
            waveforms_16khz = []

            pbar.set_description("Loading waveforms")
            results = list(executor.map(get_32khz_waveform, track_ids[i:i + BATCH_SIZE]))
            for z in results:
                if z is None:
                    print('Warning: Track Skipped', file=sys.stderr)
                    continue

                waveforms_32khz.append(z)
                waveforms_16khz.append(librosa.resample(z, orig_sr=32000, target_sr=16000))

                pbar.set_postfix({'loaded_songs': len(waveforms_32khz)})

            pbar.set_description("Creating tensor")
            x = np.array(waveforms_32khz)

            pbar.set_description("Running inference")
            pred = eval(x)

            labels = np.concatenate(
                [
                    pred['acousticness'].cpu().numpy()[:, np.newaxis],
                    pred['danceability'].cpu().numpy()[:, np.newaxis],
                    pred['energy'].cpu().numpy()[:, np.newaxis],
                    pred['instrumentalness'].cpu().numpy()[:, np.newaxis],
                    pred['liveness'].cpu().numpy()[:, np.newaxis],
                    pred['speechiness'].cpu().numpy()[:, np.newaxis],
                    pred['valence'].cpu().numpy()[:, np.newaxis]
                ],
                axis=1
            )

            batch_data.extend(waveforms_16khz)
            batch_labels.extend(labels)

            if len(batch_data) >= CHUNK_SIZE:
                np.save(f'{OUTPUT_DIR}/data/{batch_index}.npy', np.array(batch_data, dtype=np.float32))
                np.save(f'{OUTPUT_DIR}/labels/{batch_index}.npy', np.array(batch_labels, dtype=np.float32))
                batch_data = []
                batch_labels = []
                batch_index += 1

    if batch_data:
        np.save(f'{OUTPUT_DIR}/data/{batch_index}.npy', np.array(batch_data, dtype=np.float32))
        np.save(f'{OUTPUT_DIR}/labels/{batch_index}.npy', np.array(batch_labels, dtype=np.float32))