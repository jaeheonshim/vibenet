import os
import sys
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from vibenet.utils import load

SRC_ROOT = 'data/fma_full'
DEST_ROOT = 'data/fma_full_echonest'
NUM_WORKERS = 8
DRY_RUN = False

os.makedirs(DEST_ROOT, exist_ok=True)

df = load('data/fma_metadata/echonest.csv')
track_ids = df.index.to_list()

def src_path(track_id: int) -> str:
    sub = str(track_id // 1000).zfill(3)
    return os.path.join(SRC_ROOT, sub, f'{str(track_id).zfill(6)}.mp3')

def dest_path(track_id: int) -> str:
    sub = str(track_id // 1000).zfill(3)
    dest_dir = os.path.join(DEST_ROOT, sub)
    os.makedirs(dest_dir, exist_ok=True)
    return os.path.join(dest_dir, f'{str(track_id).zfill(6)}.mp3')

def move_one(track_id: int):
    try:
        src = src_path(track_id)
        if not os.path.exists(src):
            return ('missing', track_id)

        dst = dest_path(track_id)
        if os.path.exists(dst):
            return ('exists', track_id)

        if DRY_RUN:
            return ('would-move', track_id, src, dst)

        shutil.move(src, dst)
        return ('moved', track_id)
    except Exception as e:
        return ('error', track_id, repr(e))

results = {
    'moved': 0,
    'exists': 0,
    'missing': 0,
    'error': 0,
    'would-move': 0,
}

with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
    with tqdm(ex.map(move_one, track_ids), total=len(track_ids)) as pbar:
        for res in pbar:
            status = res[0]
            results[status] = results.get(status, 0) + 1

            if status == 'missing':
                _, tid = res
                print(f'Warning: missing file for track_id={tid}', file=sys.stderr)
            elif status == 'exists':
                _, tid = res
                pass
            elif status == 'error':
                _, tid, msg = res
                print(f'Error: track_id={tid}: {msg}', file=sys.stderr)
            elif status == 'would-move':
                _, tid, s, d = res
                print(f'DRY RUN: {tid}: {s} -> {d}', file=sys.stderr)

            pbar.set_postfix(results)

print('Summary:', results)
