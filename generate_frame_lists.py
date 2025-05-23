#!/usr/bin/env python3
# generate_frame_lists.py
"""
Create history_maps/<basename>.frames.txt for every image that has N_HISTORY
valid predecessors.

Unchanged input layout:
    /data/TGSSE/droneVbirds/splits/{train,val,test}/images/*.jpg
Output layout:
    /data/TGSSE/droneVbirds/splits/{train,val,test}/history_maps/*.frames.txt
"""
import re
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────
ROOT       = Path('/data/TGSSE/droneVbirds/splits')
SPLITS     = ('train', 'val', 'test')
N_HISTORY  = 5
EXTS       = {'.jpg', '.jpeg', '.png', '.bmp'}
PATTERN    = re.compile(r'^(.+?)_(\d{6})$')   # prefix + 6-digit index
# ───────────────────────────────────────────────────────────────────────────────

def build_lists(split: str):
    split_dir   = ROOT / split
    img_dir     = split_dir / 'images'
    hist_dir    = split_dir / 'history_maps'
    for img in img_dir.rglob('*'):
        if img.suffix.lower() not in EXTS:
            continue
        m = PATTERN.match(img.stem)
        if not m:
            continue
        prefix, idx_str = m.groups()
        idx = int(idx_str)
        paths = [img_dir / f'{prefix}_{idx - i:06d}{img.suffix}'
                 for i in range(N_HISTORY, -1, -1)]
        if all(p.exists() for p in paths):
            rel_subdir = img.relative_to(img_dir).parent
            out_dir = hist_dir / rel_subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f'{img.stem}.frames.txt'
            out_file.write_text('\n'.join(str(p.resolve()) for p in paths))

def main():
    for split in SPLITS:
        build_lists(split)
    print('✓  history_maps created for train / val / test')

if __name__ == '__main__':
    main()
