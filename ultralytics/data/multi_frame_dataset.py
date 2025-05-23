from pathlib import Path
from typing import List
import cv2
import numpy as np
from .dataset import YOLODataset

class YOLOMultiFrameDataset(YOLODataset):
    """Dataset that loads stacks of frames listed in *.frames.txt files."""

    def __init__(self, *args, n_history=5, history_dir="history_maps", **kwargs):
        self.n_history = n_history
        self.history_dir = history_dir
        super().__init__(*args, **kwargs)
        # Limit stored frame paths to n_history previous frames plus the target frame
        self.frame_lists = [
            self._frames_for(im)[-self.n_history - 1 :]
            for im in self.im_files
        ]

    def _frames_for(self, im_file: str) -> List[str]:
        p = Path(im_file)
        if "images" in p.parts:
            idx = p.parts.index("images")
            hist = Path(*p.parts[:idx], self.history_dir, *p.parts[idx+1:]).with_suffix(".frames.txt")
        else:
            hist = p.with_suffix(".frames.txt")
        if hist.exists():
            return hist.read_text().strip().splitlines()
        return [im_file]

    def load_image(self, i, rect_mode=True):
        paths = self.frame_lists[i]
        imgs = [cv2.imread(p) for p in paths]
        imgs = [img for img in imgs if img is not None]
        if not imgs:
            raise FileNotFoundError(f"Frames not found for index {i}")
        im, (h0, w0), (h, w) = super().load_image(i, rect_mode)
        resized = [cv2.resize(img, (w, h)) for img in imgs]
        stack = np.stack(resized, axis=0)  # (T,H,W,3)
        return stack, (h0, w0), (h, w)
