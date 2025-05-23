from pathlib import Path
from typing import List
import cv2
import numpy as np
from .dataset import YOLODataset

class YOLOMultiFrameDataset(YOLODataset):
    """Dataset that loads a stack of sequential frames for each sample.

    The dataset expects a ``.frames.txt`` file alongside every image (or in a
    ``history_dir`` folder) that lists the neighbouring frame paths. Each file is
    newline separated and should contain ``n_history`` previous frames followed by
    the current image path::

        /dataset/images/img_000123.jpg
        /dataset/images/img_000124.jpg
        ...
        /dataset/images/img_000128.jpg

    ``n_history`` defines how many historical frames are described in the text
    files. When present, ``load_image`` reads all paths, resizes them to the
    target dimensions and returns a NumPy array of shape ``(T, H, W, C)`` where
    ``T`` equals ``n_history + 1`` (the stack depth). If no ``.frames.txt`` is
    found, only the current frame is loaded.

    Example:

        >>> from ultralytics.data import YOLOMultiFrameDataset
        >>> data = {
        ...     'train': '/path/to/train/images',
        ...     'nc': 1,
        ...     'names': {0: 'object'},
        ...     'channels': 3,
        ... }
        >>> dset = YOLOMultiFrameDataset(img_path=data['train'], data=data, n_history=5)
        >>> imgs, _, _ = dset.load_image(0)
        >>> print(imgs.shape)  # (6, H, W, 3)
    """

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
