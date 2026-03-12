from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}



def _normalize_image(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
        if arr.max(initial=0.0) > 1.0:
            arr /= max(float(arr.max(initial=1.0)), 1.0)
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = arr.astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)



def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    arr = np.asarray(iio.imread(path))
    return _normalize_image(arr)



def save_image(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    image = np.clip(image.astype(np.float32), 0.0, 1.0)

    if suffix in {".jpg", ".jpeg"}:
        Image.fromarray((image * 255.0 + 0.5).astype(np.uint8)).save(
            path, quality=int(quality), subsampling=0, optimize=True
        )
        return
    if suffix == ".png":
        Image.fromarray((image * 255.0 + 0.5).astype(np.uint8)).save(path)
        return
    if suffix in {".tif", ".tiff"}:
        iio.imwrite(path, (image * 65535.0 + 0.5).astype(np.uint16))
        return
    raise ValueError(f"Unsupported output format: {suffix}")
