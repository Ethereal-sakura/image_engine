from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image



def create_demo_input(path: Path) -> None:
    h, w = 720, 1080
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = xx / (w - 1)
    y = yy / (h - 1)

    base = np.stack(
        [
            0.15 + 0.85 * x,
            0.15 + 0.75 * (1.0 - y),
            0.25 + 0.65 * (1.0 - 0.7 * x) * (0.7 + 0.3 * y),
        ],
        axis=-1,
    )

    # Simple synthetic scene: sky gradient + warm highlight block + green block.
    base[:280, :, 2] += 0.15
    base[:280, :, 1] += 0.05
    base[220:410, 300:530, 0] = 0.98
    base[220:410, 300:530, 1] = 0.62
    base[220:410, 300:530, 2] = 0.38
    base[350:630, 680:930, 0] = 0.18
    base[350:630, 680:930, 1] = 0.72
    base[350:630, 680:930, 2] = 0.34

    vignette = ((x - 0.5) ** 2 + (y - 0.5) ** 2) * 0.35
    image = np.clip(base - vignette[..., None], 0.0, 1.0)
    Image.fromarray((image * 255.0).astype(np.uint8)).save(path)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "demo_input.png"
    create_demo_input(out)
    print(f"Created {out}")
