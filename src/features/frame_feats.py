from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

@dataclass
class FrameFeatureSummary:
    n_frames: int
    mean_brightness: float
    std_brightness: float
    mean_edge_density: float

def _to_gray_np(img: Image.Image, max_size: int = 224) -> np.ndarray:
    img = img.convert("L")
    w, h = img.size
    scale = min(max_size / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def _edge_density(gray: np.ndarray) -> float:
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    return float(0.5 * (gx.mean() + gy.mean()))

def summarize_first_frames(frame_paths: List[Path]) -> FrameFeatureSummary:
    if not frame_paths:
        return FrameFeatureSummary(
            n_frames=0,
            mean_brightness=0.0,
            std_brightness=0.0,
            mean_edge_density=0.0,
        )

    brightness_means = []
    brightness_stds = []
    edges = []

    for p in frame_paths:
        img = Image.open(p)
        gray = _to_gray_np(img)
        brightness_means.append(float(gray.mean()))
        brightness_stds.append(float(gray.std()))
        edges.append(_edge_density(gray))

    return FrameFeatureSummary(
        n_frames=len(frame_paths),
        mean_brightness=float(np.mean(brightness_means)),
        std_brightness=float(np.mean(brightness_stds)),
        mean_edge_density=float(np.mean(edges)),
    )
