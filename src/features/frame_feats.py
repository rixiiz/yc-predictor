# src/features/frame_feats.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


@dataclass
class FrameSummary:
    n_frames: int
    mean_brightness: float
    std_brightness: float
    mean_edge_density: float
    mean_motion_energy: float  # NEW


def _read_gray(p: Path) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read frame: {p}")
    return img


def summarize_first_frames(frame_paths: List[Path]) -> FrameSummary:
    """
    Computes lightweight visual features from extracted frames:
    - brightness mean/std
    - edge density proxy (Canny edges fraction)
    - motion energy proxy (mean abs diff between consecutive frames)
    """
    if not frame_paths:
        return FrameSummary(
            n_frames=0,
            mean_brightness=0.0,
            std_brightness=0.0,
            mean_edge_density=0.0,
            mean_motion_energy=0.0,
        )

    frames = []
    for p in frame_paths:
        try:
            frames.append(_read_gray(p))
        except FileNotFoundError:
            continue

    if not frames:
        return FrameSummary(
            n_frames=0,
            mean_brightness=0.0,
            std_brightness=0.0,
            mean_edge_density=0.0,
            mean_motion_energy=0.0,
        )

    # brightness stats
    all_pixels = np.concatenate([f.reshape(-1) for f in frames]).astype(np.float32)
    mean_brightness = float(all_pixels.mean())
    std_brightness = float(all_pixels.std())

    # edge density
    edge_fracs = []
    for f in frames:
        edges = cv2.Canny(f, 100, 200)
        edge_fracs.append(float((edges > 0).mean()))
    mean_edge_density = float(np.mean(edge_fracs)) if edge_fracs else 0.0

    # motion energy (mean abs diff between consecutive frames)
    motion_vals = []
    for a, b in zip(frames[:-1], frames[1:]):
        diff = cv2.absdiff(a, b).astype(np.float32)
        motion_vals.append(float(diff.mean()))
    mean_motion_energy = float(np.mean(motion_vals)) if motion_vals else 0.0

    return FrameSummary(
        n_frames=len(frames),
        mean_brightness=mean_brightness,
        std_brightness=std_brightness,
        mean_edge_density=mean_edge_density,
        mean_motion_energy=mean_motion_energy,
    )