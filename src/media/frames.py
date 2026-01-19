import subprocess
from pathlib import Path

def extract_first_n_frames(video_path: Path, out_dir: Path, n: int = 10, fps: int = 1) -> list[Path]:
    """
    Extract up to the first n frames starting from time 0.
    Uses ffmpeg to sample frames at fps and stops after n frames.
    Outputs PNG frames: frame_0001.png, frame_0002.png, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean old frames if rerun
    for p in out_dir.glob("frame_*.png"):
        p.unlink()

    out_pattern = out_dir / "frame_%04d.png"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", "0",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-frames:v", str(n),
        str(out_pattern),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = sorted(out_dir.glob("frame_*.png"))
    return frames[:n]
