import subprocess
from pathlib import Path

def extract_wav(video_path: Path, out_dir: Path, sr: int = 16000) -> Path:
    """
    Extract mono 16k wav for Whisper.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / (video_path.stem + ".wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "wav",
        str(wav_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path
