import subprocess
from pathlib import Path

def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"

def download_youtube(video_id: str, out_dir: Path) -> Path:
    """
    Downloads best available format using yt-dlp.
    Returns the path to the downloaded file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = out_dir / f"{video_id}.%(ext)s"

    cmd = [
        "yt-dlp",
        "-f", "bv*+ba/best",
        "-o", str(out_template),
        youtube_url(video_id),
    ]
    subprocess.check_call(cmd)

    # Find actual file
    matches = list(out_dir.glob(f"{video_id}.*"))
    if not matches:
        raise FileNotFoundError(f"Download succeeded but file not found for {video_id}")
    # pick the newest
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]
