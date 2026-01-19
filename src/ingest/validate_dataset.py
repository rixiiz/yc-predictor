import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.config import DATASET_CSV

REQUIRED_COLS = {"youtube_id", "label", "year"}

@dataclass
class Row:
    youtube_id: str
    label: int
    year: int

def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")
        missing = REQUIRED_COLS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {reader.fieldnames}")

        for i, r in enumerate(reader, start=2):
            yt = (r.get("youtube_id") or "").strip()
            if not yt:
                continue
            try:
                label = int(str(r.get("label") or "").strip())
                year = int(str(r.get("year") or "").strip())
            except Exception:
                raise ValueError(f"Bad label/year at line {i}: {r}")

            if label not in (0, 1):
                raise ValueError(f"Label must be 0/1 at line {i}: {label}")

            rows.append(Row(yt, label, year))
    return rows

def main():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Expected dataset at {DATASET_CSV}")

    rows = read_rows(DATASET_CSV)
    print(f"Loaded rows: {len(rows)}")

    # Dedupe by youtube_id (keep first occurrence)
    seen = set()
    deduped = []
    dupes = 0
    for r in rows:
        if r.youtube_id in seen:
            dupes += 1
            continue
        seen.add(r.youtube_id)
        deduped.append(r)

    if dupes:
        print(f"Removed duplicates by youtube_id: {dupes}")
    rows = deduped

    label_counts = Counter(r.label for r in rows)
    print(f"Label counts: {dict(label_counts)}")

    year_counts = Counter(r.year for r in rows)
    print(f"Year counts: {dict(sorted(year_counts.items()))}")

    # Year x label table (small print)
    year_label = defaultdict(Counter)
    for r in rows:
        year_label[r.year][r.label] += 1

    print("\nCounts by year:")
    for y in sorted(year_label):
        c = year_label[y]
        print(f"  {y}: total={sum(c.values())}  accepted(1)={c[1]}  rejected(0)={c[0]}")

    # Quick baseline accuracy if always predicting majority
    majority = 1 if label_counts[1] >= label_counts[0] else 0
    baseline_acc = label_counts[majority] / max(1, len(rows))
    print(f"\nMajority-class baseline accuracy: {baseline_acc:.3f} (predict always {majority})")

    print("\nDataset looks OK ✅")

if __name__ == "__main__":
    main()
