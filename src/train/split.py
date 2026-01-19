import csv
from pathlib import Path

def choose_year_holdout(transcripts_csv: Path) -> int:
    years = set()
    with transcripts_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            years.add(int(r["year"]))
    if not years:
        raise ValueError("No years found.")
    return max(years)
