from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from src.utils import DATA_RAW, ensure_project_dirs, print_section


OFFICIAL_DATASET_PAGE = "https://www.unb.ca/cic/datasets/ids-2017.html"
OFFICIAL_DOWNLOAD_FORM = "https://cicresearch.ca/CICDataset/CIC-IDS-2017/"
OFFICIAL_CSV_ARCHIVES = {
    "MachineLearningCSV.zip": "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip",
    "GeneratedLabelledFlows.zip": "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/GeneratedLabelledFlows.zip",
}


def _looks_like_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 4:
        return False
    with path.open("rb") as handle:
        return handle.read(4).startswith(b"PK")


def extract_archives(raw_dir: Path = DATA_RAW) -> list[Path]:
    extracted: list[Path] = []
    for archive in sorted(raw_dir.rglob("*.zip")):
        if not _looks_like_zip(archive):
            continue
        target_dir = raw_dir / archive.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
        extracted.append(target_dir)
    return extracted


def attempt_official_download(raw_dir: Path = DATA_RAW, archive_name: str = "MachineLearningCSV.zip") -> Path:
    ensure_project_dirs()
    raw_dir.mkdir(parents=True, exist_ok=True)
    if archive_name not in OFFICIAL_CSV_ARCHIVES:
        raise ValueError(f"Unsupported archive {archive_name!r}. Choose one of {sorted(OFFICIAL_CSV_ARCHIVES)}")

    target = raw_dir / archive_name
    url = OFFICIAL_CSV_ARCHIVES[archive_name]
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 CICIDS2017 academic downloader"})
    try:
        with urlopen(request, timeout=60) as response:
            content_type = response.headers.get("Content-Type", "")
            first = response.read(4)
            remainder = response.read()
    except URLError as exc:
        raise RuntimeError(f"Unable to reach official CIC/UNB download host: {exc}") from exc

    target.write_bytes(first + remainder)
    if not first.startswith(b"PK"):
        target.unlink(missing_ok=True)
        raise RuntimeError(
            "The official CIC/UNB download endpoint did not return a ZIP archive. "
            f"It returned Content-Type={content_type!r}. Complete the official CIC form, "
            f"then place MachineLearningCSV.zip or extracted CSV files in {raw_dir}."
        )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/extract official CICIDS2017 CSV archives when available.")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--archive", choices=sorted(OFFICIAL_CSV_ARCHIVES), default="MachineLearningCSV.zip")
    parser.add_argument("--extract-only", action="store_true", help="Only extract ZIP files already present in data/raw.")
    args = parser.parse_args()

    print_section("CICIDS2017 Official Dataset Acquisition")
    print(f"Official dataset page: {OFFICIAL_DATASET_PAGE}")
    print(f"Official CIC download form: {OFFICIAL_DOWNLOAD_FORM}")
    print(f"Raw data directory: {args.raw_dir}")

    if not args.extract_only:
        try:
            archive = attempt_official_download(args.raw_dir, args.archive)
            print(f"Downloaded official archive: {archive}")
        except RuntimeError as exc:
            print(f"Automatic download unavailable: {exc}")
            print_required_files(args.raw_dir)
            return 2

    extracted = extract_archives(args.raw_dir)
    if extracted:
        print("Extracted archives:")
        for path in extracted:
            print(f"  - {path}")
    else:
        print("No ZIP archives extracted.")
    return 0


def print_required_files(raw_dir: Path = DATA_RAW) -> None:
    print("\nRequired official CICIDS2017 files:")
    print(f"  1. Preferred: MachineLearningCSV.zip from {OFFICIAL_DATASET_PAGE}")
    print("     Extract it, or place the ZIP directly here:")
    print(f"     {raw_dir}")
    print("  2. Acceptable alternative: GeneratedLabelledFlows.zip from the same official CIC/UNB source.")
    print("  3. The training pipeline needs the CSV files inside those archives, not PCAP-only data.")


if __name__ == "__main__":
    raise SystemExit(main())
