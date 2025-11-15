"""
Utility script for downloading College Scorecard CSVs on-demand so the
repository stays small and compliant with GitHub's 100 MB file limit.

Usage examples:
    python scripts/download_scorecard_data.py                      # baseline (institution file)
    python scripts/download_scorecard_data.py --bundle baseline field_of_study
    python scripts/download_scorecard_data.py --bundle full --overwrite
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, Iterable, List
from urllib import error, request
import shutil

BASE_URL = "https://ed-public-download.app.cloud.gov/downloads"
DEFAULT_DEST = Path("data/raw/scorecard/public")


def build_merged_files(start: int = 1996, end: int = 2023) -> List[str]:
    files: List[str] = []
    for year in range(start, end + 1):
        suffix = str(year + 1)[-2:]
        files.append(f"CollegeScorecard_Raw_Data/MERGED{year}_{suffix}_PP.csv")
    return files


FIELD_OF_STUDY_ARCHIVE = [
    "CollegeScorecard_Raw_Data/FieldOfStudyData1415_1516_PP.csv",
    "CollegeScorecard_Raw_Data/FieldOfStudyData1516_1617_PP.csv",
    "CollegeScorecard_Raw_Data/FieldOfStudyData1617_1718_PP.csv",
    "CollegeScorecard_Raw_Data/FieldOfStudyData1718_1819_PP.csv",
    "CollegeScorecard_Raw_Data/FieldOfStudyData1819_1920_PP.csv",
    "CollegeScorecard_Raw_Data/FieldOfStudyData1920_2021_PP.csv",
]

CROSSWALK_FILES = [
    *(f"CollegeScorecard_Raw_Data/Crosswalks/Crosswalks/CW{year}.xlsx" for year in range(2000, 2022)),
    "CollegeScorecard_Raw_Data/Crosswalks/Crosswalks/CW2022_preliminary.xlsx",
    "CollegeScorecard_Raw_Data/Crosswalks/Crosswalks/CW2023_prelim.xlsx",
]

BUNDLES: Dict[str, List[str]] = {
    "baseline": [
        "Most-Recent-Cohorts-Institution.csv",
    ],
    "field_of_study": [
        "Most-Recent-Cohorts-Field-of-Study.csv",
        *FIELD_OF_STUDY_ARCHIVE,
    ],
    "merged": build_merged_files(),
    "crosswalks": list(CROSSWALK_FILES),
}

BUNDLES["full"] = list(
    dict.fromkeys(
        itertools.chain.from_iterable(
            BUNDLES[key] for key in ("baseline", "field_of_study", "merged", "crosswalks")
        )
    )
)


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def build_url(rel_path: str) -> str:
    rel_path = rel_path.lstrip("/")
    return f"{BASE_URL}/{rel_path}"


def prepare_file_list(selected_bundles: Iterable[str], extras: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    bundles = list(selected_bundles) or ["baseline"]
    for bundle in bundles:
        ordered.extend(BUNDLES[bundle])
    ordered.extend(extras or [])
    # Preserve order while removing duplicates
    return list(dict.fromkeys(ordered))


def download_file(url: str, destination: Path, overwrite: bool) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"[skip] {destination} already exists (use --overwrite to refresh)")
        return True

    print(f"[fetch] {url}")
    req = request.Request(url, headers={"User-Agent": "student-loan-default-risk-lab/1.0"})
    try:
        with request.urlopen(req, timeout=120) as response, open(destination, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except error.HTTPError as exc:
        print(f"[error] HTTP {exc.code} for {url}")
        return False
    except error.URLError as exc:
        print(f"[error] Network error for {url}: {exc.reason}")
        return False

    size = destination.stat().st_size
    print(f"[done] {destination} ({human_bytes(size)})")
    return True


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download College Scorecard files without committing them to Git."
    )
    parser.add_argument(
        "--bundle",
        "-b",
        choices=sorted(BUNDLES.keys()),
        action="append",
        help="Named bundle(s) to download (default: baseline). You may pass multiple.",
    )
    parser.add_argument(
        "--extra",
        nargs="+",
        default=[],
        metavar="PATH",
        help="Additional file paths relative to DOE's download root.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination root (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    files_to_fetch = prepare_file_list(args.bundle or ["baseline"], args.extra)
    if not files_to_fetch:
        print("No files requested; nothing to do.")
        return 0

    print(
        f"Downloading {len(files_to_fetch)} file(s) to {args.dest} "
        f"(bundles: {', '.join(args.bundle or ['baseline'])})"
    )
    failures = 0
    for rel_path in files_to_fetch:
        url = build_url(rel_path)
        destination = (args.dest / Path(rel_path)).resolve()
        success = download_file(url, destination, overwrite=args.overwrite)
        if not success:
            failures += 1

    if failures:
        print(f"Completed with {failures} failure(s). See log above.")
        return 1

    print("All downloads completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

