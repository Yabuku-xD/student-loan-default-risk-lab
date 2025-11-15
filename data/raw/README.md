# Raw Data Downloads

This directory is intentionally empty in Git because the College Scorecard
exports weigh tens of megabytes per file and easily exceed GitHub's 50 MB
comfort threshold (and the 100 MB hard limit). Keep the raw files on your
machine only and regenerate them as needed.

## Quick start

```bash
python scripts/download_scorecard_data.py
```

The default run grabs the **Most-Recent-Cohorts-Institution.csv** file that the
pipeline expects and writes it to `data/raw/scorecard/public/`.

### Need more files?

The downloader supports additive bundles:

```bash
# Institution file + field-of-study bundle
python scripts/download_scorecard_data.py --bundle baseline field_of_study

# Everything DOE publishes, including MERGED history + crosswalks
python scripts/download_scorecard_data.py --bundle full
```

Use `python scripts/download_scorecard_data.py --help` to see every option and
the exact DOE URL that will be pulled.

## Manual download (fallback)

1. Visit https://collegescorecard.ed.gov/data/ and download the desired CSVs.
2. Drop the files under `data/raw/scorecard/public/`.
3. Re-run `python src/student_loan_default_analysis.py`.

> Tip: Keep the ZIP files outside the repo (e.g., `~/Downloads/scorecard/`) so
> they never get staged accidentally.

