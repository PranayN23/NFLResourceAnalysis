"""
Split scheme CSV files by year (2015_schemes.csv, 2016_schemes.csv, etc.)
"""
import pandas as pd
from pathlib import Path

try:
    from .config import SCHEME_DATA_DIR
except ImportError:
    import os
    SCHEME_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def split_scheme_by_year(
    input_csv: str | Path,
    output_dir: str | Path | None = None,
    prefix: str = "schemes",
) -> dict[int, Path]:
    """
    Split a scheme CSV by year, saving separate files.
    Returns dict mapping year -> output file path.
    """
    df = pd.read_csv(input_csv)
    if "season" not in df.columns:
        raise ValueError(f"CSV must have 'season' column. Columns: {list(df.columns)}")
    
    output_dir = Path(output_dir or SCHEME_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    year_files = {}
    for year in sorted(df["season"].unique()):
        year_df = df[df["season"] == year].copy()
        out_path = output_dir / f"{year}_{prefix}.csv"
        year_df.to_csv(out_path, index=False)
        year_files[year] = out_path
        print(f"Saved {len(year_df)} rows for {year} to {out_path}")
    
    return year_files


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "scheme/data/scheme_from_pbp_clustered.csv"
    split_scheme_by_year(input_file)
