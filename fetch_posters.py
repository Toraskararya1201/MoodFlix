"""
OPTIONAL: download poster links for every movie once, so the app never waits on TMDB.

    export TMDB_API_KEY=your_key        # Windows PowerShell: $env:TMDB_API_KEY="your_key"
    python fetch_posters.py             # about 8,000 movies, a few minutes

Results go to data/poster_cache.json (the app reads it automatically).
You can stop with Ctrl+C and run it again later; it continues where it stopped.
"""
import os
import sys

import pandas as pd

import posters
from recommender import DATA_DIR


def main() -> None:
    key = posters.get_api_key(os.environ.get("TMDB_API_KEY", ""))
    if not key:
        sys.exit("Set the TMDB_API_KEY environment variable first (see README).")
    ids = pd.read_csv(DATA_DIR / "movies.csv", dtype={"movie_id": str})["movie_id"].tolist()
    batch = 200
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        _, status = posters.get_posters(chunk, key)
        print(f"{min(start + batch, len(ids)):>5}/{len(ids)}  status={status}")
        if status in ("bad_key", "offline"):
            sys.exit(f"Stopped: {status}. Check your key / internet and run again.")
    print("Done. Poster cache saved to data/poster_cache.json")


if __name__ == "__main__":
    main()
