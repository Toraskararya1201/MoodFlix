"""
Download the MovieTweetings dataset and build the cleaned CSVs used by the app.

MovieTweetings (Dooms et al., 2013) is a public dataset of movie ratings
(0-10 scale) taken from tweets, with IMDb ids, titles, years and genres.
Source: https://github.com/sidooms/MovieTweetings

Run:  python prepare_data.py
Output: data/movies.csv, data/ratings.csv
"""
import html
import io
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/"
DATA = Path(__file__).parent / "data"

MIN_MOVIE_RATINGS = 10   # drop movies with too few ratings (too noisy)
MIN_USER_RATINGS = 10    # drop users with too few ratings


def fetch(name: str) -> str:
    print(f"Downloading {name} ...")
    with urllib.request.urlopen(BASE + name, timeout=120) as r:
        return r.read().decode("utf-8", errors="replace")


def main() -> None:
    DATA.mkdir(exist_ok=True)

    # ---------- movies ----------
    raw = fetch("movies.dat")
    movies = pd.read_csv(
        io.StringIO(raw), sep="::", engine="python", header=None,
        names=["movie_id", "title_year", "genres"], dtype=str,
    )
    # "Title (1999)" -> title, year
    extracted = movies["title_year"].str.extract(r"^(.*)\s\((\d{4})\)\s*$")
    movies["title"] = extracted[0].str.strip().map(lambda t: html.unescape(t) if isinstance(t, str) else t)
    movies["year"] = pd.to_numeric(extracted[1], errors="coerce")
    movies = movies.dropna(subset=["title", "year", "genres"])
    movies["year"] = movies["year"].astype(int)
    movies["genres"] = movies["genres"].str.strip()
    movies = movies[movies["genres"] != ""]
    movies = movies.drop_duplicates("movie_id")

    # ---------- ratings ----------
    raw = fetch("ratings.dat")
    ratings = pd.read_csv(
        io.StringIO(raw), sep="::", engine="python", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": int, "movie_id": str, "rating": int, "timestamp": int},
    )
    ratings = ratings.drop_duplicates(["user_id", "movie_id"], keep="last")
    ratings = ratings[ratings["rating"].between(0, 10)]
    ratings = ratings[ratings["movie_id"].isin(movies["movie_id"])]

    # ---------- filter sparse movies / users (iterate once, enough here) ----------
    counts = ratings["movie_id"].value_counts()
    keep_movies = counts[counts >= MIN_MOVIE_RATINGS].index
    ratings = ratings[ratings["movie_id"].isin(keep_movies)]
    ucounts = ratings["user_id"].value_counts()
    keep_users = ucounts[ucounts >= MIN_USER_RATINGS].index
    ratings = ratings[ratings["user_id"].isin(keep_users)]

    # recompute per-movie stats on the final ratings
    stats = ratings.groupby("movie_id")["rating"].agg(n_ratings="count", avg_rating="mean")
    movies = movies.merge(stats, left_on="movie_id", right_index=True)
    movies = movies[["movie_id", "title", "year", "genres", "n_ratings", "avg_rating"]]
    movies["avg_rating"] = movies["avg_rating"].round(3)
    movies = movies.sort_values("n_ratings", ascending=False).reset_index(drop=True)

    movies.to_csv(DATA / "movies.csv", index=False)
    ratings[["user_id", "movie_id", "rating", "timestamp"]].to_csv(DATA / "ratings.csv", index=False)

    print(f"movies : {len(movies):,}")
    print(f"users  : {ratings['user_id'].nunique():,}")
    print(f"ratings: {len(ratings):,}")


if __name__ == "__main__":
    main()
