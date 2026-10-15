"""
Live movie catalogue from TMDB (The Movie Database): every movie, always up to date.

Uses GET /3/discover/movie sorted by rating (highest first). Each result already
contains title, release date, rating, vote count, poster and a plot overview, so no
extra requests are needed. TMDB allows up to 500 pages x 20 movies (10,000 movies)
per query; change the filters (years, genre, language) to reach other movies.

Both kinds of TMDB credentials work:
  * the short "API Key (v3 auth)"  (32 characters)
  * the long "API Read Access Token" (starts with eyJ...)

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

IMG_BASE = "https://image.tmdb.org/t/p/w342"
MAX_PAGES = 500  # TMDB hard limit

# category -> (TMDB original language, TMDB origin country, default min votes)
# "votes" is the default minimum number of votes. It keeps a film with a single
# 10/10 vote from beating a classic; you can set it to 0 in the app.
CATEGORIES: dict[str, dict] = {
    "🌍 All movies": {"lang": "", "country": "", "votes": 1000},
    "🎬 Hollywood (English / USA)": {"lang": "en", "country": "US", "votes": 1000},
    "🇮🇳 Bollywood (Hindi)": {"lang": "hi", "country": "", "votes": 100},
    "🎥 Tollywood (Telugu)": {"lang": "te", "country": "", "votes": 50},
    "🎥 Kollywood (Tamil)": {"lang": "ta", "country": "", "votes": 50},
    "🎥 Mollywood (Malayalam)": {"lang": "ml", "country": "", "votes": 30},
    "🎥 Sandalwood (Kannada)": {"lang": "kn", "country": "", "votes": 20},
    "🎥 Marathi": {"lang": "mr", "country": "", "votes": 10},
    "🎥 Bengali": {"lang": "bn", "country": "", "votes": 10},
    "🎥 Punjabi": {"lang": "pa", "country": "", "votes": 10},
    "🇬🇧 British": {"lang": "en", "country": "GB", "votes": 500},
    "🇰🇷 Korean": {"lang": "ko", "country": "", "votes": 300},
    "🇯🇵 Japanese (incl. anime)": {"lang": "ja", "country": "", "votes": 300},
    "🇨🇳 Chinese": {"lang": "zh", "country": "", "votes": 100},
    "🇫🇷 French": {"lang": "fr", "country": "", "votes": 300},
    "🇪🇸 Spanish": {"lang": "es", "country": "", "votes": 300},
    "🇩🇪 German": {"lang": "de", "country": "", "votes": 300},
    "🇮🇹 Italian": {"lang": "it", "country": "", "votes": 300},
    "🇷🇺 Russian": {"lang": "ru", "country": "", "votes": 100},
    "🇹🇷 Turkish": {"lang": "tr", "country": "", "votes": 100},
    "🇹🇭 Thai": {"lang": "th", "country": "", "votes": 50},
}

GENRES: dict[str, int] = {
    "Action": 28, "Adventure": 12, "Animation": 16, "Comedy": 35, "Crime": 80,
    "Documentary": 99, "Drama": 18, "Family": 10751, "Fantasy": 14, "History": 36,
    "Horror": 27, "Music": 10402, "Mystery": 9648, "Romance": 10749,
    "Science Fiction": 878, "Thriller": 53, "War": 10752, "Western": 37,
}
GENRE_NAMES = {v: k for k, v in GENRES.items()}

# The offline dataset / mood rules use IMDb-style genre names; map them to TMDB's.
GENRE_ALIASES = {
    "Sci-Fi": "Science Fiction", "Musical": "Music", "Biography": "History",
    "Film-Noir": "Crime",
}


def to_tmdb_genres(names: list[str]) -> list[str]:
    """IMDb-style genre names -> TMDB genre names (unknown ones dropped, no duplicates)."""
    out: list[str] = []
    for n in names:
        n = GENRE_ALIASES.get(n, n)
        if n in GENRES and n not in out:
            out.append(n)
    return out


class BadKey(Exception):
    pass


class Offline(Exception):
    pass


def discover(
    key: str,
    category: str,
    page: int = 1,
    year_from: int = 1900,
    year_to: int | None = None,
    min_votes: int = 100,
    min_rating: float = 0.0,
    genre: str = "",
    genres: list[str] | None = None,
    avoid_genres: list[str] | None = None,
) -> tuple[list[dict], int, int]:
    """
    One page (20 movies) sorted by rating, highest first.
    `genres` = movie must have ANY of these; `avoid_genres` = must have NONE of these.
    Returns (movies, total_pages, total_results).
    """
    today = date.today()
    year_to = min(year_to or today.year, today.year)
    last_day = today.isoformat() if year_to == today.year else f"{year_to}-12-31"
    cat = CATEGORIES.get(category, CATEGORIES["🌍 All movies"])
    p = {
        "language": "en-US",
        "sort_by": "vote_average.desc",
        "include_adult": "false",
        "include_video": "false",
        "vote_count.gte": int(min_votes),
        "vote_average.gte": float(min_rating),
        "primary_release_date.gte": f"{int(year_from)}-01-01",
        "primary_release_date.lte": last_day,  # never later than today
        "page": int(page),
    }
    if cat["lang"]:
        p["with_original_language"] = cat["lang"]
    if cat["country"]:
        p["with_origin_country"] = cat["country"]

    wanted = list(genres or ([genre] if genre else []))
    ids = [str(GENRES[g]) for g in wanted if g in GENRES]
    if ids:
        p["with_genres"] = "|".join(ids)  # "|" = OR in TMDB
    bad = [str(GENRES[g]) for g in (avoid_genres or []) if g in GENRES and g not in wanted]
    if bad:
        p["without_genres"] = ",".join(bad)

    headers = {"Accept": "application/json"}
    key = key.strip()
    if len(key) > 60:  # long "Read Access Token" (JWT) -> Bearer header
        headers["Authorization"] = f"Bearer {key}"
    else:
        p["api_key"] = key
    url = "https://api.themoviedb.org/3/discover/movie?" + urllib.parse.urlencode(p)
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise BadKey from e
        raise Offline from e
    except Exception as e:
        raise Offline from e

    movies = []
    for m in data.get("results", []):
        rd = m.get("release_date") or ""
        avg = float(m.get("vote_average") or 0)
        movies.append({
            "id": m["id"],
            "title": m.get("title") or m.get("original_title") or "Untitled",
            "year": int(rd[:4]) if rd[:4].isdigit() else 0,
            "score": avg,                      # exact value, used for sorting
            "rating": round(avg, 1),           # shown on the card
            "votes": int(m.get("vote_count") or 0),
            "genres": [GENRE_NAMES[g] for g in m.get("genre_ids", []) if g in GENRE_NAMES],
            "overview": (m.get("overview") or "").strip(),
            "poster": IMG_BASE + m["poster_path"] if m.get("poster_path") else None,
            "url": f"https://www.themoviedb.org/movie/{m['id']}",
        })
    pages = min(int(data.get("total_pages", 1)), MAX_PAGES)
    return movies, pages, int(data.get("total_results", len(movies)))
