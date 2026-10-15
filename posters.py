"""
Movie posters from TMDB (The Movie Database).

The dataset stores IMDb ids, and TMDB can look a movie up by its IMDb id, so we call
    GET /3/find/tt<imdb_id>?external_source=imdb_id
and read `poster_path` and `overview` from the result.

* Needs a free TMDB API key (see README). Without a key the app still works and shows
  coloured placeholder cards instead of posters.
* Results are cached in data/poster_cache.json, so each movie is fetched only once.

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CACHE_PATH = Path(__file__).parent / "data" / "poster_cache.json"
IMG_BASE = "https://image.tmdb.org/t/p/w342"

_lock = threading.Lock()
_cache: dict | None = None


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text())
        except Exception:
            _cache = {}
    return _cache


def _save_cache() -> None:
    try:
        CACHE_PATH.write_text(json.dumps(_cache))
    except Exception:
        pass  # read-only folder (e.g. some hosts): just skip saving


def _http_get_json(url: str, timeout: float = 8.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class BadKey(Exception):
    pass


def _fetch_one(imdb_id: str, key: str) -> dict | None:
    """Return {'poster': url|None, 'overview': str}, or None if the request failed."""
    url = f"https://api.themoviedb.org/3/find/tt{imdb_id}?external_source=imdb_id&api_key={key}"
    try:
        data = _http_get_json(url)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise BadKey from e
        return None
    except Exception:
        return None
    results = data.get("movie_results") or []
    if not results:
        return {"poster": None, "overview": ""}
    m = results[0]
    path = m.get("poster_path")
    return {"poster": IMG_BASE + path if path else None, "overview": m.get("overview", "")}


def get_api_key(session_value: str = "") -> str:
    """Key from: sidebar box > Streamlit secrets > TMDB_API_KEY environment variable."""
    if session_value and session_value.strip():
        return session_value.strip()
    try:
        import streamlit as st

        v = st.secrets.get("TMDB_API_KEY", "")
        if v:
            return str(v).strip()
    except Exception:
        pass
    return os.environ.get("TMDB_API_KEY", "").strip()


def get_posters(imdb_ids: list[str], key: str) -> tuple[dict, str]:
    """
    Returns ({imdb_id: {'poster', 'overview'}}, status).
    status: 'ok' | 'no_key' | 'bad_key' | 'offline'
    """
    with _lock:
        cache = _load_cache()
        found = {i: cache[i] for i in imdb_ids if i in cache}
    missing = [i for i in imdb_ids if i not in found]
    if not missing:
        return found, "ok"
    if not key:
        return found, "no_key"

    status = "ok"
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            fetched = list(pool.map(lambda i: (i, _fetch_one(i, key)), missing))
    except BadKey:
        return found, "bad_key"

    new = {i: r for i, r in fetched if r is not None}
    if len(new) < len(missing) and not new:
        status = "offline"
    if new:
        with _lock:
            _load_cache().update(new)
            _save_cache()
        found.update(new)
    return found, status
