"""
Core recommendation engine (no Streamlit code in here, so it can be tested
and evaluated on its own).

Methods
-------
1. Popularity      : IMDb-style weighted rating (a movie with few ratings is
                     pulled towards the global average).
2. Content-based   : TF-IDF on genres (+ decade) -> cosine similarity.
3. Collaborative   : item-item cosine similarity on mean-centred user ratings,
                     with significance shrinkage for pairs with few co-raters.
4. Hybrid          : blend of the above:
                     score = (1-beta) * (alpha*CF + (1-alpha)*Content) + beta*Popularity
5. Mood filter     : maps a mood to genres to include / avoid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
LIKE_THRESHOLD = 7  # rating (0-10) at or above this counts as "liked"

# --------------------------------------------------------------------------
# Moods -> genres
# --------------------------------------------------------------------------
MOODS: dict[str, dict] = {
    "Happy / feel-good": {
        "emoji": "😄",
        "genres": ["Comedy", "Family", "Animation", "Musical"],
        "avoid": ["Horror", "War", "Crime"],
        "blurb": "Light, funny and uplifting.",
    },
    "Sad / need a good cry": {
        "emoji": "😢",
        "genres": ["Drama", "Romance", "Biography"],
        "avoid": ["Comedy", "Horror", "Action"],
        "blurb": "Emotional stories that hit hard.",
    },
    "Excited / want adrenaline": {
        "emoji": "🔥",
        "genres": ["Action", "Adventure", "Sci-Fi", "Thriller"],
        "avoid": ["Documentary", "Musical", "Romance"],
        "blurb": "Fast, loud and high-energy.",
    },
    "Relaxed / chilling": {
        "emoji": "😌",
        "genres": ["Comedy", "Family", "Animation", "Music", "Romance"],
        "avoid": ["Horror", "Thriller", "War", "Crime"],
        "blurb": "Easy to watch, no stress.",
    },
    "Romantic": {
        "emoji": "💘",
        "genres": ["Romance", "Drama", "Comedy", "Musical"],
        "avoid": ["Horror", "War", "Crime", "Sci-Fi"],
        "blurb": "Love stories for two (or one).",
    },
    "Want to be scared": {
        "emoji": "😱",
        "genres": ["Horror", "Thriller", "Mystery"],
        "avoid": ["Family", "Animation", "Musical", "Comedy"],
        "blurb": "Lights off, volume up.",
    },
    "Curious / mind-bending": {
        "emoji": "🧠",
        "genres": ["Mystery", "Sci-Fi", "Thriller", "Fantasy"],
        "avoid": ["Family", "Musical", "Romance"],
        "blurb": "Twists, puzzles and big ideas.",
    },
    "Inspired / motivated": {
        "emoji": "💪",
        "genres": ["Biography", "Sport", "History", "Drama"],
        "avoid": ["Horror", "Comedy"],
        "blurb": "True stories and comebacks.",
    },
    "Adventurous / want to escape": {
        "emoji": "🧭",
        "genres": ["Adventure", "Fantasy", "Sci-Fi", "Western"],
        "avoid": ["Documentary", "Romance"],
        "blurb": "Other worlds, big journeys.",
    },
    "Nostalgic": {
        "emoji": "📼",
        "genres": ["Family", "Adventure", "Comedy", "Music"],
        "avoid": ["Horror", "Documentary"],
        "blurb": "Older favourites (defaults to movies up to 2005).",
        "year_max": 2005,
    },
}


# A movie must contain at least one CORE genre to qualify for a mood
# (moods not listed here use all of their genres as core genres).
_CORE = {
    "Romantic": ["Romance"],
    "Want to be scared": ["Horror", "Thriller"],
    "Curious / mind-bending": ["Mystery", "Sci-Fi"],
    "Inspired / motivated": ["Biography", "Sport", "History"],
    "Adventurous / want to escape": ["Adventure", "Fantasy", "Western"],
    "Relaxed / chilling": ["Comedy", "Family", "Animation", "Music"],
}
for _m, _c in _CORE.items():
    MOODS[_m]["core"] = _c


def split_genres(g: str) -> list[str]:
    return [x for x in str(g).split("|") if x]


# --------------------------------------------------------------------------
# Free-text mood -> genres (simple, transparent keyword rules)
# --------------------------------------------------------------------------
# (keyword stems, genres to look for, genres to avoid)
_KEYWORD_RULES = [
    ("scary|horror|creepy|spooky|terrif|frighten|haunt", ["Horror", "Thriller"], ["Family", "Animation", "Comedy"]),
    ("dark|psycholog|disturb|twisted|gritty|intense", ["Thriller", "Mystery", "Crime", "Drama"], ["Family", "Animation"]),
    ("mind|twist|puzzle|confus|complex|brain|think|plot", ["Mystery", "Sci-Fi", "Thriller"], []),
    ("space|alien|future|robot|sci|galaxy|planet", ["Sci-Fi"], []),
    ("laugh|funny|comed|hilarious|silly|fun\\b|joke", ["Comedy"], ["Horror"]),
    ("kid|child|year old|family|toddler|cartoon|animat", ["Family", "Animation"], ["Horror", "Crime", "War", "Thriller"]),
    ("love|romantic|romance|date night|crush|heart", ["Romance"], ["Horror", "War"]),
    ("sad|cry|emotional|tear|heartbreak|lonely|depress|miss", ["Drama", "Romance"], ["Comedy", "Horror"]),
    ("happy|cheer|uplift|feel.?good|good mood|smile", ["Comedy", "Family", "Musical", "Animation"], ["Horror", "War", "Crime"]),
    ("action|adrenaline|explosi|fight|thrill|chase|epic", ["Action", "Thriller", "Adventure"], ["Documentary", "Musical"]),
    ("adventure|journey|quest|explor|escape|fantas|magic", ["Adventure", "Fantasy"], ["Documentary"]),
    ("inspir|motivat|true story|biograph|real life|underdog|sport|comeback", ["Biography", "Sport", "History", "Drama"], ["Horror"]),
    ("histor|period|war\\b|world war", ["History", "War"], []),
    ("crime|heist|gangster|detective|murder|mafia|cop|investigat", ["Crime", "Mystery", "Thriller"], ["Family", "Animation"]),
    ("music|song|dance|concert|band", ["Music", "Musical"], []),
    ("western|cowboy", ["Western"], []),
    ("documentar|learn|educat", ["Documentary"], []),
    ("chill|relax|calm|cozy|lazy|easy|light.?heart|unwind", ["Comedy", "Family", "Animation", "Romance"], ["Horror", "Thriller", "War", "Crime"]),
]


def parse_mood_text(text: str) -> tuple[list[str], list[str], list[str]]:
    """Return (genres to look for, genres to avoid, matched keywords) for a sentence."""
    import re

    t = text.lower()
    want: dict[str, int] = {}
    avoid: set[str] = set()
    hits: list[str] = []
    for pattern, genres, bad in _KEYWORD_RULES:
        m = re.search(r"\b(?:" + pattern + ")", t)
        if m:
            hits.append(m.group(0).strip())
            for g in genres:
                want[g] = want.get(g, 0) + 1
            avoid.update(bad)
    genres = [g for g, _ in sorted(want.items(), key=lambda kv: -kv[1])]
    if re.search(r"\b(?:kid|child|year old|toddler|cartoon|animat)", t):
        genres = [g for g in genres if g in ("Family", "Animation")] or ["Family", "Animation"]
    avoid -= set(genres)  # something explicitly asked for is never avoided
    return genres, sorted(avoid), hits


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(data_dir / "movies.csv", dtype={"movie_id": str})
    ratings = pd.read_csv(
        data_dir / "ratings.csv",
        dtype={"user_id": np.int64, "movie_id": str, "rating": np.int8, "timestamp": np.int64},
    )
    return movies, ratings


def load_defaults() -> dict:
    """alpha / beta chosen on the validation users by evaluate.py (if available)."""
    path = DATA_DIR / "eval_results.json"
    if path.exists():
        try:
            best = json.loads(path.read_text())["best_params"]
            return {"alpha": float(best["alpha"]), "beta": float(best["beta"])}
        except Exception:
            pass
    return {"alpha": 0.6, "beta": 0.1}


# --------------------------------------------------------------------------
# Recommender
# --------------------------------------------------------------------------
class Recommender:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame, shrinkage: float = 10.0, top_k: int = 200):
        self.movies = movies[["movie_id", "title", "year", "genres"]].reset_index(drop=True).copy()
        self.n_items = len(self.movies)
        self.id_to_idx = {m: i for i, m in enumerate(self.movies["movie_id"])}
        self.ratings = ratings[ratings["movie_id"].isin(self.id_to_idx)].copy()
        self.shrinkage = shrinkage
        self.top_k = top_k

        self._build_popularity()
        self._build_content()
        self._build_collaborative()

    # ---- popularity ------------------------------------------------------
    def _build_popularity(self) -> None:
        r = self.ratings
        stats = r.groupby("movie_id")["rating"].agg(["count", "mean"])
        n = self.movies["movie_id"].map(stats["count"]).fillna(0).to_numpy(dtype=float)
        avg = self.movies["movie_id"].map(stats["mean"]).fillna(0).to_numpy(dtype=float)
        c = r["rating"].mean()                      # global mean rating
        m = np.percentile(n[n > 0], 60)             # minimum "votes" to be trusted
        wr = (n / (n + m)) * avg + (m / (n + m)) * c
        self.n_ratings = n
        self.avg_rating = avg
        self.weighted_rating = wr
        span = wr.max() - wr.min()
        self.pop_norm = (wr - wr.min()) / (span if span > 0 else 1.0)

    # ---- content ---------------------------------------------------------
    def _build_content(self) -> None:
        """Sparse TF-IDF item vectors. Similarity is computed on the fly (X @ X.T)."""
        def tokens(row) -> str:
            gs = [g.replace("-", "").lower() for g in split_genres(row.genres)]
            decade = f"decade{(int(row.year) // 10) * 10}"
            return " ".join(gs + [decade])

        docs = [tokens(r) for r in self.movies.itertuples(index=False)]
        tfidf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False)
        self.X_content = normalize(tfidf.fit_transform(docs)).tocsr()
        self.genre_sets = [set(split_genres(g)) for g in self.movies["genres"]]

    # ---- collaborative (item-item) --------------------------------------
    def _build_collaborative(self, chunk: int = 400) -> None:
        """
        Item-item cosine similarity on mean-centred ratings, shrunk for pairs with few
        co-raters, keeping only the top-K neighbours of each item (sparse, memory-safe).
        """
        r = self.ratings
        u = pd.factorize(r["user_id"])[0]
        i = r["movie_id"].map(self.id_to_idx).to_numpy()
        user_mean = r.groupby("user_id")["rating"].transform("mean").to_numpy()
        centred = (r["rating"].to_numpy() - user_mean).astype(np.float32)

        n_users = u.max() + 1
        X = sp.csr_matrix((centred, (i, u)), shape=(self.n_items, n_users))
        B = sp.csr_matrix((np.ones_like(centred), (i, u)), shape=(self.n_items, n_users))
        Xn, XnT = normalize(X).tocsr(), normalize(X).T.tocsr()
        BT = B.T.tocsr()

        k = min(self.top_k, self.n_items - 1)
        rows, cols, vals = [], [], []
        for a in range(0, self.n_items, chunk):
            b = min(a + chunk, self.n_items)
            sim = (Xn[a:b] @ XnT).toarray().astype(np.float32)
            co = (B[a:b] @ BT).toarray().astype(np.float32)
            sim *= co / (co + self.shrinkage)            # shrink pairs with few co-raters
            np.clip(sim, 0.0, None, out=sim)             # positive similarity only
            sim[np.arange(b - a), np.arange(a, b)] = 0.0  # no self-similarity
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            v = np.take_along_axis(sim, top, axis=1)
            keep = v > 0
            rr = np.repeat(np.arange(a, b), k).reshape(b - a, k)
            rows.append(rr[keep]); cols.append(top[keep]); vals.append(v[keep])
        self.S_cf = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n_items, self.n_items),
        )

    # ---- scoring ---------------------------------------------------------
    @staticmethod
    def _row_norm(M: np.ndarray) -> np.ndarray:
        mx = M.max(axis=1, keepdims=True)
        mx[mx == 0] = 1.0
        return M / mx

    def raw_scores(self, P: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
        """
        P: (n_profiles x n_items) sparse, rows = liked-item weights summing to 1.
        Returns (collaborative, content) score matrices, each row scaled to max 1,
        with the profile's own movies set to 0.
        """
        P = P.tocsr()
        cf = np.asarray((P @ self.S_cf).todense(), dtype=np.float32)
        ct = np.asarray(((P @ self.X_content) @ self.X_content.T).todense(), dtype=np.float32)
        own = P.toarray() > 0
        cf[own] = 0.0
        ct[own] = 0.0
        return self._row_norm(cf), self._row_norm(ct)

    def combine(self, cf: np.ndarray, ct: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        base = alpha * cf + (1.0 - alpha) * ct
        return (1.0 - beta) * base + beta * self.pop_norm[None, :].astype(np.float32)

    def score_profiles(self, P: sp.spmatrix, alpha: float, beta: float) -> np.ndarray:
        cf, ct = self.raw_scores(P)
        return self.combine(cf, ct, alpha, beta)

    def profile_matrix(self, liked_lists: list[list[int]]) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for r, liked in enumerate(liked_lists):
            w = 1.0 / max(len(liked), 1)
            for j in liked:
                rows.append(r)
                cols.append(j)
                vals.append(w)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(liked_lists), self.n_items))

    # ---- mood ------------------------------------------------------------
    def mood_mask(self, mood: str | None) -> np.ndarray:
        if not mood or mood not in MOODS:
            return np.ones(self.n_items, dtype=bool)
        core = set(MOODS[mood].get("core", MOODS[mood]["genres"]))
        avoid = set(MOODS[mood]["avoid"])
        return np.array([bool(core & g) and not (avoid & g) for g in self.genre_sets])

    def mood_fit(self, mood: str) -> np.ndarray:
        """Share of a movie's genres that belong to the mood (0-1)."""
        want = set(MOODS[mood]["genres"])
        return np.array([len(want & g) / max(len(g), 1) for g in self.genre_sets])

    # ---- public API ------------------------------------------------------
    def recommend_by_mood(
        self,
        mood: str,
        n: int = 10,
        year_range: tuple[int, int] = (1900, 2100),
        min_ratings: int = 0,
        surprise: int | None = None,
    ) -> pd.DataFrame:
        mask = self.mood_mask(mood)
        mask &= self.movies["year"].between(*year_range).to_numpy()
        mask &= self.n_ratings >= min_ratings
        if not mask.any():
            return pd.DataFrame()

        fit = self.mood_fit(mood)
        score = self.weighted_rating * (0.5 + 0.5 * fit)
        score = np.where(mask, score, -np.inf)

        order = np.argsort(-score)
        if surprise is None:
            top = order[:n]
        else:  # draw n picks at random from the best 4n (weighted by score)
            pool = order[: max(n * 4, n)]
            pool = pool[np.isfinite(score[pool])]
            rng = np.random.default_rng(surprise)
            w = score[pool] - score[pool].min() + 1e-6
            top = rng.choice(pool, size=min(n, len(pool)), replace=False, p=w / w.sum())
            top = top[np.argsort(-score[top])]

        out = self._format(top, extra={"mood_fit": fit[top]})
        out["why"] = f"Matches {mood.split(' /')[0].lower()} genres"
        return out

    def recommend_similar(
        self,
        liked_ids: list[str],
        n: int = 10,
        alpha: float = 0.6,
        beta: float = 0.1,
        mood: str | None = None,
        year_range: tuple[int, int] = (1900, 2100),
        min_rating: float = 0.0,
    ) -> pd.DataFrame:
        liked = [self.id_to_idx[m] for m in liked_ids if m in self.id_to_idx]
        if not liked:
            return pd.DataFrame()
        P = self.profile_matrix([liked])
        score = self.score_profiles(P, alpha, beta)[0]

        mask = self.mood_mask(mood)
        mask &= self.movies["year"].between(*year_range).to_numpy()
        mask &= self.avg_rating >= min_rating
        mask[liked] = False
        score = np.where(mask, score, -np.inf)
        top = np.argsort(-score)[:n]
        top = top[np.isfinite(score[top])]

        # which liked movie is each pick closest to? (for the "because you liked" line)
        cf_pair = self.S_cf[liked][:, top].toarray()
        ct_pair = (self.X_content[liked] @ self.X_content[top].T).toarray()
        pair = alpha * cf_pair + (1 - alpha) * ct_pair
        closest = np.array(liked)[pair.argmax(axis=0)]
        out = self._format(top, extra={"score": score[top]})
        out["why"] = [f"Because you liked {self.movies.at[c, 'title']}" for c in closest]
        mx = out["score"].max() or 1.0
        out["match"] = (out["score"] / mx * 100).round(0).astype(int)
        return out

    def recommend_by_genres(
        self,
        genres: list[str],
        avoid: list[str] | tuple = (),
        n: int = 10,
        year_min: int = 1900,
        min_avg_rating: float = 0.0,
        min_ratings: int = 30,
        sort: str = "best",
    ) -> pd.DataFrame:
        """Movies having at least one of `genres` and none of `avoid`."""
        g, av = set(genres), set(avoid)
        mask = np.array([bool(g & s) and not (av & s) for s in self.genre_sets])
        mask &= self.movies["year"].to_numpy() >= year_min
        mask &= self.avg_rating >= min_avg_rating
        mask &= self.n_ratings >= min_ratings
        if not mask.any():
            return pd.DataFrame()
        fit = np.array([len(g & s) / max(len(s), 1) for s in self.genre_sets])
        if sort == "popular":
            score = self.n_ratings.astype(float)
        elif sort == "newest":
            score = self.movies["year"].to_numpy() + self.weighted_rating / 100.0
        else:  # "best": weighted rating, boosted when more of the movie's genres fit
            score = self.weighted_rating * (0.5 + 0.5 * fit)
        score = np.where(mask, score, -np.inf)
        top = np.argsort(-score)[:n]
        top = top[np.isfinite(score[top])]
        out = self._format(top, extra={"mood_fit": fit[top]})
        out["why"] = ""
        return out

    # ---- helpers ---------------------------------------------------------
    def _format(self, idx: np.ndarray, extra: dict | None = None) -> pd.DataFrame:
        df = self.movies.iloc[idx].copy()
        df["avg_rating"] = self.avg_rating[idx].round(2)
        df["n_ratings"] = self.n_ratings[idx].astype(int)
        df["imdb"] = "https://www.imdb.com/title/tt" + df["movie_id"] + "/"
        for k, v in (extra or {}).items():
            df[k] = v
        return df.reset_index(drop=True)
