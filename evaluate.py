"""
Offline evaluation of the recommenders.

Protocol (leave-last-out, no data leakage)
------------------------------------------
* Eligible users: at least 5 "liked" ratings (rating >= 7).
* For each sampled user, the MOST RECENT liked movie (by timestamp) is hidden.
  The user's other liked movies form the profile.
* ALL hidden ratings are removed from the data the models are built on, so the
  similarity matrices and popularity never see them.
* The model ranks every movie the user has not rated. We check where the hidden
  movie lands.
* Users are split into a VALIDATION set (used to choose alpha / beta) and a
  separate TEST set (used for the reported numbers).

Metrics
-------
HitRate@10 : fraction of users whose hidden movie is in their top 10
NDCG@10    : like hit rate, but rewards a higher position
Coverage@10: share of the catalogue that appears in anyone's top 10

Run:  python evaluate.py
Output: data/eval_results.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from recommender import DATA_DIR, LIKE_THRESHOLD, Recommender, load_data

SEED = 42
N_VAL, N_TEST = 1500, 2000
K = 10
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85]


def split_holdout(ratings: pd.DataFrame, rng: np.random.Generator):
    liked = ratings[ratings["rating"] >= LIKE_THRESHOLD]
    counts = liked.groupby("user_id").size()
    eligible = np.array(counts[counts >= 5].index)
    rng.shuffle(eligible)
    val_users = eligible[:N_VAL]
    test_users = eligible[N_VAL : N_VAL + N_TEST]

    chosen = np.concatenate([val_users, test_users])
    last = (
        liked[liked["user_id"].isin(chosen)]
        .sort_values(["user_id", "timestamp"])
        .groupby("user_id")
        .tail(1)
    )
    held = dict(zip(last["user_id"], last["movie_id"]))
    held_keys = set(zip(last["user_id"], last["movie_id"]))
    keep = [k not in held_keys for k in zip(ratings["user_id"], ratings["movie_id"])]
    train = ratings[np.array(keep)].reset_index(drop=True)
    return train, val_users, test_users, held


def build_users(rec: Recommender, train: pd.DataFrame, users, held):
    """Profile (liked items), seen mask and held-out item index for each user."""
    tr = train[train["user_id"].isin(users)]
    seen_by_user = tr.groupby("user_id")["movie_id"].apply(list).to_dict()
    liked_by_user = (
        tr[tr["rating"] >= LIKE_THRESHOLD].groupby("user_id")["movie_id"].apply(list).to_dict()
    )
    profiles, seen_rows, seen_cols, held_idx = [], [], [], []
    for r, u in enumerate(users):
        profiles.append([rec.id_to_idx[m] for m in liked_by_user.get(u, []) if m in rec.id_to_idx])
        for m in seen_by_user.get(u, []):
            if m in rec.id_to_idx:
                seen_rows.append(r)
                seen_cols.append(rec.id_to_idx[m])
        held_idx.append(rec.id_to_idx[held[u]])
    seen = np.zeros((len(users), rec.n_items), dtype=bool)
    seen[seen_rows, seen_cols] = True
    return profiles, seen, np.array(held_idx)


def rank_and_top(scores: np.ndarray, seen: np.ndarray, held_idx: np.ndarray, rng):
    """Rank of the held-out movie (ties share the average position) and each user's top-K."""
    n_users = scores.shape[0]
    ar = np.arange(n_users)
    held_score = scores[ar, held_idx].copy()
    scores = scores.copy()
    scores[seen] = -np.inf
    scores[ar, held_idx] = held_score  # the held-out movie stays a candidate

    greater = (scores > held_score[:, None]).sum(axis=1)
    equal = (scores == held_score[:, None]).sum(axis=1) - 1
    rank = greater + equal / 2.0 + 1.0

    noisy = scores + rng.random(scores.shape, dtype=np.float32) * 1e-7  # random tie-break
    top = np.argpartition(-noisy, K, axis=1)[:, :K]
    return rank, top


def summarise(ranks: np.ndarray, tops: np.ndarray, n_items: int) -> dict:
    hit = (ranks <= K).astype(float)
    ndcg = np.where(ranks <= K, 1.0 / np.log2(ranks + 1.0), 0.0)
    return {
        "hit_rate@10": float(hit.mean()),
        "ndcg@10": float(ndcg.mean()),
        "coverage@10": float(len(np.unique(tops)) / n_items),
    }


def evaluate_scores(score_fn, n_users: int, seen, held_idx, n_items: int, rng, chunk: int = 500) -> dict:
    """score_fn(start, end) -> (end-start, n_items) score matrix for those users."""
    ranks, tops = [], []
    for a in range(0, n_users, chunk):
        b = min(a + chunk, n_users)
        r, t = rank_and_top(score_fn(a, b), seen[a:b], held_idx[a:b], rng)
        ranks.append(r)
        tops.append(t)
    return summarise(np.concatenate(ranks), np.concatenate(tops), n_items)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    movies, ratings = load_data()
    train, val_users, test_users, held = split_holdout(ratings, rng)
    print(f"train ratings: {len(train):,} | val users: {len(val_users)} | test users: {len(test_users)}")

    rec = Recommender(movies, train)
    print(f"models built in {time.time() - t0:.1f}s")

    val_profiles, val_seen, val_held = build_users(rec, train, val_users, held)
    test_profiles, test_seen, test_held = build_users(rec, train, test_users, held)
    P_val = rec.profile_matrix(val_profiles)
    P_test = rec.profile_matrix(test_profiles)
    n_val, n_test, n_items = len(val_users), len(test_users), rec.n_items

    def cache_raw(P, n):
        """Collaborative / content score matrices for all users (computed once)."""
        cf = np.zeros((n, n_items), dtype=np.float32)
        ct = np.zeros((n, n_items), dtype=np.float32)
        for a in range(0, n, 500):
            b = min(a + 500, n)
            cf[a:b], ct[a:b] = rec.raw_scores(P[a:b])
        return cf, ct

    cf_val, ct_val = cache_raw(P_val, n_val)
    cf_test, ct_test = cache_raw(P_test, n_test)
    print(f"raw scores ready in {time.time() - t0:.1f}s")

    # ---- choose alpha / beta on the VALIDATION users ----------------------
    grid = []
    for a in ALPHAS:
        for b in BETAS:
            fn = lambda s, e, a=a, b=b: rec.combine(cf_val[s:e], ct_val[s:e], a, b)
            m = evaluate_scores(fn, n_val, val_seen, val_held, n_items, rng)
            grid.append({"alpha": a, "beta": b, **m})
    grid_df = pd.DataFrame(grid).sort_values("ndcg@10", ascending=False)
    best = grid_df.iloc[0]
    best_params = {"alpha": float(best["alpha"]), "beta": float(best["beta"])}
    print("best on validation:", best_params, f"NDCG@10={best['ndcg@10']:.4f}")

    # ---- final numbers on the TEST users ----------------------------------
    def hybrid(a, b):
        return lambda s, e: rec.combine(cf_test[s:e], ct_test[s:e], a, b)

    pop_fn = lambda s, e: np.tile(rec.pop_norm.astype(np.float32), (e - s, 1))
    rand_fn = lambda s, e: rng.random((e - s, n_items), dtype=np.float32)
    ev = lambda fn: evaluate_scores(fn, n_test, test_seen, test_held, n_items, rng)
    results = {
        "Random": ev(rand_fn),
        "Popularity (weighted rating)": ev(pop_fn),
        "Content-based (genres)": ev(hybrid(0.0, 0.0)),
        "Collaborative (item-item)": ev(hybrid(1.0, 0.0)),
        "Final model (collaborative + popularity)": ev(hybrid(best_params["alpha"], best_params["beta"])),
    }

    out = {
        "protocol": {
            "split": "leave-last-out on most recent liked movie; hidden ratings removed from training",
            "like_threshold": LIKE_THRESHOLD,
            "k": K,
            "seed": SEED,
            "val_users": int(len(val_users)),
            "test_users": int(n_test),
            "train_ratings": int(len(train)),
            "n_movies": int(rec.n_items),
        },
        "best_params": best_params,
        "results": results,
        "validation_grid": grid_df.round(4).to_dict(orient="records"),
    }
    (DATA_DIR / "eval_results.json").write_text(json.dumps(out, indent=2))

    print("\nTEST RESULTS")
    print(pd.DataFrame(results).T.round(4).to_string())
    print(f"\nsaved data/eval_results.json  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
