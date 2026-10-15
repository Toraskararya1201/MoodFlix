# 🎭 MoodFlix: Mood-Based Movie Recommender

A movie recommender with a dark, poster-based Streamlit UI. Describe your mood in plain words (or tap a quick mood), browse by genre, or pick a movie you love and find similar ones. The recommendation model is evaluated offline against baselines.

**Tabs:** Find Movies (live TMDB), Mood Discovery, Browse (offline data), Find Similar, Model evaluation, Data explorer.

## Live catalogue (all movies up to today)

With a free TMDB key, **Find Movies** and **Mood Discovery** search TMDB's full catalogue (Hollywood, Bollywood, Tollywood, Kollywood, Korean, Japanese, British, French and more, released up to today):

- Press **Find movies**: you get the **top 10 by rating, highest first** (every search is sorted the same way).
- Press **Explore more movies** to load the next 10, as many times as you like (TMDB allows 10,000 movies per search; change years, genre or category to reach others).
- Each card shows rank, rating, year, genres, vote count and a short description (hover for the full text).
- **Min votes** stops movies with 1-2 votes from showing a fake 10/10. Set it to 0 to include every movie.
- The key can be the short API key or the long "API Read Access Token".

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The cleaned dataset is already in `data/`, so the app runs straight away.

## Turn on movie posters (free, 2 minutes)

The dataset has no images, so posters come from [TMDB](https://www.themoviedb.org/) using each movie's IMDb id. Without a key the app still works and shows coloured placeholder cards.

1. Create a free account at themoviedb.org, then go to **Settings > API** and request an API key (choose "Developer", personal use).
2. Either paste the key into the **sidebar box** in the app, or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and put the key there.
3. Optional: run `python fetch_posters.py` once to download all poster links in advance (a few minutes). Otherwise posters are fetched as you browse and cached in `data/poster_cache.json`.

This product uses the TMDB API but is not endorsed or certified by TMDB.

## What's inside

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (5 tabs) |
| `posters.py` | TMDB poster lookup with a local cache and a placeholder fallback |
| `fetch_posters.py` | Optional: download all poster links in advance |
| `recommender.py` | Models: popularity, content-based, item-item collaborative filtering, blend, mood filter |
| `evaluate.py` | Offline evaluation with a validation/test split. Writes `data/eval_results.json` |
| `prepare_data.py` | Downloads MovieTweetings and rebuilds `data/movies.csv` and `data/ratings.csv` |
| `data/` | Cleaned dataset + evaluation results |

## Dataset

[MovieTweetings](https://github.com/sidooms/MovieTweetings) (Dooms et al., 2013): movie ratings (0-10) taken from public tweets, with IMDb ids, titles, years and genres.

The raw data has about 38,000 movies, but most have only 1-2 ratings. After cleaning (movies with at least 10 ratings, users with at least 10): **8,275 movies, 14,882 users, 723,817 ratings**. You can change `MIN_MOVIE_RATINGS` in `prepare_data.py` and re-run it to get more or fewer movies (at 5 it gives about 12,000 movies, but with noisier ratings).

Note: this dataset has genres but no plot descriptions, so the content-based part uses **genres + decade**.

## How it works

1. **Popularity**: IMDb-style weighted rating, so a movie with 3 ratings of 10 doesn't beat a classic with 2,000 ratings of 8.5.
2. **Content-based**: TF-IDF on genres and decade, then cosine similarity between movies.
3. **Collaborative (item-item)**: cosine similarity on mean-centred user ratings. Pairs with few common raters are shrunk towards zero, and only each movie's 200 nearest neighbours are kept so it scales to thousands of movies.
4. **Blend**: `score = (1 - beta) * (alpha * collaborative + (1 - alpha) * content) + beta * popularity`. The best alpha and beta are chosen automatically on validation users.
5. **Mood Discovery**: the sentence you type is turned into genres to look for and genres to avoid using simple keyword rules (for example "dark and psychological" gives Thriller, Mystery, Crime, Drama; "with my 6 year old" gives Family, Animation and avoids Horror, Crime, War). Quick-mood buttons add their own genres. Movies are filtered by your minimum rating and release year, then ranked by weighted rating, boosted when more of their genres fit.
6. **Find Similar**: uses the collaborative model for one movie, with a low popularity weight so the results are similar movies rather than just famous ones.

## Evaluation

Protocol: for each user, hide the most recent movie they liked (rating >= 7), build the models without the hidden ratings, and check whether the hidden movie lands in the top 10 out of 8,275 movies. Alpha and beta are chosen on 1,500 validation users; the numbers below are on **2,000 separate test users**.

| Model | Hit rate @10 | NDCG @10 |
|---|---|---|
| Random | 0.1% | 0.0005 |
| Popularity | 2.5% | 0.0093 |
| Content-based (genres only) | 0.6% | 0.0025 |
| Collaborative (item-item) | 6.8% | 0.0326 |
| **Final: collaborative + popularity prior (alpha 1.0, beta 0.7)** | **7.8%** | **0.0386** |

The final model is about 3x better than the popularity baseline on hit rate. An honest finding: genre-only content is weak, and on this data adding it to the blend did **not** improve validation accuracy, so the chosen blend uses no genre similarity. Genres still drive the mood picks and the "genre-content" slider in the app. To reproduce: `python evaluate.py` (about 20 seconds).

## Limitations (good to know for interviews)

- Content features are only genres and decade (no plot text, cast or director), which is why they did not help the blend.
- Mood understanding is keyword rules plus a hand-written mood-to-genre map, not a learned language model, so unusual phrasing may not be understood.
- There is no runtime filter, because the dataset has no runtimes (TMDB could provide them).
- The final model leans on popularity (beta = 0.7), so only about 3% of the catalogue ever appears in anyone's top 10. That is a trade-off for accuracy. Lower the popularity slider in the app for less obvious picks.
- Ratings come from Twitter users, so the audience skews towards recent, mainstream films.
- Evaluation is offline; there is no live user feedback.

## Ideas to extend

- Add plot descriptions (for example from TMDB) and use sentence embeddings for the content part.
- Learn the mood mapping from user tags or reviews.
- Add matrix factorisation (ALS/SVD) as another model and compare.
- Deploy on Streamlit Community Cloud and add the live link to your resume.
