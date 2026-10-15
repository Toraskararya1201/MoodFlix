"""
MoodFlix - a mood-based movie recommender (Streamlit).

Run:  streamlit run app.py
"""
import html
import json
from datetime import date

import pandas as pd
import streamlit as st

import posters
import tmdb_catalog
from recommender import DATA_DIR, MOODS, Recommender, load_data, load_defaults, parse_mood_text

st.set_page_config(page_title="MoodFlix", page_icon="🎭", layout="wide")

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.mf-logo {font-size: 2.1rem; font-weight: 800; color: #E5484D; letter-spacing: .5px; margin-bottom: 0;}
.mf-tag  {color: #9a9ab5; margin-top: -4px; margin-bottom: 14px;}
.mf-banner {background: rgba(46, 160, 67, .18); border: 1px solid rgba(46, 160, 67, .45);
            border-radius: 8px; padding: 8px 14px; margin: 10px 0 6px 0; font-size: .9rem;}
.mf-section {font-weight: 700; font-size: 1.15rem; margin: 14px 0 4px 0;
             border-bottom: 2px solid #E5484D; padding-bottom: 6px;}
.mf-card {position: relative; background: #1b1b2f; border: 1px solid #2a2a45; border-radius: 12px;
          overflow: hidden; margin-bottom: 16px; transition: transform .15s;}
.mf-card:hover {transform: translateY(-3px); border-color: #E5484D;}
.mf-poster {width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block;}
.mf-ph {width: 100%; aspect-ratio: 2/3; display: flex; align-items: center; justify-content: center;
        text-align: center; padding: 12px; font-weight: 700; font-size: .95rem; color: #fff;}
.mf-badge {position: absolute; top: 8px; right: 8px; background: #E5484D; color: #fff;
           border-radius: 8px; padding: 2px 8px; font-size: .78rem; font-weight: 700;}
.mf-body {padding: 8px 10px 10px 10px;}
.mf-title {font-weight: 600; font-size: .88rem; line-height: 1.25; color: #EDEDF4 !important;
           text-decoration: none !important;}
.mf-meta {color: #9a9ab5; font-size: .74rem; margin-top: 3px;}
.mf-why {color: #E5484D; font-size: .72rem; margin-top: 3px;}
.mf-rank {position: absolute; top: 8px; left: 8px; background: rgba(0,0,0,.72); color: #fff;
          border-radius: 8px; padding: 2px 8px; font-size: .78rem; font-weight: 700; z-index: 2;}
.mf-desc {color: #b9b9d0; font-size: .74rem; line-height: 1.35; margin-top: 6px;}
/* equal-size cards: fixed text area, long text is clipped */
.mf-body {height: 7.6rem; overflow: hidden;}
.mf-card.mf-live .mf-body {height: 12.5rem;}
.mf-title {display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;}
.mf-meta {display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;}
.mf-desc {display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and building the models (a few seconds)...")
def get_recommender() -> Recommender:
    movies, ratings = load_data()
    return Recommender(movies, ratings)


@st.cache_data
def get_eval() -> dict | None:
    path = DATA_DIR / "eval_results.json"
    return json.loads(path.read_text()) if path.exists() else None


rec = get_recommender()
defaults = load_defaults()
YEAR_MIN, YEAR_MAX = int(rec.movies["year"].min()), int(rec.movies["year"].max())
TODAY_YEAR = date.today().year  # live TMDB catalogue goes up to today

QUICK_MOODS = {
    "😱 Scare me": "Want to be scared",
    "😂 Make me laugh": "Happy / feel-good",
    "🔥 Thrill me": "Excited / want adrenaline",
    "😌 Chill": "Relaxed / chilling",
    "💘 Romance": "Romantic",
    "🧠 Mind-bending": "Curious / mind-bending",
    "💪 Inspire me": "Inspired / motivated",
    "🧭 Adventure": "Adventurous / want to escape",
    "😢 Cry": "Sad / need a good cry",
    "📼 Nostalgia": "Nostalgic",
}
GENRE_COUNTS = rec.movies["genres"].str.split("|").explode().value_counts()
BROWSE_GENRES = sorted(g for g, c in GENRE_COUNTS.items() if c >= 20)


# --------------------------------------------------------------------------
# Sidebar: TMDB key for posters
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🎞️ Posters")
    key_input = st.text_input(
        "TMDB API key (optional)", type="password", key="tmdb_key",
        help="Free at themoviedb.org > Settings > API. You can also put TMDB_API_KEY in .streamlit/secrets.toml.",
    )
    TMDB_KEY = posters.get_api_key(key_input)
    st.caption("TMDB connected: live catalogue + posters on" if TMDB_KEY else
               "No key yet: add a free TMDB key to unlock every movie, Bollywood, Hollywood and more.")
    st.divider()
    st.write(
        f"**{rec.n_items:,}** movies  \n"
        f"**{rec.ratings['user_id'].nunique():,}** users  \n"
        f"**{len(rec.ratings):,}** ratings"
    )
    st.caption("Find Movies + Mood Discovery use the live TMDB catalogue (all movies up to today). "
               "Other tabs use the offline MovieTweetings dataset.")
    st.caption("This product uses the TMDB API but is not endorsed or certified by TMDB.")


# --------------------------------------------------------------------------
# Poster grid
# --------------------------------------------------------------------------
def _placeholder_color(title: str) -> str:
    hue = sum(ord(c) for c in title) % 360
    return f"linear-gradient(160deg, hsl({hue},45%,30%), hsl({(hue + 40) % 360},45%,16%))"


def card_html(row: pd.Series, info: dict | None) -> str:
    title = html.escape(str(row["title"]))
    poster = (info or {}).get("poster")
    if poster:
        art = f'<img class="mf-poster" src="{html.escape(poster)}" alt="{title}" loading="lazy">'
    else:
        art = f'<div class="mf-ph" style="background:{_placeholder_color(row["title"])}">{title}</div>'
    genres = html.escape(str(row["genres"]).replace("|", " · "))
    why = f'<div class="mf-why">{html.escape(str(row["why"]))}</div>' if row.get("why") else ""
    return (
        f'<div class="mf-card"><div class="mf-badge">★ {row["avg_rating"]:.1f}</div>{art}'
        f'<div class="mf-body"><a class="mf-title" href="{html.escape(row["imdb"])}" target="_blank">{title}</a>'
        f'<div class="mf-meta">{int(row["year"])} · {genres}</div>{why}</div></div>'
    )


def render_grid(df: pd.DataFrame, per_row: int = 5) -> None:
    if df.empty:
        st.info("No movies match these filters. Try lowering the minimum rating or the release year.")
        return
    infos, status = posters.get_posters([str(m) for m in df["movie_id"]], TMDB_KEY)
    if status == "bad_key":
        st.warning("TMDB rejected the API key, so posters are hidden. Check the key in the sidebar.")
    elif status == "offline" and TMDB_KEY:
        st.warning("Couldn't reach TMDB just now, so posters are hidden. Check your internet connection.")
    for start in range(0, len(df), per_row):
        cols = st.columns(per_row)
        for col, (_, row) in zip(cols, df.iloc[start : start + per_row].iterrows()):
            col.markdown(card_html(row, infos.get(str(row["movie_id"]))), unsafe_allow_html=True)



# --------------------------------------------------------------------------
# Live TMDB catalogue: top 10 first, always sorted by rating, "Explore more"
# --------------------------------------------------------------------------
PER_VIEW = 10


@st.cache_data(ttl=3600, show_spinner=False)
def cached_discover(key: str, page: int, **params):
    return tmdb_catalog.discover(key, page=page, **params)


def short_text(text: str, limit: int = 150) -> str:
    text = (text or "").strip()
    if not text:
        return "No description available."
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;: ") + "…"


def live_card_html(m: dict, rank: int) -> str:
    title = html.escape(m["title"])
    if m["poster"]:
        art = f'<img class="mf-poster" src="{html.escape(m["poster"])}" alt="{title}" loading="lazy">'
    else:
        art = f'<div class="mf-ph" style="background:{_placeholder_color(m["title"])}">{title}</div>'
    genres = html.escape(" · ".join(m["genres"][:3]))
    year = m["year"] or "TBA"
    return (
        f'<div class="mf-card mf-live"><div class="mf-rank">#{rank}</div><div class="mf-badge">★ {m["rating"]:.1f}</div>{art}'
        f'<div class="mf-body"><a class="mf-title" href="{html.escape(m["url"])}" target="_blank">{title}</a>'
        f'<div class="mf-meta">{year} · {genres}</div>'
        f'<div class="mf-meta">{m["votes"]:,} votes</div>'
        f'<div class="mf-desc" title="{html.escape(m["overview"])}">{html.escape(short_text(m["overview"]))}</div>'
        f"</div></div>"
    )


def start_search(prefix: str, key: str, params: dict) -> None:
    """Fresh search: fetch the first page, show the top 10."""
    state = {"params": params, "movies": [], "page": 1, "total_pages": 1, "total": 0,
             "shown": PER_VIEW, "error": None}
    try:
        with st.spinner("Searching TMDB..."):
            movies, pages, total = cached_discover(key, page=1, **params)
        state.update(movies=movies, total_pages=pages, total=total)
    except tmdb_catalog.BadKey:
        state["error"] = "bad_key"
    except tmdb_catalog.Offline:
        state["error"] = "offline"
    st.session_state[f"{prefix}_live"] = state


def explore_more(prefix: str, key: str) -> None:
    """Button callback: show 10 more, fetching the next TMDB page when needed."""
    s = st.session_state.get(f"{prefix}_live")
    if not s:
        return
    s["shown"] += PER_VIEW
    seen = {m["id"] for m in s["movies"]}
    while len(s["movies"]) < s["shown"] and s["page"] < s["total_pages"]:
        try:
            movies, _, _ = cached_discover(key, page=s["page"] + 1, **s["params"])
        except tmdb_catalog.BadKey:
            s["error"] = "bad_key"
            return
        except tmdb_catalog.Offline:
            s["error"] = "offline"
            s["shown"] -= PER_VIEW
            return
        s["page"] += 1
        s["movies"] += [m for m in movies if m["id"] not in seen]
        seen |= {m["id"] for m in movies}


def render_live(prefix: str, key: str, per_row: int = 5) -> None:
    s = st.session_state.get(f"{prefix}_live")
    if not s:
        return
    if s["error"] == "bad_key":
        st.error("TMDB rejected the API key. Check the key in the sidebar.")
        return
    if s["error"] == "offline":
        st.error("Couldn't reach TMDB. Check your internet connection and press the Find button again.")
        return
    # Always sorted: highest rating first (ties: more votes first)
    movies = sorted(s["movies"], key=lambda m: (-m["score"], -m["votes"]))
    if not movies:
        st.info("No movies match these filters. Try a lower minimum rating or fewer minimum votes, "
                "a wider year range, or another category.")
        return
    shown = movies[: s["shown"]]
    st.markdown(
        f'<div class="mf-banner">Showing top {len(shown)} of {s["total"]:,} matching movies · '
        f"sorted by rating, highest first · released up to {date.today():%d %b %Y}</div>",
        unsafe_allow_html=True,
    )
    for start in range(0, len(shown), per_row):
        cols = st.columns(per_row)
        for i, (col, m) in enumerate(zip(cols, shown[start : start + per_row])):
            col.markdown(live_card_html(m, start + i + 1), unsafe_allow_html=True)

    has_more = s["shown"] < len(movies) or s["page"] < s["total_pages"]
    if has_more:
        st.button("➕ Explore more movies", key=f"{prefix}_more", on_click=explore_more,
                  args=(prefix, key), type="primary")
    elif s["total_pages"] >= tmdb_catalog.MAX_PAGES:
        st.caption("That's the most TMDB returns for one search (10,000 movies). "
                   "Change the years, genre or category to explore different movies.")
    else:
        st.caption("You've seen every movie that matches. Loosen the filters to see more.")


def need_key_message() -> None:
    st.info(
        "🔑 **Add your free TMDB API key to unlock every movie (Hollywood, Bollywood and more, up to today).**  \n"
        "1. Sign up at themoviedb.org, then Settings > API > request a key.  \n"
        "2. Paste it in the sidebar, or put `TMDB_API_KEY` in `.streamlit/secrets.toml`."
    )


# --------------------------------------------------------------------------
# Header + tabs
# --------------------------------------------------------------------------
st.markdown('<div class="mf-logo">🎭 MoodFlix</div>', unsafe_allow_html=True)
st.markdown('<div class="mf-tag">Tell us how you feel, and we\'ll find the perfect film. Not just genre filters.</div>',
            unsafe_allow_html=True)

tab_live, tab_mood, tab_genre, tab_sim, tab_eval, tab_data = st.tabs(
    ["🎬 Find Movies", "🎭 Mood Discovery", "🎞️ Browse (offline data)", "🔎 Find Similar",
     "📊 Model evaluation", "🧮 Data explorer"]
)

# ---- Find Movies (live TMDB) ----------------------------------------------
with tab_live:
    st.subheader("Find movies: every film up to today")
    if not TMDB_KEY:
        need_key_message()
    category = st.selectbox("Category", list(tmdb_catalog.CATEGORIES), key="f_cat")
    c1, c2, c3 = st.columns(3)
    f_genres = c1.multiselect("Genres (any of these, leave empty for all)", list(tmdb_catalog.GENRES), key="f_genres")
    f_rating = c2.slider("Min rating ⭐", 0.0, 9.5, 0.0, 0.1, key="f_rating")
    default_votes = tmdb_catalog.CATEGORIES[category]["votes"]
    f_votes = c3.slider(
        "Min votes", 0, 5000, default_votes, 10, key=f"f_votes_{category}",
        help="Movies with very few votes can show a perfect 10 by luck. Set to 0 to include every movie.",
    )
    f_years = st.slider("Release year", 1900, TODAY_YEAR, (1990, TODAY_YEAR), key="f_years")
    if st.button("🔍 Find movies", type="primary", key="f_go", disabled=not TMDB_KEY):
        start_search("f", TMDB_KEY, dict(
            category=category, year_from=f_years[0], year_to=f_years[1], min_votes=f_votes,
            min_rating=f_rating, genres=f_genres,
        ))
    if TMDB_KEY:
        if "f_live" in st.session_state:
            render_live("f", TMDB_KEY)
        else:
            st.caption("Pick a category and press **Find movies**: you get the top 10 by rating, "
                       "then use **Explore more** for the next ones.")

# ---- Mood Discovery -------------------------------------------------------
with tab_mood:
    st.subheader("How are you feeling right now?")
    text = st.text_area(
        "Describe your mood",
        placeholder="e.g. I want something dark and psychological that will mess with my head  ·  "
                    "Need a fun movie to watch with my 6 year old  ·  Something epic about space exploration",
        label_visibility="collapsed", height=90, key="mood_text",
    )
    st.caption("Quick moods")
    quick = st.pills("Quick moods", list(QUICK_MOODS), label_visibility="collapsed", key="quick_mood")

    if TMDB_KEY:
        m_cat = st.selectbox("Category", list(tmdb_catalog.CATEGORIES), key="m_cat")
        c1, c2 = st.columns(2)
        min_rating = c1.slider("Min rating ⭐", 0.0, 9.5, 6.5, 0.1, key="m_rating_live")
        m_years = c2.slider("Release year", 1900, TODAY_YEAR, (1990, TODAY_YEAR), key="m_years")
        m_votes = st.slider(
            "Min votes", 0, 5000, tmdb_catalog.CATEGORIES[m_cat]["votes"], 10, key=f"m_votes_{m_cat}",
            help="Movies with very few votes can show a perfect 10 by luck. Set to 0 to include every movie.",
        )
        n_mood = PER_VIEW
        released_after = YEAR_MIN
    else:
        c1, c2, c3 = st.columns(3)
        min_rating = c1.slider("Min rating ⭐", 0.0, 9.0, 6.5, 0.1, key="m_rating")
        released_after = c2.slider("Released after", YEAR_MIN, YEAR_MAX, 1980, key="m_year")
        n_mood = c3.select_slider("How many movies", [5, 10, 15, 20], value=10, key="m_n")
    mood_go = st.button("🔍 Find my movies", type="primary", key="mood_go")

    genres, avoid, hits = [], [], []
    if text.strip():
        genres, avoid, hits = parse_mood_text(text)
    if quick:
        mood = MOODS[QUICK_MOODS[quick]]
        for g in mood.get("core", mood["genres"]):
            if g not in genres:
                genres.append(g)
        avoid = sorted(set(avoid) | (set(mood["avoid"]) - set(genres)))
        if mood.get("year_max") and not text.strip():
            st.caption(f"Tip: {quick} works best with older movies. Lower 'Released after' to see them.")

    if not text.strip() and not quick:
        st.info("Describe your mood in your own words, or tap a quick mood above.")
    elif not genres:
        st.warning("I couldn't pick up a mood from that. Try words like funny, dark, romantic, scary, space, "
                   "family, inspiring, or tap a quick mood.")
    elif TMDB_KEY:
        live_genres = tmdb_catalog.to_tmdb_genres(genres)
        live_avoid = [g for g in tmdb_catalog.to_tmdb_genres(avoid) if g not in live_genres]
        if mood_go:
            start_search("mood", TMDB_KEY, dict(
                category=m_cat, year_from=m_years[0], year_to=m_years[1], min_votes=m_votes,
                min_rating=min_rating, genres=live_genres, avoid_genres=live_avoid,
            ))
        if "mood_live" in st.session_state:
            st.markdown(
                f'<div class="mf-banner">Genres: {html.escape(", ".join(live_genres[:5]))}'
                + (f' · Understood: {html.escape(", ".join(hits))}' if hits else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown('<div class="mf-section">Your Recommendations</div>', unsafe_allow_html=True)
            render_live("mood", TMDB_KEY)
        else:
            st.caption("Press **Find my movies** to see the top 10 for this mood, then **Explore more**.")
    else:
        need_key_message()
        picks = rec.recommend_by_genres(
            genres, avoid, n=n_mood, year_min=released_after, min_avg_rating=min_rating, min_ratings=30,
        )
        st.markdown(
            f'<div class="mf-banner">{len(picks)} films matched · Genres: {html.escape(", ".join(genres[:5]))}'
            + (f' · Understood: {html.escape(", ".join(hits))}' if hits else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="mf-section">Your Recommendations</div>', unsafe_allow_html=True)
        render_grid(picks)

# ---- Browse by Genre ------------------------------------------------------
with tab_genre:
    st.subheader("Browse by Genre + Filters")
    c1, c2, c3, c4 = st.columns(4)
    genre = c1.selectbox("Genre", BROWSE_GENRES, index=BROWSE_GENRES.index("Romance"), key="g_genre")
    g_rating = c2.slider("Min rating ⭐", 0.0, 9.0, 6.0, 0.1, key="g_rating")
    g_year = c3.slider("Min year", YEAR_MIN, YEAR_MAX, 1990, key="g_year")
    sort_label = c4.selectbox("Sort by", ["Highest rated", "Most popular", "Newest"], key="g_sort")
    g_n = st.select_slider("How many movies", [5, 10, 15, 20, 30], value=15, key="g_n")

    res = rec.recommend_by_genres(
        [genre], n=g_n, year_min=g_year, min_avg_rating=g_rating, min_ratings=30,
        sort={"Highest rated": "best", "Most popular": "popular", "Newest": "newest"}[sort_label],
    )
    st.markdown(f'<div class="mf-banner">{len(res)} movies · {html.escape(genre)}</div>', unsafe_allow_html=True)
    render_grid(res)

# ---- Find Similar ---------------------------------------------------------
with tab_sim:
    st.subheader("Find movies like one you love")
    popular_first = list(rec.movies.assign(n=rec.n_ratings).sort_values("n", ascending=False).index)
    chosen = st.selectbox(
        "Pick a movie (type to search)", popular_first,
        format_func=lambda i: f"{rec.movies.at[i, 'title']} ({rec.movies.at[i, 'year']})", key="s_movie",
    )
    c1, c2 = st.columns(2)
    s_rating = c1.slider("Min rating ⭐", 0.0, 9.0, 6.0, 0.1, key="s_rating")
    s_n = c2.select_slider("How many movies", [5, 10, 15, 20], value=10, key="s_n")
    with st.expander("Advanced: tune the model"):
        alpha = st.slider("Collaborative vs genre-content weight", 0.0, 1.0, float(defaults["alpha"]), 0.05,
                          help="1 = only 'viewers who liked this also liked that', 0 = only genres.")
        beta = st.slider("Popularity weight", 0.0, 1.0, 0.15, 0.05,
                         help="Higher = safer, better-known picks. (The evaluation tab uses a higher value "
                              "because it predicts what a user watches next; for 'similar movies' a low value fits better.)")

    st.button("🔎 Find Similar", type="primary")
    movie_row = rec.movies.iloc[chosen]
    recs = rec.recommend_similar(
        [movie_row["movie_id"]], n=s_n, alpha=alpha, beta=beta, min_rating=s_rating,
    )

    base = rec._format([chosen]).iloc[0]
    base["why"] = ""
    base_info, _ = posters.get_posters([str(movie_row["movie_id"])], TMDB_KEY)
    left, right = st.columns([1, 4])
    left.markdown(card_html(base, base_info.get(str(movie_row["movie_id"]))), unsafe_allow_html=True)
    right.markdown(f"### Because you like: {html.escape(movie_row['title'])}")
    right.caption(f"{movie_row['genres'].replace('|', ' · ')} · {movie_row['year']} · ★ {base['avg_rating']:.1f}")
    if base_info.get(str(movie_row["movie_id"]), {}).get("overview"):
        right.write(base_info[str(movie_row["movie_id"])]["overview"])

    st.markdown('<div class="mf-section">You Might Also Like</div>', unsafe_allow_html=True)
    if not recs.empty:
        recs["why"] = recs["match"].map(lambda m: f"{m}% match")
    render_grid(recs)

# ---- Model evaluation -----------------------------------------------------
with tab_eval:
    ev = get_eval()
    if ev is None:
        st.warning("No evaluation file found. Run `python evaluate.py` first.")
    else:
        p = ev["protocol"]
        st.subheader("How well does it work?")
        st.write(
            f"For each of **{p['test_users']:,} test users** we hid the last movie they liked "
            f"(rating 7 or more), asked each model to rank all **{p['n_movies']:,}** movies, and checked "
            f"whether the hidden movie appeared in the **top {p['k']}**. The models never saw the hidden ratings."
        )
        res_df = pd.DataFrame(ev["results"]).T
        res_df.columns = ["Hit rate @10", "NDCG @10", "Coverage @10"]
        final = res_df.loc[[i for i in res_df.index if i.startswith("Final")][0]]
        pop = res_df.loc["Popularity (weighted rating)"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Final model hit rate @10", f"{final['Hit rate @10'] * 100:.1f}%",
                  f"{(final['Hit rate @10'] / pop['Hit rate @10'] - 1) * 100:+.0f}% vs popularity")
        m2.metric("Final model NDCG @10", f"{final['NDCG @10']:.3f}",
                  f"{(final['NDCG @10'] / pop['NDCG @10'] - 1) * 100:+.0f}% vs popularity")
        m3.metric("Catalogue coverage @10", f"{final['Coverage @10'] * 100:.1f}%")

        st.dataframe(
            res_df.style.format({"Hit rate @10": "{:.2%}", "NDCG @10": "{:.4f}", "Coverage @10": "{:.1%}"}),
            width="stretch",
        )
        st.bar_chart(res_df["Hit rate @10"])
        st.markdown(
            f"""
**Reading the table**
- **Hit rate @10**: share of users whose hidden movie was in the top 10. With {p['n_movies']:,} movies, random guessing gets about 0.1%.
- **NDCG @10**: like hit rate, but a higher position counts more.
- **Coverage @10**: how much of the catalogue gets recommended to anyone. Lower means the model leans on popular movies.
- The final settings (collaborative weight **{ev['best_params']['alpha']}**, popularity weight **{ev['best_params']['beta']}**) were picked on **{p['val_users']:,} separate validation users**, not on the test users.
"""
        )
        with st.expander("Validation grid (all alpha / beta combinations)"):
            st.dataframe(pd.DataFrame(ev["validation_grid"]), width="stretch")

# ---- Data explorer --------------------------------------------------------
with tab_data:
    st.subheader("Dataset overview")
    c1, c2 = st.columns(2)
    c1.write("**Rating distribution**")
    c1.bar_chart(rec.ratings["rating"].value_counts().sort_index())

    movies = rec.movies.copy()
    movies["decade"] = (movies["year"] // 10 * 10).astype(str) + "s"
    c2.write("**Movies per decade**")
    c2.bar_chart(movies["decade"].value_counts().sort_index())

    c1.write("**Movies per genre**")
    c1.bar_chart(GENRE_COUNTS)

    per_user = rec.ratings.groupby("user_id").size()
    buckets = pd.cut(per_user, [0, 20, 50, 100, 200, 500, 5000]).value_counts().sort_index()
    buckets.index = buckets.index.astype(str)
    c2.write("**Ratings per user (how active users are)**")
    c2.bar_chart(buckets)

    st.write("**Top rated movies** (weighted rating, so tiny sample sizes don't dominate)")
    top = movies.assign(
        weighted_rating=rec.weighted_rating.round(2),
        avg_rating=rec.avg_rating.round(2),
        n_ratings=rec.n_ratings.astype(int),
    ).sort_values("weighted_rating", ascending=False).head(25)
    st.dataframe(
        top[["title", "year", "genres", "avg_rating", "n_ratings", "weighted_rating"]],
        width="stretch", hide_index=True,
    )