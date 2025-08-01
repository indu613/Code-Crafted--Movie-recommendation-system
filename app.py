import streamlit as st
import pandas as pd
import numpy as np
import pickle
from surprise import SVD, NMF, KNNBasic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Genre mapping from ID to name
genre_map = {
    0: "Action", 1: "Adventure", 2: "Animation", 3: "Children",
    4: "Comedy", 5: "Crime", 6: "Documentary", 7: "Drama",
    8: "Fantasy", 9: "Film-Noir", 10: "Horror", 11: "Musical",
    12: "Mystery", 13: "Romance", 14: "Sci-Fi", 15: "Thriller",
    16: "War", 17: "Western"
}

# Load data
movies_df = pd.read_csv("movieLens/movies.csv")
ratings_df = pd.read_csv("movieLens/ratings.csv")

movies_df['poster_url'] = movies_df['poster_url'].fillna("https://via.placeholder.com/180x270?text=No+Image")

# Map genre IDs to names
def decode_genres(genre_str):
    if pd.isna(genre_str): return ""
    ids = map(int, genre_str.split(","))
    return ", ".join([genre_map.get(i, str(i)) for i in ids])

movies_df['genres'] = movies_df['movie_genres'].fillna("").apply(decode_genres)

# TF-IDF for content-based filtering
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b")
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_df.index, index=movies_df['movie_title']).drop_duplicates()

movieId_to_title = dict(zip(movies_df['movie_id'], movies_df['movie_title']))
title_to_movieId = dict(zip(movies_df['movie_title'], movies_df['movie_id']))
movie_indices = pd.Series(movies_df.index, index=movies_df['movie_id'])

# Load models
with open("svd_model.pkl", "rb") as f: svd_model = pickle.load(f)
with open("nmf_model.pkl", "rb") as f: nmf_model = pickle.load(f)
with open("user_knn.pkl", "rb") as f: user_knn = pickle.load(f)
with open("item_knn.pkl", "rb") as f: item_knn = pickle.load(f)
with open("trainset.pkl", "rb") as f: trainset = pickle.load(f)

# Hybrid recommender
def hybrid_recommender(user_id, top_n=10):
    rated_ids = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist()
    unseen_ids = list(set(movies_df['movie_id'].tolist()) - set(rated_ids))

    svd_preds = [(mid, svd_model.predict(user_id, mid).est) for mid in unseen_ids]
    svd_preds = sorted(svd_preds, key=lambda x: x[1], reverse=True)[:top_n]

    recommended = []
    for mid, score in svd_preds:
        row = movies_df[movies_df['movie_id'] == mid].iloc[0]
        recommended.append((row['movie_title'], round(score, 2), row['genres'], row['poster_url']))

    if len(recommended) < top_n and rated_ids:
        last_rated = rated_ids[-1]
        idx = movie_indices.get(last_rated)
        if idx is not None:
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            for i, sim in sim_scores:
                row = movies_df.iloc[i]
                if row['movie_id'] not in rated_ids and row['movie_title'] not in [r[0] for r in recommended]:
                    recommended.append((row['movie_title'], round(sim, 2), row['genres'], row['poster_url']))
                if len(recommended) >= top_n:
                    break

    return recommended

# Content-based recommendation
def get_content_recommendations(title, top_n=10):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_idxs = [i[0] for i in sim_scores[1:top_n+1]]
    return movies_df.iloc[movie_idxs][['movie_title', 'genres', 'poster_url']].values.tolist()

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎥 Personalized Movie Recommendation System")
st.markdown("Get recommendations based on trained models.")

# Sidebar for model selection and top-rated toggle
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["🎯 Personalized", "🍿 Browse Top Rated"])

if section == "🎯 Personalized":
    user_id = st.sidebar.selectbox("Select User ID", sorted(ratings_df['user_id'].unique()))
    model_type = st.sidebar.selectbox("Select Recommendation Type", ["User-Based CF", "Item-Based CF", "SVD", "NMF", "Content-Based"])
    top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# Display results
def display_movie_cards(movie_data):
    card_css = """
        <style>
        .stColumn {
            padding: 0.2rem 0.5rem !important;
        }
        .element-container {
            margin-bottom: 0.5rem;
        }
        </style>
    """
    st.markdown(card_css, unsafe_allow_html=True)

    cols = st.columns([1, 1, 1, 1])  # 4 columns
    for i, (title, score, genres, poster_url) in enumerate(movie_data):
        with cols[i % 4]:  # Use modulo 4 since we have 4 columns
            st.image(poster_url, width=160)
            st.markdown(f"**{title}**", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 13px;'>Genres: <i>{genres}</i></span>", unsafe_allow_html=True)
            if score is not None:
                st.markdown(f"<span style='font-size: 13px;'>Predicted Rating: {score}</span>", unsafe_allow_html=True)

if section == "🎯 Personalized":
    if model_type == "Content-Based":
        movie_title = st.text_input("Enter a movie title", "Toy Story (1995)")
        if st.button("🎬 Get Content-Based Recommendations"):
            results = get_content_recommendations(movie_title, top_n)
            if results:
                st.subheader(f"Top {top_n} Movies Similar to '{movie_title}'")
                
                cols = st.columns(4)  # Create 4 columns
                for i, row in enumerate(results):
                    with cols[i % 4]:  # Spread over 4 columns
                        if isinstance(row[2], str) and row[2].startswith("http"):
                            st.image(row[2], width=180)
                        else:
                            st.image("https://via.placeholder.com/180x270?text=No+Image", width=180)
                        st.markdown(f"**{row[0]}**")
                        st.markdown(f"Genres: _{row[1]}_")
            else:
                st.warning("No recommendations found.")


    elif model_type == "Hybrid (SVD + Content)":
        if st.sidebar.button("🌿 Get Hybrid Recommendations"):
            hybrid_results = hybrid_recommender(user_id, top_n)
            st.subheader(f"Top {top_n} Hybrid Recommendations for User {user_id}")
            display_movie_cards(hybrid_results)

    else:
        if model_type == "User-Based CF":
            model = user_knn
        elif model_type == "Item-Based CF":
            model = item_knn
        elif model_type == "SVD":
            model = svd_model
        elif model_type == "NMF":
            model = nmf_model

        def get_top_n(model, user_id, trainset, n=10):
            try:
                inner_id = trainset.to_inner_uid(user_id)
            except ValueError:
                return []

            movie_ids = trainset.all_items()
            predictions = []

            for iid in movie_ids:
                raw_iid = trainset.to_raw_iid(iid)
                pred = model.predict(user_id, raw_iid)
                predictions.append((raw_iid, pred.est))

            predictions.sort(key=lambda x: x[1], reverse=True)
            top_n = predictions[:n]

            results = []
            for mid, score in top_n:
                row = movies_df[movies_df['movie_id'] == int(mid)].iloc[0]
                results.append((row['movie_title'], round(score, 2), row['genres'], row['poster_url']))

            return results

        if st.sidebar.button("🌟 Get Recommendations"):
            recs = get_top_n(model, user_id, trainset, top_n)
            st.subheader(f"Top {top_n} Recommendations for User {user_id} using {model_type}")
            display_movie_cards(recs)

if section == "🍿 Browse Top Rated":
    st.subheader("🍿 Top Rated Movies in Dataset")
    # 👇 Define top_n before using it
    top_n = st.sidebar.slider("Number of Top Rated Movies", min_value=5, max_value=50, value=20, step=5)
    
    top_movies = ratings_df.groupby('movie_id')['user_rating'].mean().sort_values(ascending=False).head(top_n).reset_index()
    top_movies = top_movies.merge(movies_df, on='movie_id')
    
    cols = st.columns(4)
    for i, row in top_movies.iterrows():
        with cols[i % 4]:
            poster = row['poster_url']
            if isinstance(poster, str) and poster.startswith("http"):
                st.image(poster, width=180)
            else:
                st.image("https://via.placeholder.com/180x270?text=No+Image", width=180)

            st.markdown(f"**{row['movie_title']}**")
            st.markdown(f"Genres: _{row['movie_genres']}_" if pd.notna(row['movie_genres']) else "_Unknown_")
            st.markdown(f"Average Rating: {round(row['user_rating'], 2)} ⭐")
