import numpy as np
import pandas as pd
import os
from .recommend_content import recommend
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = "./data/processed"

ratings = pd.read_csv(os.path.join(PROCESSED_DIR, "ratings.csv"))
movies = pd.read_csv(os.path.join(PROCESSED_DIR, "movies.csv"))

def evaluate(top_k=10):
    user_groups = ratings.groupby("userId")
    ratings_sorted = ratings.sort_values(
        by=["userId", "timestamp", "rating"],
        ascending=[True, False, False]  # timestamp desc, rating desc
    )
    positive_interactions = ratings_sorted[ratings_sorted["rating"] >= 4]

    # Take the **first row per user** (most recent positive)
    latest_positive = positive_interactions.groupby("userId").first().reset_index()

    # Merge latest_positive timestamps to ratings
    ratings_with_latest = ratings.merge(
        latest_positive[['userId', 'timestamp']], 
        on='userId', 
        suffixes=('', '_latest')
    )

    # Keep only ratings **before the latest positive**
    train_ratings = ratings_with_latest[ratings_with_latest['timestamp'] < ratings_with_latest['timestamp_latest']]

    train_ratings = train_ratings.drop(columns=['timestamp_latest'])
    
    train_merged = train_ratings.merge(
    movies, on='movieId', how='left')

    feature_cols = movies.columns.drop(['movieId', 'title'])  # all genre columns
    for col in feature_cols:
        train_merged[col] = train_merged[col] * train_merged['rating']

    user_profiles = train_merged.groupby('userId')[feature_cols].sum()
    rating_sum = train_ratings.groupby('userId')['rating'].sum()
    user_profiles = user_profiles.div(rating_sum, axis=0)


    user_seen = train_ratings.groupby('userId')['movieId'].apply(set).to_dict()


    recommendations = {}

    for user_id, user_vector in user_profiles.iterrows():
        # Movies not seen by the user
        seen = user_seen.get(user_id, set())
        candidate_movies = movies[~movies['movieId'].isin(seen)]
        
        # Compute cosine similarity
        sim = cosine_similarity(
            user_vector.values.reshape(1, -1),  # 1 x features
            candidate_movies[feature_cols].values  # n_movies x features
        ).flatten()
        
        # Get top K movies
        top_indices = np.argsort(sim)[-top_k:][::-1]  # descending order
        top_movies = candidate_movies.iloc[top_indices]['movieId'].tolist()
        
        # Store recommendations
        recommendations[user_id] = top_movies

    hit_ratio_at_k(recommendations, latest_positive, top_k) 


def hit_ratio_at_k(recommendations, latest_positive, k):   
    hits = []
    for _, row in latest_positive.iterrows():
        user_id = row['userId']
        test_movie = row['movieId']
        
        top_10 = recommendations.get(user_id, [])
        
        hit = 1 if test_movie in top_10 else 0
        hits.append(hit)

    HR_10 = sum(hits) / len(hits)
    print(f"Hit Ratio @10: {HR_10:.4f}")

if __name__ == "__main__":
    metrics = evaluate(top_k=10)
    print(metrics)
