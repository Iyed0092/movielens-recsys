import numpy as np
import pandas as pd
import os

PROCESSED_DIR = "./data/processed"
MODEL_DIR = "./models"

ratings = pd.read_csv(os.path.join(PROCESSED_DIR, "ratings.csv"))
movies = pd.read_csv(os.path.join(PROCESSED_DIR, "movies.csv"))
similarity_matrix = np.load(os.path.join(MODEL_DIR, "content_similarity.npy"))

def recommend(user_id, top_k=10):
    user_ratings = ratings[ratings['userId'] == user_id]
    rated_items = user_ratings['movieId'].values
    scores = np.zeros(similarity_matrix.shape[0])
    
    for item_id, rating in zip(user_ratings['movieId'], user_ratings['rating']):
        item_index = np.where(movies['movieId'].values == item_id)[0][0]
        scores += similarity_matrix[item_index] * rating
    
    for item_id in rated_items:
        item_index = np.where(movies['movieId'].values == item_id)[0][0]
        scores[item_index] = -1e9
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    recommended_movies = movies.iloc[top_indices]
    return recommended_movies[['movieId','title']]

if __name__ == "__main__":
    user_id = 1
    top_movies = recommend(user_id, top_k=10)
    print(top_movies)
