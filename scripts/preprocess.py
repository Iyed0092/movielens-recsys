import pandas as pd
import numpy as np
import os

RAW_DIR = "./data/raw"
PROCESSED_DIR = "./data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

movies = pd.read_csv(os.path.join(RAW_DIR, "movies.dat"), sep="::", engine='python', header=None, names=['movieId','title','genres'], encoding="latin-1")
ratings = pd.read_csv(os.path.join(RAW_DIR, "ratings.dat"), sep="::", engine='python', header=None, names=['userId','movieId','rating','timestamp'], encoding="latin-1")

movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))


all_genres = sorted(list({genre for sublist in movies['genres'] for genre in sublist}))

print(all_genres)

for genre in all_genres:
    movies[genre] = movies['genres'].apply(lambda x: int(genre in x))

movies = movies.drop(columns=['genres'])
movies.to_csv(os.path.join(PROCESSED_DIR, "movies.csv"), index=False)
ratings.to_csv(os.path.join(PROCESSED_DIR, "ratings.csv"), index=False)

item_features = movies[all_genres].values
np.save(os.path.join(PROCESSED_DIR, "item_features.npy"), item_features)

print("Preprocessing complete. Processed files saved in 'data/processed'.")
