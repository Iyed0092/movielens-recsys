import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_DIR = "./data/processed"
MODEL_DIR = "./models"

os.makedirs(MODEL_DIR, exist_ok=True)

item_features = np.load(os.path.join(PROCESSED_DIR, "item_features.npy"))

similarity_matrix = cosine_similarity(item_features)

np.save(os.path.join(MODEL_DIR, "content_similarity.npy"), similarity_matrix)

print("Content-based similarity matrix saved in 'models/content_similarity.npy'.")
