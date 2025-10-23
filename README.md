🎬 MovieLens Recommender Systems from Scratch

This project is a hands-on implementation of different state-of-the-art recommender system algorithms on the MovieLens 1M dataset. The goal is to build, compare, and benchmark multiple recommendation approaches using a reproducible and modular pipeline.

🚀 Project Overview

We implement and evaluate several recommender system techniques from scratch:

1- Content-Based Filtering

 * Leverages movie metadata (e.g., genres, tags) to recommend similar items.

 * Similarity computed using cosine similarity, TF-IDF, or Jaccard distance.

2- Collaborative Filtering

 * User-based CF using Pearson correlation or cosine similarity between users.

 * Item-based CF to recommend items similar to those a user has liked.

3- Neural Collaborative Filtering (NCF)

 * Deep learning approach using user and item embeddings.

 * Implements matrix factorization with neural layers for more flexible patterns.

4- Evaluation & Benchmarking

 * Metrics include Precision@K, Recall@K, NDCG@K.

 * Leave-One-Out (LOO) cross-validation to simulate realistic recommendation scenarios.

Benchmarked all algorithms on the same dataset to highlight strengths and weaknesses.