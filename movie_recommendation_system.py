"""
Movie Recommendation System
===========================
A complete Python implementation of collaborative filtering for movie recommendations.
This system implements both user-based and item-based collaborative filtering algorithms.

Author: Data Science Project
Date: December 2024
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import argparse
import requests
import zipfile
import os
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: DATASET CREATION AND LOADING
# =============================================================================

def create_sample_dataset():
    """
    Create a sample dataset of movies and user ratings.
    
    Returns:
        movies_df: DataFrame containing movie information
        ratings_df: DataFrame containing user ratings for movies
    """
    
    # Sample movie data with movie_id, title, and genre
    movies_data = {
        'movie_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'title': [
            'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
            'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
            'Goodfellas', 'Fight Club', 'Interstellar', 'The Silence of the Lambs',
            'Saving Private Ryan', 'The Green Mile', 'Gladiator', 'Titanic'
        ],
        'genre': [
            'Drama', 'Crime', 'Action', 'Crime', 'Drama', 'Sci-Fi', 'Sci-Fi',
            'Crime', 'Drama', 'Sci-Fi', 'Thriller', 'War', 'Drama', 'Action', 'Romance'
        ]
    }
    
    # Sample user ratings data
    # Rating scale: 1-5 (1 = poor, 5 = excellent)
    # NaN means the user hasn't rated the movie
    ratings_data = {
        'user_id': [1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4,
                    5, 5, 5, 5, 5, 5, 5,
                    6, 6, 6, 6, 6,
                    7, 7, 7, 7, 7, 7,
                    8, 8, 8, 8, 8, 8],
        'movie_id': [1, 2, 3, 5, 6, 9, 13,
                     1, 3, 4, 6, 7, 10,
                     2, 4, 5, 8, 11, 12, 14,
                     1, 3, 6, 7, 10, 15,
                     2, 4, 5, 8, 9, 11, 13,
                     1, 5, 6, 10, 15,
                     2, 3, 4, 7, 8, 14,
                     1, 5, 9, 12, 13, 15],
        'rating': [5, 4, 5, 4, 5, 4, 5,
                   4, 5, 4, 5, 5, 4,
                   5, 5, 3, 5, 4, 4, 5,
                   5, 4, 5, 4, 5, 3,
                   4, 5, 4, 5, 5, 4, 4,
                   5, 5, 4, 4, 4,
                   5, 4, 5, 5, 4, 5,
                   4, 5, 4, 5, 5, 3]
    }
    
    movies_df = pd.DataFrame(movies_data)
    ratings_df = pd.DataFrame(ratings_data)
    
    print("=" * 60)
    print("SAMPLE DATASET CREATED")
    print("=" * 60)
    print(f"\nNumber of movies: {len(movies_df)}")
    print(f"Number of users: {ratings_df['user_id'].nunique()}")
    print(f"Number of ratings: {len(ratings_df)}")
    print(f"Average rating: {ratings_df['rating'].mean():.2f}")
    
    return movies_df, ratings_df


def download_movielens_100k(data_dir: str = "data") -> None:
    """
    Download and extract MovieLens 100K dataset.
    
    Args:
        data_dir: Directory to store the dataset
    """
    
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    extract_path = os.path.join(data_dir, "ml-100k")
    
    if os.path.exists(extract_path):
        print("MovieLens 100K dataset already exists.")
        return
    
    print("Downloading MovieLens 100K dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        print("MovieLens 100K dataset downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading MovieLens dataset: {e}")
        print("Falling back to sample dataset...")


def load_movielens_100k(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens 100K dataset.
    
    Args:
        data_dir: Directory containing the dataset
    
    Returns:
        movies_df: DataFrame containing movie information
        ratings_df: DataFrame containing user ratings for movies
    """
    
    ml_path = os.path.join(data_dir, "ml-100k")
    
    if not os.path.exists(ml_path):
        download_movielens_100k(data_dir)
    
    if not os.path.exists(ml_path):
        print("Failed to download MovieLens dataset. Using sample dataset instead.")
        return create_sample_dataset()
    
    try:
        # Load ratings data
        ratings_path = os.path.join(ml_path, "u.data")
        ratings_df = pd.read_csv(
            ratings_path,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            usecols=['user_id', 'movie_id', 'rating']
        )
        
        # Load movie data
        movies_path = os.path.join(ml_path, "u.item")
        movies_df = pd.read_csv(
            movies_path,
            sep='|',
            encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
                  [f'genre_{i}' for i in range(19)],
            usecols=['movie_id', 'title'] + [f'genre_{i}' for i in range(19)]
        )
        
        # Extract primary genre
        genre_cols = [f'genre_{i}' for i in range(19)]
        genre_names = [
            'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
            'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        
        def get_primary_genre(row):
            for i, genre_col in enumerate(genre_cols):
                if row[genre_col] == 1:
                    return genre_names[i]
            return 'unknown'
        
        movies_df['genre'] = movies_df.apply(get_primary_genre, axis=1)
        movies_df = movies_df[['movie_id', 'title', 'genre']]
        
        # Filter to users and movies with sufficient ratings
        user_counts = ratings_df['user_id'].value_counts()
        movie_counts = ratings_df['movie_id'].value_counts()
        
        # Keep users with at least 20 ratings and movies with at least 20 ratings
        active_users = user_counts[user_counts >= 20].index
        popular_movies = movie_counts[movie_counts >= 20].index
        
        ratings_df = ratings_df[
            (ratings_df['user_id'].isin(active_users)) &
            (ratings_df['movie_id'].isin(popular_movies))
        ]
        
        movies_df = movies_df[movies_df['movie_id'].isin(popular_movies)]
        
        # Take a subset for faster computation (can be removed for full dataset)
        top_users = ratings_df['user_id'].value_counts().head(100).index
        top_movies = ratings_df['movie_id'].value_counts().head(200).index
        
        ratings_df = ratings_df[
            (ratings_df['user_id'].isin(top_users)) &
            (ratings_df['movie_id'].isin(top_movies))
        ]
        
        movies_df = movies_df[movies_df['movie_id'].isin(top_movies)]
        
        print("=" * 60)
        print("MOVIELENS 100K DATASET LOADED")
        print("=" * 60)
        print(f"\nNumber of movies: {len(movies_df)}")
        print(f"Number of users: {ratings_df['user_id'].nunique()}")
        print(f"Number of ratings: {len(ratings_df)}")
        print(f"Average rating: {ratings_df['rating'].mean():.2f}")
        print(f"Rating range: {ratings_df['rating'].min()}-{ratings_df['rating'].max()}")
        
        return movies_df, ratings_df
        
    except Exception as e:
        print(f"Error loading MovieLens dataset: {e}")
        print("Using sample dataset instead.")
        return create_sample_dataset()


# =============================================================================
# SECTION 2: DATA PREPROCESSING
# =============================================================================

def create_user_item_matrix(ratings_df):
    """
    Create a user-item matrix from the ratings DataFrame.
    
    The matrix has users as rows and movies as columns.
    Each cell contains the rating given by a user to a movie.
    Missing ratings are filled with 0.
    
    Args:
        ratings_df: DataFrame with columns ['user_id', 'movie_id', 'rating']
    
    Returns:
        user_item_matrix: Pivot table with users as rows, movies as columns
    """
    
    # Create the user-item matrix using pivot table
    user_item_matrix = ratings_df.pivot_table(
        index='user_id',      # Rows are users
        columns='movie_id',   # Columns are movies
        values='rating',      # Values are ratings
        fill_value=0          # Fill missing ratings with 0
    )
    
    print("\n" + "=" * 60)
    print("USER-ITEM MATRIX CREATED")
    print("=" * 60)
    print(f"\nMatrix shape: {user_item_matrix.shape}")
    print(f"(Users: {user_item_matrix.shape[0]}, Movies: {user_item_matrix.shape[1]})")
    
    return user_item_matrix


def normalize_ratings(user_item_matrix):
    """
    Normalize ratings by subtracting each user's mean rating.
    
    This helps account for different rating scales used by different users.
    Some users might rate everything high, others might rate everything low.
    
    Args:
        user_item_matrix: User-item rating matrix
    
    Returns:
        normalized_matrix: Matrix with mean-centered ratings
    """
    
    # Calculate mean rating for each user (ignoring zeros)
    user_means = user_item_matrix.replace(0, np.nan).mean(axis=1)
    
    # Subtract user mean from each rating
    normalized_matrix = user_item_matrix.sub(user_means, axis=0)
    
    # Replace NaN values (from the subtraction of mean from 0) with 0
    normalized_matrix = normalized_matrix.fillna(0)
    
    return normalized_matrix, user_means


# =============================================================================
# SECTION 3: SIMILARITY COMPUTATION
# =============================================================================

def compute_user_similarity(user_item_matrix):
    """
    Compute similarity between users using cosine similarity.
    
    Cosine similarity measures the angle between two user rating vectors.
    Users with similar rating patterns will have high similarity scores.
    
    Args:
        user_item_matrix: User-item rating matrix
    
    Returns:
        user_similarity_df: DataFrame containing user-user similarity scores
    """
    
    # Compute cosine similarity between all users
    user_similarity = cosine_similarity(user_item_matrix)
    
    # Convert to DataFrame for easier manipulation
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    print("\n" + "=" * 60)
    print("USER SIMILARITY MATRIX COMPUTED")
    print("=" * 60)
    print("\nSample of user similarity matrix:")
    print(user_similarity_df.round(3).head())
    
    return user_similarity_df


def compute_item_similarity(user_item_matrix):
    """
    Compute similarity between items (movies) using cosine similarity.
    
    Items that are rated similarly by users will have high similarity scores.
    
    Args:
        user_item_matrix: User-item rating matrix
    
    Returns:
        item_similarity_df: DataFrame containing item-item similarity scores
    """
    
    # Transpose the matrix to have items as rows
    item_user_matrix = user_item_matrix.T
    
    # Compute cosine similarity between all items
    item_similarity = cosine_similarity(item_user_matrix)
    
    # Convert to DataFrame
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )
    
    print("\n" + "=" * 60)
    print("ITEM SIMILARITY MATRIX COMPUTED")
    print("=" * 60)
    print("\nSample of item similarity matrix:")
    print(item_similarity_df.round(3).iloc[:5, :5])
    
    return item_similarity_df


# =============================================================================
# SECTION 4: USER-BASED COLLABORATIVE FILTERING
# =============================================================================

def predict_rating_user_based(user_id, movie_id, user_item_matrix, user_similarity_df, k=3):
    """
    Predict a user's rating for a movie using user-based collaborative filtering.
    
    The prediction is based on the weighted average of ratings from similar users.
    
    Args:
        user_id: The user for whom to predict the rating
        movie_id: The movie to predict the rating for
        user_item_matrix: User-item rating matrix
        user_similarity_df: User-user similarity matrix
        k: Number of similar users to consider
    
    Returns:
        predicted_rating: The predicted rating for the user-movie pair
    """
    
    # Check if user and movie exist
    if user_id not in user_item_matrix.index:
        return 0
    if movie_id not in user_item_matrix.columns:
        return 0
    
    # Get similarity scores for the target user with all other users
    user_similarities = user_similarity_df.loc[user_id].drop(user_id)
    
    # Get users who have rated this movie (non-zero ratings)
    movie_ratings = user_item_matrix[movie_id]
    users_who_rated = movie_ratings[movie_ratings > 0].index
    users_who_rated = users_who_rated[users_who_rated != user_id]
    
    if len(users_who_rated) == 0:
        return 0
    
    # Filter similarities to only users who rated this movie
    relevant_similarities = user_similarities.loc[users_who_rated]
    
    # Get top k similar users
    top_k_users = relevant_similarities.nlargest(k)
    
    if top_k_users.sum() == 0:
        return 0
    
    # Calculate weighted average of ratings
    weighted_sum = 0
    similarity_sum = 0
    
    for similar_user, similarity in top_k_users.items():
        rating = user_item_matrix.loc[similar_user, movie_id]
        weighted_sum += similarity * rating
        similarity_sum += abs(similarity)
    
    if similarity_sum == 0:
        return 0
    
    predicted_rating = weighted_sum / similarity_sum
    
    return predicted_rating


def get_user_based_recommendations(user_id, user_item_matrix, user_similarity_df, 
                                    movies_df, n_recommendations=5, k=3):
    """
    Get top N movie recommendations for a user using user-based collaborative filtering.
    
    Args:
        user_id: The user ID to get recommendations for
        user_item_matrix: User-item rating matrix
        user_similarity_df: User-user similarity matrix
        movies_df: DataFrame containing movie information
        n_recommendations: Number of recommendations to return
        k: Number of similar users to consider for predictions
    
    Returns:
        recommendations: DataFrame with recommended movies and predicted ratings
    """
    
    print(f"\n{'=' * 60}")
    print(f"USER-BASED RECOMMENDATIONS FOR USER {user_id}")
    print("=" * 60)
    
    # Check if user exists
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found in the dataset.")
        return pd.DataFrame()
    
    # Get movies the user hasn't rated yet
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index.tolist()
    
    print(f"\nUser {user_id} has rated {(user_ratings > 0).sum()} movies")
    print(f"Unrated movies to consider: {len(unrated_movies)}")
    
    if len(unrated_movies) == 0:
        print("User has rated all movies!")
        return pd.DataFrame()
    
    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        predicted_rating = predict_rating_user_based(
            user_id, movie_id, user_item_matrix, user_similarity_df, k
        )
        if predicted_rating > 0:
            predictions.append({
                'movie_id': movie_id,
                'predicted_rating': predicted_rating
            })
    
    if len(predictions) == 0:
        print("No predictions could be made.")
        return pd.DataFrame()
    
    # Create recommendations DataFrame
    recommendations_df = pd.DataFrame(predictions)
    
    # Sort by predicted rating and get top N
    recommendations_df = recommendations_df.sort_values(
        'predicted_rating', ascending=False
    ).head(n_recommendations)
    
    # Merge with movie information
    recommendations_df = recommendations_df.merge(
        movies_df, on='movie_id'
    )[['movie_id', 'title', 'genre', 'predicted_rating']]
    
    print(f"\nTop {n_recommendations} Recommendations:")
    print("-" * 60)
    
    for idx, row in recommendations_df.iterrows():
        print(f"  {row['title']} ({row['genre']}) - Predicted Rating: {row['predicted_rating']:.2f}")
    
    return recommendations_df


# =============================================================================
# SECTION 5: ITEM-BASED COLLABORATIVE FILTERING
# =============================================================================

def predict_rating_item_based(user_id, movie_id, user_item_matrix, item_similarity_df, k=3):
    """
    Predict a user's rating for a movie using item-based collaborative filtering.
    
    The prediction is based on the user's ratings of similar movies.
    
    Args:
        user_id: The user for whom to predict the rating
        movie_id: The movie to predict the rating for
        user_item_matrix: User-item rating matrix
        item_similarity_df: Item-item similarity matrix
        k: Number of similar items to consider
    
    Returns:
        predicted_rating: The predicted rating for the user-movie pair
    """
    
    # Check if user and movie exist
    if user_id not in user_item_matrix.index:
        return 0
    if movie_id not in item_similarity_df.index:
        return 0
    
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    
    # Get movies the user has rated
    rated_movies = user_ratings[user_ratings > 0].index.tolist()
    
    if len(rated_movies) == 0:
        return 0
    
    # Get similarity scores between target movie and rated movies
    if movie_id in rated_movies:
        rated_movies.remove(movie_id)
    
    if len(rated_movies) == 0:
        return 0
    
    movie_similarities = item_similarity_df.loc[movie_id, rated_movies]
    
    # Get top k similar movies
    top_k_movies = movie_similarities.nlargest(k)
    
    if top_k_movies.sum() == 0:
        return 0
    
    # Calculate weighted average of ratings
    weighted_sum = 0
    similarity_sum = 0
    
    for similar_movie, similarity in top_k_movies.items():
        rating = user_ratings[similar_movie]
        weighted_sum += similarity * rating
        similarity_sum += abs(similarity)
    
    if similarity_sum == 0:
        return 0
    
    predicted_rating = weighted_sum / similarity_sum
    
    return predicted_rating


def get_item_based_recommendations(user_id, user_item_matrix, item_similarity_df, 
                                    movies_df, n_recommendations=5, k=3):
    """
    Get top N movie recommendations for a user using item-based collaborative filtering.
    
    Args:
        user_id: The user ID to get recommendations for
        user_item_matrix: User-item rating matrix
        item_similarity_df: Item-item similarity matrix
        movies_df: DataFrame containing movie information
        n_recommendations: Number of recommendations to return
        k: Number of similar items to consider for predictions
    
    Returns:
        recommendations: DataFrame with recommended movies and predicted ratings
    """
    
    print(f"\n{'=' * 60}")
    print(f"ITEM-BASED RECOMMENDATIONS FOR USER {user_id}")
    print("=" * 60)
    
    # Check if user exists
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found in the dataset.")
        return pd.DataFrame()
    
    # Get movies the user hasn't rated yet
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index.tolist()
    
    print(f"\nUser {user_id} has rated {(user_ratings > 0).sum()} movies")
    print(f"Unrated movies to consider: {len(unrated_movies)}")
    
    if len(unrated_movies) == 0:
        print("User has rated all movies!")
        return pd.DataFrame()
    
    # Predict ratings for unrated movies
    predictions = []
    for movie_id in unrated_movies:
        predicted_rating = predict_rating_item_based(
            user_id, movie_id, user_item_matrix, item_similarity_df, k
        )
        if predicted_rating > 0:
            predictions.append({
                'movie_id': movie_id,
                'predicted_rating': predicted_rating
            })
    
    if len(predictions) == 0:
        print("No predictions could be made.")
        return pd.DataFrame()
    
    # Create recommendations DataFrame
    recommendations_df = pd.DataFrame(predictions)
    
    # Sort by predicted rating and get top N
    recommendations_df = recommendations_df.sort_values(
        'predicted_rating', ascending=False
    ).head(n_recommendations)
    
    # Merge with movie information
    recommendations_df = recommendations_df.merge(
        movies_df, on='movie_id'
    )[['movie_id', 'title', 'genre', 'predicted_rating']]
    
    print(f"\nTop {n_recommendations} Recommendations:")
    print("-" * 60)
    
    for idx, row in recommendations_df.iterrows():
        print(f"  {row['title']} ({row['genre']}) - Predicted Rating: {row['predicted_rating']:.2f}")
    
    return recommendations_df


# =============================================================================
# SECTION 6: HYBRID RECOMMENDATION SYSTEM
# =============================================================================

def get_hybrid_recommendations(user_id, user_item_matrix, user_similarity_df, 
                               item_similarity_df, movies_df, n_recommendations=5,
                               user_weight=0.5, item_weight=0.5):
    """
    Get recommendations using a hybrid approach combining user-based and item-based methods.
    
    Args:
        user_id: The user ID to get recommendations for
        user_item_matrix: User-item rating matrix
        user_similarity_df: User-user similarity matrix
        item_similarity_df: Item-item similarity matrix
        movies_df: DataFrame containing movie information
        n_recommendations: Number of recommendations to return
        user_weight: Weight for user-based predictions (0-1)
        item_weight: Weight for item-based predictions (0-1)
    
    Returns:
        recommendations: DataFrame with recommended movies and predicted ratings
    """
    
    print(f"\n{'=' * 60}")
    print(f"HYBRID RECOMMENDATIONS FOR USER {user_id}")
    print(f"(User-based weight: {user_weight}, Item-based weight: {item_weight})")
    print("=" * 60)
    
    # Check if user exists
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found in the dataset.")
        return pd.DataFrame()
    
    # Get movies the user hasn't rated yet
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings == 0].index.tolist()
    
    if len(unrated_movies) == 0:
        print("User has rated all movies!")
        return pd.DataFrame()
    
    # Predict ratings using both methods
    predictions = []
    for movie_id in unrated_movies:
        user_pred = predict_rating_user_based(
            user_id, movie_id, user_item_matrix, user_similarity_df
        )
        item_pred = predict_rating_item_based(
            user_id, movie_id, user_item_matrix, item_similarity_df
        )
        
        # Combine predictions with weights
        if user_pred > 0 and item_pred > 0:
            hybrid_pred = (user_weight * user_pred) + (item_weight * item_pred)
        elif user_pred > 0:
            hybrid_pred = user_pred
        elif item_pred > 0:
            hybrid_pred = item_pred
        else:
            continue
        
        predictions.append({
            'movie_id': movie_id,
            'predicted_rating': hybrid_pred,
            'user_based_pred': user_pred,
            'item_based_pred': item_pred
        })
    
    if len(predictions) == 0:
        print("No predictions could be made.")
        return pd.DataFrame()
    
    # Create recommendations DataFrame
    recommendations_df = pd.DataFrame(predictions)
    
    # Sort by predicted rating and get top N
    recommendations_df = recommendations_df.sort_values(
        'predicted_rating', ascending=False
    ).head(n_recommendations)
    
    # Merge with movie information
    recommendations_df = recommendations_df.merge(
        movies_df, on='movie_id'
    )[['movie_id', 'title', 'genre', 'predicted_rating']]
    
    print(f"\nTop {n_recommendations} Recommendations:")
    print("-" * 60)
    
    for idx, row in recommendations_df.iterrows():
        print(f"  {row['title']} ({row['genre']}) - Predicted Rating: {row['predicted_rating']:.2f}")
    
    return recommendations_df


# =============================================================================
# SECTION 7: UTILITY FUNCTIONS
# =============================================================================

def find_similar_users(user_id, user_similarity_df, n=5):
    """
    Find the most similar users to a given user.
    
    Args:
        user_id: The user to find similar users for
        user_similarity_df: User-user similarity matrix
        n: Number of similar users to return
    
    Returns:
        similar_users: Series of top n similar users with similarity scores
    """
    
    if user_id not in user_similarity_df.index:
        print(f"User {user_id} not found.")
        return pd.Series()
    
    # Get similarities, excluding self-similarity
    similarities = user_similarity_df.loc[user_id].drop(user_id)
    
    # Get top n similar users
    similar_users = similarities.nlargest(n)
    
    print(f"\nTop {n} users similar to User {user_id}:")
    print("-" * 40)
    for other_user, similarity in similar_users.items():
        print(f"  User {other_user}: Similarity = {similarity:.4f}")
    
    return similar_users


def find_similar_movies(movie_id, item_similarity_df, movies_df, n=5):
    """
    Find the most similar movies to a given movie.
    
    Args:
        movie_id: The movie to find similar movies for
        item_similarity_df: Item-item similarity matrix
        movies_df: DataFrame containing movie information
        n: Number of similar movies to return
    
    Returns:
        similar_movies: DataFrame of top n similar movies
    """
    
    if movie_id not in item_similarity_df.index:
        print(f"Movie {movie_id} not found.")
        return pd.DataFrame()
    
    # Get the movie title
    movie_title = movies_df[movies_df['movie_id'] == movie_id]['title'].values[0]
    
    # Get similarities, excluding self-similarity
    similarities = item_similarity_df.loc[movie_id].drop(movie_id)
    
    # Get top n similar movies
    top_n = similarities.nlargest(n)
    
    print(f"\nTop {n} movies similar to '{movie_title}':")
    print("-" * 50)
    
    for similar_movie_id, similarity in top_n.items():
        similar_title = movies_df[movies_df['movie_id'] == similar_movie_id]['title'].values[0]
        similar_genre = movies_df[movies_df['movie_id'] == similar_movie_id]['genre'].values[0]
        print(f"  {similar_title} ({similar_genre}) - Similarity: {similarity:.4f}")
    
    return top_n


def get_user_profile(user_id, user_item_matrix, movies_df):
    """
    Display a user's rating profile.
    
    Args:
        user_id: The user to get the profile for
        user_item_matrix: User-item rating matrix
        movies_df: DataFrame containing movie information
    """
    
    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found.")
        return
    
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0]
    
    print(f"\n{'=' * 60}")
    print(f"USER {user_id} PROFILE")
    print("=" * 60)
    print(f"\nTotal movies rated: {len(rated_movies)}")
    print(f"Average rating: {rated_movies.mean():.2f}")
    print(f"\nRated movies:")
    print("-" * 50)
    
    for movie_id, rating in rated_movies.items():
        movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        print(f"  {movie_info['title']} ({movie_info['genre']}) - Rating: {rating}")


# =============================================================================
# SECTION 8: EVALUATION METRICS
# =============================================================================

def split_ratings_by_user(ratings_df: pd.DataFrame, test_size: float = 0.2, 
                         min_ratings: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into train and test sets per user.
    
    Args:
        ratings_df: DataFrame with user ratings
        test_size: Fraction of ratings to use for testing
        min_ratings: Minimum ratings per user to include in split
    
    Returns:
        train_ratings: Training set
        test_ratings: Test set
    """
    
    train_list = []
    test_list = []
    
    for user_id in ratings_df['user_id'].unique():
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        if len(user_ratings) < min_ratings:
            # If user has too few ratings, put all in training
            train_list.append(user_ratings)
            continue
        
        # Split user's ratings
        train_user, test_user = train_test_split(
            user_ratings, test_size=test_size, random_state=42
        )
        
        train_list.append(train_user)
        test_list.append(test_user)
    
    train_ratings = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    test_ratings = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
    
    return train_ratings, test_ratings


def compute_prediction_metrics(actual_ratings: List[float], 
                              predicted_ratings: List[float]) -> Dict[str, float]:
    """
    Compute MAE and RMSE for rating predictions.
    
    Args:
        actual_ratings: True ratings
        predicted_ratings: Predicted ratings
    
    Returns:
        Dictionary with MAE and RMSE
    """
    
    actual = np.array(actual_ratings)
    predicted = np.array(predicted_ratings)
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    return {'MAE': mae, 'RMSE': rmse}


def compute_ndcg_at_k(recommended_items: List, relevant_items: List, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at k.
    
    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (liked) item IDs
        k: Cut-off rank
    
    Returns:
        NDCG@k score
    """
    
    if not recommended_items or not relevant_items:
        return 0.0
    
    # Take top k recommendations
    recommended_k = recommended_items[:k]
    
    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
    
    # Compute Ideal DCG
    ideal_dcg = 0.0
    for i in range(min(len(relevant_items), k)):
        ideal_dcg += 1.0 / np.log2(i + 2)
    
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg


def compute_recommendation_metrics(user_item_matrix: pd.DataFrame,
                                 user_similarity_df: pd.DataFrame,
                                 item_similarity_df: pd.DataFrame,
                                 test_ratings: pd.DataFrame,
                                 mode: str = 'hybrid',
                                 k: int = 10,
                                 rating_threshold: float = 4.0) -> Dict[str, float]:
    """
    Compute Precision@K, Recall@K, and NDCG@K for recommendations.
    
    Args:
        user_item_matrix: User-item rating matrix
        user_similarity_df: User similarity matrix
        item_similarity_df: Item similarity matrix
        test_ratings: Test set ratings
        mode: Recommendation mode ('user', 'item', 'hybrid')
        k: Number of recommendations to evaluate
        rating_threshold: Threshold for considering a rating as "relevant"
    
    Returns:
        Dictionary with precision, recall, and NDCG metrics
    """
    
    precisions = []
    recalls = []
    ndcgs = []
    
    for user_id in test_ratings['user_id'].unique():
        if user_id not in user_item_matrix.index:
            continue
        
        # Get user's test ratings
        user_test = test_ratings[test_ratings['user_id'] == user_id]
        relevant_items = user_test[user_test['rating'] >= rating_threshold]['movie_id'].tolist()
        
        if not relevant_items:
            continue
        
        # Get recommendations
        try:
            if mode == 'user':
                recs_df = get_user_based_recommendations(
                    user_id, user_item_matrix, user_similarity_df, 
                    pd.DataFrame(), k, 3  # Empty movies_df for speed
                )
            elif mode == 'item':
                recs_df = get_item_based_recommendations(
                    user_id, user_item_matrix, item_similarity_df,
                    pd.DataFrame(), k, 3
                )
            else:  # hybrid
                recs_df = get_hybrid_recommendations(
                    user_id, user_item_matrix, user_similarity_df,
                    item_similarity_df, pd.DataFrame(), k
                )
            
            if recs_df.empty:
                continue
            
            recommended_items = recs_df['movie_id'].tolist()
            
            # Compute metrics
            relevant_recommended = set(recommended_items) & set(relevant_items)
            
            precision = len(relevant_recommended) / len(recommended_items) if recommended_items else 0
            recall = len(relevant_recommended) / len(relevant_items) if relevant_items else 0
            ndcg = compute_ndcg_at_k(recommended_items, relevant_items, k)
            
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)
            
        except Exception:
            continue
    
    if not precisions:
        return {'Precision@K': 0.0, 'Recall@K': 0.0, 'NDCG@K': 0.0}
    
    return {
        f'Precision@{k}': np.mean(precisions),
        f'Recall@{k}': np.mean(recalls),
        f'NDCG@{k}': np.mean(ndcgs)
    }


def evaluate_system(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                   mode: str = 'hybrid', k: int = 10) -> None:
    """
    Evaluate the recommendation system using train/test split.
    
    Args:
        ratings_df: DataFrame with user ratings
        movies_df: DataFrame with movie information
        mode: Recommendation mode to evaluate
        k: Number of recommendations for evaluation
    """
    
    print(f"\n{'=' * 60}")
    print(f"EVALUATING {mode.upper()} RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Split data
    train_ratings, test_ratings = split_ratings_by_user(ratings_df)
    
    print(f"\nDataset split:")
    print(f"Training ratings: {len(train_ratings)}")
    print(f"Test ratings: {len(test_ratings)}")
    
    if test_ratings.empty:
        print("No test data available for evaluation.")
        return
    
    # Create matrices from training data
    train_user_item_matrix = create_user_item_matrix(train_ratings)
    train_user_similarity = compute_user_similarity(train_user_item_matrix)
    train_item_similarity = compute_item_similarity(train_user_item_matrix)
    
    # Evaluate rating predictions
    actual_ratings = []
    predicted_ratings = []
    
    print("\nEvaluating rating predictions...")
    
    for _, row in test_ratings.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        actual_rating = row['rating']
        
        try:
            if mode == 'user':
                pred_rating = predict_rating_user_based(
                    user_id, movie_id, train_user_item_matrix, train_user_similarity
                )
            elif mode == 'item':
                pred_rating = predict_rating_item_based(
                    user_id, movie_id, train_user_item_matrix, train_item_similarity
                )
            else:  # hybrid
                user_pred = predict_rating_user_based(
                    user_id, movie_id, train_user_item_matrix, train_user_similarity
                )
                item_pred = predict_rating_item_based(
                    user_id, movie_id, train_user_item_matrix, train_item_similarity
                )
                
                if user_pred > 0 and item_pred > 0:
                    pred_rating = 0.5 * user_pred + 0.5 * item_pred
                elif user_pred > 0:
                    pred_rating = user_pred
                elif item_pred > 0:
                    pred_rating = item_pred
                else:
                    continue
            
            if pred_rating > 0:
                actual_ratings.append(actual_rating)
                predicted_ratings.append(pred_rating)
                
        except Exception:
            continue
    
    # Compute prediction metrics
    if actual_ratings:
        pred_metrics = compute_prediction_metrics(actual_ratings, predicted_ratings)
        
        print(f"\nRating Prediction Metrics:")
        print(f"  MAE: {pred_metrics['MAE']:.3f}")
        print(f"  RMSE: {pred_metrics['RMSE']:.3f}")
        print(f"  Predictions made: {len(actual_ratings)}/{len(test_ratings)}")
    else:
        print("\nNo rating predictions could be made.")
    
    # Compute recommendation metrics
    print(f"\nEvaluating recommendation quality...")
    
    rec_metrics = compute_recommendation_metrics(
        train_user_item_matrix, train_user_similarity, train_item_similarity,
        test_ratings, mode, k
    )
    
    print(f"\nRecommendation Quality Metrics (Top-{k}):")
    for metric, value in rec_metrics.items():
        print(f"  {metric}: {value:.3f}")


# =============================================================================
# SECTION 9: MOVIE RECOMMENDATION SYSTEM CLASS
# =============================================================================

class MovieRecommendationSystem:
    """
    A complete Movie Recommendation System class that encapsulates all functionality.
    
    This class provides a clean interface for:
    - Loading or creating movie and rating data
    - Computing similarity matrices
    - Generating user-based, item-based, and hybrid recommendations
    - Evaluating system performance
    """
    
    def __init__(self, movies_df: Optional[pd.DataFrame] = None, 
                 ratings_df: Optional[pd.DataFrame] = None, 
                 use_movielens: bool = False, 
                 verbose: bool = True):
        """
        Initialize the recommendation system.
        
        Args:
            movies_df: DataFrame with movie information (optional)
            ratings_df: DataFrame with user ratings (optional)
            use_movielens: Whether to load MovieLens 100K dataset
            verbose: Whether to print initialization messages
        """
        
        # Load dataset
        if use_movielens:
            self.movies_df, self.ratings_df = load_movielens_100k()
        elif movies_df is None or ratings_df is None:
            self.movies_df, self.ratings_df = create_sample_dataset()
        else:
            self.movies_df = movies_df
            self.ratings_df = ratings_df
        
        # Create user-item matrix
        self.user_item_matrix = create_user_item_matrix(self.ratings_df)
        
        # Compute similarity matrices
        if verbose:
            print("\nComputing similarity matrices...")
        
        self.user_similarity_df = compute_user_similarity(self.user_item_matrix)
        self.item_similarity_df = compute_item_similarity(self.user_item_matrix)
        
        if verbose:
            print("\n" + "=" * 60)
            print("MOVIE RECOMMENDATION SYSTEM INITIALIZED")
            print("=" * 60)
    
    def recommend_user_based(self, user_id, n=5, k=3):
        """Get user-based collaborative filtering recommendations."""
        return get_user_based_recommendations(
            user_id, self.user_item_matrix, self.user_similarity_df,
            self.movies_df, n, k
        )
    
    def recommend_item_based(self, user_id, n=5, k=3):
        """Get item-based collaborative filtering recommendations."""
        return get_item_based_recommendations(
            user_id, self.user_item_matrix, self.item_similarity_df,
            self.movies_df, n, k
        )
    
    def recommend_hybrid(self, user_id, n=5, user_weight=0.5, item_weight=0.5):
        """Get hybrid recommendations combining both methods."""
        return get_hybrid_recommendations(
            user_id, self.user_item_matrix, self.user_similarity_df,
            self.item_similarity_df, self.movies_df, n, user_weight, item_weight
        )
    
    def get_similar_users(self, user_id, n=5):
        """Find users similar to the given user."""
        return find_similar_users(user_id, self.user_similarity_df, n)
    
    def get_similar_movies(self, movie_id, n=5):
        """Find movies similar to the given movie."""
        return find_similar_movies(movie_id, self.item_similarity_df, self.movies_df, n)
    
    def show_user_profile(self, user_id):
        """Display a user's rating profile."""
        get_user_profile(user_id, self.user_item_matrix, self.movies_df)
    
    def evaluate(self, mode: str = 'hybrid', k: int = 10):
        """Evaluate the recommendation system."""
        evaluate_system(self.ratings_df, self.movies_df, mode, k)


# =============================================================================
# SECTION 10: CLI INTERFACE
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description='Movie Recommendation System using Collaborative Filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full demo
  python movie_recommendation_system.py
  
  # Get hybrid recommendations for user 1
  python movie_recommendation_system.py --user-id 1 --mode hybrid --top-n 5
  
  # Use MovieLens dataset with evaluation
  python movie_recommendation_system.py --movielens --eval --mode hybrid
  
  # User-based recommendations with custom parameters
  python movie_recommendation_system.py --user-id 3 --mode user --top-n 10 --k 5
        """
    )
    
    parser.add_argument('--user-id', type=int,
                       help='Target user ID for recommendations')
    
    parser.add_argument('--mode', choices=['user', 'item', 'hybrid'],
                       default='hybrid',
                       help='Recommendation algorithm (default: hybrid)')
    
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of recommendations (default: 5)')
    
    parser.add_argument('--k', type=int, default=3,
                       help='Number of neighbors for similarity (default: 3)')
    
    parser.add_argument('--user-weight', type=float, default=0.5,
                       help='User-based weight in hybrid mode (default: 0.5)')
    
    parser.add_argument('--item-weight', type=float, default=0.5,
                       help='Item-based weight in hybrid mode (default: 0.5)')
    
    parser.add_argument('--movielens', action='store_true',
                       help='Use MovieLens 100K dataset instead of sample')
    
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation metrics on train/test split')
    
    return parser.parse_args()


def cli_mode(args):
    """Run in CLI mode with specified arguments."""
    
    # Initialize system
    print("Initializing Movie Recommendation System...")
    recommender = MovieRecommendationSystem(use_movielens=args.movielens, verbose=False)
    
    # Run evaluation if requested
    if args.eval:
        recommender.evaluate(mode=args.mode, k=args.top_n)
        return
    
    # Check if user ID is provided
    if args.user_id is None:
        print("Error: --user-id is required in CLI mode (or omit all arguments for demo)")
        return
    
    # Check if user exists
    if args.user_id not in recommender.user_item_matrix.index:
        available_users = list(recommender.user_item_matrix.index)
        print(f"Error: User {args.user_id} not found.")
        print(f"Available users: {available_users[:10]}{'...' if len(available_users) > 10 else ''}")
        return
    
    # Show user profile
    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATIONS FOR USER {args.user_id}")
    print("=" * 60)
    
    recommender.show_user_profile(args.user_id)
    
    # Get recommendations based on mode
    if args.mode == 'user':
        recommendations = recommender.recommend_user_based(
            args.user_id, n=args.top_n, k=args.k
        )
    elif args.mode == 'item':
        recommendations = recommender.recommend_item_based(
            args.user_id, n=args.top_n, k=args.k
        )
    else:  # hybrid
        recommendations = recommender.recommend_hybrid(
            args.user_id, n=args.top_n, 
            user_weight=args.user_weight, 
            item_weight=args.item_weight
        )
    
    if recommendations.empty:
        print(f"\nNo recommendations could be generated for user {args.user_id}")
    else:
        print(f"\n{args.mode.upper()} MODE RECOMMENDATIONS:")
        print("-" * 40)
        for idx, row in recommendations.iterrows():
            print(f"{row['title']} ({row['genre']}) - Rating: {row['predicted_rating']:.2f}")


# =============================================================================
# SECTION 11: EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

def main():
    """
    Main function demonstrating the complete Movie Recommendation System.
    """
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 15 + "MOVIE RECOMMENDATION SYSTEM DEMO" + " " * 21 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # Initialize the recommendation system
    # This will create sample data and compute similarity matrices
    recommender = MovieRecommendationSystem()
    
    # =========================================================================
    # EXAMPLE 1: Show a user's profile
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("EXAMPLE 1: USER PROFILE")
    print("=" * 70)
    
    recommender.show_user_profile(user_id=1)
    
    # =========================================================================
    # EXAMPLE 2: User-based recommendations
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: USER-BASED COLLABORATIVE FILTERING")
    print("=" * 70)
    
    user_based_recs = recommender.recommend_user_based(user_id=1, n=5)
    
    # =========================================================================
    # EXAMPLE 3: Item-based recommendations
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: ITEM-BASED COLLABORATIVE FILTERING")
    print("=" * 70)
    
    item_based_recs = recommender.recommend_item_based(user_id=1, n=5)
    
    # =========================================================================
    # EXAMPLE 4: Hybrid recommendations
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: HYBRID RECOMMENDATIONS")
    print("=" * 70)
    
    hybrid_recs = recommender.recommend_hybrid(user_id=1, n=5)
    
    # =========================================================================
    # EXAMPLE 5: Find similar users
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("EXAMPLE 5: FINDING SIMILAR USERS")
    print("=" * 70)
    
    similar_users = recommender.get_similar_users(user_id=1, n=3)
    
    # =========================================================================
    # EXAMPLE 6: Find similar movies
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("EXAMPLE 6: FINDING SIMILAR MOVIES")
    print("=" * 70)
    
    # Find movies similar to "The Shawshank Redemption" (movie_id=1)
    similar_movies = recommender.get_similar_movies(movie_id=1, n=5)
    
    # =========================================================================
    # EXAMPLE 7: Recommendations for different users
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("EXAMPLE 7: RECOMMENDATIONS FOR MULTIPLE USERS")
    print("=" * 70)
    
    for user_id in [2, 3, 4]:
        print(f"\n--- Recommendations for User {user_id} ---")
        recommender.show_user_profile(user_id)
        recommender.recommend_hybrid(user_id, n=3)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 20 + "DEMONSTRATION COMPLETE" + " " * 26 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    print("""
    
    SUMMARY OF THE MOVIE RECOMMENDATION SYSTEM:
    ============================================
    
    1. USER-BASED COLLABORATIVE FILTERING:
       - Finds users with similar rating patterns
       - Recommends movies that similar users liked
       - Best for: Finding diverse recommendations
    
    2. ITEM-BASED COLLABORATIVE FILTERING:
       - Finds movies similar to what the user has liked
       - Recommends movies similar to user's favorites
       - Best for: Consistent, predictable recommendations
    
    3. HYBRID APPROACH:
       - Combines both methods with weighted averaging
       - Balances diversity and consistency
       - Generally provides the best overall recommendations
    
    KEY FUNCTIONS:
    - recommend_user_based(user_id, n): Get user-based recommendations
    - recommend_item_based(user_id, n): Get item-based recommendations  
    - recommend_hybrid(user_id, n): Get hybrid recommendations
    - get_similar_users(user_id, n): Find similar users
    - get_similar_movies(movie_id, n): Find similar movies
    - show_user_profile(user_id): View user's rating history
    
    """)
    
    return recommender


# Run the system when the script is executed directly
if __name__ == "__main__":
    import sys
    
    # Check if any arguments were provided
    if len(sys.argv) > 1:
        # CLI mode
        args = parse_arguments()
        cli_mode(args)
    else:
        # Demo mode
        recommender = main()
