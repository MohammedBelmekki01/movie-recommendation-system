"""
Test suite for Movie Recommendation System.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from movie_recommendation_system import (
    create_sample_dataset,
    create_user_item_matrix,
    compute_user_similarity,
    compute_item_similarity,
    predict_rating_user_based,
    predict_rating_item_based,
    get_user_based_recommendations,
    get_item_based_recommendations,
    get_hybrid_recommendations,
    MovieRecommendationSystem,
    compute_prediction_metrics,
    compute_ndcg_at_k
)


class TestDataCreation:
    """Test data creation and loading functions."""
    
    def test_create_sample_dataset(self):
        """Test sample dataset creation."""
        movies_df, ratings_df = create_sample_dataset()
        
        # Check basic structure
        assert isinstance(movies_df, pd.DataFrame)
        assert isinstance(ratings_df, pd.DataFrame)
        
        # Check required columns
        assert 'movie_id' in movies_df.columns
        assert 'title' in movies_df.columns
        assert 'genre' in movies_df.columns
        
        assert 'user_id' in ratings_df.columns
        assert 'movie_id' in ratings_df.columns
        assert 'rating' in ratings_df.columns
        
        # Check data validity
        assert len(movies_df) > 0
        assert len(ratings_df) > 0
        assert ratings_df['rating'].min() >= 1
        assert ratings_df['rating'].max() <= 5
    
    def test_create_user_item_matrix(self):
        """Test user-item matrix creation."""
        _, ratings_df = create_sample_dataset()
        user_item_matrix = create_user_item_matrix(ratings_df)
        
        # Check matrix properties
        assert isinstance(user_item_matrix, pd.DataFrame)
        assert user_item_matrix.shape[0] > 0  # Has users
        assert user_item_matrix.shape[1] > 0  # Has movies
        
        # Check that all values are non-negative
        assert (user_item_matrix >= 0).all().all()
        
        # Check that ratings are in valid range (0 for missing, 1-5 for actual ratings)
        non_zero_ratings = user_item_matrix[user_item_matrix > 0]
        assert non_zero_ratings.min().min() >= 1
        assert non_zero_ratings.max().max() <= 5


class TestSimilarityComputation:
    """Test similarity computation functions."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        movies_df, ratings_df = create_sample_dataset()
        user_item_matrix = create_user_item_matrix(ratings_df)
        return movies_df, ratings_df, user_item_matrix
    
    def test_compute_user_similarity(self, setup_data):
        """Test user similarity computation."""
        _, _, user_item_matrix = setup_data
        user_similarity_df = compute_user_similarity(user_item_matrix)
        
        # Check basic properties
        assert isinstance(user_similarity_df, pd.DataFrame)
        assert user_similarity_df.shape[0] == user_item_matrix.shape[0]
        assert user_similarity_df.shape[1] == user_item_matrix.shape[0]
        
        # Check similarity properties (allow small floating point errors)
        assert (user_similarity_df >= -1.000001).all().all()  # Cosine similarity >= -1
        assert (user_similarity_df <= 1.000001).all().all()   # Cosine similarity <= 1
        
        # Check diagonal is 1 (self-similarity)
        np.testing.assert_allclose(np.diag(user_similarity_df), 1.0, rtol=1e-10)
        
        # Check symmetry
        assert user_similarity_df.equals(user_similarity_df.T)
    
    def test_compute_item_similarity(self, setup_data):
        """Test item similarity computation."""
        _, _, user_item_matrix = setup_data
        item_similarity_df = compute_item_similarity(user_item_matrix)
        
        # Check basic properties
        assert isinstance(item_similarity_df, pd.DataFrame)
        assert item_similarity_df.shape[0] == user_item_matrix.shape[1]
        assert item_similarity_df.shape[1] == user_item_matrix.shape[1]
        
        # Check similarity properties (allow small floating point errors)
        assert (item_similarity_df >= -1.000001).all().all()
        assert (item_similarity_df <= 1.000001).all().all()
        
        # Check diagonal is 1 (self-similarity)
        np.testing.assert_allclose(np.diag(item_similarity_df), 1.0, rtol=1e-10)
        
        # Check symmetry
        assert item_similarity_df.equals(item_similarity_df.T)


class TestPredictionFunctions:
    """Test rating prediction functions."""
    
    @pytest.fixture
    def setup_recommender(self):
        """Setup recommender system."""
        return MovieRecommendationSystem(verbose=False)
    
    def test_predict_rating_user_based(self, setup_recommender):
        """Test user-based rating prediction."""
        recommender = setup_recommender
        
        # Test with valid user and movie
        user_id = list(recommender.user_item_matrix.index)[0]
        movie_id = list(recommender.user_item_matrix.columns)[0]
        
        prediction = predict_rating_user_based(
            user_id, movie_id,
            recommender.user_item_matrix,
            recommender.user_similarity_df
        )
        
        # Check prediction is non-negative and finite
        assert prediction >= 0
        assert np.isfinite(prediction)
        
        # Test with invalid user
        invalid_prediction = predict_rating_user_based(
            999999, movie_id,
            recommender.user_item_matrix,
            recommender.user_similarity_df
        )
        assert invalid_prediction == 0
    
    def test_predict_rating_item_based(self, setup_recommender):
        """Test item-based rating prediction."""
        recommender = setup_recommender
        
        # Test with valid user and movie
        user_id = list(recommender.user_item_matrix.index)[0]
        movie_id = list(recommender.user_item_matrix.columns)[0]
        
        prediction = predict_rating_item_based(
            user_id, movie_id,
            recommender.user_item_matrix,
            recommender.item_similarity_df
        )
        
        # Check prediction is non-negative and finite
        assert prediction >= 0
        assert np.isfinite(prediction)
        
        # Test with invalid user
        invalid_prediction = predict_rating_item_based(
            999999, movie_id,
            recommender.user_item_matrix,
            recommender.item_similarity_df
        )
        assert invalid_prediction == 0


class TestRecommendationFunctions:
    """Test recommendation generation functions."""
    
    @pytest.fixture
    def setup_recommender(self):
        """Setup recommender system."""
        return MovieRecommendationSystem(verbose=False)
    
    def test_get_user_based_recommendations(self, setup_recommender):
        """Test user-based recommendations."""
        recommender = setup_recommender
        user_id = list(recommender.user_item_matrix.index)[0]
        
        recommendations = get_user_based_recommendations(
            user_id,
            recommender.user_item_matrix,
            recommender.user_similarity_df,
            recommender.movies_df,
            n_recommendations=3
        )
        
        if not recommendations.empty:
            # Check structure
            expected_columns = ['movie_id', 'title', 'genre', 'predicted_rating']
            assert all(col in recommendations.columns for col in expected_columns)
            
            # Check that predictions are sorted in descending order
            ratings = recommendations['predicted_rating'].values
            assert all(ratings[i] >= ratings[i+1] for i in range(len(ratings)-1))
            
            # Check that all predicted ratings are positive
            assert all(rating > 0 for rating in ratings)
    
    def test_get_item_based_recommendations(self, setup_recommender):
        """Test item-based recommendations."""
        recommender = setup_recommender
        user_id = list(recommender.user_item_matrix.index)[0]
        
        recommendations = get_item_based_recommendations(
            user_id,
            recommender.user_item_matrix,
            recommender.item_similarity_df,
            recommender.movies_df,
            n_recommendations=3
        )
        
        if not recommendations.empty:
            # Check structure
            expected_columns = ['movie_id', 'title', 'genre', 'predicted_rating']
            assert all(col in recommendations.columns for col in expected_columns)
            
            # Check that predictions are sorted in descending order
            ratings = recommendations['predicted_rating'].values
            assert all(ratings[i] >= ratings[i+1] for i in range(len(ratings)-1))
            
            # Check that all predicted ratings are positive
            assert all(rating > 0 for rating in ratings)
    
    def test_get_hybrid_recommendations(self, setup_recommender):
        """Test hybrid recommendations."""
        recommender = setup_recommender
        user_id = list(recommender.user_item_matrix.index)[0]
        
        recommendations = get_hybrid_recommendations(
            user_id,
            recommender.user_item_matrix,
            recommender.user_similarity_df,
            recommender.item_similarity_df,
            recommender.movies_df,
            n_recommendations=3
        )
        
        if not recommendations.empty:
            # Check structure
            expected_columns = ['movie_id', 'title', 'genre', 'predicted_rating']
            assert all(col in recommendations.columns for col in expected_columns)
            
            # Check that predictions are sorted in descending order
            ratings = recommendations['predicted_rating'].values
            assert all(ratings[i] >= ratings[i+1] for i in range(len(ratings)-1))
            
            # Check that all predicted ratings are positive
            assert all(rating > 0 for rating in ratings)


class TestMovieRecommendationSystem:
    """Test the main MovieRecommendationSystem class."""
    
    def test_initialization(self):
        """Test system initialization."""
        recommender = MovieRecommendationSystem(verbose=False)
        
        # Check that all components are created
        assert hasattr(recommender, 'movies_df')
        assert hasattr(recommender, 'ratings_df')
        assert hasattr(recommender, 'user_item_matrix')
        assert hasattr(recommender, 'user_similarity_df')
        assert hasattr(recommender, 'item_similarity_df')
        
        # Check that dataframes are not empty
        assert len(recommender.movies_df) > 0
        assert len(recommender.ratings_df) > 0
        assert recommender.user_item_matrix.shape[0] > 0
        assert recommender.user_item_matrix.shape[1] > 0
    
    def test_recommendation_methods(self):
        """Test recommendation methods."""
        recommender = MovieRecommendationSystem(verbose=False)
        user_id = list(recommender.user_item_matrix.index)[0]
        
        # Test all recommendation methods
        user_recs = recommender.recommend_user_based(user_id, n=3, k=2)
        item_recs = recommender.recommend_item_based(user_id, n=3, k=2)
        hybrid_recs = recommender.recommend_hybrid(user_id, n=3)
        
        # All should return DataFrames (could be empty)
        assert isinstance(user_recs, pd.DataFrame)
        assert isinstance(item_recs, pd.DataFrame)
        assert isinstance(hybrid_recs, pd.DataFrame)


class TestEvaluationMetrics:
    """Test evaluation metric functions."""
    
    def test_compute_prediction_metrics(self):
        """Test prediction metrics computation."""
        actual = [4.0, 3.5, 5.0, 2.0, 4.5]
        predicted = [4.2, 3.3, 4.8, 2.1, 4.3]
        
        metrics = compute_prediction_metrics(actual, predicted)
        
        # Check that metrics are computed
        assert 'MAE' in metrics
        assert 'RMSE' in metrics
        
        # Check that metrics are non-negative
        assert metrics['MAE'] >= 0
        assert metrics['RMSE'] >= 0
        
        # Check that RMSE >= MAE (always true)
        assert metrics['RMSE'] >= metrics['MAE']
    
    def test_compute_ndcg_at_k(self):
        """Test NDCG@K computation."""
        # Perfect ranking
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3, 4, 5]
        ndcg = compute_ndcg_at_k(recommended, relevant, 5)
        assert ndcg == 1.0
        
        # No relevant items
        recommended = [1, 2, 3]
        relevant = [4, 5, 6]
        ndcg = compute_ndcg_at_k(recommended, relevant, 3)
        assert ndcg == 0.0
        
        # Partial relevance
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5]
        ndcg = compute_ndcg_at_k(recommended, relevant, 5)
        assert 0 <= ndcg <= 1


if __name__ == "__main__":
    pytest.main([__file__])