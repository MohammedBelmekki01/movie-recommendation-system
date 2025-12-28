# Movie Recommendation System

A comprehensive Python implementation of collaborative filtering algorithms for movie recommendations. This system demonstrates user-based, item-based, and hybrid recommendation approaches with both synthetic and real datasets.

##  Overview

This project implements three collaborative filtering techniques:

- **User-Based Collaborative Filtering**: Recommends movies based on similar users' preferences
- **Item-Based Collaborative Filtering**: Recommends movies similar to those the user has already liked
- **Hybrid Approach**: Combines both methods for balanced, robust recommendations

The system uses cosine similarity to compute user-user and item-item relationships, then predicts ratings and generates top-N recommendations.

##  Quick Start

### Installation

```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
pip install -r requirements.txt
```

### Basic Usage

#### Run Full Demo

```bash
python movie_recommendation_system.py
```

#### CLI Mode Examples

```bash
# Get hybrid recommendations for user 1
python movie_recommendation_system.py --user-id 1 --mode hybrid --top-n 5

# User-based recommendations with custom parameters
python movie_recommendation_system.py --user-id 3 --mode user --top-n 10 --k 5

# Item-based with MovieLens dataset
python movie_recommendation_system.py --movielens --user-id 1 --mode item --top-n 8

# Evaluate system performance
python movie_recommendation_system.py --eval --mode hybrid --movielens
```

##  Features

### Core Algorithms

- **User-Based CF**: Finds k-nearest neighbors among users using cosine similarity
- **Item-Based CF**: Computes item similarities and recommends based on user's rating history
- **Hybrid Method**: Weighted combination of user-based and item-based predictions

### Evaluation Metrics

- **Rating Prediction**: Mean Absolute Error (MAE), Root Mean Square Error (RMSE)
- **Recommendation Quality**: Precision@K, Recall@K, Normalized Discounted Cumulative Gain (NDCG@K)

### Datasets

- **Built-in Sample**: 15 popular movies, 8 users, 50 ratings for quick testing
- **MovieLens 100K**: Real-world dataset with 100,000 ratings, 1,682 movies, 943 users

### Utilities

- User profile analysis and rating history
- Similar user/movie discovery
- Comprehensive similarity matrix computation

##  CLI Reference

```
usage: movie_recommendation_system.py [options]

optional arguments:
  --user-id USER_ID     Target user ID for recommendations (default: demo mode)
  --mode {user,item,hybrid}  Recommendation algorithm (default: hybrid)
  --top-n TOP_N         Number of recommendations (default: 5)
  --k K                 Number of neighbors for similarity (default: 3)
  --user-weight WEIGHT  User-based weight in hybrid mode (default: 0.5)
  --item-weight WEIGHT  Item-based weight in hybrid mode (default: 0.5)
  --movielens           Use MovieLens 100K dataset instead of sample
  --eval                Run evaluation metrics on train/test split
```

##  Example Output

### User Profile

```
============================================================
USER 1 PROFILE
============================================================

Total movies rated: 7
Average rating: 4.57

Rated movies:
--------------------------------------------------
  The Shawshank Redemption (Drama) - Rating: 5.0
  The Godfather (Crime) - Rating: 4.0
  The Dark Knight (Action) - Rating: 5.0
```

### Recommendations

```
============================================================
HYBRID RECOMMENDATIONS FOR USER 1
============================================================

Top 5 Recommendations:
------------------------------------------------------------
  The Matrix (Sci-Fi) - Predicted Rating: 4.80
  Interstellar (Sci-Fi) - Predicted Rating: 4.67
  Gladiator (Action) - Predicted Rating: 4.62
```

### Evaluation Metrics

```
============================================================
EVALUATION RESULTS
============================================================

Rating Prediction Metrics:
  MAE: 0.642
  RMSE: 0.891

Recommendation Quality (Top-10):
  Precision@10: 0.731
  Recall@10: 0.584
  NDCG@10: 0.692
```

##  Architecture

```
movie_recommendation_system.py
├── Data Creation & Loading
│   ├── create_sample_dataset()
│   └── load_movielens_data()
├── Data Preprocessing
│   ├── create_user_item_matrix()
│   └── normalize_ratings()
├── Similarity Computation
│   ├── compute_user_similarity()
│   └── compute_item_similarity()
├── Recommendation Algorithms
│   ├── User-Based Collaborative Filtering
│   ├── Item-Based Collaborative Filtering
│   └── Hybrid Recommendations
├── Evaluation Framework
│   ├── train_test_split()
│   ├── compute_prediction_metrics()
│   └── compute_recommendation_metrics()
└── MovieRecommendationSystem Class
    └── Unified API interface
```

##  Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_similarity.py
pytest tests/test_predictions.py
pytest tests/test_recommendations.py
```

##  Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- requests >= 2.28.0 (for MovieLens download)
- pytest >= 7.0.0 (for testing)

##  Development Roadmap

### Completed 

- [x] User-based and item-based collaborative filtering
- [x] Hybrid recommendation approach
- [x] Sample dataset generation
- [x] CLI interface with comprehensive options
- [x] Evaluation metrics (MAE, RMSE, Precision@K, Recall@K, NDCG@K)
- [x] MovieLens 100K integration
- [x] Unit test suite

### Future Enhancements 

- [ ] Matrix factorization techniques (SVD, NMF)
- [ ] Deep learning approaches (Neural Collaborative Filtering)
- [ ] Content-based filtering integration
- [ ] Real-time recommendation API
- [ ] Web interface with Flask/FastAPI
- [ ] Scalability improvements (sparse matrices, approximate methods)
- [ ] Cold-start problem handling
- [ ] Bias detection and fairness metrics

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  References

- [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121)
- [Item-Based Collaborative Filtering Recommendation Algorithms](https://dl.acm.org/doi/10.1145/371920.372071)
- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Evaluating Recommender Systems](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_8)

##  Contact

Mohammed Belmekki - belmekki.meh@gmail.com

Project Link: [https://github.com/MohammedBelmekki01/movie-recommendation-system](https://github.com/MohammedBelmekki01/movie-recommendation-system)
