import pandas as pd
from pathlib import Path


def load_movielens(path='data/ml-latest-small', min_interactions=5):
    """
    Load MovieLens dataset.

    Parameters
    ----------
    path : str
        Path to the dataset folder containing ratings.csv and movies.csv.
    min_interactions : int
        Drop users/items with fewer than this many interactions.
        Ensures every user and item in train also appears enough times.

    Returns
    -------
    ratings : pd.DataFrame
        Columns: user_id, item_id, rating, timestamp
    movies : pd.DataFrame
        Columns: movieId, title, genres
    """
    ratings = pd.read_csv(Path(path) / 'ratings.csv')
    ratings.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    movies = pd.read_csv(Path(path) / 'movies.csv')

    # Filter out users and items with too few interactions
    # (iterative filtering until stable)
    before = 0
    while before != len(ratings):
        before = len(ratings)
        user_counts = ratings['user_id'].value_counts()
        item_counts = ratings['item_id'].value_counts()
        ratings = ratings[
            ratings['user_id'].isin(user_counts[user_counts >= min_interactions].index) &
            ratings['item_id'].isin(item_counts[item_counts >= min_interactions].index)
        ]

    print(f"Loaded {len(ratings)} ratings, "
          f"{ratings['user_id'].nunique()} users, "
          f"{ratings['item_id'].nunique()} items")

    return ratings, movies


def temporal_train_test_split(ratings, test_ratio=0.2):
    """
    Split by timestamp: each user's most recent interactions go to test.

    This is the standard evaluation protocol for recommender systems --
    it simulates the real scenario where we predict future interactions
    from past ones.

    Parameters
    ----------
    ratings : pd.DataFrame
        Must have columns: user_id, item_id, rating, timestamp
    test_ratio : float
        Fraction of each user's interactions to hold out for testing.

    Returns
    -------
    train : pd.DataFrame
    test : pd.DataFrame
    """
    ratings = ratings.sort_values('timestamp')

    test_dfs = []
    train_dfs = []

    for user_id, group in ratings.groupby('user_id'):
        n_test = max(1, int(len(group) * test_ratio))
        train_dfs.append(group.iloc[:-n_test])
        test_dfs.append(group.iloc[-n_test:])

    train = pd.concat(train_dfs).reset_index(drop=True)
    test = pd.concat(test_dfs).reset_index(drop=True)

    print(f"Train: {len(train)} ratings | Test: {len(test)} ratings")
    print(f"Test users: {test['user_id'].nunique()} | "
          f"Test items: {test['item_id'].nunique()}")

    return train, test

def load_movielens_1m(path='data/ml-1m', min_interactions=5):
    """
    Load MovieLens 1M dataset.

    Returns:
        ratings: DataFrame with columns
                 [user_id, item_id, rating, timestamp]
        movies:  DataFrame with columns
                 [movieId, title, genres]
    """

    ratings = pd.read_csv(
        Path(path) / 'ratings.dat',
        sep='::',
        engine='python',
        header=None,
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )

    movies = pd.read_csv(
        Path(path) / 'movies.dat',
        sep='::',
        engine='python',
        header=None,
        names=['movieId', 'title', 'genres'],
        encoding='latin-1'
    )

    # Optional: filter low-interaction users/items (same as small)
    before = 0
    while before != len(ratings):
        before = len(ratings)
        user_counts = ratings['user_id'].value_counts()
        item_counts = ratings['item_id'].value_counts()
        ratings = ratings[
            ratings['user_id'].isin(user_counts[user_counts >= min_interactions].index) &
            ratings['item_id'].isin(item_counts[item_counts >= min_interactions].index)
        ]

    print(f"Loaded {len(ratings)} ratings | "
          f"{ratings['user_id'].nunique()} users | "
          f"{ratings['item_id'].nunique()} items")

    return ratings, movies
