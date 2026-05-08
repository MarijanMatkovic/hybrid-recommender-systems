import numpy as np
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


def random_train_test_split(ratings, test_ratio=0.2, seed=0):
    """
    Per-user random train/test split with a reproducible RNG seed.

    Useful for multi-seed experiments where we want to report mean ± std
    across different data splits (as opposed to the fixed temporal split).
    Within each user, ``test_ratio`` fraction of interactions is assigned
    to the test set uniformly at random; the remainder goes to train.

    ``max(1, int(len(group) * test_ratio))`` guarantees every user has at
    least one test interaction, matching the contract of
    ``temporal_train_test_split``.

    Parameters
    ----------
    ratings : pd.DataFrame
        Must have columns: user_id, item_id, rating, timestamp
    test_ratio : float
        Fraction of each user's interactions to hold out for testing.
    seed : int
        RNG seed -- different seeds yield different splits, same seed
        always produces an identical split (for reproducibility).

    Returns
    -------
    train : pd.DataFrame
    test : pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    test_dfs = []
    train_dfs = []

    for user_id, group in ratings.groupby('user_id', sort=True):
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        test_dfs.append(group.iloc[test_idx])
        train_dfs.append(group.iloc[train_idx])

    train = pd.concat(train_dfs).reset_index(drop=True)
    test = pd.concat(test_dfs).reset_index(drop=True)

    print(f"[random split seed={seed}]  "
          f"Train: {len(train)} ratings | Test: {len(test)} ratings")
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


def load_netflix_prize(path='data/netflixprize', min_interactions=20,
                       subsample_users=None, seed=42, use_cache=True):
    """Load the Netflix Prize dataset.

    The dataset ships as 4 ``combined_data_X.txt`` files with a
    movie-id-then-ratings layout::

        <movie_id>:
        <user_id>,<rating>,<date>
        <user_id>,<rating>,<date>
        ...
        <next_movie_id>:
        ...

    First-time parse takes ~5-10 min (100M lines). The result is cached
    as ``ratings_cache.parquet`` next to the source files; subsequent
    loads take <30s.

    Parameters
    ----------
    path : str
        Path to the netflixprize folder containing the 4 combined_data
        files and ``movie_titles.csv``.
    min_interactions : int
        Iterative drop of users/items with fewer than this many ratings.
        Default 20 (more aggressive than ml-1m's 5 because Netflix is
        denser; lowering memory pressure on the EASE inversion).
    subsample_users : int or None
        If given, sample this many users uniformly at random *before*
        the min_interactions filter. Useful for development runs where
        full ~480k users is overkill. Default None (use all).
    seed : int
        Subsample RNG seed. Only used when ``subsample_users`` is set.
    use_cache : bool
        Cache the parsed (un-filtered) ratings DataFrame as parquet for
        fast reload. Default True.

    Returns
    -------
    ratings : pd.DataFrame
        Columns: user_id, item_id, rating, timestamp (Unix seconds)
    movies : pd.DataFrame or None
        Columns: movieId, year, title (loaded from movie_titles.csv).
    """
    path = Path(path)
    cache_path = path / 'ratings_cache.parquet'

    if use_cache and cache_path.exists():
        print(f"[netflix] loading cached ratings from {cache_path}")
        ratings = pd.read_parquet(cache_path)
    else:
        print("[netflix] parsing combined_data_*.txt (slow first time)")
        # Pre-allocate arrays — 100M rows in a list is slow because of
        # Python object overhead. Numpy arrays grow with realloc which
        # is fine for ~100M ints.
        users = []
        items = []
        ratings_arr = []
        timestamps = []
        for fname in ('combined_data_1.txt', 'combined_data_2.txt',
                      'combined_data_3.txt', 'combined_data_4.txt'):
            fpath = path / fname
            if not fpath.exists():
                print(f"  WARNING: {fpath} not found, skipping")
                continue
            print(f"  parsing {fname}...")
            current_movie = None
            n_in_file = 0
            with open(fpath, 'r') as f:
                for line in f:
                    line = line.rstrip('\r\n')
                    if not line:
                        continue
                    if line.endswith(':'):
                        current_movie = int(line[:-1])
                        continue
                    # user_id,rating,YYYY-MM-DD
                    comma1 = line.index(',')
                    comma2 = line.index(',', comma1 + 1)
                    users.append(int(line[:comma1]))
                    ratings_arr.append(int(line[comma1 + 1:comma2]))
                    items.append(current_movie)
                    timestamps.append(line[comma2 + 1:])
                    n_in_file += 1
            print(f"    {n_in_file:,} ratings parsed")

        ratings = pd.DataFrame({
            'user_id': np.asarray(users, dtype=np.int32),
            'item_id': np.asarray(items, dtype=np.int32),
            'rating':  np.asarray(ratings_arr, dtype=np.int8),
            'date':    pd.to_datetime(timestamps, format='%Y-%m-%d'),
        })
        # Use unix-seconds timestamp to match the MovieLens schema
        ratings['timestamp'] = (
            ratings['date'].astype('int64') // 10**9).astype(np.int64)
        ratings = ratings[['user_id', 'item_id', 'rating', 'timestamp']]

        if use_cache:
            try:
                ratings.to_parquet(cache_path, index=False)
                print(f"[netflix] cached parsed data to {cache_path}")
            except Exception as exc:
                print(f"[netflix] could not write cache: {exc}")

    # Optional user subsampling for development runs
    if subsample_users is not None:
        rng = np.random.default_rng(seed)
        all_users = ratings['user_id'].unique()
        if subsample_users < len(all_users):
            keep = rng.choice(all_users, size=int(subsample_users),
                              replace=False)
            ratings = ratings[ratings['user_id'].isin(keep)].copy()
            print(f"[netflix] subsampled to {subsample_users:,} users "
                  f"(seed={seed}) -> {len(ratings):,} ratings")

    # Iterative density filter (drops sparse users/items)
    before = 0
    while before != len(ratings):
        before = len(ratings)
        user_counts = ratings['user_id'].value_counts()
        item_counts = ratings['item_id'].value_counts()
        ratings = ratings[
            ratings['user_id'].isin(
                user_counts[user_counts >= min_interactions].index)
            & ratings['item_id'].isin(
                item_counts[item_counts >= min_interactions].index)
        ]

    # Load movie titles (best-effort; some titles have stray commas)
    titles_path = path / 'movie_titles.csv'
    movies = None
    if titles_path.exists():
        try:
            movies = pd.read_csv(
                titles_path, header=None,
                names=['movieId', 'year', 'title'],
                encoding='latin-1', on_bad_lines='skip')
        except Exception as exc:
            print(f"[netflix] could not parse movie_titles.csv: {exc}")

    print(f"Loaded {len(ratings):,} ratings | "
          f"{ratings['user_id'].nunique():,} users | "
          f"{ratings['item_id'].nunique():,} items "
          f"(min_interactions={min_interactions})")

    return ratings, movies
