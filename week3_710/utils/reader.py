import os
import pandas as pd
from sklearn import model_selection


def get_data(data_path, movie_name, rating_name):    
    movie_path = os.path.join(data_path, movie_name)
    rating_path = os.path.join(data_path, rating_name)

    movie_data = pd.read_csv(movie_path)
    rating_data = pd.read_csv(rating_path)[['userId', 'movieId', 'rating']]
    
    return movie_data, rating_data


def get_train_valid():
    data_path = './movie_len_small/'    
    movie_data, rating_data = get_data(data_path, "movies.csv", "ratings.csv")
    
    max_user = max(rating_data.userId)
    max_movie = max(rating_data.movieId)

    
    df_train, df_valid = model_selection.train_test_split(
                            rating_data,
                            test_size = 0.1,
                            random_state = 42,
                            stratify = rating_data.rating.values)

    return df_train, df_valid, max_user, max_movie