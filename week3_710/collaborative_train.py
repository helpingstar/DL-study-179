from turtle import forward
from sklearn import model_selection
import csv
import pandas as pd
import os
import numpy as np
import torch

from sklearn import metrics
from utils.dataset import MovieDataset
from utils.reader import get_data


def build_rating_sparse_tensor(train_df):
    user_indices = np.array(train_df['userId'].values)
    user_max = np.max(user_indices)
    movie_indices = np.array(train_df['movieId'].values)
    movie_max = np.max(movie_indices)
    
    values = train_df['rating'].values
    
    return torch.sparse_coo_tensor(
                            indices = [user_indices, movie_indices], 
                            values = values, 
                            size = (user_max+1, movie_max+1)).to_dense()

# def sparse_mean_error(sparse_ratings, user_embed, movie_embed):
    


if __name__ == "__main__":
    data_path = './movie_len_small/'    
    movie_data, rating_data = get_data(data_path, "movies.csv", "ratings.csv")
    
    df_train, df_valid = model_selection.train_test_split(
                            rating_data,
                            test_size = 0.1,
                            random_state = 42,
                            stratify = rating_data.rating.values)
    
    # print(max(movie_data['movieId']))
    # # print(df_train[:10])
    # # print(df_train['rating'].value_counts())
    # # print(df_valid[:10])
    # # print(df_valid['rating'].value_counts())

    train_matrix = build_rating_sparse_tensor(df_train)

    train_dataset = MovieDataset(df_train.userId.values, 
                                 df_train.movieId.values, 
                                 df_train.rating.values)
    
    valid_dataset = MovieDataset(df_valid.userId.values,
                                 df_valid.movieId.values,
                                 df_valid.rating.values)
    