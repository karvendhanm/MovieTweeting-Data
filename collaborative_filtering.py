# collaborative filtering is a method of recommendation based on user-item
# interaction.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t
from scipy.sparse import csr_matrix
import pickle

data_dir = 'C:/Users/John/PycharmProjects/MovieTweeting-Data/data/'
movies = pd.read_csv(data_dir + 'movies_clean.csv')
reviews = pd.read_csv(data_dir + 'reviews_clean.csv')

del movies['Unnamed: 0']
del reviews['Unnamed: 0']

# Measures of Similiarity
# when using neighborhood based collaborative filtering, it is important to
# understand how to measure the similiarity to users or items to one another.

user_items = reviews[['user_id', 'movie_id', 'rating']]

# creating a user, movie, rating matrix.
user_by_movie = user_items.pivot(index = 'user_id', columns = 'movie_id',values = 'rating')

# Create a dictionary with users and corresponding movies seen
def movies_watched(user_id):
    '''
    INPUT:
    user_id - the user_id of an individual as int
    OUTPUT:
    movies - an array of movies the user has watched
    '''
    movie_ids = user_items['movie_id'][
        user_items['user_id'].isin([user_id])].values
    # movie_names = movies[movies['movie_id'].isin(user_items['movie_id'][user_items['user_id'].isin([user_id])])]['movie'].values

    return movie_ids


def create_user_movie_dict():
    '''
    INPUT: None
    OUTPUT: movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids

    Creates the movies_seen dictionary
    '''

    movies_seen = {}
    for user_id in user_items['user_id'].unique():
        movies_seen[user_id] = movies_watched(user_id)

    return movies_seen


# Use your function to return dictionary
movies_seen = create_user_movie_dict()


# Remove individuals who have watched 2 or fewer movies - don't have enough data to make recs
def create_movies_to_analyze(movies_seen, lower_bound=2):
    '''
    INPUT:
    movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids
    lower_bound - (an int) a user must have more movies seen than the lower bound to be added to the movies_to_analyze dictionary

    OUTPUT:
    movies_to_analyze - a dictionary where each key is a user_id and the value is an array of movie_ids

    The movies_seen and movies_to_analyze dictionaries should be the same except that the output dictionary has removed

    '''

    # Do things to create updated dictionary
    movies_to_analyze = {}
    for user_id in movies_seen.keys():
        if len(movies_seen[user_id]) > lower_bound:
            movies_to_analyze[user_id] = movies_seen[user_id]

    return movies_to_analyze


# Use your function to return your updated dictionary
movies_to_analyze = create_movies_to_analyze(movies_seen)


def compute_correlation(user1, user2):
    '''
    INPUT
    user1 - int user_id
    user2 - int user_id
    OUTPUT
    the correlation between the matching ratings between the two users
    '''

    all_columns = []
    all_columns.extend(movies_to_analyze[user1])
    all_columns.extend(movies_to_analyze[user2])

    df = user_by_movie.loc[[user1, user2], list(set(all_columns))]
    df = df.transpose()
    corr = df.corr().iloc[0, 1]

    return corr  # return the correlation

compute_correlation(2, 104)


def compute_euclidean_dist(user1, user2):
    '''
    INPUT
    user1 - int user_id
    user2 - int user_id
    OUTPUT
    the euclidean distance between user1 and user2
    '''

    all_columns = []
    all_columns.extend(movies_to_analyze[user1])
    all_columns.extend(movies_to_analyze[user2])

    df = user_by_movie.loc[[user1, user2], list(set(all_columns))]
    df = df.transpose()
    df = df[
        np.logical_or(df.iloc[:, 0].isnull(), df.iloc[:, 1].isnull()) == False]
    dist = np.sqrt(sum((df.iloc[:, 0] - df.iloc[:, 1]) ** 2))

    return dist  # return the euclidean distance

df_dists = pd.read_pickle(data_dir + "dists.p")

# Using the Nearest Neighbors to Make Recommendations























