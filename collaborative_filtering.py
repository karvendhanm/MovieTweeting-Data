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
    movie_ids = user_items['movie_id'][user_items['user_id'].isin([user_id])].values
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

def find_closest_neighbors(user):
    '''
    INPUT:
        user - (int) the user_id of the individual you want to find the closest users
    OUTPUT:
        closest_neighbors - an array of the id's of the users sorted from closest to farthest away
    '''
    # I treated ties as arbitrary and just kept whichever was easiest to keep using the head method
    # You might choose to do something less hand wavy - order the neighbors

    df = df_dists[df_dists['user1'].isin([user])].reset_index(drop=True)
    df = df.sort_values(by=['eucl_dist'], ascending=[True])
    closest_neighbors = df['user2'][1:].values

    return closest_neighbors

closest_neighbors  = find_closest_neighbors(2)

def movies_liked(user_id, min_rating=7):
    '''
    INPUT:
    user_id - the user_id of an individual as int
    min_rating - the minimum rating considered while still a movie is still a "like" and not a "dislike"
    OUTPUT:
    movies_liked - an array of movies the user has watched and liked
    '''

    series_ = user_by_movie.loc[user_id, movies_watched(user_id)]
    movies_liked = series_[series_ >= min_rating].index.values

    return movies_liked

def movie_names(movie_ids):
    '''
    INPUT
    movie_ids - a list of movie_ids
    OUTPUT
    movies - a list of movie names associated with the movie_ids

    '''

    movie_lst = movies[movies['movie_id'].isin(movie_ids)]['movie'].values
    return movie_lst


def make_recommendations(user, num_recs=10):
    '''
    INPUT:
        user - (int) a user_id of the individual you want to make recommendations for
        num_recs - (int) number of movies to return
    OUTPUT:
        recommendations - a list of movies - if there are "num_recs" recommendations return this many
                          otherwise return the total number of recommendations available for the "user"
                          which may just be an empty list
    '''
    # I wanted to make recommendations by pulling different movies than the user has already seen
    # Go in order from closest to farthest to find movies you would recommend
    # I also only considered movies where the closest user rated the movie as a 9 or 10

    # movies_seen by user (we don't want to recommend these)
    movies_seen = movies_watched(user)
    closest_neighbors = find_closest_neighbors(user)

    # Keep the recommended movies here
    recs = np.array([])

    # Go through the neighbors and identify movies they like the user hasn't seen
    for neighbor in closest_neighbors:
        neighbs_likes = movies_liked(neighbor)

        # Obtain recommendations for each neighbor
        new_recs = np.setdiff1d(neighbs_likes, movies_seen, assume_unique=True)

        # Update recs with new recs
        recs = np.unique(np.concatenate([new_recs, recs], axis=0))

        # If we have enough recommendations exit the loop
        if len(recs) > num_recs - 1:
            break

    # Pull movie titles using movie ids
    recommendations = movie_names(recs)

    return recommendations


def all_recommendations(num_recs=10):
    '''
    INPUT
        num_recs (int) the (max) number of recommendations for each user
    OUTPUT
        all_recs - a dictionary where each key is a user_id and the value is an array of recommended movie titles
    '''

    # Make the recommendations for each user

    all_recs = {}
    for user in user_items['user_id'].unique():
        all_recs[user] = make_recommendations(user, num_recs)

    return all_recs


all_recs = all_recommendations(10)
















































