import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t


# Read in the datasets
movies = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat',
                     encoding="utf8", delimiter='::', header=None, names=[
        'movie_id', 'movie', 'genre'], dtype={'movie_id': object}, engine='python')
reviews = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat', delimiter='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], dtype={'movie_id': object, 'user_id': object, 'timestamp': object}, engine='python')

# The number of different genres
df_genre = movies['genre'].str.split("|", expand = True).fillna('None')
df_genre = pd.concat([df_genre[0], df_genre[1], df_genre[2], df_genre[3],
                    df_genre[4]], axis = 0)

number_of_different_genre = len(df_genre.unique()) -1

# The number of unique users in the dataset
number_of_unique_users = reviews['user_id'].nunique()

# The number missing ratings in the reviews dataset
number_of_missing_rating = reviews['rating'].isna()


dict_sol1 = {
'The number of movies in the dataset':movies.shape[0],
'The number of ratings in the dataset':reviews.shape[0],
'The number of different genres':number_of_different_genre,
'The number of unique users in the dataset':number_of_unique_users,
'The number missing ratings in the reviews dataset':number_of_missing_rating,
'The average rating given across all ratings':,
'The minimum rating given across all ratings':,
'The maximum rating given across all ratings':,
}