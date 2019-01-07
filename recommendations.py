import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t
from datetime import datetime
import re


# Read in the datasets
movies = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat',
                     encoding="utf8", delimiter='::', header=None, names=[
        'movie_id', 'movie', 'genre'], dtype={'movie_id': object}, engine='python')
reviews = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat', delimiter='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], dtype={'movie_id': object, 'user_id': object, 'timestamp': object}, engine='python')

# The number of different genres
df_genre = movies['genre'].str.split("|", expand = True).fillna('None')
df_genre = pd.concat([df_genre[0], df_genre[1], df_genre[2], df_genre[3],
                    df_genre[4]], axis = 0)

number_of_different_genre = len(df_genre.unique()) - 1

# The number of unique users in the dataset
number_of_unique_users = reviews['user_id'].nunique()

# The number of missing ratings in the reviews dataset
number_of_missing_rating = (reviews['rating'].isnull() == True).sum()

# average rating across all rating
average_rating_across_all_rating = reviews['rating'].mean()

# Minimum rating across all rating
Minimum_rating_across_all_rating = reviews['rating'].min()

# Maximum rating across all rating
Maximum_rating_across_all_rating = reviews['rating'].max()

# dict_sol1 = {
# # 'The number of movies in the dataset':movies.shape[0],
# # 'The number of ratings in the dataset':reviews.shape[0],
# # 'The number of different genres':number_of_different_genre,
# # 'The number of unique users in the dataset':number_of_unique_users,
# # 'The number missing ratings in the reviews dataset':number_of_missing_rating,
# # 'The average rating given across all ratings':average_rating_across_all_rating,
# # 'The minimum rating given across all ratings':Minimum_rating_across_all_rating,
# # 'The maximum rating given across all ratings':Maximum_rating_across_all_rating
# # }

# Data Cleaning
# Pull the date from the title and create new column
movies['year'] = movies['movie'].apply(lambda x: x[-5:-1] if x[-1] is ')'
                                                            else np.nan)
movies['year'] = movies['year'].astype(int)
movies['century'] = movies['year']/100
movies['century'] = movies['century'].astype(int)

# Dummy the date column with 1's and 0's for each century of a movie (1800's, 1900's, and 2000's)
temp = pd.get_dummies(data=movies['century'])
movies.drop('century', axis = 1, inplace = True)
movies = pd.concat([movies, temp], axis = 1)


#Dummy column the genre with 1's and 0's
for genre in df_genre.unique():
    if genre == 'None':
        pass
    else:
        movies[genre] = movies['genre'].apply(lambda x: 0 if re.search(genre,
                                    str(x)) == None else 1)

# Timestamp to date and time
reviews['timestamp'] = reviews['timestamp'].astype(int)

reviews['timestamp'] = reviews['timestamp'].apply(lambda x:
            datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))


reviews.to_csv('./data/reviews_clean.csv',index = False, index_label = False)
movies.to_csv('./data/movies_clean.csv',index = False, index_label = False)


































