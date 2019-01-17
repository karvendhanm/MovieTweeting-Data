import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t
from datetime import datetime

# Read in the datasets
movies = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat', delimiter='::', encoding="utf8", header=None, names=['movie_id', 'movie', 'genre'], dtype={'movie_id': object}, engine='python')
reviews = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat', delimiter='::', encoding="utf8", header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'], dtype={'movie_id': object, 'user_id': object, 'timestamp': object}, engine='python')

# The number of different genres
genre_list = []
for genre in movies['genre']:
    try:
        genre_list.extend(genre.split("|"))
    except:
        continue

genre_list = set(genre_list)
n_genres = len(genre_list)

movies['year'] = movies['movie'].apply(lambda x: x[-5:-1])

# Movie in which century

for cen in ['18', '19', '20']:
    movies[cen + '00s'] = movies['year'].apply(lambda x: 1 if x.startswith(cen) == True else 0)

for genre in genre_list:
    movies[genre] = movies['genre'].apply(lambda x: 1 if str(x).find(genre) >= 0 else 0)

# Create a date out of time stamp
reviews['timestamp'] = reviews['timestamp'].astype(int)

reviews['timestamp'] = reviews['timestamp'].apply(lambda x:
                                                  datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

