import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t

data_dir = './data/'

movies = pd.read_csv(data_dir+'movies_clean.csv')
reviews = pd.read_csv(data_dir+'reviews_clean.csv')
del movies['Unnamed: 0']
del reviews['Unnamed: 0']

# Find the most popular movies:
# The task is no matter the user, we need to provide a list of the
# recommendations based on simply the most popular items.

For this task, we will consider what is "most popular" based on the following criteria:

A movie with the highest average rating is considered best
With ties, movies that have more ratings are better
A movie must have a minimum of 5 ratings to be considered among the best movies
If movies are tied in their average rating and number of ratings, the ranking is determined by the movie that is the most recent rating




