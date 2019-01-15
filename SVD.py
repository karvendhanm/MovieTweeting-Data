import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import svd_tests as t

data_dir = 'C:/Users/John/PycharmProjects/MovieTweeting-Data/data/'

# Read in the datasets
movies = pd.read_csv(data_dir + 'movies_clean.csv')
reviews = pd.read_csv(data_dir + 'reviews_clean.csv')

del movies['Unnamed: 0']
del reviews['Unnamed: 0']

# Create user-by-item matrix
user_items = reviews[['user_id', 'movie_id', 'rating']]
user_by_movie = user_items.groupby(['user_id','movie_id'])['rating'].max().unstack()

user_movie_subset = user_by_movie[[73486, 75314,  68646, 99685]].dropna(axis=0)
print(user_movie_subset)


u, s, vt = np.linalg.svd(user_movie_subset)
print('the shape of u: {}'.format(u.shape))
print('the shape of s: {}'.format(s.shape))
print('the shape of vt: {}'.format(vt.shape))

sigma_square = np.diag(s)
u = u[:,:4]

user_movie_subset_recreate = np.dot(np.dot(u,sigma_square), vt)

u_2 = u[:,:2]
s_2 = np.diag(s[:2])
vt = vt[:2,:]

pred_ratings = np.dot(np.dot(u_2,s_2), vt)

np.sum(np.sum((user_movie_subset - pred_ratings)**2))
