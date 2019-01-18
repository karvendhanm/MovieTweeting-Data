import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

data_dir = 'C:/Users/John/PycharmProjects/MovieTweeting-Data/data/'

# Read in the datasets
movies = pd.read_csv(data_dir + 'movies_clean.csv')
reviews = pd.read_csv(data_dir + 'reviews_clean.csv')

del movies['Unnamed: 0']
del reviews['Unnamed: 0']

# Create user-by-item matrix
user_items = reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()


# Create data subset
#user_movie_subset = user_by_movie[[73486, 75314,  68646, 99685]].dropna(axis=0)
user_movie_subset = pd.read_csv(data_dir + 'temp.csv')
ratings_mat = np.matrix(user_movie_subset)


def FunkSVD(ratings_mat, latent_features=4, learning_rate=0.0001, iters=100):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization

    INPUT:
    ratings_mat - (numpy array) a matrix with users as rows, movies as columns, and ratings as values
    latent_features - (int) the number of latent features used
    learning_rate - (float) the learning rate
    iters - (int) the number of iterations

    OUTPUT:
    user_mat - (numpy array) a user by latent feature matrix
    movie_mat - (numpy array) a latent feature by movie matrix
    '''

    # Set up useful values to be used through the rest of the function
    n_users = ratings_mat.shape[0]  # number of rows in the matrix
    n_movies = ratings_mat.shape[1]  # number of movies in the matrix
    num_ratings = n_users * n_movies  # total number of ratings in the matrix

    # initialize the user and movie matrices with random values
    # helpful link: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html
    user_mat = np.random.rand(n_users,
                              latent_features)  # user matrix filled with random values of shape user x latent
    movie_mat = np.random.rand(latent_features,
                               n_movies)  # movie matrix filled with random values of shape latent x movies

    # initialize sse at 0 for first iteration
    sse_accum = 0

    # header for running results
    print("Optimization Statistics")
    print("Iterations | Mean Squared Error ")

    # for each iteration
    for iteration in range(iters):

        # update our sse
        old_sse = sse_accum
        sse_accum = 0

        # For each user-movie pair
        for i in range(n_users):
            for j in range(n_movies):

                # if the rating exists
                if ratings_mat[i, j] > 0:

                    # compute the error as the actual minus the dot product of the user and movie latent features
                    diff = ratings_mat[i, j] - np.dot(user_mat[i, :],
                                                      movie_mat[:, j])

                    # Keep track of the sum of squared errors for the matrix
                    sse_accum += diff ** 2

                    # update the values in each matrix in the direction of the gradient
                    for k in range(latent_features):
                        user_mat[i, k] += learning_rate * (
                                    2 * diff * movie_mat[k, j])
                        movie_mat[k, j] += learning_rate * (
                                    2 * diff * user_mat[i, k])

        # print results for iteration
        print("%d \t\t %f" % (iteration + 1, sse_accum / num_ratings))

    return user_mat, movie_mat

user_mat, movie_mat = FunkSVD(ratings_mat, latent_features=4, learning_rate=0.005, iters=10)

print(user_mat)

#Compare the predicted and actual results
print(np.dot(user_mat, movie_mat))
print(ratings_mat)

user_mat, movie_mat = FunkSVD(ratings_mat, latent_features=4, learning_rate=0.005, iters=250)

#Compare the predicted and actual results
print(np.dot(user_mat, movie_mat))
print(ratings_mat)

# Here we are placing a nan into our original subset matrix
ratings_mat[0, 0] = np.nan
ratings_mat

# run SVD on the matrix with the missing value
user_mat, movie_mat = FunkSVD(ratings_mat, latent_features=4, learning_rate=0.005, iters=250)

preds = np.dot(user_mat, movie_mat)
print("The predicted value for the missing rating is {}:".format(preds[0,0]))

# Setting up a matrix of the first 1000 users with movie ratings
first_1000_users = np.matrix(user_by_movie.head(1000))

# perform funkSVD on the matrix of the top 1000 users
user_mat, movie_mat = FunkSVD(first_1000_users, latent_features=4, learning_rate=0.005, iters=20)

preds = np.dot(user_mat, movie_mat)
print(preds)

# Replace each of the comments below with the correct values
num_ratings = np.count_nonzero(~np.isnan(first_1000_users))
print("The number of actual ratings in the first_1000_users is {}.".format(num_ratings))
print()

# How many ratings did we make for user-movie pairs that didn't actually have ratings
ratings_for_missing = first_1000_users.shape[0]*first_1000_users.shape[1] - num_ratings
print("The number of ratings made for user-movie pairs that didn't have ratings is {}".format(ratings_for_missing))

# Test your results against the solution
assert num_ratings == 10852, "Oops!  The number of actual ratings doesn't quite look right."
assert ratings_for_missing == 31234148, "Oops!  The number of movie-user pairs that you made ratings for that didn't actually have ratings doesn't look right."

# Make sure you made predictions on all the missing user-movie pairs
preds = np.dot(user_mat, movie_mat)
assert np.isnan(preds).sum() == 0
print("Nice job!  Looks like you have predictions made for all the missing user-movie pairs! But I still have one question... How good are they?")


# my own twist to the funk svd function
def FunkSVD(ratings_mat, latent_features=4, learning_rate=0.0001, iters=100):

    n_users = ratings_mat.shape[0]
    n_movies = ratings_mat.shape[1]
    num_rating = np.count_nonzero(~np.isnan(ratings_mat))

    user_mat = np.random.rand(n_users, latent_features)
    movie_mat = np.random.rand(latent_features, n_movies)

    for iter in range(iters):
        sse = 0
        for i in range(n_users):
            for j in range(n_movies):
                if np.isnan(ratings_mat[i,j]) == False:
                    actual_rating = ratings_mat[i,j]
                    predicted_rating = np.dot(user_mat[i,:], movie_mat[:,j])

                    sse += (actual_rating - predicted_rating)**2
                    error = actual_rating - predicted_rating

                    # for lat in range(latent_features):
                    #     user_mat[i, lat] += (learning_rate * 2 * error * movie_mat[lat, j])
                    #     movie_mat[lat, j] += (learning_rate * 2 * error * user_mat[i, lat])

                    user_mat[i, :] += (learning_rate * 2 * error * movie_mat[:, j])
                    movie_mat[:, j] += (learning_rate * 2 * error * user_mat[i, :])

        print('the mean squared error on iteration: {} is: {}'.format(iter+1, sse/num_rating))
    return user_mat, movie_mat






