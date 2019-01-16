import numpy as np
import pandas as pd

data_dir = 'C:/Users/John/PycharmProjects/MovieTweeting-Data/data/'

# Read in the datasets
movies = pd.read_csv(data_dir + 'movies_clean.csv')
reviews = pd.read_csv(data_dir + 'reviews_clean.csv')

del movies['Unnamed: 0']
del reviews['Unnamed: 0']


def create_train_test(reviews, order_by, training_size, testing_size):
    '''
    INPUT:
    reviews - (pandas df) dataframe to split into train and test
    order_by - (string) column name to sort by
    training_size - (int) number of rows in training set
    testing_size - (int) number of columns in the test set

    OUTPUT:
    training_df -  (pandas df) dataframe of the training set
    validation_df - (pandas df) dataframe of the test set
    '''

    df = reviews.sort_values(order_by, ascending=True, axis=0)
    df = df.iloc[:(training_size + testing_size), :].reset_index(drop=True)

    training_df = df.iloc[:training_size, :]
    validation_df = df.iloc[training_size:(training_size + testing_size),
                    :].reset_index(drop=True)

    return training_df, validation_df

# Nothing to change in this or the next cell
# Use our function to create training and test datasets
train_df, val_df = create_train_test(reviews, 'date', 8000, 2000)


def FunkSVD(ratings_mat, latent_features=12, learning_rate=0.0001, iters=100):
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
    n_users = ratings_mat.shape[0]
    n_movies = ratings_mat.shape[1]
    num_ratings = np.count_nonzero(~np.isnan(ratings_mat))

    # initialize the user and movie matrices with random values
    user_mat = np.random.rand(n_users, latent_features)
    movie_mat = np.random.rand(latent_features, n_movies)

    # initialize sse at 0 for first iteration
    sse_accum = 0

    # keep track of iteration and MSE
    print("Optimizaiton Statistics")
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

        # print results
        print("%d \t\t %f" % (iteration + 1, sse_accum / num_ratings))

    return user_mat, movie_mat

# Create user-by-item matrix - nothing to do here
train_user_item = train_df[['user_id', 'movie_id', 'rating', 'timestamp']]
train_data_df = train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
train_data_np = np.array(train_data_df)

# Fit FunkSVD with the specified hyper parameters to the training data
user_mat, movie_mat = FunkSVD(train_data_np, latent_features=15, learning_rate=0.005, iters=250)


def predict_rating(user_matrix, movie_matrix, user_id, movie_id):
    '''
    INPUT:
    user_matrix - user by latent factor matrix
    movie_matrix - latent factor by movie matrix
    user_id - the user_id from the reviews df
    movie_id - the movie_id according the movies df

    OUTPUT:
    pred - the predicted rating for user_id-movie_id according to FunkSVD
    '''
    # Use the training data to create a series of users and movies that matches the ordering in training data
    user_ids_series = train_data_df.index.values
    movie_ids_series = train_data_df.columns.values

    # User row and Movie Column
    user_row = np.where(user_ids_series == user_id)[0][0]
    movie_col = np.where(movie_ids_series == movie_id)[0][0]

    # Take dot product of that row and column in U and V to make prediction
    pred = np.dot(user_matrix[user_row, :], movie_matrix[:, movie_col])

    return pred

pred_val = predict_rating(user_mat, movie_mat, 8, 2844)


def print_prediction_summary(user_id, movie_id, prediction):
    '''
    INPUT:
    user_id - the user_id from the reviews df
    movie_id - the movie_id according the movies df
    prediction - the predicted rating for user_id-movie_id

    OUTPUT:
    None - prints a statement about the user, movie, and prediction made

    '''
    movie_name = \
    ((movies[movies['movie_id'] == movie_id]['movie']).reset_index(drop=True))[
        0]
    print('For user {} we predict a {} rating for the movie {}'.format(user_id,
                                    prediction, movie_name))

# Test your function the the results of the previous function
print_prediction_summary(8, 2844, pred_val)






