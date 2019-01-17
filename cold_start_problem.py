import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
    reviews_new = reviews.sort_values(order_by)
    training_df = reviews_new.head(training_size)
    validation_df = reviews_new.iloc[
                    training_size:training_size + testing_size]

    return training_df, validation_df


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
    # Create series of users and movies in the right order
    user_ids_series = np.array(train_data_df.index)
    movie_ids_series = np.array(train_data_df.columns)

    # User row and Movie Column
    user_row = np.where(user_ids_series == user_id)[0][0]
    movie_col = np.where(movie_ids_series == movie_id)[0][0]

    # Take dot product of that row and column in U and V to make prediction
    pred = np.dot(user_matrix[user_row, :], movie_matrix[:, movie_col])

    return pred


# Use our function to create training and test datasets
train_df, val_df = create_train_test(reviews, 'date', 8000, 2000)

# Create user-by-item matrix - this will keep track of order of users and movies in u and v
train_user_item = train_df[['user_id', 'movie_id', 'rating', 'timestamp']]
train_data_df = train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
train_data_np = np.array(train_data_df)

# Read in user and movie matrices
user_file = open(data_dir + "user_matrix", 'rb')
user_mat = pickle.load(user_file)
user_file.close()

movie_file = open(data_dir + "movie_matrix", 'rb')
movie_mat = pickle.load(movie_file)
movie_file.close()

# Validating Predictions
# Unfortunately, you weren't able to make predictions on every user-movie combination in the test set, as some of these users or movies were new.
# However, we can validate your predictions for the user-movie pairs that do exist in the user_mat and movie_mat matrices.

def validation_comparison(val_df, user_mat=user_mat, movie_mat=movie_mat):
    '''
    INPUT:
    val_df - the validation dataset created in the third cell above
    user_mat - U matrix in FunkSVD
    movie_mat - V matrix in FunkSVD

    OUTPUT:
    rmse - RMSE of how far off each value is from it's predicted value
    perc_rated - percent of predictions out of all possible that could be rated
    actual_v_pred - a 10 x 10 grid with counts for actual vs predicted values
    preds - (list) predictions for any user-movie pairs where it was possible to make a prediction
    acts - (list) actual values for any user-movie pairs where it was possible to make a prediction
    '''

    val_users = val_df.user_id.values
    val_movies = val_df.movie_id.values
    val_ratings = val_df.rating.values

    sse = 0
    num_rated = 0
    preds, acts = [], []
    actual_v_pred = np.zeros((10, 10))
    for idx in range(len(val_users)):
        try:
            pred = predict_rating(user_mat, movie_mat, val_users[idx], val_movies[idx])
            sse += (val_ratings[idx] - pred)**2
            num_rated += 1
            preds.append(pred)
            acts.append(val_ratings[idx])
            actual_v_pred[11 - int(val_ratings[idx] - 1), int(round(pred) - 1)] += 1
        except:
            continue

    rmse = np.sqrt(sse/num_rated)
    perc_rated = num_rated / len(val_users)
    return rmse, perc_rated, actual_v_pred, preds, acts

rmse, perc_rated, actual_v_pred, preds, acts = validation_comparison(val_df)
print(rmse, perc_rated)
sns.heatmap(actual_v_pred)
plt.xticks(ticks = np.arange(10), labels = np.arange(1, 11))
plt.yticks(ticks = np.arange(10), labels = np.arange(1, 11))
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs. Predicted Values')

plt.figure(figsize=(8,8))
plt.hist(acts, normed=True, alpha=.5, label='actual');
plt.hist(preds, normed=True, alpha=.5, label='predicted');
plt.legend(loc=2, prop={'size': 15});
plt.xlabel('Rating');
plt.title('Predicted vs. Actual Rating');

# From the above, this can be calculated as follows:
print("Number not rated {}".format(int(len(val_df['rating'])*(1-perc_rated))))
print("Number rated {}.".format(int(len(val_df['rating'])*perc_rated)))

# Subset so movie_content is only using the dummy variables for each genre and the 3 century based year dummy columns
movie_content = np.array(movies.iloc[:, 4:])

# Take the dot product to obtain a movie x movie matrix of similarities
dot_prod_movies = movie_content.dot(np.transpose(movie_content))


def find_similar_movies(movie_id):
    '''
    INPUT
    movie_id - a movie_id
    OUTPUT
    similar_movies - an array of the most similar movies by title
    '''
    # find the row of each movie id
    movie_idx = np.where(movies['movie_id'] == movie_id)[0][0]

    # find the most similar movie indices - to start I said they need to be the same for all content
    similar_idxs = \
    np.where(dot_prod_movies[movie_idx] == np.max(dot_prod_movies[movie_idx]))[
        0]

    # pull the movie titles based on the indices
    similar_movies = np.array(movies.iloc[similar_idxs,]['movie'])

    return similar_movies


def get_movie_names(movie_ids):
    '''
    INPUT
    movie_ids - a list of movie_ids
    OUTPUT
    movies - a list of movie names associated with the movie_ids

    '''
    movie_lst = list(movies[movies['movie_id'].isin(movie_ids)]['movie'])

    return movie_lst

def create_ranked_df(movies, reviews):
    '''
    INPUT
    movies - the movies dataframe
    reviews - the reviews dataframe

    OUTPUT
    ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews,
                    then time, and must have more than 4 ratings
    '''

    # Pull the average ratings and number of ratings for each movie
    movie_ratings = reviews.groupby('movie_id')['rating']
    avg_ratings = movie_ratings.mean()
    num_ratings = movie_ratings.count()
    last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
    last_rating.columns = ['last_rating']

    # Add Dates
    rating_count_df = pd.DataFrame(
        {'avg_rating': avg_ratings, 'num_ratings': num_ratings})
    rating_count_df = rating_count_df.join(last_rating)

    # merge with the movies dataset
    movie_recs = movies.set_index('movie_id').join(rating_count_df)

    # sort by top avg rating and number of ratings
    ranked_movies = movie_recs.sort_values(
        ['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

    # for edge cases - subset the movie list to those with only 5 or more reviews
    ranked_movies = ranked_movies[ranked_movies['num_ratings'] > 4]

    return ranked_movies


def popular_recommendations(user_id, n_top, ranked_movies):
    '''
    INPUT:
    user_id - the user_id (str) of the individual you are making recommendations for
    n_top - an integer of the number recommendations you want back
    ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time

    OUTPUT:
    top_movies - a list of the n_top recommended movies by movie title in order best to worst
    '''

    top_movies = list(ranked_movies['movie'][:n_top])

    return top_movies











