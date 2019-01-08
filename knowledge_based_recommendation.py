import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tests as t

data_dir = 'C:/Users/John/PycharmProjects/MovieTweeting-Data/data/'

movies = pd.read_csv(data_dir+'movies_clean.csv')
reviews = pd.read_csv(data_dir+'reviews_clean.csv')
del movies['Unnamed: 0']
del reviews['Unnamed: 0']


# Find the most popular movies:
# The task is no matter the user, we need to provide a list of the
# recommendations based on simply the most popular items.

#For this task, we will consider what is "most popular" based on the
# following criteria:

# 1) A movie with the highest average rating is considered best
# 2) With ties, movies that have more ratings are better
# 3) A movie must have a minimum of 5 ratings to be considered among the best
# movies
# 4) If movies are tied in their average rating and number of ratings,
# the ranking is determined by the movie that is the most recent rating.

def popular_recommendations(user_id, n_top):
    '''
    INPUT:
    user_id - the user_id of the individual you are making recommendations for
    n_top - an integer of the number recommendations you want back
    OUTPUT:
    top_movies - a list of the n_top recommended movies by movie title in order best to worst
    '''

    rating_df = reviews.groupby('movie_id')['rating'].agg(['mean', 'size'])
    latest_review_df = reviews.groupby('movie_id')['date'].agg(['max'])
    pop_movies = pd.merge(rating_df, latest_review_df, left_index=True,
                          right_index=True)
    pop_movies.drop(index=pop_movies.index[(pop_movies['size'] < 5)], axis=0,
                    inplace=True)
    pop_movies.sort_values(by=['mean', 'size', 'max'],
                                ascending=[False,False,False], inplace=True)
    top_movies_idx = pop_movies.index[:n_top]

    top_movies = []
    for idx in top_movies_idx:
        temp = movies.loc[movies['movie_id'] == idx,'movie'].reset_index(drop = True)
        top_movies.append(temp[0])

    return top_movies  # a list of the n_top movies as recommended


recs_20_for_1 = popular_recommendations(1, 20)
recs_5_for_53968 = popular_recommendations(53968, 5)
recs_100_for_70000 = popular_recommendations(70000, 100)
recs_35_for_43 = popular_recommendations(43, 35)

ranked_movies = t.create_ranked_df(movies, reviews) # only run this once - it is not fast

# check 1
assert t.popular_recommendations('1', 20, ranked_movies) == recs_20_for_1,  "The first check failed..."
# check 2
assert t.popular_recommendations('53968', 5, ranked_movies) == recs_5_for_53968,  "The second check failed..."
# check 3
assert t.popular_recommendations('70000', 100, ranked_movies) == recs_100_for_70000,  "The third check failed..."
# check 4
assert t.popular_recommendations('43', 35, ranked_movies) == recs_35_for_43,  "The fourth check failed..."

print("If you got here, looks like you are good to go!  Nice job!")


