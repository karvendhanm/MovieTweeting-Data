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

def popular_recommendations(user_id, n_top, years, genres):
    '''
    INPUT:
    user_id - the user_id of the individual you are making recommendations for
    n_top - an integer of the number recommendations you want back
    OUTPUT:
    top_movies - a list of the n_top recommended movies by movie title in order best to worst
    '''

    list_df_year = []
    list_df_genre = []

    rating_df = reviews.groupby('movie_id')['rating'].agg(['mean', 'size'])
    latest_review_df = reviews.groupby('movie_id')['date'].agg(['max'])
    pop_movies = pd.merge(rating_df, latest_review_df, left_index=True,
                          right_index=True)
    pop_movies.drop(index=pop_movies.index[(pop_movies['size'] < 5)], axis=0,
                    inplace=True)
    pop_movies.sort_values(by=['mean', 'size', 'max'],
                                ascending=[False,False,False], inplace=True)
    # top_movies_idx = pop_movies.index[:n_top]
    top_movies_idx = pop_movies.index

    # Filtering for the years
    for year in years:
        list_df_year.append(movies[movies['date'] == int(year)])
    df_year = pd.concat(list_df_year)
    df_year.drop_duplicates(inplace=True)

    # Filtering for the genre
    for genre in genres:
        list_df_genre.append(movies[movies[genre] == 1])
    df_genre = pd.concat(list_df_genre)
    df_genre.drop_duplicates(inplace=True)

    # merging dataframes using index as index here is movie_ids
    # when genres or years is sent in empty, the following code handles it
    if (df_year.shape[0] > 0) & (df_genre.shape[0] > 0):
        df_movie_year_genre = pd.merge(df_year, df_genre, right_index=True,
                                       left_index=True)
    elif (df_year.shape[0] == 0):
        df_movie_year_genre = df_genre

    elif (df_genre.shape[0] == 0):
        df_movie_year_genre = df_year

    else:
        return np.nan

    # making a list of top movies as per the criteria defined above
    top_movies = []
    for idx in top_movies_idx:
        temp = df_movie_year_genre.loc[df_movie_year_genre['movie_id_x'] ==
                                    idx,'movie_x'].reset_index(drop = True)
        if(temp.shape[0] > 0):
            top_movies.append(temp[0])

    return top_movies[:n_top]  # a list of the n_top movies as recommended

popular_recommendations(1, 20,years=['2015', '2016', '2017','2018'], genres=['History'])


popular_recommendations(1, 20,years=['2015', '2016', '2017',
                                    '2018'], genres=['History'])
