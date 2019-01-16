

data_dir = 'C:/Users/John/PycharmProjects/MovieTweeting-Data/data/'

# Read in the datasets
movies = pd.read_csv(data_dir + 'movies_clean.csv')
reviews = pd.read_csv(data_dir + 'reviews_clean.csv')

del movies['Unnamed: 0']
del reviews['Unnamed: 0']