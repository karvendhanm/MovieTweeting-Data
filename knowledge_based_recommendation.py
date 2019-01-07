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




