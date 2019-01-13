import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import tests as t
import helper as h


play_data = pd.DataFrame({'x1': [-3, -2, -1, 0, 1, 2, 3],
               'x2': [1, 2, 3, 4, 5, 6, 7],
               'x3': [2,5,10,17,26,37,50],
               'x4': [2, 5, 15, 27, 28, 30, 31]
})


# Pearson's Correlation Coefficient
def pearson_corr(x, y):
    '''
    INPUT
    x - an array of matching length to array y
    y - an array of matching length to array x
    OUTPUT
    corr - the pearson correlation coefficient for comparing x and y
    '''
    x_bar = x.mean()
    y_bar = y.mean()

    numerator = np.sum((x - x_bar) * (y - y_bar))
    denominator = np.sqrt(np.sum((x - x_bar) ** 2)) * np.sqrt(
        np.sum((y - y_bar) ** 2))

    corr = numerator / denominator

    return corr

pearson_corr(play_data['x2'],play_data['x3'])

# # This cell will test your function against the built in numpy function
# assert pearson_corr(play_data['x1'], play_data['x2']) == np.corrcoef(play_data['x1'], play_data['x2'])[0][1], 'Oops!  The correlation between the first two columns should be 0, but your function returned {}.'.format(pearson_corr(play_data['x1'], play_data['x2']))
# assert round(pearson_corr(play_data['x1'], play_data['x3']), 2) == np.corrcoef(play_data['x1'], play_data['x3'])[0][1], 'Oops!  The correlation between the first and third columns should be {}, but your function returned {}.'.format(np.corrcoef(play_data['x1'], play_data['x3'])[0][1], pearson_corr(play_data['x1'], play_data['x3']))
# assert round(pearson_corr(play_data['x3'], play_data['x4']), 2) == round(np.corrcoef(play_data['x3'], play_data['x4'])[0][1], 2), 'Oops!  The correlation between the first and third columns should be {}, but your function returned {}.'.format(np.corrcoef(play_data['x3'], play_data['x4'])[0][1], pearson_corr(play_data['x3'], play_data['x4']))
# print("If this is all you see, it looks like you are all set!  Nice job coding up Pearson's correlation coefficient!")


# Spearman's Correlation Coefficient
# Very similar to pearson's correlation coefficient.  But instead of using
# values themselves we use the rank.
def corr_spearman(x, y):
    '''
    INPUT
    x - an array of matching length to array y
    y - an array of matching length to array x
    OUTPUT
    corr - the spearman correlation coefficient for comparing x and y
    '''

    x_bar = x.rank().mean()
    y_bar = y.rank().mean()

    numerator = np.sum((x.rank() - x_bar) * (y.rank() - y_bar))
    denominator = np.sqrt(np.sum((x.rank() - x_bar) ** 2)) * np.sqrt(
        np.sum((y.rank() - y_bar) ** 2))

    corr = numerator / denominator

    return corr


# # This cell will test your function against the built in scipy function
# assert corr_spearman(play_data['x1'], play_data['x2']) == spearmanr(play_data['x1'], play_data['x2'])[0], 'Oops!  The correlation between the first two columns should be 0, but your function returned {}.'.format(compute_corr(play_data['x1'], play_data['x2']))
# assert round(corr_spearman(play_data['x1'], play_data['x3']), 2) == spearmanr(play_data['x1'], play_data['x3'])[0], 'Oops!  The correlation between the first and third columns should be {}, but your function returned {}.'.format(np.corrcoef(play_data['x1'], play_data['x3'])[0][1], compute_corr(play_data['x1'], play_data['x3']))
# assert round(corr_spearman(play_data['x3'], play_data['x4']), 2) == round(spearmanr(play_data['x3'], play_data['x4'])[0], 2), 'Oops!  The correlation between the first and third columns should be {}, but your function returned {}.'.format(np.corrcoef(play_data['x3'], play_data['x4'])[0][1], compute_corr(play_data['x3'], play_data['x4']))
# print("If this is all you see, it looks like you are all set!  Nice job coding up Spearman's correlation coefficient!")



# Kendall's Tau
def kendalls_tau(x, y):
    '''
    INPUT
    x - an array of matching length to array y
    y - an array of matching length to array x
    OUTPUT
    tau - the kendall's tau for comparing x and y
    '''

    x = x.rank()
    y = y.rank()
    n = len(x)

    sum_vals = 0
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        for j, (x_j, y_j) in enumerate(zip(x, y)):
            if i < j:
                sum_vals += np.sign(x_i - x_j) * np.sign(y_i - y_j)

    tau = (2 / (n * (n - 1))) * (sum_vals)

    return tau

# Distance Metrics
# Euclidean Distance
def eucl_dist(x, y):
    '''
    INPUT
    x - an array of matching length to array y
    y - an array of matching length to array x
    OUTPUT
    euc - the euclidean distance between x and y
    '''

    eucl_dist = np.sqrt(sum((x - y) ** 2))
    return eucl_dist

# Manhattan distance
def manhat_dist(x, y):
    '''
    INPUT
    x - an array of matching length to array y
    y - an array of matching length to array x
    OUTPUT
    manhat - the manhattan distance between x and y
    '''

    manhat_dist = sum(np.abs(y - x))
    return manhat_dist


















