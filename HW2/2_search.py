# 기본 패키지 import
import numpy as np
from utils import load_data
from utils import eval_explicit
import warnings
import random
import warnings

import numpy as np
import random
import matplotlib.pyplot as plt

import torch


warnings.filterwarnings('ignore')

def seed_everything(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

seed = 1
seed_everything(seed)

from models.BiasedMF_explicit import BiasedMF_explicit
from models.SVDpp_explicit import SVDpp_explicit

"""
dataset loading
"""
dataset = "naver_movie_dataset_100k.csv" # "naver_movie_dataset_100k.csv", "movielens_1m.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=False)

# ========================= EDIT HERE ========================
"""
Specify values of the parameter to search.
"""
_rank = [1, 10, 100, 1000, 2000]
# ============================================================


biased_mf_test_rmse = []
for i, space in enumerate(_rank):
    model = BiasedMF_explicit(train=np.copy(train_data), valid=valid_data, n_features=space)

    model.fit()
    rmse = eval_explicit(model, train_data+valid_data, test_data)

    print("BiasedMF RSME (rank=%d): " % space, rmse)
    biased_mf_test_rmse.append(rmse)


SVDpp_test_rmse = []
for i, space in enumerate(_rank):
    model = SVDpp_explicit(train=np.copy(train_data), valid=valid_data, n_features=space)

    model.fit()
    rmse = eval_explicit(model, train_data+valid_data, test_data)

    print("SVD++ RSME (rank=%d): " % space, rmse)
    SVDpp_test_rmse.append(rmse)

"""
Draw scatter plot of search results.
- X-axis: search paramter
- Y-axis: RMSE (Train, Test respectively)

Put title, X-axis name, Y-axis name in your plot.

Resources
------------
Official document: https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.scatter.html
"Data Visualization in Python": https://medium.com/python-pandemonium/data-visualization-in-python-scatter-plots-in-matplotlib-da90ac4c99f9
"""

num_space = len(_rank)
plt.scatter(_rank, biased_mf_test_rmse, label='BiasedMF', marker='x', s=150)
plt.scatter(_rank, SVDpp_test_rmse, label='SVD++', marker='o', s=150)
plt.legend()
plt.title('Search results_naver_movie_dataset_100k.csv')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.savefig('Search results_naver_movie_dataset_100k.csv.png')
