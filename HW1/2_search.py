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


warnings.filterwarnings('ignore')

def seed_everything(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

seed = 1
seed_everything(seed)


from models.UserKNN_explicit import UserKNN_explicit
from models.ItemKNN_explicit import ItemKNN_explicit

"""
dataset loading
"""
dataset = "movielens_100k.csv" #"naver_movie_dataset_small.csv" # "movielens_100k.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=False)

# ========================= EDIT HERE ========================
"""
Specify values of the parameter to search.
"""
_userk = [1, 10, 100, 1000, 2000]
_itemk = [1, 5, 10, 100, 500]
# ============================================================

user_knn_test_rmse = []
for i, space in enumerate(_userk):
    model = UserKNN_explicit(train=np.copy(train_data), valid=valid_data, top_k=space)

    model.fit()
    rmse = eval_explicit(model, train_data+valid_data, test_data)

    print("UserKNN RSME (k=%d): " % space, rmse)
    user_knn_test_rmse.append(rmse)

item_knn_test_rmse = []
for i, space in enumerate(_itemk):
    model = ItemKNN_explicit(train=np.copy(train_data), valid=valid_data, top_k=space)

    model.fit()
    rmse = eval_explicit(model, train_data+valid_data, test_data)

    print("ItemKNN RSME (k=%d): " % space, rmse)
    item_knn_test_rmse.append(rmse)

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

num_space = 5
plt.scatter(_userk, user_knn_test_rmse, label='UserKNN', marker='x', s=150)
plt.scatter(_itemk, item_knn_test_rmse, label='ItemKNN', marker='o', s=150)
plt.legend()
plt.title('Search results2')
plt.xlabel('k')
plt.ylabel('RMSE')
plt.savefig('Search results2.png')
