# 기본 패키지 import
from time import time
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
dataset = "naver_movie_dataset_small.csv" # "movielens_100k.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=False)

"""
model training
"""
print("model training...")
time_start = time()
userknn = ItemKNN_explicit(train=np.copy(train_data), valid=valid_data, top_k=10)
# itemknn = ItemKNN_explicit(train=np.copy(train_data), valid=valid_data, top_k=10)    

userknn.fit()
# itemknn.fit()
print("training time: ", time()-time_start)
"""
model evaluation
"""
print("model evaluation")
userknn_rmse = eval_explicit(userknn, train_data+valid_data, test_data)
# itemknn_rmse = eval_explicit(itemknn, train_data+valid_data, test_data)
print("evaluation time: ", time()-time_start)

print("RMSE on Test Data")
print("UserKNN: %f"%(userknn_rmse))
# print("ItemKNN: %f"%(itemknn_rmse))