from time import sleep
from models.ItemKNN_explicit import ItemKNN_explicit
from utils import load_data
from utils import eval_explicit
import warnings
import random
import warnings

import numpy as np
import random


warnings.filterwarnings('ignore')


def seed_everything(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)


seed = 1
seed_everything(seed)

sample_data = [[1, 1, 1, 2, 0],
               [2, 0, 2, 0, 1], 
               [0, 0, 0, 1, 1], 
               [0, 2, 0, 1, 3], 
               [3, 3, 0, 3, 2]]
sample_data = np.array(sample_data, dtype=np.float32)

itemknn = ItemKNN_explicit(train=np.copy(sample_data), valid=sample_data, top_k=5)

itemknn.fit()
if abs(-4.4522711634635925 - itemknn.item_item_sim_matrix.sum()) < 0.00001:
    print("ItemKNN_explicit.fit() is correct!")
else:
    print("ItemKNN_explicit.fit() is wrong!")

itemknn.item_item_sim_matrix = sample_data
if abs(6.666667 - np.sum(itemknn.predict(item_id=1, user_ids=[2, 3, 4]))) < 0.00001:
    print("ItemKNN_explicit.predict() is correct!")
else:
    print("ItemKNN_explicit.predict() is wrong!")