# 기본 패키지 import
from time import time
import numpy as np

from utils import load_data_CTR
from utils import eval_implicit_CTR
import warnings
import random
import warnings
import torch

import numpy as np
import random
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def seed_everything(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

seed = 1
seed_everything(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.FM_implicit import FM_implicit
from models.FFM_implicit import FFM_implicit
from models.DeepFM_implicit import DeepFM_implicit
"""
dataset loading
"""
dataset = "naver_movie_dataset.csv"

use_features_list = [['user_id', 'item_id', 'people', 'country', 'genre'], ['item_id', 'people', 'country', 'genre'], ['user_id', 'item_id', 'country', 'genre'], ['item_id', 'country', 'genre']]
total_field_dims = []
fm_params = []
ffm_params = []
deepfm_params = []

for use_features in use_features_list:
    train_arr, train_rating, valid_arr, valid_rating, test_arr, test_rating, field_dims = load_data_CTR(dataset, use_features, pos_threshold=6)

    print("# of total field dims: ", np.array(field_dims).sum())
    total_field_dims.append(np.array(field_dims).sum())

    fm = FM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                    learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
    ffm = FFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                    learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
    deepfm = DeepFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                            mlp_dims=[500, 500], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)

    print("# of fm parmas :", sum(p.numel() for p in fm.parameters() if p.requires_grad))
    print("# of ffm parmas :", sum(p.numel() for p in ffm.parameters() if p.requires_grad))
    print("# of deepfm parmas :", sum(p.numel() for p in deepfm.parameters() if p.requires_grad))

    fm_params.append(sum(p.numel() for p in fm.parameters() if p.requires_grad))
    ffm_params.append(sum(p.numel() for p in ffm.parameters() if p.requires_grad))
    deepfm_params.append(sum(p.numel() for p in deepfm.parameters() if p.requires_grad))


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

# scatter plot
plt.plot(total_field_dims, fm_params, label='FM', marker='x', color='red')
plt.plot(total_field_dims, ffm_params, label='FFM', marker='o', color='blue')
plt.plot(total_field_dims, deepfm_params, label='DeepFM', marker='*', color='green')
plt.legend()
plt.title('# of params over field dims')
plt.xlabel('# of field dims')
plt.ylabel('# of params')
plt.savefig('#_of_params_over_field_dims.png')
