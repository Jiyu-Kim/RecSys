# 기본 패키지 import
from time import time
import numpy as np

import warnings
import random
import warnings
import torch

import numpy as np
import random

warnings.filterwarnings('ignore')

def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

seed = 1
seed_everything(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import load_data_CTR
from utils import eval_implicit_CTR
from models.FM_implicit import FM_implicit
from models.FFM_implicit import FFM_implicit
from models.DeepFM_implicit import DeepFM_implicit
"""
dataset loading
"""
dataset = "naver_movie_dataset.csv" # "movielens_100k.csv" , "naver_movie_dataset.csv"
use_features = ['user_id', 'item_id', 'people', 'country', 'genre']
train_arr, train_rating, valid_arr, valid_rating, test_arr, test_rating, field_dims = load_data_CTR(dataset, use_features, pos_threshold=6)
top_k = 50

"""
model training
"""
print("model training...")
time_start = time()
fm = FM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                 learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
ffm = FFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                   learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)
deepfm = DeepFM_implicit(train_arr, train_rating, valid_arr, valid_rating, field_dims, num_epochs=100, embed_dim=20,
                         mlp_dims=[20, 20], dropout=0.2, learning_rate=0.01, reg_lambda=0.001, batch_size=1024, early_stop_trial=20, device=device)


fm.fit()
ffm.fit()
deepfm.fit()
print("training time: ", time()-time_start)

"""
model evaluation
"""
print("model evaluation")

FM_AUC, FM_logloss = eval_implicit_CTR(fm, test_arr, test_rating)
FFM_AUC, FFM_logloss = eval_implicit_CTR(ffm, test_arr, test_rating)
DeepFM_AUC, DeepFM_logloss = eval_implicit_CTR(deepfm, test_arr, test_rating)

print(f"[FM]\t Test_AUC = {FM_AUC:.4f} Test_logloss = {FM_logloss:.4f}")
print(f"[FFM]\t Test_AUC = {FFM_AUC:.4f} Test_logloss = {FFM_logloss:.4f}")
print(f"[DeepFM]\t Test_AUC = {DeepFM_AUC:.4f} Test_logloss = {DeepFM_logloss:.4f}")

"""
[FM]     Test_AUC = 0.8079 Test_logloss = 0.2291
[FFM]    Test_AUC = 0.8062 Test_logloss = 0.2340
[DeepFM]         Test_AUC = 0.8147 Test_logloss = 0.2322
"""