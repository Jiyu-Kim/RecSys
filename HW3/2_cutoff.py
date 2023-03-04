# 기본 패키지 import
from time import time
import numpy as np

from utils import load_data
from utils import eval_implicit
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

from models.MF_implicit import MF_implicit
from models.WMF_implicit import WMF_implicit
from models.BPR_implicit import BPR_implicit
from models.EASE_implicit import EASE_implicit
#from models.FISM_implicit import FISM_implicit
"""
dataset loading
"""
dataset = "naver_movie_dataset_small.csv" # "movielens_100k.csv" , "naver_movie_dataset_small.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=True)
"""
model training
"""
print("model training...")
time_start = time()
mf = MF_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, learning_rate=0.1, num_epochs=200, device=device)
wmf = WMF_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, num_epochs=10)
bpr = BPR_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, num_epochs=30, learning_rate=0.1, device=device)
#fism = FISM_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, num_epochs=30, learning_rate=0.1, device=device)
ease = EASE_implicit(train=np.copy(train_data), reg_lambda=1000)


mf.fit()
wmf.fit()
bpr.fit()
#fism.fit()
ease.fit()
print("training time: ", time()-time_start)
"""
model evaluation
"""
print("model evaluation")

top_k_list = [1, 3, 5, 10, 20, 50, 100]
mf_ndcg_list, mf_mrr_list, mf_map_list = [], [], []
wmf_ndcg_list, wmf_mrr_list, wmf_map_list = [], [], []
bpr_ndcg_list, bpr_mrr_list, bpr_map_list = [], [], []
#fism_ndcg_list, fism_mrr_list, fism_map_list = [], [], []
ease_ndcg_list, ease_mrr_list, ease_map_list = [], [], []

for top_k in top_k_list:
    mf_prec, mf_recall, mf_ndcg, mf_mrr, mf_map = eval_implicit(mf, train_data, test_data, top_k)
    wmf_prec, wmf_recall, wmf_ndcg, wmf_mrr, wmf_map = eval_implicit(wmf, train_data, test_data, top_k)
    bpr_prec, bpr_recall, bpr_ndcg, bpr_mrr, bpr_map = eval_implicit(bpr, train_data, test_data, top_k)
    #fism_prec, fism_recall, fism_ndcg, fism_mrr, fism_map = eval_implicit(fism, train_data, test_data, top_k)
    ease_prec, ease_recall, ease_ndcg, ease_mrr, ease_map = eval_implicit(ease, train_data, test_data, top_k)

    mf_ndcg_list.append(mf_ndcg); mf_mrr_list.append(mf_mrr); mf_map_list.append(mf_map)
    wmf_ndcg_list.append(wmf_ndcg); wmf_mrr_list.append(wmf_mrr); wmf_map_list.append(wmf_map)
    bpr_ndcg_list.append(bpr_ndcg); bpr_mrr_list.append(bpr_mrr); bpr_map_list.append(bpr_map)
    #fism_ndcg_list.append(fism_ndcg); fism_mrr_list.append(fism_mrr); fism_map_list.append(fism_map)
    ease_ndcg_list.append(ease_ndcg); ease_mrr_list.append(ease_mrr); ease_map_list.append(ease_map)
print("evaluation time: ", time()-time_start)

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

# scatter plot ndcg
plt.plot(top_k_list, mf_ndcg_list, label='MF', marker='x', color='red')
plt.plot(top_k_list, wmf_ndcg_list, label='WMF', marker='o', color='blue')
plt.plot(top_k_list, bpr_ndcg_list, label='BPR', marker='*', color='green')
#plt.plot(top_k_list, fism_ndcg_list, label='FISM', marker='+', color='black')
plt.plot(top_k_list, ease_ndcg_list, label='EASE', marker='.', color='yellow')
plt.legend()
plt.title(f'ndcg cutoff results {dataset}')
plt.xlabel('k')
plt.ylabel('ndcg')
plt.savefig(f'cutoff results ndcg {dataset}.png')

plt.clf()

#  scatter plot mrr
plt.plot(top_k_list, mf_mrr_list, label='MF', marker='x', color='red')
plt.plot(top_k_list, wmf_mrr_list, label='WMF', marker='o', color='blue')
plt.plot(top_k_list, bpr_mrr_list, label='BPR', marker='*', color='green')
#plt.plot(top_k_list, fism_mrr_list, label='FISM', marker='+', color='black')
plt.plot(top_k_list, ease_mrr_list, label='EASE', marker='.', color='yellow')
plt.legend()
plt.title(f'mrr cutoff results {dataset}')
plt.xlabel('k')
plt.ylabel('mrr')
plt.savefig(f'cutoff results mrr {dataset}.png')

plt.clf()

# scatter plot map
plt.plot(top_k_list, mf_map_list, label='MF', marker='x', color='red')
plt.plot(top_k_list, wmf_map_list, label='WMF', marker='o', color='blue')
plt.plot(top_k_list, bpr_map_list, label='BPR', marker='*', color='green')
#plt.plot(top_k_list, fism_map_list, label='FISM', marker='+', color='black')
plt.plot(top_k_list, ease_map_list, label='EASE', marker='.', color='yellow')
plt.legend()
plt.title(f'map cutoff results {dataset}')
plt.xlabel('k')
plt.ylabel('map')
plt.savefig(f'cutoff results map {dataset}.png')
