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
from models.AE_implicit import AE_implicit
from models.CDAE_implicit import CDAE_implicit
from models.MultVAE_implicit import MultVAE_implicit
from models.LightGCN_implicit import LightGCN_implicit
"""
dataset loading
"""
dataset = "movielens_100k.csv" # "movielens_100k.csv" , "naver_movie_dataset_small.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=True)
"""
model training
"""
print("model training...")
time_start = time()
mf = MF_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, learning_rate=0.1, num_epochs=200, device=device)
ae = AE_implicit(train=np.copy(train_data), valid=valid_data, hidden_dim=500, num_epochs=150, learning_rate=0.01, reg_lambda=0.0001, device = device)
cdae = CDAE_implicit(train=np.copy(train_data), valid=valid_data, hidden_dim=500, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.001, device = device)
multvae = MultVAE_implicit(train=np.copy(train_data), valid=valid_data, hidden_dim=500, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.001, device = device) 
lightgcn = LightGCN_implicit(train=np.copy(train_data), valid=valid_data, learning_rate=0.005, regs=0.001, batch_size=2048, num_epochs=70, emb_size=400, num_layers=1, node_dropout=0.0, device=device)



mf.fit()
ae.fit()
cdae.fit()
multvae.fit()
lightgcn.fit()

print("training time: ", time()-time_start)
"""
model evaluation
"""
print("model evaluation")

top_k_list = [1, 10, 20, 50, 100]
mf_prec_list, mf_recall_list, mf_ndcg_list = [], [], []
ae_prec_list, ae_recall_list, ae_ndcg_list = [], [], []
cdae_prec_list, cdae_recall_list, cdae_ndcg_list = [], [], []
multvae_prec_list, multvae_recall_list, multvae_ndcg_list = [], [], []
lightgcn_prec_list, lightgcn_recall_list, lightgcn_ndcg_list = [], [], []

for top_k in top_k_list:
    mf_prec, mf_recall, mf_ndcg = eval_implicit(mf, train_data, test_data, top_k)
    ae_prec, ae_recall, ae_ndcg = eval_implicit(ae, train_data, test_data, top_k)
    cdae_prec, cdae_recall, cdae_ndcg = eval_implicit(cdae, train_data, test_data, top_k)
    multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
    lightgcn_prec, lightgcn_recall, lightgcn_ndcg = eval_implicit(lightgcn, train_data, test_data, top_k)

    mf_prec_list.append(mf_prec); mf_recall_list.append(mf_recall); mf_ndcg_list.append(mf_ndcg)
    ae_prec_list.append(ae_prec); ae_recall_list.append(ae_recall); ae_ndcg_list.append(ae_ndcg)
    cdae_prec_list.append(cdae_prec); cdae_recall_list.append(cdae_recall); cdae_ndcg_list.append(cdae_ndcg)
    multvae_prec_list.append(multvae_prec); multvae_recall_list.append(multvae_recall); multvae_ndcg_list.append(multvae_ndcg)
    lightgcn_prec_list.append(lightgcn_prec); lightgcn_recall_list.append(lightgcn_recall); lightgcn_ndcg_list.append(lightgcn_ndcg)
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
plt.plot(top_k_list, mf_prec_list, label='MF', marker='x', color='red')
plt.plot(top_k_list, ae_prec_list, label='AE', marker='o', color='blue')
plt.plot(top_k_list, cdae_prec_list, label='CDAE', marker='*', color='green')
plt.plot(top_k_list, multvae_prec_list, label='MultVAE', marker='+', color='black')
plt.plot(top_k_list, lightgcn_prec_list, label='LightGCN', marker='.', color='yellow')
plt.legend()
plt.title(f'precision cutoff results {dataset}')
plt.xlabel('k')
plt.ylabel('precision')
plt.savefig(f'cutoff results precision {dataset}.png')

plt.clf()

#  scatter plot prec
plt.plot(top_k_list, mf_recall_list, label='MF', marker='x', color='red')
plt.plot(top_k_list, ae_recall_list, label='AE', marker='o', color='blue')
plt.plot(top_k_list, cdae_recall_list, label='CDAE', marker='*', color='green')
plt.plot(top_k_list, multvae_recall_list, label='MultVAE', marker='+', color='black')
plt.plot(top_k_list, lightgcn_recall_list, label='LightGCN', marker='.', color='yellow')
plt.legend()
plt.title(f'recall cutoff results {dataset}')
plt.xlabel('k')
plt.ylabel('recall')
plt.savefig(f'cutoff results recall {dataset}.png')

plt.clf()

# scatter plot recall
plt.plot(top_k_list, mf_ndcg_list, label='MF', marker='x', color='red')
plt.plot(top_k_list, ae_ndcg_list, label='AE', marker='o', color='blue')
plt.plot(top_k_list, cdae_ndcg_list, label='CDAE', marker='*', color='green')
plt.plot(top_k_list, multvae_ndcg_list, label='MultVAE', marker='+', color='black')
plt.plot(top_k_list, lightgcn_ndcg_list, label='LightGCN', marker='.', color='yellow')
plt.legend()
plt.title(f'ndcg cutoff results {dataset}')
plt.xlabel('k')
plt.ylabel('ndcg')
plt.savefig(f'cutoff results ndcg {dataset}.png')
