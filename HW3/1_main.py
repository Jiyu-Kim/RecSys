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
from models.FISM_implicit import FISM_implicit
"""
dataset loading
"""
dataset = "movielens_100k.csv" # "movielens_100k.csv" , "naver_movie_dataset_small.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=True)
top_k = 50

"""
model training
"""
print("model training...")
time_start = time()
mf = MF_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, learning_rate=0.1, num_epochs=200, device=device)
#wmf = WMF_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, num_epochs=10)
#bpr = BPR_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, num_epochs=30, learning_rate=0.1, device=device)
fism = FISM_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, num_epochs=30, learning_rate=0.1, device=device)
#ease = EASE_implicit(train=np.copy(train_data), reg_lambda=1000)


mf.fit()
#wmf.fit()
#bpr.fit()
fism.fit()
#ease.fit()
print("training time: ", time()-time_start)
"""
model evaluation
"""
print("model evaluation")

mf_prec, mf_recall, mf_ndcg, mf_mrr, mf_map = eval_implicit(mf, train_data, test_data, top_k)
#wmf_prec, wmf_recall, wmf_ndcg, wmf_mrr, wmf_map = eval_implicit(wmf, train_data, test_data, top_k)
#bpr_prec, bpr_recall, bpr_ndcg, bpr_mrr, bpr_map = eval_implicit(bpr, train_data, test_data, top_k)
fism_prec, fism_recall, fism_ndcg, fism_mrr, fism_map = eval_implicit(fism, train_data, test_data, top_k)
#ease_prec, ease_recall, ease_ndcg, ease_mrr, ease_map = eval_implicit(ease, train_data, test_data, top_k)
print("evaluation time: ", time()-time_start)

print(f"MF: prec@{top_k} {mf_prec}, recall@{top_k} {mf_recall}, ndcg@{top_k} {mf_ndcg} mrr@{top_k} {mf_mrr} map@{top_k} {mf_map}")
#print(f"WMF: prec@{top_k} {wmf_prec}, recall@{top_k} {wmf_recall}, ndcg@{top_k} {wmf_ndcg} mrr@{top_k} {wmf_mrr} map@{top_k} {wmf_map}")
#print(f"BPR: prec@{top_k} {bpr_prec}, recall@{top_k} {bpr_recall}, ndcg@{top_k} {bpr_ndcg} mrr@{top_k} {bpr_mrr} map@{top_k} {bpr_map}")
print(f"FISM: prec@{top_k} {fism_prec}, recall@{top_k} {fism_recall}, ndcg@{top_k} {fism_ndcg} mrr@{top_k} {fism_mrr} map@{top_k} {fism_map}")
#print(f"EASE: prec@{top_k} {ease_prec}, recall@{top_k} {ease_recall}, ndcg@{top_k} {ease_ndcg} mrr@{top_k} {ease_mrr} map@{top_k} {ease_map}")
