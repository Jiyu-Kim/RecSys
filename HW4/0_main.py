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
from models.AE_implicit import AE_implicit
from models.CDAE_implicit import CDAE_implicit
from models.MultVAE_implicit import MultVAE_implicit
from models.LightGCN_implicit import LightGCN_implicit


"""
dataset loading
"""
dataset = "naver_movie_dataset_small.csv" # "movielens_100k.csv" , "naver_movie_dataset_small.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=True)
top_k = 50


"""
model training
"""
print("model training...")
time_start = time()
#mf = MF_implicit(train=np.copy(train_data), valid=valid_data, n_features=10, learning_rate=0.1, num_epochs=200, device=device)
#ae = AE_implicit(train=np.copy(train_data), valid=valid_data, hidden_dim=500, num_epochs=150, learning_rate=0.01, reg_lambda=0.0001, device = device)
#cdae = CDAE_implicit(train=np.copy(train_data), valid=valid_data, hidden_dim=500, num_epochs=150, learning_rate=0.01, dropout=0.5, reg_lambda=0.001, device = device)
multvae = MultVAE_implicit(train=np.copy(train_data), valid=valid_data, hidden_dim=500, dropout=0.5, num_epochs=150, learning_rate=0.001, reg_lambda=0.001, device = device) 
#lightgcn = LightGCN_implicit(train=np.copy(train_data), valid=valid_data, learning_rate=0.005, regs=0.001, batch_size=2048, num_epochs=70, emb_size=400, num_layers=1, node_dropout=0.0, device=device)


#mf.fit()
#ae.fit()
#cdae.fit()
multvae.fit()
#lightgcn.fit()

print("training time: ", time()-time_start)
"""
model evaluation
"""
print("model evaluation")

#mf_prec, mf_recall, mf_ndcg = eval_implicit(mf, train_data, test_data, top_k)
#ae_prec, ae_recall, ae_ndcg = eval_implicit(ae, train_data, test_data, top_k)
#cdae_prec, cdae_recall, cdae_ndcg = eval_implicit(cdae, train_data, test_data, top_k)
multvae_prec, multvae_recall, multvae_ndcg = eval_implicit(multvae, train_data, test_data, top_k)
#lightgcn_prec, lightgcn_recall, lightgcn_ndcg = eval_implicit(lightgcn, train_data, test_data, top_k)

print("evaluation time: ", time()-time_start)

#print(f"MF: prec@{top_k} {mf_prec}, recall@{top_k} {mf_recall}, ndcg@{top_k} {mf_ndcg}")
#print(f"AE: prec@{top_k} {ae_prec}, recall@{top_k} {ae_recall}, ndcg@{top_k} {ae_ndcg}")
#print(f"CDAE: prec@{top_k} {cdae_prec}, recall@{top_k} {cdae_recall}, ndcg@{top_k} {cdae_ndcg}")
print(f"MultVAE: prec@{top_k} {multvae_prec}, recall@{top_k} {multvae_recall}, ndcg@{top_k} {multvae_ndcg}")
#print(f"LightGCN: prec@{top_k} {lightgcn_prec}, recall@{top_k} {lightgcn_recall}, ndcg@{top_k} {lightgcn_ndcg}")
