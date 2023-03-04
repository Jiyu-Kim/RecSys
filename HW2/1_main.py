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


from models.MF_explicit import MF_explicit
from models.BiasedMF_explicit import BiasedMF_explicit
from models.SVDpp_explicit import SVDpp_explicit

"""
dataset loading
"""
dataset = "movielens_1m.csv" # "movielens_1m.csv" , "naver_movie_dataset_100k.csv"
train_data, valid_data, test_data = load_data(dataset, implicit=False)

"""
model training
"""
print("model training...")
time_start = time()
#mf = MF_explicit(train=np.copy(train_data), valid=valid_data, n_features=10)
#biasedmf = BiasedMF_explicit(train=np.copy(train_data), valid=valid_data, n_features=10)
svdpp = SVDpp_explicit(train=np.copy(train_data), valid=valid_data, n_features=10)

#mf.fit()
#biasedmf.fit()
svdpp.fit()
print("training time: ", time()-time_start)
"""
model evaluation
"""
print("model evaluation")
#mf_rmse = eval_explicit(mf, train_data, test_data)
#biasedmf_rmse = eval_explicit(biasedmf, train_data, test_data)
svdpp_rmse = eval_explicit(svdpp, train_data, test_data)
print("evaluation time: ", time()-time_start)

print("RMSE on Test Data")
#print("MF: %f"%(mf_rmse))
#print("BiasedMF: %f"%(biasedmf_rmse))
print("SVD++: %f"%(svdpp_rmse))
