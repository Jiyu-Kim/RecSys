from models.SlopeOnePredictor_explicit import SlopeOnePredictor_explicit
from models.SVDpp_explicit import SVDpp_explicit
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

sample_data = [
    [2, 5, 0],
    [3, 2, 5],
    [4, 0, 3],
]
sample_data = np.array(sample_data, dtype=np.float32)

sop = SlopeOnePredictor_explicit(train=np.copy(sample_data), valid=sample_data)
sop.fit()
try:
    assert (4.333333333333333 - sop.predict(user_id=0, item_ids=[2])[0]) < 0.00001
    print("SlopeOnePredictor_explicit.predict() is correct")
except:
    print("SlopeOnePredictor_explicit.predict() is wrong")

"""
try:
    svdpp = SVDpp_explicit(train=np.copy(sample_data), valid=sample_data, num_epochs=1)
    svdpp.fit()
    assert np.sum(np.linalg.norm(svdpp.implicit_ratings, axis=1) - np.array([1, 1, 1])) < 0.00001
    print("SVDpp_explicit.implicit_ratings is correct")
except:
    print("SVDpp_explicit.implicit_ratings is wrong or SVDpp_explicit_model is not implemented")
"""

