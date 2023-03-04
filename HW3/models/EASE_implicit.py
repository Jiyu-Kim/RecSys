"""
Embarrassingly shallow autoencoders for sparse data, 
Harald Steck,
Arxiv.
"""
import os
import math
import numpy as np

class EASE_implicit():
    def __init__(self, train, reg_lambda):
        self.train = train
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.reg_lambda = reg_lambda

    def fit(self):   
        '''
        Implement fit function
        '''
        self.B = np.zeros((self.num_users, self.num_items))
        # ========================= EDIT HERE ========================
        G = self.train.T.dot(self.train)
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.reg_lambda
        P = np.linalg.inv(G)
        self.B = P / (-np.diag(P))
        self.B[diagIndices] = 0
        # ========================= EDIT HERE ========================
        # 사용자-항목 행렬과 W 행렬의 행렬 곱을 통해 예측 값 행렬 생성
        self.reconstructed = self.train @ self.B

    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]