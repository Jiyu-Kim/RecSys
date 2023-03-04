
import numpy as np
import torch
from IPython import embed
from utils import eval_implicit
import os
import math
from time import time
import torch.nn as nn
import torch.nn.functional as F


class AE_implicit(torch.nn.Module):
    def __init__(self, train, valid, num_epochs, hidden_dim, learning_rate, reg_lambda, device='cpu'):
        super().__init__()

        self.train_mat = train
        self.valid_mat = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]

        self.num_epochs = num_epochs
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

        self.device = device

        self.build_graph()


    def build_graph(self):
        # W, W'와 b, b'만들기
        self.enc_w = nn.Parameter(torch.ones(self.num_items, self.hidden_dim))
        self.enc_b = nn.Parameter(torch.ones(self.hidden_dim))
        nn.init.xavier_uniform_(self.enc_w)
        nn.init.normal_(self.enc_b, 0, 0.001)

        self.dec_w = nn.Parameter(torch.ones(self.hidden_dim, self.num_items))
        self.dec_b = nn.Parameter(torch.ones(self.num_items))
        nn.init.xavier_uniform_(self.dec_w)
        nn.init.normal_(self.dec_b, 0, 0.001)

        # 최적화 방법 설정
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)

        # 모델을 device로 보냄
        self.to(self.device)


    def forward(self, x):
        # encoder 과정
        h = torch.sigmoid(x @ self.enc_w + self.enc_b)

        # decoder 과정
        output = torch.sigmoid(h @ self.dec_w + self.dec_b)
        return output


    def fit(self):
        train_matrix = torch.FloatTensor(self.train_mat).to(self.device)

        for epoch in range(0, self.num_epochs):
            self.train()
            
            loss = self.train_model_per_batch(train_matrix)

            if torch.isnan(loss):
                print('Loss NAN. Train finish.')
                break


            if epoch % 20 == 0:
                with torch.no_grad():
                    self.eval()
                    self.reconstructed = self.forward(train_matrix).detach().cpu().numpy()
                    
                    top_k=50
                    print("[AE] epoch %d, loss: %f"%(epoch, loss))
                    prec, recall, ndcg = eval_implicit(self, self.train_mat, self.valid_mat, top_k)
                    print(f"(AE VALID) prec@{top_k} {prec}, recall@{top_k} {recall}, ndcg@{top_k} {ndcg}")
                    self.train()
            

    def train_model_per_batch(self, train_matrix):
        # grad 초기화
        self.optimizer.zero_grad()

        # 모델 forwrad
        output = self.forward(train_matrix)

        # loss 구함
        loss = F.binary_cross_entropy(output, train_matrix, reduction='none').sum(1).mean()

        # 역전파
        loss.backward()
        
        # 최적화
        self.optimizer.step()
        return loss

    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
