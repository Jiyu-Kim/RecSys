import numpy as np
import torch


class SVDpp_explicit_model(torch.nn.Module):
    def __init__(self, num_users, num_items, n_features):
        super().__init__()
        # ========================= EDIT HERE ========================
        self.user_factors = torch.nn.Embedding(num_users, n_features+2, sparse=False)
        self.item_factors = torch.nn.Embedding(num_items, n_features+2, sparse=False)
        self.latent_item_matrix = torch.nn.Embedding(num_items, n_features+2, sparse=False)

        torch.nn.init.normal_(self.user_factors.weight, std=0.01)
        torch.nn.init.normal_(self.item_factors.weight, std=0.01)
        torch.nn.init.normal_(self.latent_item_matrix.weight, std=0.01)

        torch.nn.init.ones_(self.user_factors.weight[:,-1])
        torch.nn.init.ones_(self.item_factors.weight[:,-2])
        torch.nn.init.zeros_(self.latent_item_matrix.weight[:,-1])
        torch.nn.init.zeros_(self.latent_item_matrix.weight[:,-2])
        # ========================= EDIT HERE ========================

    def forward(self, implicit_train_matrix):
        reconstruction = None
        # ========================= EDIT HERE ========================
        uv = torch.matmul(self.user_factors.weight, self.item_factors.weight.T)
        fy = torch.matmul(implicit_train_matrix, self.latent_item_matrix.weight)
        fyv = torch.matmul(fy, self.item_factors.weight.T)
        reconstruction = uv + fyv
        # ========================= EDIT HERE ========================
        return reconstruction


class SVDpp_explicit():
    def __init__(self, train, valid, n_features=20, learning_rate = 1e-2, reg_lambda =0.1, num_epochs = 100):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.num_epcohs = num_epochs
        self.n_features = n_features

        self.y = np.zeros_like(self.train)
        for i, row in enumerate(self.train):
            self.y[i, np.where(row > 0.5)[0]] = 1.0

        self.model = SVDpp_explicit_model(self.num_users, self.num_items, self.n_features)#.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_lambda)


    def mse_loss(self, y, target, predict):
        return (y * (target - predict) ** 2).sum()


    def fit(self):
        ratings = torch.FloatTensor(self.train)#.cuda()
        weights = torch.FloatTensor(self.y)#.cuda()

        implicit_ratings = torch.FloatTensor(self.train).bool().float() 

        # TODO: normalize implicit ratings with the eplison
        # ========================= EDIT HERE ========================
        epsilon = 1e-10
        ju = torch.sum(implicit_ratings, dim=1)
        ju = torch.sqrt(ju).view((-1, 1))
        implicit_ratings = implicit_ratings / (ju + epsilon)
        # ========================= EDIT HERE ========================

        # U와 V를 업데이트 함.
        for epoch in range(self.num_epcohs):
            self.optimizer.zero_grad()

            # 예측
            prediction = self.model.forward(implicit_ratings)
            loss = self.mse_loss(weights, ratings, prediction)

            # Backpropagate
            loss.backward()

            # Update the parameters
            self.optimizer.step()


        with torch.no_grad():
            self.reconstructed = self.model.forward(implicit_ratings).cpu().numpy()
        self.implicit_ratings = implicit_ratings.cpu().numpy()

    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
