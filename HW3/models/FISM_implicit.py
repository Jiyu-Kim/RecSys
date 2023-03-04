

import numpy as np
import torch
from tqdm import tqdm
from IPython import embed
from utils import eval_implicit


class FISM_implicit_model(torch.nn.Module):
    def __init__(self, num_items, n_features, alpha):
        super().__init__()
        self.alpha = alpha

        self.item_factors_P = torch.nn.Embedding(num_items, n_features, sparse=False)
        self.item_factors_Q = torch.nn.Embedding(num_items, n_features, sparse=False)
        self.item_bias = torch.nn.Embedding(num_items, 1, sparse=False)

        torch.nn.init.normal_(self.item_factors_P.weight, std=0.01)
        torch.nn.init.normal_(self.item_factors_Q.weight, std=0.01)
        torch.nn.init.normal_(self.item_bias.weight, std=0.01)

    def forward(self, user_rating, item_ids, pos_item=False):
        #predictions = torch.tensor(0.0).to(user_rating.device)
        '''
        Implement forward pass
        '''
        # ========================= EDIT HERE ========================
        predictions = torch.matmul(self.item_factors_P(item_ids).sum(axis=0), self.item_factors_Q.weight.T)
        predictions = predictions * np.power(len(item_ids)-1, -self.alpha)
        predictions = predictions + self.item_bias(item_ids)
        # ========================= EDIT HERE ========================
        return predictions

# It is FISM AUC version
class FISM_implicit():
    def __init__(self, train, valid, n_features=20, learning_rate=1e-2, reg_lambda=0.1, num_epochs=100,
                 alpha=0.5, num_negative=3, batch_size=102400, device='cpu'):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.num_epcohs = num_epochs
        self.n_features = n_features
        self.device = device
        self.alpha = alpha
        self.num_negative = num_negative
        self.batch_size = batch_size

        self.model = FISM_implicit_model(self.num_items, self.n_features, self.alpha).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_lambda)

    def fit(self):
        # ========================= EDIT HERE ========================
        user_rated_dict = dict()
        user_not_rated_dict = dict()
        for u in range(self.num_users):
            user_rated_dict[u] = np.where(self.train[u, :] > 0)[0]
            user_not_rated_dict[u] = np.setdiff1d(np.arange(self.num_items), user_rated_dict[u], assume_unique=True)

        for epoch in range(self.num_epcohs):
            train_data = []
            '''
            Implement making training data
            e.g.) train_data = [(user_id, pos_item_id, neg_item_id), ...]
            '''
            # IMPLEMENT HERE
            
            for user_id in range(self.num_users):
                for pos_item_id in user_rated_dict[u]:
                    for _ in range(self.num_negative):
                        neg_item_id = np.random.choice(user_not_rated_dict[u], 1).item()
                        while [user_id, pos_item_id, neg_item_id] in train_data:
                            neg_item_id = np.random.choice(user_not_rated_dict[u], 1).item()
                        train_data.append([user_id, pos_item_id, neg_item_id])
            

            train_data = torch.tensor(np.array(train_data))

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            epoch_loss = 0
            for train in train_loader:
                user_ratings = torch.Tensor(self.train[train[:, 0]]).to(self.device).to(torch.float32)
                item_is = train[:, 1].to(self.device) # pos_item_ids
                item_js = train[:, 2].to(self.device) # neg_item_ids

                '''
                Implement prediction and loss
                '''
                # IMPLEMENT HERE
                prediction_is = self.model.forward(user_ratings, item_is, pos_item=True)
                prediction_js = self.model.forward(user_ratings, item_js, pos_item=False)
                loss =  torch.pow((1 - (prediction_is - prediction_js)), 2).sum()
                #loss = torch.Tensor([0.0]).to(self.device)

                epoch_loss += loss.item() / len(train)

                # gradient reset
                self.optimizer.zero_grad()

                # Backpropagate
                loss.backward()

                # Update the parameters
                self.optimizer.step()

            if epoch % 1 == 0:
                top_k = 50
                prec, recall, ndcg, mrr, mAP = eval_implicit(self, self.train, self.valid, top_k)
                print("[FISM] epoch %d, loss: %f" % (epoch, epoch_loss/len(train_loader)))
                print(f"(FISM VALID) prec@{top_k} {prec}, recall@{top_k} {recall}, ndcg@{top_k} {ndcg}, mrr@{top_k} {mrr}, map@{top_k} {mAP}")
        # ========================= EDIT HERE ========================

    def predict(self, user_id, item_ids):
        with torch.no_grad():
            user_rating = torch.tensor([self.train[user_id]]).to(self.device).to(torch.float32)
            item_ids = torch.tensor(item_ids).to(self.device)
            prediction = self.model.forward(user_rating, item_ids, pos_item=False)
            return prediction.cpu().numpy()
