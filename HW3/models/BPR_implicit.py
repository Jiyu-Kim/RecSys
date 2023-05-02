

import numpy as np
import torch
from tqdm import tqdm
from IPython import embed
from utils import eval_implicit

class BPR_implicit_model(torch.nn.Module):
    def __init__(self, num_users, num_items, n_features):
        super().__init__()
        self.user_factors = torch.nn.Embedding(num_users, n_features, sparse=False)
        self.item_factors = torch.nn.Embedding(num_items, n_features, sparse=False)
        self.item_bias = torch.nn.Embedding(num_items, 1, sparse=False)

        torch.nn.init.normal_(self.user_factors.weight, std=0.01)
        torch.nn.init.normal_(self.item_factors.weight, std=0.01)
        torch.nn.init.normal_(self.item_bias.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        user_embs = self.user_factors(user_ids)
        item_embs = self.item_factors(item_ids)
        item_biases = self.item_bias(item_ids).squeeze()

        predictions = torch.sum(user_embs * item_embs, dim=1) + item_biases
        return predictions


class BPR_implicit():
    def __init__(self, train, valid, n_features=20, learning_rate = 1e-2, reg_lambda =0.1, num_epochs = 100, batch_size=102400, num_negative=3, device='cpu'):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.num_epcohs = num_epochs
        self.n_features = n_features
        self.batch_size = batch_size
        self.num_negative = num_negative
        self.device = device

        self.model = BPR_implicit_model(self.num_users, self.num_items, self.n_features).to(device)
        self.BCE_loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_lambda)

    def fit(self):
        user_rated_dict = dict()
        user_not_rated_dict = dict()
        for u in range(self.num_users):
            user_rated_dict[u] = np.where(self.train[u, :] > 0)[0]
            user_not_rated_dict[u] = np.setdiff1d(np.arange(self.num_items), user_rated_dict[u], assume_unique=True)

        for epoch in range(self.num_epcohs):
            train_data = []
            
            for user_id in range(self.num_users):
                for pos_item_id in user_rated_dict[u]:
                    for _ in range(self.num_negative):
                        neg_item_id = np.random.choice(user_not_rated_dict[u], 1).item()
                        while [user_id, pos_item_id, neg_item_id] in train_data:
                            neg_item_id = np.random.choice(user_not_rated_dict[u], 1).item()
                        train_data.append([user_id, pos_item_id, neg_item_id])
            
            #print(train_data)
            train_data = torch.tensor(np.array(train_data))

            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            epoch_loss = 0
            for train in train_loader:
                users = train[:, 0].to(self.device)
                item_is = train[:, 1].to(self.device)
                item_js = train[:, 2].to(self.device)

                prediction_is = self.model.forward(users, item_is)
                prediction_js = self.model.forward(users, item_js)
                loss = -(prediction_is - prediction_js).sigmoid().log().sum()
                #loss = torch.Tensor([0.0]).to(self.device)

                epoch_loss += loss.item() / len(train)

                # gradient reset
                self.optimizer.zero_grad()

                # Backpropagate
                loss.backward()

                # Update the parameters
                self.optimizer.step()

            if epoch % 1 == 0:
                top_k=50
                prec, recall, ndcg, mrr, mAP = eval_implicit(self, self.train, self.valid, top_k)
                print("[BPR] epoch %d, loss: %f"%(epoch, epoch_loss/len(train_loader)))
                print(f"(BPR VALID) prec@{top_k} {prec}, recall@{top_k} {recall}, ndcg@{top_k} {ndcg}, mrr@{top_k} {mrr}, map@{top_k} {mAP}")

    def predict(self, user_id, item_ids):
        with torch.no_grad():
            user_id = torch.tensor([user_id]).to(self.device)
            item_ids = torch.tensor(item_ids).to(self.device)
            prediction = self.model.forward(user_id, item_ids)
            return prediction.cpu().numpy()
