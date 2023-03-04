import numpy as np

class SlopeOnePredictor_explicit():
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]

        for i, row in enumerate(self.train):
            self.train[i, np.where(row < 0.5)[0]] = np.nan

    def fit(self):
        """
        You can pre-calculate deviation in here or calculate in predict().
        """
        # ========================= EDIT HERE ========================
        def get_dev_val(i, j):
            dev_val = 0
            users = 0
            for row in range(self.num_users):
                if (~np.isnan(self.train[row][i])) and (~np.isnan(self.train[row][j])):
                    users += 1
                    dev_val += self.train[row][i] - self.train[row][j]
                    
            if users == 0:
                ret = 0
            else:
                ret = dev_val / users
            return ret, users

        self.dev = np.zeros((self.num_items, self.num_items))
        self.evaled_users_mat = np.zeros((self.num_items, self.num_items))
        for i in range(self.num_items):
            for j in range(self.num_items):
                if i == j:
                    break
                else:
                    dev_temp, users = get_dev_val(i, j)
                    self.dev[i][j] = dev_temp
                    self.dev[j][i] = (-1) * dev_temp
                    self.evaled_users_mat[i][j] = users
                    self.evaled_users_mat[j][i] = users
        
        #return dev, evaled_users_mat
        # ============================================================
        #pass

    def predict(self, user_id, item_ids):

        predicted_values = []
        # user i가 시청한 item들
        rated_items = np.where(~np.isnan(self.train[user_id,:]))[0]
        for one_missing_item in item_ids:
            # ========================= EDIT HERE ========================
            predicted_rate = np.sum((self.dev[one_missing_item][rated_items] + self.train[user_id][rated_items]) * self.evaled_users_mat[one_missing_item][rated_items]) / np.sum(self.evaled_users_mat[one_missing_item][rated_items])
            predicted_values.append(predicted_rate)
            # ============================================================
        return predicted_values


