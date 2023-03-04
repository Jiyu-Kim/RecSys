import numpy as np
from sqlalchemy import desc
from tqdm import tqdm

class UserKNN_explicit():
    def __init__(self, train, valid, top_k):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.top_k = top_k

        for i, row in enumerate(self.train):
            self.train[i, np.where(row < 0.5)[0]] = np.nan
        
        self.user_mean = np.nanmean(self.train, axis=1)
        self.user_mean[np.isnan(self.user_mean)] = 0.0
        self.train = self.train - self.user_mean[:,None]

    def fit(self):
        user_user_sim_matrix = np.zeros((self.num_users, self.num_users))

        for user_i in tqdm(range(0, self.num_users), desc='user_user_sim_matrix (k=%d)' % self.top_k):
            for user_j in range(user_i+1, self.num_users):
                a = self.train[user_i]
                b = self.train[user_j]

                co_rated = ~np.logical_or(np.isnan(a), np.isnan(b))
                a = np.compress(co_rated, a)
                b = np.compress(co_rated, b) 

                if len(a) == 0:
                    continue 

                dot_a_b = np.dot(a, b)
                if dot_a_b == 0:
                    continue

                user_user_sim_matrix[user_i, user_j] = dot_a_b / (np.linalg.norm(a) * np.linalg.norm(b))

        self.user_user_sim_matrix = (user_user_sim_matrix + user_user_sim_matrix.T)
        

    def predict(self, user_id, item_ids):
        predicted_values = []

        for one_missing_item in item_ids:

            # item i를 시청한 사용자들
            rated_users = np.where(~np.isnan(self.train[:, one_missing_item]))[0]
            unsorted_sim = self.user_user_sim_matrix[user_id, rated_users]

            # 유사도 정렬
            sorted_users = np.argsort(unsorted_sim) # 오름차순으로 정렬된 인덱스
            sorted_users = sorted_users[::-1] # 내림차순으로 정렬된 인덱스

            # Top K 이웃 구하기
            if(self.top_k > len(sorted_users)):
                top_k = len(sorted_users)
            else:
                top_k = self.top_k 
            sorted_users = sorted_users[0:top_k]
            top_k_users = rated_users[sorted_users]

            # 예측 값 구하기
            if(top_k == 0):
                predicted_values.append(0.0)
            else:
                users_rate = self.train[top_k_users, one_missing_item] # top_k_users가 target item에 준 rating
                users_sim = self.user_user_sim_matrix[user_id, top_k_users] # target user와 top_k_users 간의 코사인 유사도

                users_sim[users_sim < 0.0] = 0.0 # negative correlation은 0으로 설정
                
                if np.sum(users_sim) == 0.0: # correlation이 전부 0인 경우
                    predicted_rate = self.user_mean[user_id]
                else:
                    predicted_rate = self.user_mean[user_id] + np.sum(users_rate*users_sim)/np.sum(users_sim)

                predicted_values.append(predicted_rate)

        return predicted_values


