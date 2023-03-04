import os
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def load_data(data_name, implicit=True):
    data_path = './data/%s' % (data_name)

    column_names = ['user_id', 'item_id', 'rating', 'timestamp']
    movie_data = pd.read_csv(data_path, names=column_names)

    if implicit:
        movie_data['rating'] = 1

    user_list = list(movie_data['user_id'].unique())
    item_list = list(movie_data['item_id'].unique())

    num_users = len(user_list)
    num_items = len(item_list)
    num_ratings = len(movie_data)

    user_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(user_list)}
    movie_data.user_id = [user_id_dict[x] for x in movie_data.user_id.tolist()]

    item_id_dict = {old_uid: new_uid for new_uid, old_uid in enumerate(item_list)}
    movie_data.item_id = [item_id_dict[x] for x in movie_data.item_id.tolist()]
    print(f"# of users: {num_users},  # of items: {num_items},  # of ratings: {num_ratings}")

    movie_data = movie_data[['user_id', 'item_id', 'rating']]
    movie_data = movie_data.sort_values(by="user_id", ascending=True)
    
    train_valid, test = train_test_split(movie_data, test_size=0.2, stratify=movie_data['user_id'], random_state=1234)
    train, valid = train_test_split(train_valid, test_size=0.1, stratify=train_valid['user_id'], random_state=1234)

    train = train.to_numpy()
    valid = valid.to_numpy()
    test = test.to_numpy()

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in train:
        matrix[u, i] = r
    train = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in valid:
        matrix[u, i] = r
    valid = sparse.csr_matrix(matrix)

    matrix = sparse.lil_matrix((num_users, num_items))
    for (u, i, r) in test:
        matrix[u, i] = r
    test = sparse.csr_matrix(matrix)

    return train.toarray(), valid.toarray(), test.toarray()


def eval_explicit(model, train_data, test_data):
    rmse_list = []
    if 'Item' in model.__class__.__name__:
        num_users, num_items = train_data.shape
        pred_matrix = np.zeros((num_users, num_items))

        for item_id in range(len(train_data.T)):
            train_by_item = test_data[:, item_id]
            missing_user_ids = np.where(train_by_item >= 0.5)[0]

            pred_u_score = model.predict(item_id, missing_user_ids)
            pred_matrix[missing_user_ids, item_id] = pred_u_score

        for user_id in range(len(train_data)):
            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]
            target_u_score = test_by_user[target_u]

            pred_u_score = pred_matrix[user_id, target_u]

            rmse = mean_squared_error(target_u_score, pred_u_score)
            rmse_list.append(rmse)
    else:
        for user_id in range(len(train_data)):
            test_by_user = test_data[user_id]
            target_u = np.where(test_by_user >= 0.5)[0]
            target_u_score = test_by_user[target_u]

            pred_u_score = model.predict(user_id, target_u)

            # RMSE 계산
            rmse = mean_squared_error(target_u_score, pred_u_score)
            rmse_list.append(rmse)

    return np.mean(rmse_list)
