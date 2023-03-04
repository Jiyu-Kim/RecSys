import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss

import pickle
import ast


def load_data_CTR(data_name, use_features, pos_threshold=6):
    data_path = './data/%s' % (data_name)

    column_names = ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'people', 'country', 'genre']
    movie_data = pd.read_csv(data_path, names=column_names)
    movie_data = movie_data.fillna('[]')

    movie_data['rating'] = movie_data['rating'].apply(lambda x: 1 if x >= pos_threshold else 0)

    user_list = list(movie_data['user_id'].unique())
    item_list = list(movie_data['item_id'].unique())

    num_users = len(user_list)
    num_items = len(item_list)

    all_genre_dict = {'None': 0}
    all_country_dict = {'None': 0}
    all_people_dict = {'None': 0}

    dict_path = data_path + '_dict'

    if not os.path.exists(dict_path):
        for index, row in tqdm(movie_data.iterrows(), total=len(movie_data), desc='check genre, country, people', dynamic_ncols=True):
            genres = row["genre"]
            coutries = row["country"]
            people = row["people"]
            genres = ast.literal_eval(genres)
            coutries = ast.literal_eval(coutries)
            people = ast.literal_eval(people)

            for genre in genres:
                if all_genre_dict.get(genre) is None:
                    all_genre_dict[genre] = len(all_genre_dict)
            for country in coutries:
                if all_country_dict.get(country) is None:
                    all_country_dict[country] = len(all_country_dict)
            for person in people:
                if all_people_dict.get(person) is None:
                    all_people_dict[person] = len(all_people_dict)
        pickle.dump([all_genre_dict, all_country_dict, all_people_dict], open(dict_path, 'wb'), protocol=4)
    else:
        all_genre_dict, all_country_dict, all_people_dict = pickle.load(open(dict_path, 'rb'))

    num_genres = len(all_genre_dict)
    num_countries = len(all_country_dict)
    num_people = len(all_people_dict)

    idx2title = {}
    idx2genre = {}
    idx2country = {}
    idx2people = {}

    item_data = movie_data[['item_id', 'title', 'people', 'country', 'genre']]
    item_data = item_data.drop_duplicates()
    for idx, title, people, country, genre in zip(item_data['item_id'], item_data['title'], item_data['people'], item_data['country'], item_data['genre']):
        idx2title[idx] = title
        idx2genre[idx] = people
        idx2country[idx] = country
        idx2people[idx] = genre

    train_valid, test = train_test_split(movie_data, test_size=0.2, stratify=movie_data['rating'], random_state=1234)
    train, valid = train_test_split(train_valid, test_size=0.1, stratify=train_valid['rating'], random_state=1234)

    num_fields = len(use_features)

    def df_to_array(df):
        final_array = np.zeros((len(df), num_fields))
        for index, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc='convert df to array', dynamic_ncols=True):
            features = []

            for feature in use_features:
                if feature == "user_id":
                    user_id = row["user_id"]
                    features.append(user_id)
                if feature == "item_id":
                    item_id = row["item_id"]
                    features.append(item_id)
                if feature == "genre":
                    genres = row["genre"]
                    genres = ast.literal_eval(genres)
                    genre_id = all_genre_dict[genres[0]] if len(genres) > 0 else 0
                    features.append(genre_id)
                if feature == "country":
                    coutries = row["country"]
                    coutries = ast.literal_eval(coutries)
                    country_id = all_country_dict[coutries[0]] if len(coutries) > 0 else 0
                    features.append(country_id)
                if feature == "people":
                    people = row["people"]
                    people = ast.literal_eval(people)
                    people_id = all_people_dict[people[0]] if len(people) > 0 else 0
                    features.append(people_id)

            final_array[index] = features

        return final_array

    train_arr = df_to_array(train)
    valid_arr = df_to_array(valid)
    test_arr = df_to_array(test)

    train_rating = train['rating'].values
    valid_rating = valid['rating'].values
    test_rating = test['rating'].values
    field_dims = []
    for feature in use_features:
        if feature == 'user_id': field_dims.append(num_users)
        if feature == 'item_id': field_dims.append(num_items)
        if feature == 'genre': field_dims.append(num_genres)
        if feature == 'country': field_dims.append(num_countries)
        if feature == 'people': field_dims.append(num_people)

    return train_arr, train_rating, valid_arr, valid_rating, test_arr, test_rating, field_dims

def eval_implicit_CTR(model, test_data, test_label):

    predict_test = model.predict(test_data)

    auc = roc_auc_score(test_label, predict_test)
    logloss = log_loss(test_label, predict_test)
    return auc, logloss