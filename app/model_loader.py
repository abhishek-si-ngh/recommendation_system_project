# === model_loader.py ===
import pickle
import pandas as pd

# Load trained SVD model
with open("../models/svd_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load item (movie) data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv("../data/ml-100k/u.data", sep='\t', names=column_names)
movies = pd.read_csv("../data/ml-100k/u.item", sep='|', encoding='latin-1', header=None)[[0, 1]]
movies.columns = ['item_id', 'title']

# Merge movie titles with ratings
data_merged = pd.merge(ratings, movies, on="item_id")

# Create item_id to title mapping
item_id_to_title = dict(zip(data_merged['item_id'], data_merged['title']))