# === recommender.py ===
from collections import defaultdict
from surprise import Dataset, Reader
from model_loader import model, item_id_to_title, data_merged

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        top_n[uid] = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:n]
    return top_n

def generate_recommendations(user_id, n=10):
    user_id = str(user_id)

    # Build dataset and anti-testset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(data_merged[['user_id', 'item_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    anti_testset = trainset.build_anti_testset()

    # Filter for current user only
    user_anti_testset = [entry for entry in anti_testset if entry[0] == user_id]
    predictions = model.test(user_anti_testset)

    # Get top N
    top_n = get_top_n(predictions, n=n)

    # Map item IDs to titles
    if user_id in top_n:
        return [(item_id_to_title[int(iid)], round(est, 2)) for iid, est in top_n[user_id]]
    else:
        return []
