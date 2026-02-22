from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Datasets', 'dataset.csv')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'Data Exploration', 'templates')

df = pd.read_csv(DATA_PATH)

cols_to_drop = ["Unnamed: 0", "track_id", "mode"]
cleaned_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
cleaned_df = cleaned_df.dropna()

numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
Q1 = cleaned_df[numeric_cols].quantile(0.25)
Q3 = cleaned_df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

df_clean = cleaned_df[~((cleaned_df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                        (cleaned_df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
df_clean.reset_index(drop=True, inplace=True)

if 'track_name' in df_clean.columns and 'artists' in df_clean.columns:
	df_clean = df_clean.drop_duplicates(subset=['track_name', 'artists']).reset_index(drop=True)

feature_cols = [
    "danceability", "energy", "loudness", "acousticness",
    "instrumentalness", "valence", "tempo", "duration_ms"
]

df_feat = df_clean.dropna(subset=feature_cols).reset_index(drop=True)

X_num = df_feat[feature_cols].values

genre_cols = [c for c in df_feat.columns if 'genre' in c.lower()]
if genre_cols:
    genre_col = genre_cols[0]
    genre_dummies = pd.get_dummies(df_feat[genre_col].astype('category'), prefix=genre_col)
    X = np.hstack([X_num, genre_dummies.values])
else:
    X = X_num

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

K_DEFAULT = 10
knn = NearestNeighbors(metric="euclidean", n_neighbors=K_DEFAULT)
knn.fit(X_scaled)

playlist = []

def find_song_row(query: str):
    q = query.strip().lower()
    if not q:
        return None
    mask = (
        df_feat['track_name'].str.lower().str.contains(q) |
        df_feat['artists'].str.lower().str.contains(q)
    )
    matches = df_feat[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_playlist_query_vector(playlist_indices):
    if not playlist_indices:
        return None
    idx_array = np.array(list(playlist_indices), dtype=int)
    if idx_array.size == 0:
        return None
    return X_scaled[idx_array].mean(axis=0, keepdims=True)


def recommend_knn(query_vec, excluded_indices=None, k: int = K_DEFAULT):
    if query_vec is None:
        return None
    excluded = set(excluded_indices or [])
    n_total = X_scaled.shape[0]
    n_neighbors = min(n_total, k + len(excluded) + 5)
    distances, indices = knn.kneighbors(query_vec, n_neighbors=n_neighbors)
    neighbor_indices = [i for i in indices[0] if i not in excluded]
    if not neighbor_indices:
        return None
    rec_idx = neighbor_indices[0]
    rec_vec = X_scaled[rec_idx].reshape(1, -1)
    cos_sim = float(cosine_similarity(query_vec, rec_vec)[0, 0])
    return rec_idx, cos_sim


def recommend_random(query_vec, excluded_indices=None):
    if query_vec is None:
        return None
    n = X_scaled.shape[0]
    all_indices = np.arange(n)
    excluded = set(excluded_indices or [])
    candidates = np.array([i for i in all_indices if i not in excluded])
    if candidates.size == 0:
        return None
    rec_idx = int(np.random.choice(candidates))
    rec_vec = X_scaled[rec_idx].reshape(1, -1)
    cos_sim = float(cosine_similarity(query_vec, rec_vec)[0, 0])
    return rec_idx, cos_sim


app = Flask(__name__, template_folder=TEMPLATE_DIR)


def track_info_from_index(idx: int):
    row = df_feat.iloc[idx]
    return {
        'track_name': row['track_name'],
        'artists': row['artists'],
        'album_name': row.get('album_name', ''),
        'track_id': row.get('track_id', ''),
    }


@app.route('/search_songs', methods=['POST'])
def search_songs():
    data = request.get_json(silent=True) or {}
    query = str(data.get('query', '')).strip().lower()
    if not query:
        return jsonify([])

    mask = (
        df_feat['track_name'].str.lower().str.contains(query) |
        df_feat['artists'].str.lower().str.contains(query)
    )
    matches = df_feat[mask].head(10)

    results = [
        {
            'index': int(idx),
            'track_name': row['track_name'],
            'artists': row['artists'],
        }
        for idx, row in matches.iterrows()
    ]
    return jsonify(results)


@app.route('/', methods=['GET', 'POST'])
def index():
    global playlist
    error = None
    knn_rec = None
    rand_rec = None
    current_song = None

    if request.method == 'POST':
        action = request.form.get('action')
        query = request.form.get('song_query', '')
        index_str = request.form.get('song_index', '').strip()

        if action == 'add':
            if index_str != '':
                try:
                    idx = int(index_str)
                    if 0 <= idx < len(df_feat):
                        playlist.append(idx)
                    else:
                        error = 'Selected song index is out of range.'
                except ValueError:
                    error = 'Invalid song selection.'
            else:
                row = find_song_row(query)
                if row is None:
                    error = 'No matching song found.'
                else:
                    playlist.append(int(row.name))
        elif action == 'clear':
            playlist = []

    if playlist:
        current_idx = playlist[-1]
        current_song = track_info_from_index(current_idx)

        query_vec = build_playlist_query_vector(playlist)
        excluded = set(playlist)

        knn_out = recommend_knn(query_vec, excluded_indices=excluded)
        if knn_out is not None:
            rec_idx, sim = knn_out
            info = track_info_from_index(rec_idx)
            info['similarity'] = sim
            row_rec = df_feat.iloc[rec_idx]
            info['features'] = {col: float(row_rec[col]) for col in feature_cols}
            knn_rec = info

        rand_out = recommend_random(query_vec, excluded_indices=excluded)
        if rand_out is not None:
            rec_idx, sim = rand_out
            info = track_info_from_index(rec_idx)
            info['similarity'] = sim
            row_rec = df_feat.iloc[rec_idx]
            info['features'] = {col: float(row_rec[col]) for col in feature_cols}
            rand_rec = info

    playlist_info = [track_info_from_index(i) for i in playlist]
    playlist_feature_data = []
    playlist_avg_features = None
    if playlist:
        for i in playlist:
            row = df_feat.iloc[i]
            playlist_feature_data.append({
                "index": int(i),
                "track_name": row['track_name'],
                "artists": row['artists'],
                "features": {col: float(row[col]) for col in feature_cols},
            })
        feat_matrix = df_feat.loc[playlist, feature_cols].astype(float)
        playlist_avg_features = {col: float(feat_matrix[col].mean()) for col in feature_cols}

    return render_template(
        'index.html',
        playlist=playlist_info,
        current_song=current_song,
        knn_rec=knn_rec,
        rand_rec=rand_rec,
        error=error,
        playlist_feature_data=playlist_feature_data,
        playlist_avg_features=playlist_avg_features,
    )


if __name__ == '__main__':
    app.run(debug=True)
