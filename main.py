import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Định nghĩa lớp mô hình
class TransformerRecommender(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, num_countries, num_genders, num_occupations,
                 num_directors, num_actors, tfidf_dim, embed_dim=64, num_heads=4, num_layers=2):
        super(TransformerRecommender, self).__init__()
        self.embed_dim = embed_dim

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        self.genre_embedding = nn.Embedding(num_genres, embed_dim)
        self.country_embedding = nn.Embedding(num_countries, embed_dim)
        self.gender_embedding = nn.Embedding(num_genders, embed_dim)
        self.occupation_embedding = nn.Embedding(num_occupations, embed_dim)
        self.director_embedding = nn.Embedding(num_directors, embed_dim)
        self.actor_embedding = nn.Embedding(num_actors, embed_dim)

        self.age_linear = nn.Linear(1, embed_dim)
        self.release_year_linear = nn.Linear(1, embed_dim)
        self.synopsis_linear = nn.Linear(tfidf_dim, embed_dim)

        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=0.1)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, movie_id, age, gender, occupation, release_year, country, director, genres, main_actors, synopsis_tfidf):
        batch_size = user_id.size(0)

        user_emb = self.user_embedding(user_id)
        movie_emb = self.movie_embedding(movie_id)
        gender_emb = self.gender_embedding(gender)
        occupation_emb = self.occupation_embedding(occupation)
        country_emb = self.country_embedding(country)
        director_emb = self.director_embedding(director)

        age_emb = self.age_linear(age.unsqueeze(-1))
        release_year_emb = self.release_year_linear(release_year.unsqueeze(-1))
        synopsis_emb = self.synopsis_linear(synopsis_tfidf)

        genre_emb = self.genre_embedding(genres)
        genre_emb = genre_emb.mean(dim=1) if genre_emb.size(1) > 0 else torch.zeros(batch_size, self.embed_dim).to(user_id.device)
        actor_emb = self.actor_embedding(main_actors)
        actor_emb = actor_emb.mean(dim=1) if actor_emb.size(1) > 0 else torch.zeros(batch_size, self.embed_dim).to(user_id.device)

        combined = user_emb + movie_emb + gender_emb + occupation_emb + country_emb + director_emb + \
                   age_emb + release_year_emb + synopsis_emb + genre_emb + actor_emb
        combined = combined.unsqueeze(0)  # Thêm chiều seq_len=1

        transformer_out = self.transformer(combined)
        transformer_out = transformer_out.squeeze(0)

        output = self.fc(transformer_out)
        output = self.sigmoid(output) * 5.0
        return output.squeeze()

# Tiền xử lý dữ liệu
try:
    # Đọc file CSV với phân tách là dấu ,
    movies_df = pd.read_csv("movies.csv", sep=',', names=['movieId', 'title', 'release_year', 'genres', 'synopsis', 'director', 'main_actors', 'country'], skiprows=1)
    users_df = pd.read_csv("users.csv", sep=',', names=['userId', 'age', 'gender', 'occupation'], skiprows=1)

    # Xử lý giá trị khuyết
    movies_df[['genres', 'country', 'director', 'main_actors', 'synopsis']] = movies_df[['genres', 'country', 'director', 'main_actors', 'synopsis']].fillna('Unknown')
    users_df[['occupation']] = users_df[['occupation']].fillna('Unknown')

    # Mã hóa các cột dạng categorical
    label_encoders = {}
    movies_df['genres'] = movies_df['genres'].str.split(',')
    all_genres = set(g for genres in movies_df['genres'] for g in genres)
    genre_encoder = LabelEncoder()
    genre_encoder.fit(list(all_genres))
    label_encoders['genres'] = genre_encoder

    country_encoder = LabelEncoder()
    movies_df['country'] = country_encoder.fit_transform(movies_df['country'])
    label_encoders['country'] = country_encoder

    gender_encoder = LabelEncoder()
    users_df['gender'] = gender_encoder.fit_transform(users_df['gender'])
    label_encoders['gender'] = gender_encoder

    occupation_encoder = LabelEncoder()
    users_df['occupation'] = occupation_encoder.fit_transform(users_df['occupation'])
    label_encoders['occupation'] = occupation_encoder

    director_encoder = LabelEncoder()
    movies_df['director'] = director_encoder.fit_transform(movies_df['director'])
    label_encoders['director'] = director_encoder

    movies_df['main_actors'] = movies_df['main_actors'].str.split(',')
    all_actors = set(a for actors in movies_df['main_actors'] for a in actors)
    actor_encoder = LabelEncoder()
    actor_encoder.fit(list(all_actors))
    label_encoders['main_actors'] = actor_encoder

    # Chuẩn hóa các cột số
    scaler = MinMaxScaler()
    movies_df['release_year'] = scaler.fit_transform(movies_df[['release_year']].astype(float))
    users_df['age'] = scaler.fit_transform(users_df[['age']].astype(float))

    # Chuyển synopsis thành vector TF-IDF
    tfidf = TfidfVectorizer(max_features=100)
    synopsis_tfidf = tfidf.fit_transform(movies_df['synopsis']).toarray()
    movies_df['synopsis_tfidf'] = list(synopsis_tfidf)

    # Tạo ma trận tương đồng dựa trên genres, director, main_actors, country
    def get_movie_features(row):
        genre_ids = label_encoders['genres'].transform([g for g in row['genres'] if g in label_encoders['genres'].classes_])
        actor_ids = label_encoders['main_actors'].transform([a for a in row['main_actors'] if a in label_encoders['main_actors'].classes_])
        genre_ids = genre_ids[:3]
        actor_ids = actor_ids[:3]
        return np.concatenate([
            np.pad(genre_ids, (0, 3 - len(genre_ids)), 'constant'),
            [row['country']],
            [row['director']],
            np.pad(actor_ids, (0, 3 - len(actor_ids)), 'constant')
        ])

    movie_features = np.array([get_movie_features(row) for _, row in movies_df.iterrows()])
    similarity_matrix = cosine_similarity(movie_features)

    # Khởi tạo các tham số cho mô hình
    num_users = users_df['userId'].max() + 1
    num_movies = movies_df['movieId'].max() + 1
    num_genres = len(label_encoders['genres'].classes_)
    num_countries = len(label_encoders['country'].classes_)
    num_genders = len(label_encoders['gender'].classes_)
    num_occupations = len(label_encoders['occupation'].classes_)
    num_directors = len(label_encoders['director'].classes_)
    num_actors = len(label_encoders['main_actors'].classes_)
    tfidf_dim = 100

    # Khởi tạo mô hình
    model = TransformerRecommender(
        num_users=num_users,
        num_movies=num_movies,
        num_genres=num_genres,
        num_countries=num_countries,
        num_genders=num_genders,
        num_occupations=num_occupations,
        num_directors=num_directors,
        num_actors=num_actors,
        tfidf_dim=tfidf_dim,
        embed_dim=64,
        num_heads=4,
        num_layers=2
    )

    # Tải trọng số mô hình
    model.load_state_dict(torch.load("transformer_model.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    logger.error(f"Error loading model or data: {e}")
    raise

@app.post("/recommend")
async def recommend(data: dict):
    try:
        user_id = data.get("user_id")
        movie_id = data.get("movie_id")
        if user_id is None:
            raise HTTPException(status_code=400, detail="Missing user_id in request")

        if user_id not in users_df['userId'].values:
            logger.warning(f"User ID {user_id} not found in users.csv")
            raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")

        # Lấy thông tin người dùng từ users.csv
        user_data = users_df[users_df['userId'] == user_id].iloc[0]
        age = float(user_data['age'])
        gender = int(user_data['gender'])
        occupation = int(user_data['occupation'])

        # Chuẩn bị dữ liệu đầu vào cho mô hình
        movie_ids = torch.tensor(movies_df['movieId'].values, dtype=torch.long)
        num_movies = len(movie_ids)

        age_tensor = torch.tensor([age] * num_movies, dtype=torch.float)
        gender_tensor = torch.tensor([gender] * num_movies, dtype=torch.long)
        occupation_tensor = torch.tensor([occupation] * num_movies, dtype=torch.long)
        release_year = torch.tensor(movies_df['release_year'].values, dtype=torch.float)
        country = torch.tensor(movies_df['country'].values, dtype=torch.long)
        director = torch.tensor(movies_df['director'].values, dtype=torch.long)

        # Xử lý genres và main_actors
        max_genres = 3
        max_actors = 3
        genres_padded = np.zeros((num_movies, max_genres), dtype=np.int64)
        actors_padded = np.zeros((num_movies, max_actors), dtype=np.int64)

        for i in range(num_movies):
            genres = movies_df['genres'].iloc[i]
            actors = movies_df['main_actors'].iloc[i]
            genre_ids = label_encoders['genres'].transform([g for g in genres if g in label_encoders['genres'].classes_])
            actor_ids = label_encoders['main_actors'].transform([a for a in actors if a in label_encoders['main_actors'].classes_])
            genre_ids = genre_ids[:max_genres]
            actor_ids = actor_ids[:max_actors]
            genres_padded[i] = np.pad(genre_ids, (0, max_genres - len(genre_ids)), 'constant')
            actors_padded[i] = np.pad(actor_ids, (0, max_actors - len(actor_ids)), 'constant')

        genres_tensor = torch.tensor(genres_padded, dtype=torch.long)
        main_actors_tensor = torch.tensor(actors_padded, dtype=torch.long)

        # Tạo synopsis_tfidf
        synopsis_tfidf = torch.tensor(movies_df['synopsis_tfidf'].tolist(), dtype=torch.float)

        # Dự đoán điểm số với mô hình
        with torch.no_grad():
            output = model(
                torch.tensor([user_id] * num_movies, dtype=torch.long),
                movie_ids,
                age_tensor,
                gender_tensor,
                occupation_tensor,
                release_year,
                country,
                director,
                genres_tensor,
                main_actors_tensor,
                synopsis_tfidf
            )

        # Xử lý gợi ý
        if movie_id is not None:
            # Trường hợp có movie_id: kết hợp điểm từ mô hình và độ tương đồng với movie_id
            if movie_id not in movies_df['movieId'].values:
                logger.warning(f"Movie ID {movie_id} not found in movies.csv")
                raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found")

            movie_idx = movies_df.index[movies_df['movieId'] == movie_id][0]
            similarity_scores = similarity_matrix[movie_idx]
            combined_scores = output.numpy() * 0.7 + similarity_scores * 0.3  # Kết hợp 70% từ mô hình, 30% từ tương đồng

            # Sao chép mảng để tránh stride âm
            filtered_movie_ids = movie_ids.numpy().copy()
            filtered_scores = combined_scores.copy()

            # Loại bỏ movie_id đã chọn khỏi danh sách gợi ý
            mask = filtered_movie_ids != movie_id
            filtered_movie_ids = filtered_movie_ids[mask]
            filtered_scores = filtered_scores[mask]

            # Lấy top 5 phim gợi ý
            top_indices = np.argsort(filtered_scores)[::-1][:5]
            top_movie_ids = filtered_movie_ids[top_indices].tolist()
        else:
            # Trường hợp chỉ có user_id: gợi ý dựa trên điểm từ mô hình
            top_indices = torch.topk(output, 5).indices
            top_movie_ids = movie_ids[top_indices].tolist()

        return top_movie_ids
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error in recommend endpoint: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
