import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

class TransformerRecommender(nn.Module):
    def __init__(self, num_users, num_movies, num_genres, num_countries, num_genders, num_occupations,
                 num_directors, num_actors, tfidf_dim, embed_dim=64, num_heads=4, num_layers=2):
        super(TransformerRecommender, self).__init__()
        self.embed_dim = embed_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        self.genre_embedding = nn.Embedding(num_genres, embed_dim)
        self.country_embedding = nn.Embedding(num_countries, embed_dim)
        self.gender_embedding = nn.Embedding(num_genders, embed_dim)
        self.occupation_embedding = nn.Embedding(num_occupations, embed_dim)
        self.director_embedding = nn.Embedding(num_directors, embed_dim)
        self.actor_embedding = nn.Embedding(num_actors, embed_dim)

        # Linear layers for additional features
        self.age_linear = nn.Linear(1, embed_dim)
        self.release_year_linear = nn.Linear(1, embed_dim)
        self.synopsis_linear = nn.Linear(tfidf_dim, embed_dim)

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=0.1)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, movie_ids, age, gender, occupation, release_year, country, director, genres, main_actors, synopsis_tfidf):
        batch_size = user_id.size(0)
        num_movies = movie_ids.size(1)

        # Lặp qua từng cặp (user_id, movie_id) trong batch
        user_emb = self.user_embedding(user_id).unsqueeze(1).expand(-1, num_movies, -1)  # [batch_size, num_movies, embed_dim]
        movie_emb = self.movie_embedding(movie_ids)  # [batch_size, num_movies, embed_dim]
        gender_emb = self.gender_embedding(gender).unsqueeze(1).expand(-1, num_movies, -1)
        occupation_emb = self.occupation_embedding(occupation).unsqueeze(1).expand(-1, num_movies, -1)
        country_emb = self.country_embedding(country).unsqueeze(1).expand(-1, num_movies, -1)
        director_emb = self.director_embedding(director).unsqueeze(1).expand(-1, num_movies, -1)

        age_emb = self.age_linear(age.unsqueeze(-1)).unsqueeze(1).expand(-1, num_movies, -1)
        release_year_emb = self.release_year_linear(release_year.unsqueeze(-1)).unsqueeze(1).expand(-1, num_movies, -1)
        synopsis_emb = self.synopsis_linear(synopsis_tfidf)

        # Xử lý genres và main_actors (giả sử lấy trung bình)
        genre_emb = self.genre_embedding(genres)  # [batch_size, max_genres, embed_dim]
        genre_emb = genre_emb.mean(dim=1).unsqueeze(1).expand(-1, num_movies, -1)
        actor_emb = self.actor_embedding(main_actors)  # [batch_size, max_actors, embed_dim]
        actor_emb = actor_emb.mean(dim=1).unsqueeze(1).expand(-1, num_movies, -1)

        # Kết hợp tất cả embedding
        combined = (user_emb + movie_emb + gender_emb + occupation_emb + country_emb + director_emb +
                    age_emb + release_year_emb + synopsis_emb + genre_emb + actor_emb)
        combined = combined.permute(1, 0, 2)  # [num_movies, batch_size, embed_dim] cho transformer

        # Transformer
        transformer_out = self.transformer(combined)  # [num_movies, batch_size, embed_dim]
        transformer_out = transformer_out.permute(1, 0, 2)  # [batch_size, num_movies, embed_dim]

        # Output layer
        output = self.fc(transformer_out)  # [batch_size, num_movies, 1]
        output = self.sigmoid(output) * 5.0  # Chuẩn hóa thành thang điểm 0-5
        return output.squeeze(-1)  # [batch_size, num_movies]

# Khởi tạo và tải mô hình
movies_df = pd.read_csv("movies.csv", names=['movieId', 'title', 'release_year', 'genres', 'synopsis', 'director', 'main_actors', 'country'], skiprows=1)
num_users = 944  # Từ dữ liệu huấn luyện
num_movies = movies_df['movieId'].max() + 1
num_genres = 589  # Từ label_encoders
num_countries = 45
num_genders = 2
num_occupations = 21
num_directors = 955
num_actors = 1636
tfidf_dim = 100

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

model.load_state_dict(torch.load("transformer_recommender_stable.pth", map_location=torch.device('cpu')))
model.eval()
model = torch.compile(model, mode="reduce-overhead")

@app.post("/recommend")
async def recommend(data: dict):
    user_id = data.get("user_id")
    user_features = np.array(data.get("user_features"), dtype=np.float32)  # [100] là synopsis_tfidf

    # Lấy dữ liệu user từ users.csv (giả sử đã tiền xử lý)
    users_df = pd.read_csv("users.csv", names=['userId', 'age', 'gender', 'occupation'], skiprows=1)
    user_data = users_df[users_df['userId'] == user_id].iloc[0]
    age = float(user_data['age'])
    gender = int(user_data['gender'])
    occupation = int(user_data['occupation'])

    # Lấy tất cả movie_ids từ movies.csv
    movie_ids = torch.tensor(movies_df['movieId'].values, dtype=torch.long)
    num_movies = len(movie_ids)

    # Chuẩn bị dữ liệu cố định (lặp lại cho tất cả phim)
    age_tensor = torch.tensor([age] * num_movies, dtype=torch.float).unsqueeze(0)
    gender_tensor = torch.tensor([gender] * num_movies, dtype=torch.long).unsqueeze(0)
    occupation_tensor = torch.tensor([occupation] * num_movies, dtype=torch.long).unsqueeze(0)
    release_year = torch.zeros(num_movies, dtype=torch.float).unsqueeze(0)  # Cần lấy từ movies.csv
    country = torch.zeros(num_movies, dtype=torch.long).unsqueeze(0)
    director = torch.zeros(num_movies, dtype=torch.long).unsqueeze(0)
    genres = torch.zeros((1, 3), dtype=torch.long)  # Giả sử tối đa 3 thể loại
    main_actors = torch.zeros((1, 3), dtype=torch.long)  # Giả sử tối đa 3 diễn viên
    synopsis_tfidf = torch.tensor([user_features] * num_movies, dtype=torch.float)  # Lặp synopsis cho tất cả phim

    # Dự đoán điểm số cho tất cả phim
    with torch.no_grad():
        output = model(
            torch.tensor([user_id] * num_movies, dtype=torch.long),
            movie_ids.unsqueeze(0),
            age_tensor,
            gender_tensor,
            occupation_tensor,
            release_year,
            country,
            director,
            genres,
            main_actors,
            synopsis_tfidf
        )  # [1, num_movies]

    # Lấy top 5 movie_ids dựa trên điểm số
    top_indices = torch.topk(output[0], 5).indices
    top_movie_ids = movie_ids[top_indices].tolist()

    return top_movie_ids
