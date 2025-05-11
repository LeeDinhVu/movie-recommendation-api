import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastapi import FastAPI
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = FastAPI()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def forward(self, user_id, movie_ids, age, gender, occupation, release_year, country, director, genres, main_actors, synopsis_tfidf):
        batch_size = user_id.size(0)
        num_movies = movie_ids.size(1)

        user_emb = self.user_embedding(user_id).unsqueeze(1).expand(-1, num_movies, -1)
        movie_emb = self.movie_embedding(movie_ids)
        gender_emb = self.gender_embedding(gender).unsqueeze(1).expand(-1, num_movies, -1)
        occupation_emb = self.occupation_embedding(occupation).unsqueeze(1).expand(-1, num_movies, -1)
        country_emb = self.country_embedding(country).unsqueeze(1).expand(-1, num_movies, -1)
        director_emb = self.director_embedding(director).unsqueeze(1).expand(-1, num_movies, -1)

        age_emb = self.age_linear(age.unsqueeze(-1)).unsqueeze(1).expand(-1, num_movies, -1)
        release_year_emb = self.release_year_linear(release_year.unsqueeze(-1)).unsqueeze(1).expand(-1, num_movies, -1)
        synopsis_emb = self.synopsis_linear(synopsis_tfidf)

        genre_emb = self.genre_embedding(genres)
        genre_emb = genre_emb.mean(dim=1).unsqueeze(1).expand(-1, num_movies, -1)
        actor_emb = self.actor_embedding(main_actors)
        actor_emb = actor_emb.mean(dim=1).unsqueeze(1).expand(-1, num_movies, -1)

        combined = (user_emb + movie_emb + gender_emb + occupation_emb + country_emb + director_emb +
                    age_emb + release_year_emb + synopsis_emb + genre_emb + actor_emb)
        combined = combined.permute(1, 0, 2)

        transformer_out = self.transformer(combined)
        transformer_out = transformer_out.permute(1, 0, 2)

        output = self.fc(transformer_out)
        output = self.sigmoid(output) * 5.0
        return output.squeeze(-1)

# Tiền xử lý dữ liệu
try:
    movies_df = pd.read_csv("movies.csv", names=['movieId', 'title', 'release_year', 'genres', 'synopsis', 'director', 'main_actors', 'country'], skiprows=1)
    users_df = pd.read_csv("users.csv", names=['userId', 'age', 'gender', 'occupation'], skiprows=1)

    # Tiền xử lý movies.csv
    le_country = LabelEncoder()
    le_director = LabelEncoder()
    le_genres = LabelEncoder()
    le_actors = LabelEncoder()
    scaler = MinMaxScaler()

    movies_df['country'] = movies_df['country'].fillna('Unknown')
    movies_df['director'] = movies_df['director'].fillna('Unknown')
    movies_df['genres'] = movies_df['genres'].fillna('Unknown')
    movies_df['main_actors'] = movies_df['main_actors'].fillna('Unknown')
    movies_df['synopsis'] = movies_df['synopsis'].fillna('')

    movies_df['country_encoded'] = le_country.fit_transform(movies_df['country'])
    movies_df['director_encoded'] = le_director.fit_transform(movies_df['director'])

    # Xử lý genres và main_actors (giả sử là danh sách chuỗi, ví dụ: "Action|Comedy")
    movies_df['genres_list'] = movies_df['genres'].apply(lambda x: x.split('|')[:3])  # Lấy tối đa 3 thể loại
    movies_df['actors_list'] = movies_df['main_actors'].apply(lambda x: x.split('|')[:3])  # Lấy tối đa 3 diễn viên

    # Tạo danh sách tất cả genres và actors duy nhất
    all_genres = set()
    all_actors = set()
    for genres in movies_df['genres_list']:
        all_genres.update(genres)
    for actors in movies_df['actors_list']:
        all_actors.update(actors)

    # Mã hóa genres và actors
    le_genres.fit(list(all_genres))
    le_actors.fit(list(all_actors))

    movies_df['genres_encoded'] = movies_df['genres_list'].apply(lambda x: [le_genres.transform([g])[0] for g in x if g in le_genres.classes_])
    movies_df['actors_encoded'] = movies_df['actors_list'].apply(lambda x: [le_actors.transform([a])[0] for a in x if a in le_actors.classes_])

    # Chuẩn hóa release_year
    movies_df['release_year'] = movies_df['release_year'].fillna(movies_df['release_year'].median())
    movies_df['release_year_scaled'] = scaler.fit_transform(movies_df[['release_year']])

    num_users = users_df['userId'].max() + 1
    num_movies = movies_df['movieId'].max() + 1
    num_genres = len(le_genres.classes_)
    num_countries = len(le_country.classes_)
    num_genders = 2
    num_occupations = 21
    num_directors = len(le_director.classes_)
    num_actors = len(le_actors.classes_)
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
except Exception as e:
    logger.error(f"Error loading model or data: {e}")
    raise

@app.post("/recommend")
async def recommend(data: dict):
    try:
        user_id = data.get("user_id")
        user_features = np.array(data.get("user_features"), dtype=np.float32)

        if user_id not in users_df['userId'].values:
            logger.warning(f"User ID {user_id} not found in users.csv")
            return {"error": "User ID not found"}

        user_data = users_df[users_df['userId'] == user_id].iloc[0]
        age = float(user_data['age'])
        gender = int(user_data['gender'])
        occupation = int(user_data['occupation'])

        movie_ids = torch.tensor(movies_df['movieId'].values, dtype=torch.long)
        num_movies = len(movie_ids)

        # Tạo tensor từ dữ liệu đã tiền xử lý
        age_tensor = torch.tensor([age] * num_movies, dtype=torch.float).unsqueeze(0)
        gender_tensor = torch.tensor([gender] * num_movies, dtype=torch.long).unsqueeze(0)
        occupation_tensor = torch.tensor([occupation] * num_movies, dtype=torch.long).unsqueeze(0)
        release_year = torch.tensor(movies_df['release_year_scaled'].values, dtype=torch.float).unsqueeze(0)
        country = torch.tensor(movies_df['country_encoded'].values, dtype=torch.long).unsqueeze(0)
        director = torch.tensor(movies_df['director_encoded'].values, dtype=torch.long).unsqueeze(0)

        # Xử lý genres và main_actors (padding để có kích thước cố định)
        max_genres = 3
        max_actors = 3
        genres_padded = np.zeros((num_movies, max_genres), dtype=np.int64)
        actors_padded = np.zeros((num_movies, max_actors), dtype=np.int64)

        for i in range(num_movies):
            genres = movies_df['genres_encoded'].iloc[i]
            actors = movies_df['actors_encoded'].iloc[i]
            genres_padded[i, :len(genres)] = genres[:max_genres]
            actors_padded[i, :len(actors)] = actors[:max_actors]

        genres_tensor = torch.tensor(genres_padded, dtype=torch.long).unsqueeze(0)
        main_actors_tensor = torch.tensor(actors_padded, dtype=torch.long).unsqueeze(0)
        synopsis_tfidf = torch.tensor([user_features] * num_movies, dtype=torch.float)

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
                genres_tensor,
                main_actors_tensor,
                synopsis_tfidf
            )

        top_indices = torch.topk(output[0], 5).indices
        top_movie_ids = movie_ids[top_indices].tolist()

        return top_movie_ids
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {e}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
