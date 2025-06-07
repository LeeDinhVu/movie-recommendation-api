import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Load processed data
try:
    data = pd.read_csv('processed_data/processed_data.csv')
    genres = pd.read_csv('processed_data/genres.csv', index_col=0)
    synopsis_embeddings = np.load('processed_data/synopsis_embeddings.npy')
    movies_df = pd.read_csv('movies.csv', sep=',',
                            names=['movieId', 'title', 'release_year', 'genres', 'synopsis', 'director', 'main_actors', 'country'],
                            skiprows=1)
    movies_df[['genres', 'country', 'director', 'main_actors', 'synopsis']] = movies_df[['genres', 'country', 'director', 'main_actors', 'synopsis']].fillna('Unknown')
    logger.info("Data loaded successfully")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise

# Load encoders
try:
    with open('processed_data/le_user.pkl', 'rb') as f:
        le_user = pickle.load(f)
    with open('processed_data/le_movie.pkl', 'rb') as f:
        le_movie = pickle.load(f)
    with open('processed_data/le_director.pkl', 'rb') as f:
        le_director = pickle.load(f)
    with open('processed_data/le_actor.pkl', 'rb') as f:
        le_actor = pickle.load(f)
    logger.info("Encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading encoders: {str(e)}")
    raise

# Preprocess movies_df for similarity matrix
label_encoders = {}
movies_df['genres'] = movies_df['genres'].str.split(',')
all_genres = set(g for genres in movies_df['genres'] for g in genres)
genre_encoder = LabelEncoder()
genre_encoder.fit(list(all_genres))
label_encoders['genres'] = genre_encoder

country_encoder = LabelEncoder()
movies_df['country_encoded'] = country_encoder.fit_transform(movies_df['country'])
label_encoders['country'] = country_encoder

director_encoder = LabelEncoder()
movies_df['director_encoded'] = director_encoder.fit_transform(movies_df['director'])
label_encoders['director'] = director_encoder

movies_df['main_actors'] = movies_df['main_actors'].str.split(',')
all_actors = set(a for actors in movies_df['main_actors'] for a in actors)
actor_encoder = LabelEncoder()
actor_encoder.fit(list(all_actors))
label_encoders['main_actors'] = actor_encoder

scaler = MinMaxScaler()
movies_df['release_year_scaled'] = scaler.fit_transform(movies_df[['release_year']].astype(float))

tfidf = TfidfVectorizer(max_features=100)
synopsis_vectors = tfidf.fit_transform(movies_df['synopsis']).toarray()

# Create similarity matrix
def get_movie_features(row):
    try:
        genre_ids = label_encoders['genres'].transform([g for g in row['genres'] if g in label_encoders['genres'].classes_])[:3]
        actor_ids = label_encoders['main_actors'].transform([a for a in row['main_actors'] if a in label_encoders['main_actors'].classes_])[:3]
        synopsis_vec = synopsis_vectors[row.name]
        return np.concatenate([
            np.pad(genre_ids, (0, 3 - len(genre_ids)), 'constant'),
            [row['country_encoded']],
            [row['director_encoded']],
            np.pad(actor_ids, (0, 3 - len(actor_ids)), 'constant'),
            [row['release_year_scaled']],
            synopsis_vec
        ])
    except Exception as e:
        logger.error(f"Error in get_movie_features for row index {row.name}: {str(e)}")
        raise

try:
    movie_features = np.array([get_movie_features(row) for _, row in movies_df.iterrows()])
    similarity_matrix = cosine_similarity(movie_features)
    logger.info("Similarity matrix created successfully")
except Exception as e:
    logger.error(f"Error creating similarity matrix: {str(e)}")
    raise

# Define Transformer Model
class TransformerRecommender(nn.Module):
    def __init__(self, num_users, num_movies, num_occupations, num_directors, num_actors, genre_dim, embed_dim=32, dropout=0.1):
        super(TransformerRecommender, self).__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.movie_embed = nn.Embedding(num_movies, embed_dim)
        self.occupation_embed = nn.Embedding(num_occupations, 16)
        self.director_embed = nn.Embedding(num_directors, 16)
        self.actor_embed = nn.Embedding(num_actors, 16)

        self.age_fc = nn.Linear(1, 16)
        self.year_fc = nn.Linear(1, 16)
        self.genre_fc = nn.Linear(genre_dim, 32)
        self.synopsis_fc = nn.Linear(64, 32)

        self.user_combine = nn.Linear(embed_dim + 16 + 16, 64)
        self.movie_combine = nn.Linear(embed_dim + 16 + 16 + 16 + 32 + 32, 64)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True, dropout=dropout),
            num_layers=2
        )

        self.final_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_id, movie_id, age, occupation, release_year, director, top_actor, genres, synopsis):
        user_vec = self.user_embed(user_id)
        age_vec = self.age_fc(age.unsqueeze(1))
        occupation_vec = self.occupation_embed(occupation)
        user_combined = torch.cat([user_vec, age_vec, occupation_vec], dim=1)
        user_vector = self.user_combine(user_combined)

        movie_vec = self.movie_embed(movie_id)
        year_vec = self.year_fc(release_year.unsqueeze(1))
        director_vec = self.director_embed(director)
        actor_vec = self.actor_embed(top_actor)
        genre_vec = self.genre_fc(genres)
        synopsis_vec = self.synopsis_fc(synopsis)
        movie_combined = torch.cat([movie_vec, year_vec, director_vec, actor_vec, genre_vec, synopsis_vec], dim=1)
        movie_vector = self.movie_combine(movie_combined)

        interaction = torch.stack([user_vector, movie_vector], dim=1)
        transformer_out = self.transformer(interaction).mean(dim=1)

        final_input = torch.cat([user_vector, movie_vector], dim=1)
        output = self.final_estimator(final_input)
        return output.squeeze()

# Initialize model
try:
    num_users = data['userId_mapped'].nunique()
    num_movies = data['movieId_mapped'].nunique()
    num_occupations = data['occupation_encoded'].nunique()
    num_directors = data['director_encoded'].nunique()
    num_actors = data['top_actor_encoded'].nunique()
    genre_dim = genres.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerRecommender(num_users, num_movies, num_occupations, num_directors, num_actors, genre_dim, dropout=0.1)
    model.to(device)
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise

# Load best checkpoint
try:
    checkpoint_path = 'updated_checkpoint/best_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("Checkpoint loaded successfully")
except Exception as e:
    logger.error(f"Error loading checkpoint: {str(e)}")
    raise

# Pydantic model for request validation
class RecommendRequest(BaseModel):
    user_id: int
    movie_id: int | None = None
    top_k: int = 5

# Generate recommendations
async def recommend_movies(user_id: int, top_k: int = 5, movie_id: int | None = None):
    try:
        logger.info(f"Generating recommendations for user_id={user_id}, top_k={top_k}, movie_id={movie_id}")

        if user_id not in le_user.classes_:
            logger.warning(f"User ID {user_id} not found")
            raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")

        user_data = data[data['userId'] == user_id].iloc[0]
        user_id_mapped = user_data['userId_mapped']

        unique_movie_ids = data['movieId_mapped'].unique()
        unique_movie_data = data[['movieId', 'movieId_mapped', 'release_year_scaled', 'director_encoded', 'top_actor_encoded']].drop_duplicates('movieId_mapped')
        unique_movie_data = unique_movie_data[unique_movie_data['movieId_mapped'].isin(unique_movie_ids)]

        unique_indices = [data[data['movieId_mapped'] == mid].index[0] for mid in unique_movie_ids]
        unique_movie_genres = genres.loc[unique_indices]
        unique_movie_synopsis = synopsis_embeddings[unique_indices]

        inputs = {
            'user_id': torch.tensor([user_id_mapped] * len(unique_movie_ids), dtype=torch.long, device=device),
            'movie_id': torch.tensor(unique_movie_ids, dtype=torch.long, device=device),
            'age': torch.tensor([user_data['age_scaled']] * len(unique_movie_ids), dtype=torch.float, device=device),
            'occupation': torch.tensor([user_data['occupation_encoded']] * len(unique_movie_ids), dtype=torch.long, device=device),
            'release_year': torch.tensor(unique_movie_data['release_year_scaled'].values, dtype=torch.float, device=device),
            'director': torch.tensor(unique_movie_data['director_encoded'].values, dtype=torch.long, device=device),
            'top_actor': torch.tensor(unique_movie_data['top_actor_encoded'].values, dtype=torch.long, device=device),
            'genres': torch.tensor(unique_movie_genres.values, dtype=torch.float, device=device),
            'synopsis': torch.tensor(unique_movie_synopsis, dtype=torch.float, device=device)
        }

        with torch.no_grad():
            predictions = model(**inputs)
        predictions = predictions.cpu().numpy() * 4 + 1

        if movie_id is None:
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            recommended_movie_ids_mapped = unique_movie_ids[top_indices]
            recommended_movie_ids = le_movie.inverse_transform(recommended_movie_ids_mapped)
        else:
            if movie_id not in le_movie.classes_:
                logger.warning(f"Movie ID {movie_id} not found")
                raise HTTPException(status_code=404, detail=f"Movie ID {movie_id} not found")

            movie_row = movies_df[movies_df['movieId'] == movie_id]
            if movie_row.empty:
                logger.error(f"No data found for movie_id={movie_id}")
                raise HTTPException(status_code=404, detail=f"No data found for movie_id={movie_id}")
            movie_idx = movie_row.index[0]

            similarity_scores = similarity_matrix[movie_idx]

            filtered_movie_ids = unique_movie_data['movieId'].values.copy()
            filtered_scores = (predictions * 0.7 + similarity_scores * 0.3).copy()

            mask = filtered_movie_ids != movie_id
            filtered_movie_ids = filtered_movie_ids[mask]
            filtered_scores = filtered_scores[mask]

            top_indices = np.argsort(filtered_scores)[-top_k:][::-1]
            recommended_movie_ids = filtered_movie_ids[top_indices]

        logger.info(f"Generated {len(recommended_movie_ids)} recommendations")
        return recommended_movie_ids.tolist()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recommend_movies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in recommend endpoint: {str(e)}")

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    recommendations = await recommend_movies(request.user_id, request.top_k, request.movie_id)
    return recommendations  # Return list directly: [movieId1, movieId2, ...]

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
