import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import logging
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import csv

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        self.final_fc = nn.Sequential(
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
        output = self.final_fc(final_input)
        return output.squeeze()

# Load processed data and model
try:
    # Load data with robust CSV parsing
    data = pd.read_csv(
        "processed_data/processed_data.csv",
        quoting=csv.QUOTE_MINIMAL,
        delimiter=',',
        encoding='utf-8',
        on_bad_lines='skip'
    )
    genres = pd.read_csv(
        "processed_data/genres.csv",
        index_col=0,
        quoting=csv.QUOTE_MINIMAL,
        delimiter=',',
        encoding='utf-8'
    )
    synopsis_embeddings = np.load("processed_data/synopsis_embeddings.npy")
    
    # Log data shapes
    logger.info(f"Loaded data.csv with shape: {data.shape}")
    logger.info(f"Loaded genres.csv with shape: {genres.shape}")
    logger.info(f"Loaded synopsis_embeddings with shape: {synopsis_embeddings.shape}")
    
    # Load encoders
    with open("processed_data/le_user.pkl", "rb") as f:
        le_user = pickle.load(f)
    with open("processed_data/le_movie.pkl", "rb") as f:
        le_movie = pickle.load(f)
    with open("processed_data/le_occupation.pkl", "rb") as f:
        le_occupation = pickle.load(f)
    with open("processed_data/le_director.pkl", "rb") as f:
        le_director = pickle.load(f)
    with open("processed_data/le_actor.pkl", "rb") as f:
        le_actor = pickle.load(f)
    
    # Log encoder classes
    logger.info(f"le_user classes: {len(le_user.classes_)} users")
    logger.info(f"le_movie classes: {len(le_movie.classes_)} movies")
    
    # Initialize model
    num_users = data['userId_mapped'].nunique()
    num_movies = data['movieId_mapped'].nunique()
    num_occupations = data['occupation_encoded'].nunique()
    num_directors = data['director_encoded'].nunique()
    num_actors = data['top_actor_encoded'].nunique()
    genre_dim = genres.shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerRecommender(
        num_users,
        num_movies,
        num_occupations,
        num_directors,
        num_actors,
        genre_dim,
        dropout=0.1
    )
    model.to(device=device)
    
    # Load checkpoint with fallback
    checkpoint_path = "updated_checkpoint/best_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Create similarity matrix
    def get_movie_features(row):
        try:
            genre_vec = genres.loc[row.name].values
            return np.concatenate([
                genre_vec,
                [row['director_encoded']],
                [row['top_actor_encoded']],
                [row['release_year_scaled']]
            ])
        except Exception as e:
            logger.error(f"Error in get_movie_features for row index {row.name}: {str(e)}")
            raise

    movie_features = np.array([get_movie_features(row) for _, row in data.drop_duplicates('movieId_mapped').iterrows()])
    similarity_matrix = cosine_similarity(movie_features)
except Exception as e:
    logger.error(f"Failed to load model or files: {str(e)}")
    raise ValueError(f"Failed to load model or files: {str(e)}")

def generate_user_recommendations(user_id, top_k=5):
    try:
        logger.info(f"Generating recommendations for user_id={user_id}, top_k={top_k}")
        
        # Validate user_id
        if user_id not in le_user.classes_:
            logger.warning(f"User ID {user_id} not found in le_user.classes_")
            raise ValueError(f"User ID {user_id} not found")
        
        # Map user_id
        user_id_mapped = le_user.transform(np.array([user_id]))[0]
        
        # Get user data
        user_data = data[data['userId'] == user_id]
        if user_data.empty:
            logger.error(f"No data found for user_id={user_id}")
            raise ValueError(f"No data found for user_id={user_id}")
        user_data = user_data.iloc[0]
        age_scaled = user_data['age_scaled']
        occupation_encoded = user_data['occupation_encoded']
        
        # Prepare input tensors
        unique_movie_data = data[['movieId_mapped', 'release_year_scaled', 'director_encoded', 'top_actor_encoded']].drop_duplicates('movieId_mapped')
        unique_movie_ids = unique_movie_data['movieId_mapped'].values
        num_movies = len(unique_movie_ids)
        
        # Get genres and synopsis
        unique_indices = []
        for mid in unique_movie_ids:
            movie_rows = data[data['movieId_mapped'] == mid]
            if not movie_rows.empty:
                unique_indices.append(movie_rows.index[0])
            else:
                logger.warning(f"No data found for movieId_mapped={mid}")
                continue
        if not unique_indices:
            logger.error("No valid movie indices found")
            raise ValueError("No valid movie indices found")
        
        genres_tensor = torch.tensor(genres.loc[unique_indices].values, dtype=torch.float, device=device)
        synopsis_tensor = torch.tensor(synopsis_embeddings[unique_indices], dtype=torch.float, device=device)
        
        inputs = {
            'user_id': torch.tensor([user_id_mapped] * num_movies, dtype=torch.long, device=device),
            'movie_id': torch.tensor(unique_movie_ids, dtype=torch.long, device=device),
            'age': torch.tensor([age_scaled] * num_movies, dtype=torch.float, device=device),
            'occupation': torch.tensor([occupation_encoded] * num_movies, dtype=torch.long, device=device),
            'release_year': torch.tensor(unique_movie_data['release_year_scaled'].values, dtype=torch.float, device=device),
            'director': torch.tensor(unique_movie_data['director_encoded'].values, dtype=torch.long, device=device),
            'top_actor': torch.tensor(unique_movie_data['top_actor_encoded'].values, dtype=torch.long, device=device),
            'genres': genres_tensor,
            'synopsis': synopsis_tensor
        }
        
        # Generate predictions
        with torch.no_grad():
            predictions = model(**inputs)
        predictions = predictions.cpu().numpy() * 4 + 1
        
        # Get top-k movies
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_movie_ids_mapped = unique_movie_ids[top_indices]
        
        # Map back to original movie IDs
        top_movie_ids = le_movie.inverse_transform(top_movie_ids_mapped)
        
        logger.info(f"Generated {len(top_movie_ids)} recommendations for user_id={user_id}")
        return top_movie_ids.tolist()
    except Exception as e:
        logger.error(f"Error in generate_user_recommendations: {str(e)}")
        raise

def generate_recommendations(user_id, movie_id, top_k=5):
    try:
        logger.info(f"Generating recommendations for user_id={user_id}, movie_id={movie_id}, top_k={top_k}")
        
        # Validate user_id and movie_id
        if user_id not in le_user.classes_:
            logger.warning(f"User ID {user_id} not found in le_user.classes_")
            raise ValueError(f"User ID {user_id} not found")
        if movie_id not in le_movie.classes_:
            logger.warning(f"Movie ID {movie_id} not found in le_movie.classes_")
            raise ValueError(f"Movie ID {movie_id} not found")
        
        # Map user_id and movie_id
        user_id_mapped = le_user.transform(np.array([user_id]))[0]
        movie_id_mapped = le_movie.transform(np.array([movie_id]))[0]
        
        # Get user data
        user_data = data[data['userId'] == user_id]
        if user_data.empty:
            logger.error(f"No data found for user_id={user_id}")
            raise ValueError(f"No data found for user_id={user_id}")
        user_data = user_data.iloc[0]
        age_scaled = user_data['age_scaled']
        occupation_encoded = user_data['occupation_encoded']
        
        # Prepare input tensors
        unique_movie_data = data[['movieId_mapped', 'release_year_scaled', 'director_encoded', 'top_actor_encoded']].drop_duplicates('movieId_mapped')
        unique_movie_ids = unique_movie_data['movieId_mapped'].values
        num_movies = len(unique_movie_ids)
        
        # Get genres and synopsis
        unique_indices = []
        for mid in unique_movie_ids:
            movie_rows = data[data['movieId_mapped'] == mid]
            if not movie_rows.empty:
                unique_indices.append(movie_rows.index[0])
            else:
                logger.warning(f"No data found for movieId_mapped={mid}")
                continue
        if not unique_indices:
            logger.error("No valid movie indices found")
            raise ValueError("No valid movie indices found")
        
        genres_tensor = torch.tensor(genres.loc[unique_indices].values, dtype=torch.float, device=device)
        synopsis_tensor = torch.tensor(synopsis_embeddings[unique_indices], dtype=torch.float, device=device)
        
        inputs = {
            'user_id': torch.tensor([user_id_mapped] * num_movies, dtype=torch.long, device=device),
            'movie_id': torch.tensor(unique_movie_ids, dtype=torch.long, device=device),
            'age': torch.tensor([age_scaled] * num_movies, dtype=torch.float, device=device),
            'occupation': torch.tensor([occupation_encoded] * num_movies, dtype=torch.long, device=device),
            'release_year': torch.tensor(unique_movie_data['release_year_scaled'].values, dtype=torch.float, device=device),
            'director': torch.tensor(unique_movie_data['director_encoded'].values, dtype=torch.long, device=device),
            'top_actor': torch.tensor(unique_movie_data['top_actor_encoded'].values, dtype=torch.long, device=device),
            'genres': genres_tensor,
            'synopsis': synopsis_tensor
        }
        
        # Generate predictions
        with torch.no_grad():
            predictions = model(**inputs)
        predictions = predictions.cpu().numpy() * 4 + 1
        
        # Get similarity scores
        movie_rows = data[data['movieId_mapped'] == movie_id_mapped]
        if movie_rows.empty:
            logger.error(f"No data found for movieId_mapped={movie_id_mapped}")
            raise ValueError(f"No data found for movieId_mapped={movie_id_mapped}")
        movie_idx = movie_rows.index[0]
        similarity_scores = similarity_matrix[movie_idx]
        
        # Filter out input movie
        mask = unique_movie_ids != movie_id_mapped
        filtered_movie_ids = unique_movie_ids[mask]
        filtered_predictions = predictions[mask]
        filtered_similarity = similarity_scores[mask]
        
        # Combine scores
        combined_scores = filtered_predictions * 0.7 + filtered_similarity * 0.3
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        top_movie_ids_mapped = filtered_movie_ids[top_indices]
        
        # Map back to original movie IDs
        top_movie_ids = le_movie.inverse_transform(top_movie_ids_mapped)
        
        logger.info(f"Generated {len(top_movie_ids)} recommendations for user_id={user_id}, movie_id={movie_id}")
        return top_movie_ids.tolist()
    except Exception as e:
        logger.error(f"Error in generate_recommendations: {str(e)}")
        raise

@app.post("/recommend")
async def recommend(request: dict):
    try:
        user_id = request.get("user_id")
        movie_id = request.get("movie_id")
        top_k = request.get("top_k", 5)
        
        logger.info(f"Received recommend request: user_id={user_id}, movie_id={movie_id}, top_k={top_k}")
        
        if user_id is None:
            raise HTTPException(status_code=400, detail="Missing user_id in request")
        
        if movie_id is None:
            result = generate_user_recommendations(user_id, top_k)
        else:
            result = generate_recommendations(user_id, movie_id, top_k)
        
        return result
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in recommend endpoint: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
