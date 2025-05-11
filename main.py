from fastapi import FastAPI
import torch
import numpy as np

app = FastAPI()

class TransformerRecommender(torch.nn.Module):
    def __init__(self, num_users=944, num_movies=1683, num_genres=589, num_countries=45, num_genders=2, 
                 num_occupations=21, num_directors=955, num_actors=1636, embed_size=64, num_transformer_layers=2):
        super(TransformerRecommender, self).__init__()
        
        # Embedding layers
        self.user_embedding = torch.nn.Embedding(num_users, embed_size)
        self.movie_embedding = torch.nn.Embedding(num_movies, embed_size)
        self.genre_embedding = torch.nn.Embedding(num_genres, embed_size)
        self.country_embedding = torch.nn.Embedding(num_countries, embed_size)
        self.gender_embedding = torch.nn.Embedding(num_genders, embed_size)
        self.occupation_embedding = torch.nn.Embedding(num_occupations, embed_size)
        self.director_embedding = torch.nn.Embedding(num_directors, embed_size)
        self.actor_embedding = torch.nn.Embedding(num_actors, embed_size)
        
        # Linear layers for additional features
        self.age_linear = torch.nn.Linear(1, embed_size)
        self.release_year_linear = torch.nn.Linear(1, embed_size)
        self.synopsis_linear = torch.nn.Linear(100, embed_size)  # Giả sử synopsis có 100 chiều
        
        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_size, nhead=8, dim_feedforward=256, dropout=0.1)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        # Output layer
        self.fc = torch.nn.Linear(embed_size, 1)  # Đầu ra 1 chiều (theo checkpoint)

    def forward(self, x):
        # x: [batch_size, num_features] (cần xử lý để phù hợp với đầu vào transformer)
        # Giả sử x chứa user_id, movie_id, genre_id, country_id, gender_id, occupation_id, director_id, actor_id, age, release_year, synopsis
        batch_size = x.size(0)
        embeddings = []
        
        # Lấy embedding từ các đặc trưng
        user_emb = self.user_embedding(x[:, 0].long())  # user_id
        movie_emb = self.movie_embedding(x[:, 1].long())  # movie_id
        genre_emb = self.genre_embedding(x[:, 2].long())  # genre_id
        country_emb = self.country_embedding(x[:, 3].long())  # country_id
        gender_emb = self.gender_embedding(x[:, 4].long())  # gender_id
        occupation_emb = self.occupation_embedding(x[:, 5].long())  # occupation_id
        director_emb = self.director_embedding(x[:, 6].long())  # director_id
        actor_emb = self.actor_embedding(x[:, 7].long())  # actor_id
        
        # Xử lý đặc trưng số (age, release_year, synopsis)
        age_emb = self.age_linear(x[:, 8].unsqueeze(-1))  # age
        release_year_emb = self.release_year_linear(x[:, 9].unsqueeze(-1))  # release_year
        synopsis_emb = self.synopsis_linear(x[:, 10:110])  # synopsis (100 chiều)
        
        # Kết hợp tất cả embedding
        combined_emb = user_emb + movie_emb + genre_emb + country_emb + gender_emb + occupation_emb + director_emb + actor_emb + age_emb + release_year_emb + synopsis_emb
        combined_emb = combined_emb.unsqueeze(1)  # Thêm chiều sequence cho transformer [batch_size, 1, embed_size]
        
        # Transformer
        output = self.transformer(combined_emb)
        output = output.squeeze(1)  # [batch_size, embed_size]
        
        # Output layer
        output = self.fc(output)  # [batch_size, 1]
        return output

# Khởi tạo và tải mô hình
model = TransformerRecommender()
model.load_state_dict(torch.load("transformer_recommender_stable.pth", map_location=torch.device('cpu')))
model.eval()

# Tối ưu hóa với torch.compile (tùy chọn)
model = torch.compile(model, mode="reduce-overhead")

@app.post("/recommend")
async def recommend(data: dict):
    user_id = data.get("user_id")
    user_features = np.array(data.get("user_features"), dtype=np.float32)  # [100] từ synopsis
    
    # Giả sử user_features là synopsis (100 chiều), cần thêm các đặc trưng khác để khớp với mô hình
    # Tạm thời chỉ dùng synopsis để test
    input_tensor = torch.tensor(user_features).unsqueeze(0)  # [1, 100]
    
    with torch.no_grad():
        output = model(input_tensor)  # [1, 1]
    
    movie_score = output.item()  # Lấy giá trị đầu ra (1 chiều)
    # Logic để ánh xạ movie_score thành movie_ids (cần điều chỉnh dựa trên dữ liệu huấn luyện)
    # Ví dụ: Giả sử output là điểm số, lấy top 5 movie_ids từ cơ sở dữ liệu dựa trên điểm số
    movie_ids = [1, 2, 3, 4, 5]  # Placeholder, cần thay bằng logic thực tế
    
    return movie_ids
