from fastapi import FastAPI
import torch
import numpy as np

app = FastAPI()

class TransformerRecommender(torch.nn.Module):
    def __init__(self):
        super(TransformerRecommender, self).__init__()
        self.fc = torch.nn.Linear(100, 10)  # Ví dụ đơn giản

    def forward(self, x):
        return self.fc(x)

# Khởi tạo và tải mô hình
model = TransformerRecommender()
model.load_state_dict(torch.load("transformer_recommender_stable.pth", map_location=torch.device('cpu')))
model.eval()

# Tối ưu hóa với torch.compile (PyTorch 2.4.0)
model = torch.compile(model, mode="reduce-overhead")  # Tăng tốc inference trên CPU

@app.post("/recommend")
async def recommend(data: dict):
    user_id = data.get("user_id")
    user_features = np.array(data.get("user_features"), dtype=np.float32)
    
    input_tensor = torch.tensor(user_features).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    movie_ids = output.squeeze().numpy().astype(int).tolist()
    
    return movie_ids