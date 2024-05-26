import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import TextToVideoDataset
from .model import TextToVideoModel
import torch.nn as nn
import torch

def train_model(processed_data_path, batch_size=2, num_epochs=10):
    dataset = TextToVideoDataset(processed_data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TextToVideoModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for captions, frames in dataloader:
            optimizer.zero_grad()
            outputs = model(captions)
            
            loss = criterion(outputs, frames)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    torch.save(model.state_dict(), os.path.join('..', 'models', 'text_to_video_model.pth'))
    print("Training complete.")
