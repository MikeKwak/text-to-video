import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TextToVideoDataset(Dataset):
    def __init__(self, processed_data_path):
        self.processed_data_path = processed_data_path
        self.video_files = [f for f in os.listdir(processed_data_path) if f.endswith('_frames.npy')]
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_id = video_file.split('_frames.npy')[0]
        
        frames = np.load(os.path.join(self.processed_data_path, video_file))
        with open(os.path.join(self.processed_data_path, f'{video_id}_caption.txt'), 'r') as f:
            caption = f.read().split()
        
        return torch.tensor(caption), torch.tensor(frames, dtype=torch.float32)
