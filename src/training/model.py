import torch.nn as nn
import torch
from transformers import GPT2Model, GPT2Tokenizer

class TextToVideoModel(nn.Module):
    def __init__(self):
        super(TextToVideoModel, self).__init__()
        self.text_encoder = GPT2Model.from_pretrained('gpt2')
        self.video_generator = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, text):
        text_features = self.text_encoder(text)[0]
        text_features = text_features.unsqueeze(-1).unsqueeze(-1)
        video_frames = self.video_generator(text_features)
        return video_frames
