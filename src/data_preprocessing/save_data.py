import os
import numpy as np

def save_frames(frames, video_id, processed_data_path):
    output_path = os.path.join(processed_data_path, f'{video_id}_frames.npy')
    np.save(output_path, frames)

def save_text(tokens, video_id, processed_data_path):
    output_path = os.path.join(processed_data_path, f'{video_id}_caption.txt')
    with open(output_path, 'w') as f:
        f.write(' '.join(tokens))
