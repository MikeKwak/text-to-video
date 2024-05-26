# src/main.py

from inference.generate_video import main as generate_video

import json
import os

def preprocess_dataset(json_file_path, video_path, processed_data_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    info = data['info']
    videos = data['videos']
    sentences = data['sentences']
    
    for video in videos:
        video_id = video['video_id']
        split = video['split']
        video_filename = f"{video_id}.mp4"
        video_full_path = os.path.join(video_path, video_filename)
    
        if not os.path.exists(video_full_path):
            print(f"Video file {video_filename} not found, skipping.")
            continue
    
        print(f"Processing video: {video_filename}")
    
        frames = extract_frames(video_full_path)
        save_frames(frames, video_id, processed_data_path)
        print(f"Saved frames for video: {video_filename}")
    
        for sentence in sentences:
            if sentence['video_id'] == video_id:
                caption = sentence['caption']
                tokens = preprocess_text(caption)
                save_text(tokens, video_id, processed_data_path)
                print(f"Saved caption for video: {video_filename}")
    
    print("Data preprocessing complete.")

if __name__ == "__main__":
    text_prompt = "A girl smiling"
    step = 4  # Options: [1, 2, 4, 8]
    output_path = "animation.gif"
    
    generate_video(text_prompt, step, output_path)
   
