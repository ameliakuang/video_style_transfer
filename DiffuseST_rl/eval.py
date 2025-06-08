import os
import numpy as np
from PIL import Image
import torch
from eval_metrics import calculate_clip_score, compute_clip_similarities, compute_style_similarities#, calculate_fvd_score
import argparse
from tqdm import tqdm

def load_frames_from_directory(directory, subfolder_prefix=''):
    all_frames = []
    
    # Get all subfolders that start with the prefix
    subfolders = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith(subfolder_prefix)]
    subfolders.sort()  # Sort subfolders to maintain order
    
    if subfolders:
        for subfolder in subfolders:
            subfolder_path = os.path.join(directory, subfolder)
            frame_files = sorted([f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            frames = []
            for frame_file in frame_files:
                frame_path = os.path.join(subfolder_path, frame_file)
                frame = Image.open(frame_path).convert('RGB')  # Ensure RGB format
                frame = frame.resize((224, 224))  # Resize to consistent size
                frame = np.array(frame)  # Convert to numpy array
                frames.append(frame)
            
            all_frames.append(np.stack(frames))  # Stack frames for each video
    else:
        video_frames = []
        for frame_file in sorted(os.listdir(directory)):
            if frame_file.endswith(('.png', '.jpg', '.jpeg')):
                frame_path = os.path.join(directory, frame_file)
                frame = np.array(Image.open(frame_path))
                video_frames.append(frame)
        if video_frames:
            all_frames = [video_frames]  # Make it a list with one video
        else:
            raise ValueError(f"No valid frames found in {directory}")
    
    return np.stack(all_frames)  # Return shape: [num_videos, num_frames, H, W, C]

def evaluate_temporal_consistency(frames):
    # frames shape: [num_videos, num_frames, H, W, C]
    clip_similarities_per_video = []
    
    for video_frames in frames:  # Loop through each video
        clip_similarities = compute_clip_similarities([Image.fromarray(frame) for frame in video_frames])
        clip_similarities_per_video.append(np.mean(clip_similarities))
    
    return {
        'clip_similarities': clip_similarities_per_video,
        'mean_temporal_consistency': np.mean(clip_similarities_per_video),
        'std_temporal_consistency': np.std(clip_similarities_per_video)
    }

def convert_frames_to_tensor(frames, target_size=(224, 224)):
    video_tensor = []
    for frame in frames:
        frame = frame.resize(target_size)
        frame_array = np.array(frame)
        video_tensor.append(frame_array)
    
    return np.stack(video_tensor)

def main():
    parser = argparse.ArgumentParser(description='Evaluate stylized frames')
    parser.add_argument('--stylized_train_dir', type=str, default='./output/exp_20250603_032809_epochs10_lr0p0001_videotrain_5/train_eval', help='Directory containing stylized frames')
    parser.add_argument('--stylized_test_dir', type=str, default='./output/exp_20250603_032809_epochs10_lr0p0001_videotrain_5/test_eval', help='Directory containing stylized frames')
    # parser.add_argument('--stylized_train_dir', type=str, default='./output_baseline/train', help='Directory containing stylized frames')
    # parser.add_argument('--stylized_test_dir', type=str, default='./output_baseline/test', help='Directory containing stylized frames')
    parser.add_argument('--train_content_dir', type=str, default='../data2/train_5', help='Directory containing original content frames')
    parser.add_argument('--test_content_dir', type=str, default='../data2/test_2', help='Directory containing original content frames')
    parser.add_argument('--style_dir', type=str, default='./images/style', help='Directory containing style frames')
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt', help='File to save evaluation results')
    
    args = parser.parse_args()
    
    def evaluate_set(stylized_frames, content_frames, style_img, set_name):
        print(f"\nEvaluating {set_name} set...")
        results = {}
        results['temporal'] = []
        
        print("1. Evaluating temporal consistency...")
        temporal_results = evaluate_temporal_consistency(stylized_frames)
        results['temporal'] = temporal_results['mean_temporal_consistency']

        print(f"Temporal consistency results: {results['temporal']}")
        
        print("2. Calculating frame-wise content preservation...")
        content_similarities = []
        
        # Loop through each video
        for stylized_video, content_video in tqdm(zip(stylized_frames, content_frames), total=len(stylized_frames)):
            video_similarities = []
            # Compare corresponding frames
            for stylized, content in zip(stylized_video, content_video):
                sim = calculate_clip_score(Image.fromarray(stylized), Image.fromarray(content))
                video_similarities.append(sim)
            content_similarities.append(np.mean(video_similarities))
        
        results['content_similarities'] = np.mean(content_similarities)

        print(f"Content preservation results: {results['content_similarities']}")

        print("3. Calculating frame-wise style preservation...")
        style_similarities = []
        
        for stylized_video in tqdm(stylized_frames, total=len(stylized_frames)):
            style_scores = []
            # Compare corresponding frames
            for stylized in stylized_video:
                score = compute_style_similarities(stylized, style_img[0])
                style_scores.append(score)
            style_similarities.append(np.mean(style_scores))
        
        results['style_similarities'] = np.mean(style_similarities)

        print(f"Style preservation results: {results['style_similarities']}")
        
        return results
    
    print("Loading frames...")
    stylized_train_frames = load_frames_from_directory(args.stylized_train_dir, 'frames_')
    stylized_test_frames = load_frames_from_directory(args.stylized_test_dir, 'frames_')
    train_content_frames = load_frames_from_directory(args.train_content_dir, 'frames_')
    test_content_frames = load_frames_from_directory(args.test_content_dir, 'frames_')
    style_frames = load_frames_from_directory(args.style_dir)

    print(f"Loaded {stylized_train_frames.shape[0]} x {stylized_train_frames.shape[1]} stylized train frames")
    print(f"Loaded {stylized_test_frames.shape[0]} x {stylized_test_frames.shape[1]} stylized test frames")
    print(f"Loaded {train_content_frames.shape[0]} x {train_content_frames.shape[1]} train content frames")
    print(f"Loaded {test_content_frames.shape[0]} x {test_content_frames.shape[1]} test content frames")
    print(f"Loaded {style_frames.shape[0]} x {style_frames.shape[1]} style frames")
    
    # Evaluate both sets
    train_results = evaluate_set(stylized_train_frames, train_content_frames, style_frames[0], "training")
    test_results = evaluate_set(stylized_test_frames, test_content_frames, style_frames[0], "test")

    # Save results
    with open(args.output_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("==================\n\n")
        
        for set_name, results in [("Training Set", train_results), ("Test Set", test_results)]:
            f.write(f"{set_name} Results:\n")
            f.write("-----------------\n\n")
            
            f.write("Temporal Consistency (CLIP similarities):\n")
            f.write(f"Mean: {results['temporal']:.4f}\n")
            
            f.write("Content Preservation (CLIP similarities):\n")
            f.write(f"Mean: {results['content_similarities']:.4f}\n")
            
            f.write(f"Style Preservation (CLIP similarities):\n")
            f.write(f"Mean: {results['style_similarities']:.4f}\n")
            
            f.write("\n")

if __name__ == "__main__":
    main() 