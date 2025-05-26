
from pypexels import PyPexels
import requests
import os
import cv2

def segment_video_to_frames(video_path, output_dir='frames', max_frames=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()

def frames_to_video(frame_dir, output_video_path, frame_rate=30, width=None, height=None):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    if not frame_files:
        print("Not found")
        return
    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    if first_frame is None:
        print("Error reading the first frame.")
        return
    if width is None or height is None:
        height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
    out.release()
    print(f"Video saved to {output_video_path}")

def main():
    api_key = 'h768YyoLQHnO34G6aDiEiZJLIVjjUIiSEbdRepkyQd037V4JiamH2kxa'
    py_pexel = PyPexels(api_key=api_key)
    search_videos_page = py_pexel.videos_search(query=["Nature", "pets", "flower", "animals"], per_page=20)
    os.makedirs('video', exist_ok=True)
    for video in search_videos_page.entries:
        best_video_file = video.video_files[0]
        for file in video.video_files:
            if file['quality'] == 'hd' and file['width'] > best_video_file['width']:
                best_video_file = file
        video_url = best_video_file['link']
        print("Downloading from:", video_url)
        response = requests.get(video_url)
        filename = os.path.join('video', f'train{video.id}.mp4')
        with open(filename, 'wb') as f:
            f.write(response.content)
        segment_video_to_frames(filename, output_dir=os.path.join('video', f'frames_{video.id}'), max_frames=20)
    
    frames_to_video("/content/video/frames_3042473", "/content/test.mp4", frame_rate=10, width=None, height=None)

if __name__ == "__main__":
    main()
