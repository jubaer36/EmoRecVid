import cv2
import os

# Force CPU to avoid CUDA/CuDNN version mismatch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from deepface import DeepFace
from collections import Counter

def get_gender(video_path, seconds=10):
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return "Unknown"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Could not read FPS for {video_path}")
        return "Unknown"
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Analyze limited seconds or full video if shorter
    analyze_frames = int(min(duration, seconds) * fps)
    
    genders = []
    
    # Sample 1 frame per second to save time
    for i in range(0, analyze_frames, int(fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        try:
            # Enforce detection False to handle frames where face might not be clear
            # Use retinaface for better accuracy on CPU (might be slower but more accurate)
            result = DeepFace.analyze(frame, actions=['gender'], detector_backend='retinaface', enforce_detection=False, silent=True)
            if isinstance(result, list):
                result = result[0]
            genders.append(result['dominant_gender'])
            print(f"Frame {i}: {result['dominant_gender']}")
        except Exception as e:
            # print(f"Frame {i}: Error - {e}")
            pass
            
    cap.release()
    
    if not genders:
        return "Unknown"
        
    offset_gender = Counter(genders).most_common(1)[0][0]
    return offset_gender

if __name__ == "__main__":
    split_dir = "split_videos"
    videos = ["left_speaker.avi", "right_speaker.avi"]
    
    layout = {}
    
    for vid in videos:
        path = os.path.join(split_dir, vid)
        print(f"Analyzing {path}...")
        gender = get_gender(path)
        print(f"Result for {vid}: {gender}")
        layout[vid] = gender
        
    print("\nFinal Results:")
    print(layout)
