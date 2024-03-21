import os
from utils import read_video, save_video
from trackers import PlayerTracker, TennisBallTracker
from court_line_detector import CourtLineDetector


def main():
    # Define and read input video for inference
    input_video_path = './input_videos/input_video.mp4'
    video_frames = read_video(input_video_path)
    
    # Player tracker model and detect players
    player_tracker_model_name = 'yolov8x'
    player_tracker = PlayerTracker(player_tracker_model_name)
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='./tracker_stubs/player_detection.pkl') # pkl = pickle ext
    
    # Tennis ball tracker model and detect the tennis ball in the frames
    tennis_ball_tracker_model_path = './models/yolo5_best.pt'
    tennis_ball_tracker = TennisBallTracker(model_path=tennis_ball_tracker_model_path)
    ball_detections = tennis_ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path='./tracker_stubs/tennis_ball_detection.pkl')
    
    # Court Line Detector Model
    court_model_path = './models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(model_path=court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    
    # Draw outputs bounding boxes
    video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    video_frames = tennis_ball_tracker.draw_bounding_boxes(video_frames, ball_detections)
    video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)
    
    # Save the output video
    save_video(video_frames, './output_videos/output_video.avi')
    
    
if __name__ == '__main__':
    # Set the environment variable; to allow persist=True in model tracking without causing crashes
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # Verify that the variable is set
    print(os.environ['KMP_DUPLICATE_LIB_OK'])
    
    main()