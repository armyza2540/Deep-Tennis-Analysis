from ultralytics import YOLO
import pickle
import cv2


class TennisBallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect_frame(self, frame):
        # detect people/players in the given frame and returns a dictionary of players_id and associated bounding box coordinates
        results = self.model.predict(frame, conf=0.15)[0]
 
        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0] # bounding box coordinates
            # save the bounding box associated with an id in a dictionary of all persons detected
            ball_dict[1] = result
            
        return ball_dict

    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        # read saved frames from pickle file if read_from_stub is True and stub_path is provided
        if read_from_stub is True and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections
        
        # detect multiple frames utilizing the detect_frame method defined above
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict) 
        
        # Save the frames in a pickle file if stub is provided 
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
            
        return ball_detections # list of dictionaries of player id and key and bounding box coordinates and values for each frame
    
    def draw_bounding_boxes(self, video_frames, ball_detections):
        # draw bounding boxes of detected tennis ball on each frame of the video
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for ball_id, bbox in ball_dict.items():
                # extract the coordinates of bbox
                x1, y1, x2, y2 = bbox
                #  draw bounding boxes
                cv2.putText(frame, f"Tennis Ball ID: {str(ball_id)}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
            
    
            