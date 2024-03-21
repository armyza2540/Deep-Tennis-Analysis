from ultralytics import YOLO
import pickle
import cv2


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect_frame(self, frame):
        # detect people/players in the given frame and returns a dictionary of players_id and associated bounding box coordinates
        results = self.model.track(frame, persist=True)[0] # persist tracking for multiple frames(remember tracking that were done before);
        id_name_dict = results.names
        
        player_dict = {}

        for box in results.boxes:
            track_id = int(box.id.tolist()[0]) # id of bounding box
            result = box.xyxy.tolist()[0] # bounding box coordinates
            object_class_id = box.cls.tolist()[0] # class id
            object_class_name = id_name_dict[object_class_id] # class name
            # save the bounding box associated with an id in a dictionary of all persons detected
            if object_class_name == 'person':
                player_dict[track_id] = result
        return player_dict

    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        # read saved frames from pickle file if read_from_stub is True and stub_path is provided
        if read_from_stub is True and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections
        
        # detect multiple frames utilizing the detect_frame method defined above
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict) 
        
        # Save the frames in a pickle file if stub is provided 
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
            
        return player_detections # list of dictionaries of player id and key and bounding box coordinates and values for each frame
    
    def draw_bounding_boxes(self, video_frames, player_detections):
        # draw bounding boxes of detected player on each frame of the video
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for player_id, bbox in player_dict.items():
                # extract the coordinates of bbox
                x1, y1, x2, y2 = bbox
                #  draw bounding boxes
                cv2.putText(frame, f"Player ID: {str(player_id)}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
            
    
            