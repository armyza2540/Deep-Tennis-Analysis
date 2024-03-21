import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initial the ResNet 101 model
        self.model = models.resnet101(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        # Update the model with the parameters in which we fine-tuned for detecting court key points
        self.model.load_state_dict(torch.load(model_path,  map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define the transforms to be applied to the image during inference; same as during training 
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
         
    def predict(self, image):
        #  Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Unsqueeze the image to add a batch dimension; model expect a list of images to predict on(not a single item)
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            keypoints = self.model(image_tensor)
        # Squeeze the output to remove the batch dimension (since we're only predicting on one image)
        keypoints = keypoints.squeeze(0)
        keypoints = keypoints.cpu().numpy()
        original_height, original_width = image.shape[:2]
        keypoints[::2] *= original_width/224.0 # maps the image back to original width from 224
        keypoints[1::2] *= original_height/224.0 # maps the image back to original height from 224
        return keypoints
        
    def draw_keypoints(self, image, keypoints):
        # Draw the keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            
            # Draw circles as keypoints on the image
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
            
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames