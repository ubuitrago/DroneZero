import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class VideoPredictor:
    def __init__(self, model_path='gesture_classifier.h5', pose_model_path='pose_landmarker_full.task', sequences_path='pose_sequences.npz'):
        # Load the gesture classification model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load gesture label mappings
        gesture2label = np.load(sequences_path, allow_pickle=True)['gesture2label'].item()
        self.label2gesture = {v: k for k, v in gesture2label.items()}
        
        # Load context mapping
        with open('Base_Signals_Context.json', 'r') as f:
            self.context_mapping = json.load(f)['Command Gestures']
        
        # Initialize MediaPipe PoseLandmarker
        base_options = python.BaseOptions(model_asset_path=pose_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        # Initialize parameters
        self.WINDOW_SIZE = 10
        self.SMOOTHING_WINDOW = 5
        self.frame_buffer = []
        self.predictions_queue = []

    def normalize_landmarks(self, landmarks):
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        center = np.array([(left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2, (left_hip.z + right_hip.z)/2])
        normalized = []
        for lm in landmarks:
            normalized.extend([lm.x - center[0], lm.y - center[1], lm.z - center[2], lm.visibility])
        return normalized

    def wrap_landmarks(self, landmarks_list):
        landmark_proto = landmark_pb2.NormalizedLandmarkList()
        for lm in landmarks_list:
            landmark = landmark_pb2.NormalizedLandmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            )
            landmark_proto.landmark.append(landmark)
        return landmark_proto

    def draw_landmarks_on_image(self, image, landmarks):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            self.wrap_landmarks(landmarks),
            mp_pose.POSE_CONNECTIONS)
        return annotated_image

    def process_video(self, video_path, output_dir='video_predictions'):
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        start_time = time.time()
        predictions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb = np.ascontiguousarray(image_rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.pose_landmarks and detection_result.pose_landmarks[0]:
                landmarks = detection_result.pose_landmarks[0]
                features = self.normalize_landmarks(landmarks)
                self.frame_buffer.append((features, frame.copy(), landmarks))
            else:
                self.frame_buffer.append(([0]*132, frame.copy(), None))
            
            if len(self.frame_buffer) >= self.WINDOW_SIZE:
                sequence = [fb[0] for fb in self.frame_buffer[-self.WINDOW_SIZE:]]
                sequence = np.expand_dims(sequence, axis=0)
                prediction_probs = self.model.predict(sequence)[0]
                self.predictions_queue.append(prediction_probs)
                
                if len(self.predictions_queue) > self.SMOOTHING_WINDOW:
                    self.predictions_queue.pop(0)
                
                avg_prediction = np.mean(self.predictions_queue, axis=0)
                top_pred_idx = np.argmax(avg_prediction)
                top_pred_label = self.label2gesture[top_pred_idx]
                top_pred_confidence = avg_prediction[top_pred_idx]
                
                # Save prediction and frame overlay every 30 frames (1 second at 30fps)
                if frame_idx % 30 == 0:
                    # Save frame overlay
                    if self.frame_buffer[-1][2] is not None:
                        annotated_frame = self.draw_landmarks_on_image(frame, self.frame_buffer[-1][2])
                        cv2.imwrite(os.path.join(output_dir, f'frame_overlay{len(predictions)+1}.jpg'), annotated_frame)
                    
                    # Add prediction to list
                    predictions.append({
                        'frame': frame_idx,
                        'gesture': top_pred_label,
                        'confidence': float(top_pred_confidence),
                        'context': self.context_mapping.get(top_pred_label, "No context available")
                    })
            
            # Print progress
            frame_idx += 1
            if frame_idx % 30 == 0:  # Print every second (assuming 30fps)
                elapsed_time = time.time() - start_time
                print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%) - Elapsed time: {elapsed_time:.1f}s")
        
        # Save predictions to file
        with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
            for pred in predictions:
                f.write(f"Frame {pred['frame']}:\n")
                f.write(f"Gesture: {pred['gesture']}\n")
                f.write(f"Confidence: {pred['confidence']:.2f}\n")
                f.write(f"Context: {pred['context']}\n")
                f.write("-" * 50 + "\n")
        
        cap.release()
        print(f"Video processing complete. Output saved to: {output_dir}")
        return output_dir

if __name__ == "__main__":
    # Example usage
    predictor = VideoPredictor()
    video_path = "C:\\Users\\Uriel Buitrago\\Desktop\\DroneZero\\Unseen_Gestures\\move-back-10ft-e.MP4"
    results_dir = predictor.process_video(video_path) 