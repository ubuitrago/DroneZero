# 3_Live_Predict_Gesture.py

# Install dependencies (if needed)
# !pip install mediapipe opencv-python tensorflow

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# --- Load Model ---

MODEL_PATH = 'gesture_classifier.h5'
POSE_MODEL_PATH = 'pose_landmarker_full.task'
SEQUENCES_PATH = 'pose_sequences.npz'
OUTPUT_DIR = 'live_predictions'

model = tf.keras.models.load_model(MODEL_PATH)

# Load gesture label mappings
gesture2label = np.load(SEQUENCES_PATH, allow_pickle=True)['gesture2label'].item()
label2gesture = {v: k for k, v in gesture2label.items()}
num_classes = len(label2gesture)

# --- Initialize MediaPipe PoseLandmarker ---

base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

# --- Helper Functions ---

def normalize_landmarks(landmarks):
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    center = np.array([(left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2, (left_hip.z + right_hip.z)/2])
    normalized = []
    for lm in landmarks:
        normalized.extend([lm.x - center[0], lm.y - center[1], lm.z - center[2], lm.visibility])
    return normalized

def wrap_landmarks(landmarks_list):
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

def draw_landmarks_on_image(image, landmarks):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        wrap_landmarks(landmarks),
        mp_pose.POSE_CONNECTIONS)
    return annotated_image

# --- Main Loop ---

cap = cv2.VideoCapture(0)
frame_buffer = []
WINDOW_SIZE = 10
os.makedirs(OUTPUT_DIR, exist_ok=True)
predictions_queue = []
SMOOTHING_WINDOW = 5
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)

    if detection_result.pose_landmarks and detection_result.pose_landmarks[0]:
        landmarks = detection_result.pose_landmarks[0]
        features = normalize_landmarks(landmarks)
        frame_buffer.append((features, frame.copy(), landmarks))
    else:
        frame_buffer.append(([0]*132, frame.copy(), None))

    if len(frame_buffer) >= WINDOW_SIZE:
        sequence = [fb[0] for fb in frame_buffer[-WINDOW_SIZE:]]
        sequence = np.expand_dims(sequence, axis=0)
        prediction_probs = model.predict(sequence)[0]
        predictions_queue.append(prediction_probs)

        if len(predictions_queue) > SMOOTHING_WINDOW:
            predictions_queue.pop(0)

        avg_prediction = np.mean(predictions_queue, axis=0)
        top_pred_idx = np.argmax(avg_prediction)
        top_pred_label = label2gesture[top_pred_idx]
        top_pred_confidence = avg_prediction[top_pred_idx]

        display_text = f"{top_pred_label} ({top_pred_confidence:.2f})"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show probability bar
        bar_start = 60
        for idx, conf in enumerate(avg_prediction):
            cv2.rectangle(frame, (10, bar_start + idx*20), (int(10 + conf*200), bar_start + idx*20 + 15), (255, 0, 0), -1)
            cv2.putText(frame, label2gesture[idx], (220, bar_start + idx*20 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Save outputs
        first_features, first_image, first_landmarks = frame_buffer[-WINDOW_SIZE]
        last_features, last_image, last_landmarks = frame_buffer[-1]

        if first_landmarks:
            annotated_first = draw_landmarks_on_image(first_image, first_landmarks)
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'first_frame_overlay.jpg'), annotated_first)
        if last_landmarks:
            annotated_last = draw_landmarks_on_image(last_image, last_landmarks)
            cv2.imwrite(os.path.join(OUTPUT_DIR, 'last_frame_overlay.jpg'), annotated_last)

        np.save(os.path.join(OUTPUT_DIR, 'first_frame_features.npy'), np.array(first_features))
        np.save(os.path.join(OUTPUT_DIR, 'last_frame_features.npy'), np.array(last_features))
        np.save(os.path.join(OUTPUT_DIR, 'full_confidences.npy'), avg_prediction)

    # Show FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Inference ended.")
