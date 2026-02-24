import math
import time
import pandas as pd
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

# Ensure drawing.py is in the same folder
from drawing import draw_custom_skeleton, draw_hand_box_with_label, hand_bbox_from_landmarks

# --- CONFIGURATION ---
MODEL_PATH = Path("models/SVM1.4_noZvalue_noStdScale_Angle_model.pkl")
OUTPUT_VIDEO_PATH = "output_recording.mp4" # <--- NEW: Output file name
PREDICTION_WINDOW = 5
TARGET_WIDTH = 640

# Performance knobs.
MAX_NUM_HANDS = 2
MODEL_COMPLEXITY = 0          # 0 is faster than 1/2.
BASE_PROCESS_SCALE = 0.70     # Landmarks are extracted on a smaller image.
LOW_FPS_THRESHOLD = 18.0
LOW_PROCESS_SCALE = 0.55


def calculate_angle_joint(a, b, c):
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def angle_between_vectors(v1, v2):
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def extract_features(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    middle_finger_tip = hand_landmarks.landmark[12]

    scale = math.sqrt(
        (middle_finger_tip.x - wrist.x) ** 2
        + (middle_finger_tip.y - wrist.y) ** 2
    )
    if scale == 0:
        scale = 1.0

    coords = []
    for lm in hand_landmarks.landmark:
        norm_x = (lm.x - wrist.x) / scale
        norm_y = (lm.y - wrist.y) / scale
        coords.append([norm_x, norm_y, lm.z])

    coords = np.array(coords, dtype=np.float32)
    features = coords.flatten().tolist()

    finger_landmarks = {
        "Thumb": [1, 2, 3, 4],
        "Index": [5, 6, 7, 8],
        "Middle": [9, 10, 11, 12],
        "Ring": [13, 14, 15, 16],
        "Pinky": [17, 18, 19, 20],
    }

    for points in finger_landmarks.values():
        a = coords[points[0]]
        b = coords[points[1]]
        c = coords[points[2]]
        features.append(calculate_angle_joint(a, b, c))

    finger_pairs = [
        ("Thumb", "Index"),
        ("Index", "Middle"),
        ("Middle", "Ring"),
        ("Ring", "Pinky"),
    ]

    for f1, f2 in finger_pairs:
        v1 = coords[finger_landmarks[f1][3]] - coords[finger_landmarks[f1][0]]
        v2 = coords[finger_landmarks[f2][3]] - coords[finger_landmarks[f2][0]]
        features.append(angle_between_vectors(v1, v2))

    return np.array(features, dtype=np.float32).reshape(1, -1)


def stable_vote(buffer):
    counts = Counter(buffer)
    return counts.most_common(1)[0][0]


def draw_system_hud(frame, fps, hands_count, process_scale):
    status = (
        f"REC | FPS {fps:0.1f} | Hands {hands_count} | Scale {process_scale:0.2f}"
    )
    # Changed HUD color to Red to indicate Recording
    cv2.rectangle(frame, (6, 6), (620, 34), (0, 0, 0), -1)
    cv2.circle(frame, (20, 20), 6, (0, 0, 255), -1) # Red recording dot
    cv2.putText(
        frame,
        status,
        (35, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )


def main():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as exc:
        print(f"Failed to load model from '{MODEL_PATH}': {exc}")
        return

    cap = cv2.VideoCapture('Input Video\WIN_20260224_01_27_14_Pro.mp4')
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam (index 0).")
        return

    # 1. Setup Camera Properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 2. Setup Video Writer (NEW CODE)
    # We must get the *actual* size the camera provided (it might reject 640x480)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = 20 # Fixed FPS for output video
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_out, (actual_width, actual_height))
    print(f"Recording to: {OUTPUT_VIDEO_PATH} ({actual_width}x{actual_height})")

    prediction_buffers = defaultdict(lambda: deque(maxlen=PREDICTION_WINDOW))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_NUM_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    fps_ema = 0.0
    prev_time = time.perf_counter()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Flip immediately (mirror view)
            frame = cv2.flip(frame, 1)

            # --- FPS Calculation ---
            current_time = time.perf_counter()
            dt = current_time - prev_time
            prev_time = current_time
            instant_fps = 1.0 / max(dt, 1e-6)
            fps_ema = instant_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * instant_fps)

            # --- Adaptive Scaling ---
            if fps_ema < LOW_FPS_THRESHOLD:
                detail_level = "low"
                process_scale = LOW_PROCESS_SCALE
            else:
                detail_level = "full"
                process_scale = BASE_PROCESS_SCALE

            # --- Processing ---
            process_w = max(1, int(frame.shape[1] * process_scale))
            process_h = max(1, int(frame.shape[0] * process_scale))
            small_frame = cv2.resize(frame, (process_w, process_h), interpolation=cv2.INTER_LINEAR)

            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            rgb_small.flags.writeable = False
            results = hands.process(rgb_small)

            if results.multi_hand_landmarks:
                handedness = results.multi_handedness or []

                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw Skeleton
                    draw_custom_skeleton(frame, hand_landmarks, detail_level="full")

                    # Extract & Predict
                    with open('feature_names.txt', 'r') as f :
                        feature_names = f.read().split(',')

                    input_data = extract_features(hand_landmarks)

                    input_data = pd.DataFrame(input_data, columns=feature_names) # Ensure correct feature order
                    
                    prediction = model.predict(input_data)[0]

                    # Handle Hand Label (Left/Right)
                    if idx < len(handedness):
                        hand_info = handedness[idx].classification[0]
                        hand_label = hand_info.label
                        hand_score = float(hand_info.score)
                    else:
                        hand_label = f"Hand{idx}"
                        hand_score = None

                    # Smoothing
                    label_for_buffer = str(prediction)
                    color = (230, 216, 173)
                    buffer_key = hand_label
                    prediction_buffers[buffer_key].append(label_for_buffer)
                    stable_prediction = stable_vote(prediction_buffers[buffer_key])

                    # Draw Box
                    hand_text = hand_label if hand_score is None else f"{hand_label} {int(hand_score * 100)}%"
                    box_label = f"{stable_prediction} | {hand_text}"
                    bbox = hand_bbox_from_landmarks(frame.shape, hand_landmarks)
                    draw_hand_box_with_label(frame, bbox, box_label, color)

                    y = 65 + idx * 28

                    cv2.putText(
                    frame, f"{hand_label.title()} Hand | {stable_prediction.title()}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
                )
                    
            else:
                cv2.putText(
                    frame, "Waiting for hand...", (12, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255 , 255), 2, cv2.LINE_AA
                )

            hands_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            draw_system_hud(frame, fps_ema, hands_count, process_scale)

            # 3. Write Frame to Video (NEW CODE)
            # We write the full 'frame' which now has all the drawings on it
            out.write(frame)

            # --- Display ---
            if frame.shape[1] != TARGET_WIDTH:
                aspect_ratio = frame.shape[0] / frame.shape[1]
                target_height = int(TARGET_WIDTH * aspect_ratio)
                display_frame = cv2.resize(frame, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)
            else:
                display_frame = frame

            cv2.imshow("Final Hand Gesture Project", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("Cleaning up...")
        hands.close()
        cap.release()
        out.release() # <--- Important: Save the video file properly
        cv2.destroyAllWindows()
        print(f"Video saved to {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()