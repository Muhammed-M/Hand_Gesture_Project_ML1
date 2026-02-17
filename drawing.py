# File: drawing.py
import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands

# Precompute structures once to reduce per-frame overhead.
FULL_CONNECTIONS = tuple(mp_hands.HAND_CONNECTIONS)
PALM_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (17, 0),
)

SKELETON_COLOR = (20, 20, 20)
JOINT_COLOR = (40, 40, 255)


def _style_for_detail(detail_level):
    if detail_level == "low":
        return {
            "connections": PALM_CONNECTIONS,
            "line_thickness": 1,
            "joint_radius": 2,
        }

    return {
        "connections": FULL_CONNECTIONS,
        "line_thickness": 3,
        "joint_radius": 5,
    }


def draw_custom_skeleton(image, hand_landmarks, detail_level="full"):
    """
    Draws a custom colorful skeleton on the image.
    Args:
        image: The image frame to draw on.
        hand_landmarks: The list of 21 landmarks from MediaPipe.
    """
    h, w, _ = image.shape
    style = _style_for_detail(detail_level)

    # 1. Draw Connections (The "Bones") -> Green Lines
    # We draw these FIRST so they look like they are behind the joints
    for start_idx, end_idx in style["connections"]:
        start_point = hand_landmarks.landmark[start_idx]
        end_point = hand_landmarks.landmark[end_idx]

        start_x, start_y = int(start_point.x * w), int(start_point.y * h)
        end_x, end_y = int(end_point.x * w), int(end_point.y * h)

        cv2.line(
            image,
            (start_x, start_y),
            (end_x, end_y),
            (20, 20, 20),
            style["line_thickness"],
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            (start_x, start_y),
            (end_x, end_y),
            SKELETON_COLOR,
            style["line_thickness"],
            cv2.LINE_AA,
        )

    # 2. Draw Landmarks (The "Joints") -> Custom Colors
    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        radius = style["joint_radius"]
        cv2.circle(image, (cx, cy), radius + 2, (20, 20, 20), cv2.FILLED, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), radius, JOINT_COLOR, cv2.FILLED, cv2.LINE_AA)


def hand_bbox_from_landmarks(frame_shape, hand_landmarks, pad=14):
    h, w, _ = frame_shape
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]

    left = max(0, min(xs) - pad)
    top = max(0, min(ys) - pad)
    right = min(w - 1, max(xs) + pad)
    bottom = min(h - 1, max(ys) + pad)
    return left, top, right, bottom


def draw_hand_box_with_label(image, bbox, label_text, color):
    left, top, right, bottom = bbox
    cv2.rectangle(image, (left, top), (right, bottom), color, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    (label_w, label_h), baseline = cv2.getTextSize(label_text, font, scale, thickness)
    pad_x = 8
    pad_y = 5
    tag_w = label_w + (2 * pad_x)
    tag_h = label_h + baseline + (2 * pad_y)

    tag_left = left
    tag_top = max(0, top - tag_h)
    if tag_left + tag_w >= image.shape[1]:
        tag_left = max(0, image.shape[1] - tag_w - 1)

    tag_right = tag_left + tag_w
    tag_bottom = tag_top + tag_h

    cv2.rectangle(image, (tag_left, tag_top), (tag_right, tag_bottom), color, cv2.FILLED, cv2.LINE_AA)
    cv2.putText(
        image,
        label_text,
        (tag_left + pad_x, tag_bottom - baseline - pad_y),
        font,
        scale,
        (10, 10, 10),
        thickness,
        cv2.LINE_AA,
    )
