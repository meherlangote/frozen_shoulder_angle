# utils/pose_utils.py
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import io

mp_pose = mp.solutions.pose

# MediaPipe landmark indices for convenience
LANDMARKS = mp_pose.PoseLandmark

def detect_pose_landmarks(image_rgb, min_detection_confidence=0.5):
    """
    Runs MediaPipe Pose on RGB image.
    Returns results object from MediaPipe.
    """
    with mp_pose.Pose(static_image_mode=True,
                      min_detection_confidence=min_detection_confidence,
                      model_complexity=1) as pose:
        results = pose.process(image_rgb)
    return results

def landmark_to_pixel(landmark, image_width, image_height):
    return int(landmark.x * image_width), int(landmark.y * image_height)

def calculate_angle(a, b, c):
    """
    Calculate angle ABC (in degrees) where B is the vertex.
    That is, angle between vectors BA and BC.
    a, b, c are (x,y) tuples or numpy arrays.
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    BA = a - b
    BC = c - b
    # handle zero-length vectors
    denom = (np.linalg.norm(BA) * np.linalg.norm(BC))
    if denom == 0:
        return None
    cos_angle = np.dot(BA, BC) / denom
    # numerical stability
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return float(angle)

def compute_shoulder_angle(results, image_width, image_height, side='left'):
    """
    Given MediaPipe results, compute shoulder angle for 'left' or 'right'.
    Returns:
      angle (float or None), landmark_positions (dict)
    landmark_positions contains pixel coords for shoulder, elbow, hip used.
    """
    if not results.pose_landmarks:
        return None, {}

    lm = results.pose_landmarks.landmark
    if side == 'left':
        shoulder = lm[LANDMARKS.LEFT_SHOULDER]
        elbow = lm[LANDMARKS.LEFT_ELBOW]
        hip = lm[LANDMARKS.LEFT_HIP]
    else:
        shoulder = lm[LANDMARKS.RIGHT_SHOULDER]
        elbow = lm[LANDMARKS.RIGHT_ELBOW]
        hip = lm[LANDMARKS.RIGHT_HIP]

    s_px = landmark_to_pixel(shoulder, image_width, image_height)
    e_px = landmark_to_pixel(elbow, image_width, image_height)
    h_px = landmark_to_pixel(hip, image_width, image_height)

    angle = calculate_angle(e_px, s_px, h_px)  # angle at shoulder between elbow and hip
    return angle, {'shoulder': s_px, 'elbow': e_px, 'hip': h_px}

def annotate_image_pil(image_bgr, results, angle, landmark_positions, side='left'):
    """
    Annotate BGR image (numpy array) with landmarks, skeleton lines and angle text.
    Returns annotated PIL Image.
    """
    # convert to RGB for PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_im)
    w, h = pil_im.size

    # draw simple skeleton (shoulder-elbow, shoulder-hip)
    s = landmark_positions.get('shoulder')
    e = landmark_positions.get('elbow')
    hi = landmark_positions.get('hip')

    # white lines for skeleton
    if s and e:
        draw.line([s, e], width=4)
    if s and hi:
        draw.line([s, hi], width=4)

    # draw circles for points
    r = 6
    if s:
        draw.ellipse([s[0]-r, s[1]-r, s[0]+r, s[1]+r], fill=(255,0,0))
    if e:
        draw.ellipse([e[0]-r, e[1]-r, e[0]+r, e[1]+r], fill=(0,255,0))
    if hi:
        draw.ellipse([hi[0]-r, hi[1]-r, hi[0]+r, hi[1]+r], fill=(0,0,255))

    # draw text for angle
    angle_text = "Angle: N/A" if angle is None else f"Angle: {angle:.1f}°"
    # choose position near shoulder
    text_pos = (s[0] + 10, s[1] - 30) if s else (10, 10)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # black outline for readability
    x,y = text_pos
    outline_color = (0,0,0)
    for ox, oy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        draw.text((x+ox, y+oy), angle_text, font=font, fill=outline_color)
    draw.text(text_pos, angle_text, font=font, fill=(255,255,255))

    return pil_im

def pil_image_to_bytes(pil_img, img_format='PNG'):
    buf = io.BytesIO()
    pil_img.save(buf, format=img_format)
    buf.seek(0)
    return buf.read()
