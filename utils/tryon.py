import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_frame(frame, garment_name):

    garment_path = os.path.join("static","garments",garment_name)

    garment = cv2.imread(garment_path, cv2.IMREAD_UNCHANGED)

    if garment is None:
        return frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return frame

    h, w, _ = frame.shape

    left_shoulder = results.pose_landmarks.landmark[11]
    right_shoulder = results.pose_landmarks.landmark[12]

    x1 = int(left_shoulder.x * w)
    y1 = int(left_shoulder.y * h)

    x2 = int(right_shoulder.x * w)
    y2 = int(right_shoulder.y * h)

    width = abs(x2 - x1) + 120
    height = int(width * garment.shape[0] / garment.shape[1])

    garment_resized = cv2.resize(garment, (width, height))

    x = min(x1,x2) - 60
    y = min(y1,y2)

    gh, gw = garment_resized.shape[:2]

    if y+gh > h or x+gw > w or x<0 or y<0:
        return frame

    if garment_resized.shape[2] == 4:

        alpha = garment_resized[:,:,3] / 255.0

        for c in range(3):
            frame[y:y+gh, x:x+gw, c] = (
                alpha * garment_resized[:,:,c] +
                (1-alpha) * frame[y:y+gh, x:x+gw, c]
            )

    else:
        frame[y:y+gh, x:x+gw] = garment_resized

    return frame