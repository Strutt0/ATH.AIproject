import cv2
import mediapipe as mp
import numpy as np
import os

# Inizializza MediaPipe Pose per il rilevamento della posa
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Funzione per calcolare l'angolo tra tre punti (utile per valutare posture)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Funzione per elaborare ogni frame e rilevare la posa
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

        angle = calculate_angle(hip, knee, ankle)

        cv2.putText(frame, str(int(angle)),
                    tuple(np.multiply(knee, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )
        return frame, angle
    return frame, None

# Caricamento del video e elaborazione dei frame
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    rep_count = 0
    knee_bent = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, angle = process_frame(frame)

        if angle is not None:
            if angle < 70:
                feedback = "Piegamento ginocchio troppo stretto"
            elif angle > 160:
                feedback = "Piegamento ginocchio troppo largo"
            else:
                feedback = "Piegamento corretto"
            cv2.putText(frame, feedback, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if angle < 70 and not knee_bent:
                knee_bent = True
            elif angle > 160 and knee_bent:
                rep_count += 1
                knee_bent = False
                cv2.putText(frame, f"Ripetizioni: {rep_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Esercizio", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Esecuzione dello script
if __name__ == "__main__":
    video_path = "Videos/Squat Riky.mp4"
    if not os.path.exists(video_path):
        print("Errore: Video non trovato. ")
    else:
        analyze_video(video_path)
