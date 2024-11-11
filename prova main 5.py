import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Inizializza MediaPipe Pose per il rilevamento della posa
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Variabile per conteggio delle ripetizioni e dati raccolti
rep_count = 0
knee_bent = False
data_records = []  # Lista per salvare i dati delle ripetizioni e angoli

# Funzione per calcolare l'angolo tra tre punti
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Funzione per elaborare ogni frame
def process_frame(frame):
    global rep_count, knee_bent
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        # Calcola angoli specifici
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]

        # Calcola l'angolo del ginocchio e tra femore e schiena
        knee_angle = calculate_angle(hip, knee, ankle)
        back_thigh_angle = calculate_angle(shoulder, hip, knee)

        # Conteggio delle ripetizioni basato sull'angolo del ginocchio
        if knee_angle < 90 and not knee_bent:
            knee_bent = True
        elif knee_angle > 110 and knee_bent:
            rep_count += 1
            knee_bent = False
            # Aggiungi il record della ripetizione
            data_records.append({
                "Ripetizione": rep_count,
                "Angolo_Ginocchio": knee_angle,
                "Angolo_Schiena_Femore": back_thigh_angle
            })

        # Visualizzazione delle informazioni
        cv2.putText(frame, f"Ripetizioni: {rep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Angolo Ginocchio: {int(knee_angle)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Angolo Schiena-Femore: {int(back_thigh_angle)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return frame

# Funzione per analizzare il video
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Processa ogni frame
        frame = process_frame(frame)

        # Mostra il frame elaborato
        cv2.imshow("Esercizio", frame)

        # Premi 'q' per uscire dal video
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Funzione per salvare i dati raccolti in un CSV
def save_data_to_csv():
    if data_records:
        df = pd.DataFrame(data_records)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"allenamento_{timestamp}.csv"
        os.makedirs("dati_esercizi", exist_ok=True)
        df.to_csv(os.path.join("dati_esercizi", filename), index=False)
        print(f"Dati salvati in '{filename}'.")

# Esegui l'analisi e salva i dati al termine
video_path = "Videos/Squat Riky.mp4 "
analyze_video(video_path)
save_data_to_csv()
