import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, Label

# Inizializza MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Funzione per calcolare l'angolo tra tre punti
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Funzione per salvare i dati in CSV
def save_data(data):
    filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(os.getcwd(), filename)
    df = pd.DataFrame(data, columns=["Timestamp", "Repetitions", "Back-Thigh Angle"])
    df.to_csv(filepath, index=False)
    print(f"File salvato: {filepath}")

# Funzione per elaborare ogni frame e rilevare la posa
def process_frame(frame, rep_count, data):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    back_thigh_angle = None

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]

        # Calcolo angolo tra schiena e femore
        back_thigh_angle = calculate_angle(shoulder, hip, knee)

        # Mostra angolo sul video
        cv2.putText(frame, f"Angolo: {int(back_thigh_angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Controllo ripetizioni in base all'angolo
        if back_thigh_angle < 70:
            rep_count += 1
            cv2.putText(frame, f"Ripetizioni: {rep_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Salva dati per ogni ripetizione
        data.append([datetime.now().strftime('%H:%M:%S'), rep_count, back_thigh_angle])

    return frame, rep_count, data

# Funzione principale per analizzare il video
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    rep_count = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, rep_count, data = process_frame(frame, rep_count, data)

        # Mostra il frame elaborato
        cv2.imshow("Esercizio", frame)

        # Premi 'q' per uscire dal video
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Salva i dati al termine dell'analisi
    save_data(data)

# Funzione per caricare e avviare il video
def load_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if video_path:
        analyze_video(video_path)

# Configura l'interfaccia grafica
root = tk.Tk()
root.title("Palestra AI - Analisi Esercizi")
root.geometry("300x150")

# Etichette e pulsante per caricare il video
label = Label(root, text="Benvenuto in Palestra AI!")
label.pack(pady=10)
button = tk.Button(root, text="Carica Video", command=load_video)
button.pack(pady=20)

# Avvia l'interfaccia
root.mainloop()
