import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, Label
import pyttsx3  # Per il feedback vocale
import matplotlib.pyplot as plt  # Per il grafico delle ripetizioni/angoli
from fpdf import FPDF  # Per creare report PDF

# Inizializza MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Inizializza il motore di sintesi vocale
engine = pyttsx3.init()

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
    generate_report(data)  # Crea anche il report PDF

# Funzione per creare un report in PDF
def generate_report(data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(200, 10, txt="Palestra AI - Report di Allenamento", ln=True, align='C')
    pdf.ln(10)

    # Aggiungi dati delle ripetizioni e angoli
    pdf.set_font('Arial', '', 10)
    for row in data:
        pdf.cell(0, 10, f"{row[0]} - Ripetizioni: {row[1]} - Angolo: {row[2]}", ln=True)

    # Salva il report PDF
    report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(report_filename)
    print(f"Report salvato come PDF: {report_filename}")

# Funzione per il feedback vocale
def give_feedback(message):
    engine.say(message)
    engine.runAndWait()

# Funzione per elaborare ogni frame e rilevare la posa
def process_frame(frame, rep_count, data, exercise_type, angle_history, rep_time):
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

        # Aggiungi l'angolo alla cronologia
        angle_history.append(back_thigh_angle)

        # Rilevamento delle ripetizioni
        if back_thigh_angle < 70:
            rep_count += 1
            rep_time.append(datetime.now())  # Aggiungi il tempo della ripetizione
            cv2.putText(frame, f"Ripetizioni: {rep_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Feedback vocale per una ripetizione corretta
            give_feedback(f"Ottimo! Ripetizione completata. Esercizio: {exercise_type}")

        # Controllo della posizione corretta
        if back_thigh_angle > 170:
            cv2.putText(frame, "Posizione non corretta!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            give_feedback("Posizione non corretta, abbassa di più le gambe!")

        # Salva dati per ogni ripetizione
        data.append([datetime.now().strftime('%H:%M:%S'), rep_count, back_thigh_angle])

    return frame, rep_count, data, angle_history, rep_time

# Funzione per analizzare il video
def analyze_video(video_path, exercise_type):
    cap = cv2.VideoCapture(video_path)
    rep_count = 0
    data = []
    angle_history = []
    rep_time = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, rep_count, data, angle_history, rep_time = process_frame(frame, rep_count, data, exercise_type, angle_history, rep_time)

        # Mostra il frame elaborato
        cv2.imshow("Esercizio", frame)

        # Premi 'q' per uscire dal video
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Salva i dati al termine dell'analisi
    save_data(data)

    # Mostra il grafico delle ripetizioni nel tempo
    timestamps = [(time - rep_time[0]).total_seconds() for time in rep_time]
    plt.plot(timestamps, range(1, len(rep_time) + 1), label='Ripetizioni')
    plt.xlabel('Tempo (secondi)')
    plt.ylabel('Numero di ripetizioni')
    plt.title(f"Progresso - Esercizio {exercise_type}")
    plt.legend()
    plt.show()

# Funzione per caricare e avviare il video
def load_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if video_path:
        exercise_type = exercise_var.get()
        analyze_video(video_path, exercise_type)

# Funzione per attivare la fotocamera in tempo reale
def start_camera():
    cap = cv2.VideoCapture(0)
    rep_count = 0
    data = []
    angle_history = []
    rep_time = []

    # Aggiungi il tasto "Fine Allenamento"
    def stop_training():
        cap.release()
        cv2.destroyAllWindows()
        save_data(data)

        # Mostra il grafico delle ripetizioni nel tempo
        timestamps = [(time - rep_time[0]).total_seconds() for time in rep_time]
        plt.plot(timestamps, range(1, len(rep_time) + 1), label='Ripetizioni')
        plt.xlabel('Tempo (secondi)')
        plt.ylabel('Numero di ripetizioni')
        plt.title(f"Progresso - Esercizio {exercise_var.get()}")
        plt.legend()
        plt.show()

        # Disabilita il tasto Fine Allenamento
        stop_button.pack_forget()

    # Creiamo un bottone per fermare l'allenamento
    stop_button = tk.Button(root, text="Fine Allenamento", command=stop_training)
    stop_button.pack(pady=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, rep_count, data, angle_history, rep_time = process_frame(frame, rep_count, data, exercise_var.get(), angle_history, rep_time)

        # Mostra il frame della webcam
        cv2.imshow("Allenamento in corso", frame)

        # Se l'utente preme 'q' esci
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Configura l'interfaccia grafica
root = tk.Tk()
root.title("Palestra AI - Analisi Esercizi")
root.geometry("400x350")

# Etichette e pulsante per caricare il video
label = Label(root, text="Benvenuto in Palestra AI! Inserisci i tuoi dati biometrici e scegli l'esercizio")
label.pack(pady=10)

# Dati biometrici
height_label = Label(root, text="Altezza (cm):")
height_label.pack()
height_entry = tk.Entry(root)
height_entry.pack(pady=5)

weight_label = Label(root, text="Peso (kg):")
weight_label.pack()
weight_entry = tk.Entry(root)
weight_entry.pack(pady=5)

femur_length_label = Label(root, text="Lunghezza femore (cm):")
femur_length_label.pack()
femur_length_entry = tk.Entry(root)
femur_length_entry.pack(pady=5)

tibia_length_label = Label(root, text="Lunghezza tibia (cm):")
tibia_length_label.pack()
tibia_length_entry = tk.Entry(root)
tibia_length_entry.pack(pady=5)

dob_label = Label(root, text="Data di Nascita (DD/MM/YYYY):")
dob_label.pack()
dob_entry = tk.Entry(root)
dob_entry.pack(pady=5)

# Tipo di esercizio
exercise_label = Label(root, text="Seleziona esercizio:")
exercise_label.pack()

exercise_var = tk.StringVar(value="Squat")
exercise_options = ["Squat", "Affondi", "Push-up"]
exercise_menu = tk.OptionMenu(root, exercise_var, *exercise_options)
exercise_menu.pack(pady=10)

# Pulsante per avviare la webcam
start_button = tk.Button(root, text="Avvia Webcam", command=start_camera)
start_button.pack(pady=10)

# Pulsante per caricare un video
load_button = tk.Button(root, text="Carica Video", command=load_video)
load_button.pack(pady=10)

# Avvia l'interfaccia
root.mainloop()
