import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sqlite3
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib import pyplot as plt

# Costante per il database
DATABASE = "palestra_ai.db"

# Creazione del database e della tabella utenti
def initialize_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE,
                        password TEXT,
                        name TEXT,
                        surname TEXT,
                        dob TEXT,
                        height INTEGER,
                        weight INTEGER,
                        femur_length INTEGER,
                        tibia_length INTEGER
                      )''')
    conn.commit()
    conn.close()

# Funzione per effettuare il login
def login_user(email, password):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    user = cursor.fetchone()
    conn.close()
    return user

# Funzione per registrare un nuovo utente
def register_user(email, password, name, surname, dob, height, weight, femur_length, tibia_length):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    try:
        cursor.execute('''INSERT INTO users (email, password, name, surname, dob, height, weight, femur_length, tibia_length)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (email, password, name, surname, dob, height, weight, femur_length, tibia_length))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

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

# Funzione per aprire la webcam e ottenere feedback in tempo reale
def analyze_realtime():
    cap = cv2.VideoCapture(0)
    rep_count = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, rep_count, data = process_frame(frame, rep_count, data)

        # Mostra il frame elaborato
        cv2.imshow("Esercizio in tempo reale", frame)

        # Premi 'q' per uscire dal video
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Salva i dati al termine dell'analisi
    save_data(data)

# Inizializza MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Interfaccia utente principale
class PalestraAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Palestra AI")
        self.root.geometry("400x300")

        # Schermata di login
        self.show_login_screen()

    def show_login_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Login", font=("Helvetica", 16)).pack(pady=10)
        tk.Label(self.root, text="Email").pack()
        self.email_entry = tk.Entry(self.root)
        self.email_entry.pack()

        tk.Label(self.root, text="Password").pack()
        self.password_entry = tk.Entry(self.root, show="*")
        self.password_entry.pack()

        tk.Button(self.root, text="Login", command=self.login).pack(pady=10)
        tk.Button(self.root, text="Registrati", command=self.show_register_screen).pack()

    def show_register_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Registrazione", font=("Helvetica", 16)).pack(pady=10)
        tk.Label(self.root, text="Nome").pack()
        self.name_entry = tk.Entry(self.root)
        self.name_entry.pack()

        tk.Label(self.root, text="Cognome").pack()
        self.surname_entry = tk.Entry(self.root)
        self.surname_entry.pack()

        tk.Label(self.root, text="Email").pack()
        self.email_entry = tk.Entry(self.root)
        self.email_entry.pack()

        tk.Label(self.root, text="Password").pack()
        self.password_entry = tk.Entry(self.root, show="*")
        self.password_entry.pack()

        tk.Label(self.root, text="Data di nascita (AAAA-MM-GG)").pack()
        self.dob_entry = tk.Entry(self.root)
        self.dob_entry.pack()

        tk.Button(self.root, text="Registrati", command=self.register).pack(pady=10)
        tk.Button(self.root, text="Indietro", command=self.show_login_screen).pack()

    def register(self):
        email = self.email_entry.get()
        password = self.password_entry.get()
        name = self.name_entry.get()
        surname = self.surname_entry.get()
        dob = self.dob_entry.get()
        height, weight, femur_length, tibia_length = 170, 70, 40, 35  # Valori di esempio

        if register_user(email, password, name, surname, dob, height, weight, femur_length, tibia_length):
            messagebox.showinfo("Registrazione completata", "Utente registrato con successo!")
            self.show_login_screen()
        else:
            messagebox.showerror("Errore", "Email già registrata.")

    def login(self):
        email = self.email_entry.get()
        password = self.password_entry.get()
        user = login_user(email, password)
        if user:
            messagebox.showinfo("Login effettuato", "Benvenuto!")
            self.show_main_screen()
        else:
            messagebox.showerror("Errore", "Credenziali non valide.")

    def show_main_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Palestra AI", font=("Helvetica", 16)).pack(pady=10)
        tk.Button(self.root, text="Analisi Video", command=self.load_video).pack(pady=5)
        tk.Button(self.root, text="Analisi in Tempo Reale", command=analyze_realtime).pack(pady=5)
        tk.Button(self.root, text="Esci", command=self.root.quit).pack(pady=10)

    def load_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if video_path:
            analyze_video(video_path)

# Avvio del programma
initialize_db()
root = tk.Tk()
app = PalestraAIApp(root)
root.mainloop()
