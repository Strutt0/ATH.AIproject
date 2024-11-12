import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import sqlite3
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, Label
import matplotlib.pyplot as plt

# Impostazioni e configurazione del database
DATABASE = "palestra_ai.db"

# Inizializza MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Crea il database se non esiste
def create_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE,
                        password TEXT,
                        name TEXT,
                        surname TEXT,
                        dob TEXT,
                        height REAL,
                        weight REAL,
                        femur_length REAL,
                        tibia_length REAL
                    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        timestamp TEXT,
                        repetitions INTEGER,
                        file_path TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )''')
    conn.commit()
    conn.close()

create_database()

# Funzioni di autenticazione e registrazione
def register_user(email, password, name, surname, dob, height, weight, femur_length, tibia_length):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, password, name, surname, dob, height, weight, femur_length, tibia_length) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (email, password, name, surname, dob, height, weight, femur_length, tibia_length))
        conn.commit()
        messagebox.showinfo("Registrazione", "Registrazione avvenuta con successo!")
    except sqlite3.IntegrityError:
        messagebox.showerror("Errore", "Email già registrata!")
    conn.close()

def login_user(email, password):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    user = cursor.fetchone()
    conn.close()
    return user

# Funzione per calcolare l'angolo tra tre punti
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Funzione per salvare report in CSV
def save_report(user_id, repetitions, data):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"analysis_{timestamp}.csv"
    filepath = os.path.join(os.getcwd(), filename)
    pd.DataFrame(data, columns=["Timestamp", "Repetitions", "Back-Thigh Angle"]).to_csv(filepath, index=False)
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO reports (user_id, timestamp, repetitions, file_path) VALUES (?, ?, ?, ?)",
                   (user_id, timestamp, repetitions, filepath))
    conn.commit()
    conn.close()
    print(f"File salvato: {filepath}")

# Funzione per visualizzare i report
def show_reports(user_id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, repetitions, file_path FROM reports WHERE user_id = ?", (user_id,))
    reports = cursor.fetchall()
    conn.close()
    report_text = "\n".join([f"{timestamp} - {repetitions} ripetizioni - {file_path}" for timestamp, repetitions, file_path in reports])
    messagebox.showinfo("Report", report_text or "Nessun report disponibile")

# Funzione per processare i frame con feedback in tempo reale
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

        back_thigh_angle = calculate_angle(shoulder, hip, knee)
        cv2.putText(frame, f"Angolo: {int(back_thigh_angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if back_thigh_angle < 70:
            rep_count += 1
            cv2.putText(frame, f"Ripetizioni: {rep_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        data.append([datetime.now().strftime('%H:%M:%S'), rep_count, back_thigh_angle])

    return frame, rep_count, data

# Funzione per avviare l'allenamento in tempo reale con webcam
def start_realtime_analysis(user_id):
    cap = cv2.VideoCapture(0)
    rep_count = 0
    data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, rep_count, data = process_frame(frame, rep_count, data)
        cv2.imshow("Allenamento in Tempo Reale", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_report(user_id, rep_count, data)

# Classe principale dell'interfaccia grafica
class PalestraAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Palestra AI - Accesso")
        self.root.geometry("400x400")
        self.user = None

        self.show_login_screen()

    def show_login_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        Label(self.root, text="Email").pack(pady=5)
        self.email_entry = tk.Entry(self.root)
        self.email_entry.pack()

        Label(self.root, text="Password").pack(pady=5)
        self.password_entry = tk.Entry(self.root, show="*")
        self.password_entry.pack()

        tk.Button(self.root, text="Login", command=self.login).pack(pady=20)
        tk.Button(self.root, text="Registrati", command=self.show_register_screen).pack(pady=5)

    def show_register_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        Label(self.root, text="Nome").pack(pady=5)
        self.name_entry = tk.Entry(self.root)
        self.name_entry.pack()

        Label(self.root, text="Cognome").pack(pady=5)
        self.surname_entry = tk.Entry(self.root)
        self.surname_entry.pack()

        Label(self.root, text="Email").pack(pady=5)
        self.email_entry = tk.Entry(self.root)
        self.email_entry.pack()

        Label(self.root, text="Password").pack(pady=5)
        self.password_entry = tk.Entry(self.root, show="*")
        self.password_entry.pack()

        Label(self.root, text="Data di nascita (DD/MM/YYYY)").pack(pady=5)
        self.dob_entry = tk.Entry(self.root)
        self.dob_entry.pack()

        Label(self.root, text="Altezza (cm)").pack(pady=5)
        self.height_entry = tk.Entry(self.root)
        self.height_entry.pack()

        Label(self.root, text="Peso (kg)").pack(pady=5)
        self.weight_entry = tk.Entry(self.root)
        self.weight_entry.pack()

        Label(self.root, text="Lunghezza femore (cm)").pack(pady=5)
        self.femur_length_entry = tk.Entry(self.root)
        self.femur_length_entry.pack()

        Label(self.root, text="Lunghezza tibia (cm)").pack(pady=5)
        self.tibia_length_entry = tk.Entry(self.root)
        self.tibia_length_entry.pack()

        tk.Button(self.root, text="Registrati", command=self.register).pack(pady=20)
        tk.Button(self.root, text="Torna al Login", command=self.show_login_screen).pack()

    def login(self):
        email = self.email_entry.get()
        password = self.password_entry.get()
        self.user = login_user(email, password)
        if self.user:
            self.show_main_screen()
        else:
            messagebox.showerror("Errore", "Credenziali non valide.")

    def register(self):
        email = self.email_entry.get()
        password = self.password_entry.get()
        name = self.name_entry.get()
        surname = self.surname_entry.get()
        dob = self.dob_entry.get()
        height = float(self.height_entry.get())
        weight = float(self.weight_entry.get())
        femur_length = float(self.femur_length_entry.get())
        tibia_length = float(self.tibia_length_entry.get())

        register_user(email, password, name, surname, dob, height, weight, femur_length, tibia_length)
        self.show_login_screen()

    def show_main_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text=f"Benvenuto {self.user[3]}").pack(pady=10)
        tk.Button(self.root, text="Inizia Allenamento", command=lambda: start_realtime_analysis(self.user[0])).pack(pady=20)
        tk.Button(self.root, text="Visualizza Report", command=lambda: show_reports(self.user[0])).pack(pady=20)
        tk.Button(self.root, text="Logout", command=self.show_login_screen).pack(pady=20)

# Avvio dell'applicazione
if __name__ == "__main__":
    root = tk.Tk()
    app = PalestraAIApp(root)
    root.mainloop()
