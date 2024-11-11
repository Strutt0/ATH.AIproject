import cv2
import mediapipe as mp
import numpy as np

# Inizializza MediaPipe per il rilevamento della posa
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


# Funzione per calcolare l'angolo tra tre punti
def calculate_angle(a, b, c):
    a = np.array(a)  # Primo punto
    b = np.array(b)  # Punto centrale
    c = np.array(c)  # Terzo punto

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Caricamento del video e analisi delle pose
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    rep_count = 0  # Contatore delle ripetizioni
    knee_bent = False  # Stato iniziale del ginocchio

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converti il frame in RGB per MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        # Se vengono rilevati i punti di posa, esegui calcoli
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark

            # Punti per calcolare angolo ginocchio
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]

            # Calcolo dell'angolo del ginocchio
            knee_angle = calculate_angle(hip, knee, ankle)

            # Calcolo angolo tra schiena e femore (esempio semplice)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            back_thigh_angle = calculate_angle(shoulder, hip, knee)

            # Condizioni di feedback e conteggio ripetizioni
            feedback = ""
            if knee_angle < 70:
                feedback = "Piegamento ginocchio troppo stretto"
            elif knee_angle > 160:
                feedback = "Piegamento ginocchio troppo largo"
            else:
                feedback = "Piegamento corretto"

            # Aggiorna contatore ripetizioni in base all'angolo del ginocchio
            if knee_angle < 90 and not knee_bent:
                knee_bent = True
            elif knee_angle > 110 and knee_bent:
                rep_count += 1
                knee_bent = False

            # Mostra feedback e ripetizioni sullo schermo
            cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Ripetizioni: {rep_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.putText(frame, f"Angolo schiena-femore: {int(back_thigh_angle)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2, cv2.LINE_AA)

        # Mostra il frame elaborato
        cv2.imshow("Esercizio", frame)

        # Premi 'q' per uscire dal video
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Chiudi il video e la finestra
    cap.release()
    cv2.destroyAllWindows()


# Path al video di esempio
video_path = "Videos/Squat Riky.mp4"  # Modifica con il tuo percorso video
analyze_video(video_path)
