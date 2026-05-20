import cv2
import time
import numpy as np
from collections import deque

from mediapipe_module import FaceAnalyzer
from ICA_module import extract_hr_from_rgb


ICA_BUFFER_TIME_LENGHT = 15   # in secondi: era 10
ICA_STARTUP_TIME_LENGHT = 10  # in secondi: era 5
ICA_UPDATE_PERIOD = 5         # in secondi: era 2

def run_dms():
    analyzer = FaceAnalyzer()
    cap = cv2.VideoCapture(0)
    fps = 30.0

    # Variabili Buffer ICA (Battito)
    BUFFER_SIZE = int(fps * ICA_BUFFER_TIME_LENGHT)
    r_buffer = deque(maxlen=BUFFER_SIZE)
    g_buffer = deque(maxlen=BUFFER_SIZE)
    b_buffer = deque(maxlen=BUFFER_SIZE)
    current_bpm = 0
    last_bpm_calc_time = time.time()

    # Variabili di Stato DMS (Timers)
    distraction_history = deque()  # Finestra 30s per Short Distraction
    continuous_distraction_time = 0.0
    focus_time = 0.0

    continuous_closed_time = 0.0
    eyes_open_time = 0.0

    active_warning = None  # "Sleep", "Microsleep", "Distracted (long)", "Distracted (short)"
    warning_detail = ""  # Per aggiungere info tipo "- Owl"

    # ==========================================
    # SOGLIE DI CALIBRAZIONE
    # ==========================================
    OWL_RADIUS = 0.13 #0.12 # Tolleranza distanza naso dal centro (0.5, 0.5)
    LIZARD_RADIUS = 0.17  # Tolleranza distanza iride dal centro occhio (0, 0)
    EAR_THRESHOLD = 0.22  # Sotto questo valore gli occhi sono considerati chiusi

    prev_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        current_time = time.time()
        delta_t = current_time - prev_time
        prev_time = current_time

        # FIX BUG INITIALIZATION: Evita salti di tempo anomali dovuti a lag di sistema
        if delta_t > 0.5: delta_t = 0.033

        face_data = analyzer.process_frame(frame, draw=True)
        frame_distraction = None

        if face_data["face_found"]:
            # 1. ICA Update (BPM)
            # Disegna le 3 zone di analisi (Fronte + Zigomi) ---
            if "roi_viz" in face_data:
                for (pt1, pt2) in face_data["roi_viz"]:
                    cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 1)

            if face_data["roi_rgb"] is not None:
                r_val, g_val, b_val = face_data["roi_rgb"]
                r_buffer.append(r_val)
                g_buffer.append(g_val)
                b_buffer.append(b_val)

                if (current_time - last_bpm_calc_time > ICA_UPDATE_PERIOD) and len(r_buffer) > (fps * ICA_STARTUP_TIME_LENGHT):
                    try:
                        bpm = extract_hr_from_rgb(np.array(r_buffer), np.array(g_buffer), np.array(b_buffer), fps)
                        if bpm > 0:
                            if current_bpm == 0:
                                current_bpm = int(bpm)  # Prima lettura
                            else:
                                # Filtro Esponenziale (EMA) per ignorare gli sbalzi anomali di FastICA
                                current_bpm = int(0.3 * bpm + 0.7 * current_bpm)
                    except:
                        pass
                    last_bpm_calc_time = current_time

            # 2. Controllo Sonno (EAR)
            ear = face_data["ear"]
            if ear is not None and ear < EAR_THRESHOLD:
                continuous_closed_time += delta_t
                eyes_open_time = 0.0
            else:
                eyes_open_time += delta_t
                continuous_closed_time = 0.0

            # 3. Controllo Sguardo (Owl e Lizard) tramite circonferenza Euclidea (in PIXEL)
            nx, ny = face_data["nose_x"], face_data["nose_y"]
            ix, iy = face_data["iris_offset_x"], face_data["iris_offset_y"]

            if nx is not None and ny is not None:
                # Conversione in pixel per l'Owl per evitare lo schiacciamento dell'aspect ratio
                h, w = frame.shape[:2]
                nx_px, ny_px = nx * w, ny * h
                cx_px, cy_px = w / 2.0, h / 2.0

                owl_dist_px = np.hypot(nx_px - cx_px, ny_px - cy_px)
                owl_radius_px = OWL_RADIUS * w

                # Lizard rimane normalizzato perché si basa sulle proporzioni dell'occhio
                lizard_dist = np.hypot(ix, iy) if (ix is not None and iy is not None) else 0

                if owl_dist_px > owl_radius_px:
                    frame_distraction = "Owl"
                elif lizard_dist > LIZARD_RADIUS:
                    frame_distraction = "Lizard"

            # Disegno cerchio di calibrazione Owl al centro dello schermo (basato sui pixel)
            cv2.circle(frame, (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
                       int(OWL_RADIUS * frame.shape[1]), (255, 255, 255), 1)

        # ==========================================
        # AGGIORNAMENTO STORICO DISTRAZIONI
        # ==========================================
        if frame_distraction:
            distraction_history.append((current_time, delta_t, frame_distraction))
            continuous_distraction_time += delta_t
            focus_time = 0.0
        else:
            focus_time += delta_t
            continuous_distraction_time = 0.0

        # Pulizia finestra 30s
        while distraction_history and (current_time - distraction_history[0][0] > 30.0):
            distraction_history.popleft()
        cumulative_time = sum(dur for _, dur, _ in distraction_history)

        # ==========================================
        # MACCHINA A STATI E REGOLE DI RESET
        # ==========================================
        # 1. Condizioni di Reset
        if active_warning in ["Sleep", "Microsleep"]:
            if eyes_open_time >= 2.0:
                active_warning = None
        elif active_warning is not None and "Distracted" in active_warning:
            if focus_time >= 2.0:
                active_warning = None
                distraction_history.clear()

        # 2. Attivazione Allarmi (In ordine di gravità)
        if continuous_closed_time >= 7.0:
            active_warning = "Sleep"
            warning_detail = ""
        elif continuous_closed_time >= 4.0 and active_warning != "Sleep":
            active_warning = "Microsleep"
            warning_detail = ""
        elif active_warning not in ["Sleep", "Microsleep"]:
            # Gestione Distrazioni
            if continuous_distraction_time >= 5.0:
                active_warning = "Distracted (long)"
                warning_detail = f" - {frame_distraction}"
            elif cumulative_time >= 10.0 and active_warning is None:
                types = [typ for _, _, typ in distraction_history]
                predominant_type = max(set(types), key=types.count) if types else "Unknown"
                active_warning = "Distracted (short)"
                warning_detail = f" - {predominant_type}"

        # ==========================================
        # GESTIONE UI
        # ==========================================
        # Stato in basso a destra
        driver_state = f"{active_warning}{warning_detail}" if active_warning else "Focused on the road"
        color = (0, 0, 255) if active_warning else (0, 255, 0)
        cv2.putText(frame, f"State: {driver_state}", (frame.shape[1] - 450, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Info di debug (in alto a sinistra)
        cv2.putText(frame, f"Distr. Accum: {cumulative_time:.1f}s/30s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)
        cv2.putText(frame, f"Distr. Cont: {continuous_distraction_time:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)
        cv2.putText(frame, f"Eyes Closed: {continuous_closed_time:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)

        # BPM in basso a sinistra
        cv2.putText(frame, f"BPM: {current_bpm if current_bpm > 0 else 'Calc...'}", (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('Driver Monitoring System', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_dms()