import cv2
import mediapipe as mp
import numpy as np


class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True è obbligatorio per rilevare le iridi
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Liste di indici per il disegno su schermo
        self.LEFT_EYE_LMS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE_LMS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS_LMS = [473, 474, 475, 476, 477]
        self.RIGHT_IRIS_LMS = [468, 469, 470, 471, 472]
        self.NOSE_TIP_LMS = [45, 4, 275]

        # Indici per calcolo Eye Aspect Ratio (EAR)
        self.RIGHT_EYE_H = [33, 133]
        self.RIGHT_EYE_V1 = [160, 144]
        self.RIGHT_EYE_V2 = [158, 153]
        self.LEFT_EYE_H = [362, 263]
        self.LEFT_EYE_V1 = [385, 380]
        self.LEFT_EYE_V2 = [387, 373]

        self.NOSE_TIP = 4
        self.FOREHEAD_CENTER = 151
        self.RIGHT_IRIS_CENTER = 468  # Usato per il calcolo Lizard

    def _euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _calculate_ear(self, landmarks, w, h, eye_h, eye_v1, eye_v2):
        p_h1 = (landmarks[eye_h[0]].x * w, landmarks[eye_h[0]].y * h)
        p_h2 = (landmarks[eye_h[1]].x * w, landmarks[eye_h[1]].y * h)
        p_v1_top = (landmarks[eye_v1[0]].x * w, landmarks[eye_v1[0]].y * h)
        p_v1_bot = (landmarks[eye_v1[1]].x * w, landmarks[eye_v1[1]].y * h)
        p_v2_top = (landmarks[eye_v2[0]].x * w, landmarks[eye_v2[0]].y * h)
        p_v2_bot = (landmarks[eye_v2[1]].x * w, landmarks[eye_v2[1]].y * h)

        hor_dist = self._euclidean_distance(p_h1, p_h2)
        ver_dist1 = self._euclidean_distance(p_v1_top, p_v1_bot)
        ver_dist2 = self._euclidean_distance(p_v2_top, p_v2_bot)

        if hor_dist == 0: return 0
        return (ver_dist1 + ver_dist2) / (2.0 * hor_dist)

    def draw_landmarks(self, frame, landmarks):
        """Disegna i punti chiave di occhi, iridi e naso sul frame."""
        img_h, img_w, _ = frame.shape
        for idx, lm in enumerate(landmarks):
            pos = (int(lm.x * img_w), int(lm.y * img_h))
            if idx in self.LEFT_EYE_LMS or idx in self.RIGHT_EYE_LMS:
                cv2.circle(frame, pos, radius=1, color=(0, 0, 255), thickness=-1)
            if idx in self.LEFT_IRIS_LMS or idx in self.RIGHT_IRIS_LMS:
                cv2.circle(frame, pos, radius=1, color=(0, 255, 0), thickness=-1)
            if idx in self.NOSE_TIP_LMS:
                cv2.circle(frame, pos, radius=2, color=(255, 0, 0), thickness=-1)

    def process_frame(self, frame, draw=True):
        result = {
            "face_found": False, "nose_x": None, "nose_y": None,
            "ear": None, "roi_rgb": None, "iris_offset_x": None, "iris_offset_y": None
        }

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.face_mesh.process(image_rgb)

        if not mp_results.multi_face_landmarks:
            return result

        img_h, img_w, _ = frame.shape
        landmarks = mp_results.multi_face_landmarks[0].landmark

        if draw:
            self.draw_landmarks(frame, landmarks)

        # 1. Naso (Owl) - Posizione assoluta
        result["nose_x"] = landmarks[self.NOSE_TIP].x
        result["nose_y"] = landmarks[self.NOSE_TIP].y

        # 2. EAR (Sonno)
        right_ear = self._calculate_ear(landmarks, img_w, img_h, self.RIGHT_EYE_H, self.RIGHT_EYE_V1, self.RIGHT_EYE_V2)
        left_ear = self._calculate_ear(landmarks, img_w, img_h, self.LEFT_EYE_H, self.LEFT_EYE_V1, self.LEFT_EYE_V2)
        result["ear"] = (right_ear + left_ear) / 2.0

        # 3. Posizione Iridi 2D (Lizard) - Offset relativo al centro dell'occhio
        r_inner = landmarks[self.RIGHT_EYE_H[1]]
        r_outer = landmarks[self.RIGHT_EYE_H[0]]
        r_top = landmarks[self.RIGHT_EYE_V1[0]]
        r_bot = landmarks[self.RIGHT_EYE_V1[1]]
        iris = landmarks[self.RIGHT_IRIS_CENTER]

        eye_center_x = (r_inner.x + r_outer.x) / 2.0
        eye_center_y = (r_top.y + r_bot.y) / 2.0
        eye_width = abs(r_inner.x - r_outer.x)

        # FIX: Usiamo eye_width anche per l'asse Y. L'altezza dell'occhio varia
        # drasticamente con le palpebre e sballa il calcolo se usata come divisore.
        if eye_width > 0.001:
            result["iris_offset_x"] = (iris.x - eye_center_x) / eye_width
            result["iris_offset_y"] = (iris.y - eye_center_y) / eye_width
        else:
            result["iris_offset_x"] = 0.0
            result["iris_offset_y"] = 0.0

        # 4. ROI RGB (Frequenza Cardiaca) con 3 zone dinamiche
        # Definiamo i punti: 151 (Fronte), 118 (Zigomo DX), 347 (Zigomo SX)
        roi_configs = [
            {"idx": self.FOREHEAD_CENTER, "w": 20, "h": 10},
            {"idx": 118, "w": 10, "h": 10},
            {"idx": 347, "w": 10, "h": 10}
        ]

        all_means = []
        rects = []  # Per il disegno in runDMS

        for cfg in roi_configs:
            lm = landmarks[cfg["idx"]]
            cx, cy = int(lm.x * img_w), int(lm.y * img_h)

            # Calcolo dei bordi del rettangolo
            y1, y2 = max(0, cy - cfg["h"]), min(img_h, cy + cfg["h"])
            x1, x2 = max(0, cx - cfg["w"]), min(img_w, cx + cfg["w"])

            rects.append(((x1, y1), (x2, y2)))

            roi_crop = frame[y1:y2, x1:x2]
            if roi_crop.size > 0:
                # cv2.mean restituisce (B, G, R, A)
                all_means.append(cv2.mean(roi_crop)[:3])

        # Se abbiamo estratto dati, facciamo la media delle 3 zone
        if all_means:
            avg_bgr = np.mean(all_means, axis=0)
            # Salviamo in ordine R, G, B
            result["roi_rgb"] = (avg_bgr[2], avg_bgr[1], avg_bgr[0])
            result["roi_viz"] = rects  # Passiamo le coordinate dei rettangoli

        result["face_found"] = True
        return result