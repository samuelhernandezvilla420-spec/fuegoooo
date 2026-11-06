from pathlib import Path
import json
import os
import ssl
import smtplib
import sys
from collections import deque
from email.message import EmailMessage
from urllib import request

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")
else:
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "samuelhernandezvilla420@gmail.com")
EMAIL_PROVIDER = os.environ.get("EMAIL_PROVIDER", "smtp").lower()
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY")
SENDGRID_FROM = os.environ.get("SENDGRID_FROM", ALERT_EMAIL)


def send_fire_alert():
    """Envía una alerta por correo cuando se detecta fuego."""
    body = (
        "Se detectó un posible incendio en el sistema de detección. "
        "Revise la cámara o el área monitoreada de inmediato."
    )

    if EMAIL_PROVIDER == "sendgrid":
        if not SENDGRID_API_KEY:
            print("[WARN] Falta SENDGRID_API_KEY. No se pudo enviar la alerta por correo.")
            return False

        payload = {
            "personalizations": [{"to": [{"email": ALERT_EMAIL}]}],
            "from": {"email": SENDGRID_FROM},
            "subject": "ALERTA: posible incendio detectado",
            "content": [{"type": "text/plain", "value": body}],
        }
        try:
            req = request.Request(
                "https://api.sendgrid.com/v3/mail/send",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {SENDGRID_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=20) as resp:
                status = resp.status
            if 200 <= status < 300:
                print(f"[ALERTA] Correo enviado a {ALERT_EMAIL} por SendGrid.")
                return True
            print(f"[ERROR] SendGrid respondió con estado {status}.")
            return False
        except Exception as exc:
            print(f"[ERROR] No se pudo enviar la alerta por SendGrid: {exc}")
            return False

    if not SMTP_USERNAME or not SMTP_PASSWORD:
        print("[WARN] Falta SMTP_USERNAME o SMTP_PASSWORD. No se pudo enviar la alerta por correo.")
        return False

    msg = EmailMessage()
    msg["Subject"] = "ALERTA: posible incendio detectado"
    msg["From"] = SMTP_USERNAME
    msg["To"] = ALERT_EMAIL
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"[ALERTA] Correo enviado a {ALERT_EMAIL}.")
        return True
    except Exception as exc:
        print(f"[ERROR] No se pudo enviar la alerta por correo: {exc}")
        return False


def open_camera(preferred_index=None):
    """Busca una cámara disponible y la devuelve si existe."""
    requested_index = preferred_index if preferred_index is not None else int(os.environ.get("CAMERA_INDEX", "0"))
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    indices = [requested_index] + [i for i in range(5) if i != requested_index]

    for backend in backends:
        for index in indices:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[OK] Cámara encontrada: índice {index}, backend {backend}")
                return cap

            print(f"[WARN] El índice {index} con backend {backend} no devolvió imágenes válidas.")
            cap.release()

    for index in indices:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            print(f"[OK] Cámara encontrada: índice {index}, backend por defecto")
            return cap

        print(f"[WARN] El índice {index} con backend por defecto no devolvió imágenes válidas.")
        cap.release()

    print("[WARN] No se detectó ninguna cámara disponible. Se usará modo demo.")
    return None


class FireDetector:
    def __init__(self):
        self.prev_gray = None
        self.region_history = {}
        self.confirmed_frames = 0
        self.alert_sent = False

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.zeros_like(gray), 0.0

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray.copy()

        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(motion_mask, 5)
        motion_mask = cv2.dilate(motion_mask, np.ones((5, 5), np.uint8), iterations=1)

        motion_ratio = cv2.countNonZero(motion_mask) / max(motion_mask.size, 1)
        return motion_mask, float(motion_ratio)

    def detect_fire_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 60, 60], dtype=np.uint8)
        upper1 = np.array([30, 255, 255], dtype=np.uint8)

        lower2 = np.array([170, 60, 60], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, hsv

    def analyze_region(self, frame, color_mask, motion_mask):
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 180:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            box = (x, y, x + w, y + h)
            roi_color = color_mask[y : y + h, x : x + w]
            roi_motion = motion_mask[y : y + h, x : x + w]

            if roi_color.size == 0:
                continue

            bbox_area = max(w * h, 1)
            color_ratio = cv2.countNonZero(roi_color) / bbox_area
            motion_ratio = cv2.countNonZero(roi_motion) / bbox_area

            if color_ratio < 0.04 or motion_ratio < 0.008:
                continue

            hsv = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
            brightness = hsv[:, :, 2]
            fire_pixels = brightness[roi_color > 0]

            if fire_pixels.size == 0:
                continue

            history = self.region_history.setdefault(box, deque(maxlen=8))
            history.append(float(np.mean(fire_pixels)))

            if len(history) < 4:
                flicker_score = 0.0
            else:
                mean_hist = float(np.mean(history))
                std_hist = float(np.std(history))
                flicker_score = std_hist / max(mean_hist, 1.0)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1.0)
            irregularity = 1.0 - min(solidity, 1.0)

            if flicker_score < 0.01 or irregularity < 0.05:
                continue

            color_score = min(color_ratio / 0.18, 1.0)
            motion_score = min(motion_ratio / 0.06, 1.0)
            flicker_score_norm = min(flicker_score / 0.10, 1.0)
            shape_score = min(irregularity / 0.30, 1.0)

            score = (
                0.38 * color_score
                + 0.30 * motion_score
                + 0.22 * flicker_score_norm
                + 0.10 * shape_score
            )

            if score > 0.42:
                candidates.append({
                    "box": box,
                    "score": score,
                    "motion": motion_ratio,
                    "flicker": flicker_score,
                })

        return candidates

    def process(self, frame):
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        color_mask, _ = self.detect_fire_color(blurred)
        motion_mask, motion_ratio = self.detect_motion(blurred)

        candidates = self.analyze_region(frame, color_mask, motion_mask)

        best_box = None
        best_score = 0.0

        for cand in candidates:
            if cand["score"] > best_score:
                best_score = cand["score"]
                best_box = cand["box"]

        if best_box is not None and best_score >= 0.45:
            self.confirmed_frames += 1
        else:
            self.confirmed_frames = max(0, self.confirmed_frames - 1)

        fire_detected = self.confirmed_frames >= 2

        display = frame.copy()
        if fire_detected and best_box is not None:
            x1, y1, x2, y2 = best_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(display, "FUEGO DETECTADO", (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, f"Score: {best_score:.2f}", (x1, min(y2 + 20, display.shape[0] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(display, "MONITOREO NORMAL", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(display, f"Movimiento: {motion_ratio:.3f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Confirmacion: {self.confirmed_frames}/2", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        color_view = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
        color_view[color_mask > 0] = (0, 0, 255)

        motion_view = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        motion_view[motion_mask > 0] = (0, 255, 0)

        return display, color_view, motion_view, fire_detected


def generate_demo_frame(width=640, height=480, frame_index=0):
    """Genera una escena de prueba con una llama para validar el detector sin cámara."""
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    cx = width // 2
    cy = height // 2 + 50
    for i in range(1, 90):
        angle = np.linspace(0, 2 * np.pi, 100)
        r = i * 1.2 + (frame_index % 7)
        x = np.int32(cx + r * np.cos(angle) * 0.5)
        y = np.int32(cy - i * 1.5 + 20 * np.sin(frame_index * 0.3 + i * 0.18))
        for px, py in zip(x, y):
            if 0 <= px < width and 0 <= py < height:
                frame[py, px] = [0, max(50, 100 + i), 255]
    return frame


def main(preferred_index=None):
    camera = open_camera(preferred_index=preferred_index)
    detector = FireDetector()

    if camera is None:
        print("[INFO] No se encontró cámara. Ejecutando modo demostración.")
        for i in range(500):
            frame = generate_demo_frame(frame_index=i)
            output, color_view, motion_view, fire_detected = detector.process(frame)
            cv2.imshow("Demo - Fuego", output)
            cv2.imshow("Demo - Color", color_view)
            cv2.imshow("Demo - Movimiento", motion_view)

            if fire_detected and not detector.alert_sent:
                print("[ALERTA] Fuego detectado en modo demo.")
                send_fire_alert()
                detector.alert_sent = True
            elif not fire_detected:
                detector.alert_sent = False

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return

    print("Sistema de detección de incendios iniciado. Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[ERROR] La cámara no está entregando imágenes válidas.")
                break

            output, color_view, motion_view, fire_detected = detector.process(frame)
            cv2.imshow("Cámara", output)
            cv2.imshow("Detección de color", color_view)
            cv2.imshow("Movimiento", motion_view)

            if fire_detected and not detector.alert_sent:
                cv2.putText(output, "ALERTA: POSIBLE INCENDIO", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Cámara", output)
                send_fire_alert()
                detector.alert_sent = True
            elif not fire_detected:
                detector.alert_sent = False

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    preferred_index = None
    if "--camera-index" in sys.argv:
        try:
            idx = sys.argv[sys.argv.index("--camera-index") + 1]
            preferred_index = int(idx)
        except (IndexError, ValueError):
            print("[WARN] Uso esperado: --camera-index 0")

    if "--test-email" in sys.argv:
        print(f"[INFO] Probando envío a {ALERT_EMAIL} usando SMTP {SMTP_HOST}:{SMTP_PORT}")
        send_fire_alert()
    elif "--demo" in sys.argv:
        print("Modo demo activado.")
        detector = FireDetector()
        for i in range(500):
            frame = generate_demo_frame(frame_index=i)
            output, color_view, motion_view, fire_detected = detector.process(frame)
            cv2.imshow("Demo - Fuego", output)
            cv2.imshow("Demo - Color", color_view)
            cv2.imshow("Demo - Movimiento", motion_view)
            if fire_detected and not detector.alert_sent:
                print("[ALERTA] Fuego detectado en modo demo.")
                send_fire_alert()
                detector.alert_sent = True
            elif not fire_detected:
                detector.alert_sent = False
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    else:
        main(preferred_index=preferred_index)
