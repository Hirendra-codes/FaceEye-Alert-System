import cv2
import subprocess
import time
from collections import deque

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


cap = cv2.VideoCapture(0)

EYE_BUFFER_SIZE = 8          
CLOSED_THRESHOLD = 0.65      
ALARM_DELAY = 0.4            
RESET_DELAY = 0.25           

ALARM_PATH = "alarm.wav"

eye_history = deque(maxlen=EYE_BUFFER_SIZE)

alarm_on = False
alarm_process = None
alarm_started_at = None
eyes_open_since = None

print("✅ Fast & Smooth Drowsiness Detection Started")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False
    now = time.time()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(30, 30)
        )

        if len(eyes) >= 2:
            eyes_detected = True

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                frame,
                (x+ex, y+ey),
                (x+ex+ew, y+ey+eh),
                (255, 0, 0),
                2
            )

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    eye_history.append(1 if eyes_detected else 0)
    closed_ratio = 1 - (sum(eye_history) / len(eye_history))

    if closed_ratio >= CLOSED_THRESHOLD:
        eyes_open_since = None

        if alarm_started_at is None:
            alarm_started_at = now

        if now - alarm_started_at >= ALARM_DELAY:
            cv2.putText(
                frame, "DROWSINESS ALERT!",
                (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3
            )

            if not alarm_on:
                alarm_process = subprocess.Popen(
                    ["afplay", ALARM_PATH]
                )
                alarm_on = True
    else:
        alarm_started_at = None

        if eyes_open_since is None:
            eyes_open_since = now

        if alarm_on and now - eyes_open_since >= RESET_DELAY:
            alarm_process.terminate()
            alarm_on = False


    cv2.putText(
        frame,
        f"Closed ratio: {closed_ratio:.2f}",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    cv2.imshow("Driver Drowsiness Detection (FAST)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

if alarm_on:
    alarm_process.terminate()

cap.release()
cv2.destroyAllWindows()
