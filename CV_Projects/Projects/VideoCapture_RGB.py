import cv2
import time
import glob
import os
from datetime import datetime

# OutPut Path
output_dir = "../Productions"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None
status_list = []
recording = False

# Sensitivity threshold for motion detection
SENSITIVITY_THRESHOLD = 500

# Initialize video writer variable
out = None

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    status = 0
    check, frame = video.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if first_frame is None:
        first_frame = gray_frame_gau
        continue

    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)
    thresh_frame = cv2.threshold(delta_frame, 75, 255, cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    contours, _ = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        faces = face_cascade.detectMultiScale(gray_frame_gau, scaleFactor=1.1, minNeighbors=5)
        for (fx, fy, fw, fh) in faces:
            center_x = fx + fw // 2
            center_y = fy + fh // 2
            radius = max(fw, fh) // 2
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)

    # Start recording if motion detected
    if not recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"Capture_RGB_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        recording = True

    status = 1
    # Adding timestamp
    timestamp = datetime.now().strftime("Cam1: Date:%d/%m/%Y:Time:%H:%M:%S")
    cv2.putText(frame, timestamp, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

    if recording:
        out.write(frame)

    status_list.append(status)
    status_list = status_list[-2:]

    # Stop recording if no motion is detected
    if status == 0 and recording:
        recording = False
        out.release()

    # Display the Detection Screen
    cv2.imshow("RGB Screen", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the video capture and writer
if out:
    out.release()
video.release()
cv2.destroyAllWindows()
