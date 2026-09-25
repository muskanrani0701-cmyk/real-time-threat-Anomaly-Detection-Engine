import cv2
import requests
import urllib.request
import os
from datetime import datetime

# 1. Automatically download the missing Cascade file if it isn't in your folder
cascade_file = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_file):
    print("Downloading Face Detection Model...")
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, cascade_file)

detector = cv2.CascadeClassifier(cascade_file)

# 2. Use CAP_DSHOW to bypass Windows MSMF camera lock bugs
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Live Camera active. Press 'q' in the video window to stop.")

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Failed to grab frame. Camera might still be locked.")
        break
    
    # Resize and convert to grayscale for faster processing
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect upper bodies/faces
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, "HUMAN TARGET DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        try:
            requests.post("http://127.0.0.1:5000/api/v1/events", json={
                "source_id": "laptop_cam_1",
                "risk_score": 0.89,
                "severity": "HIGH",
                "captured_at": datetime.utcnow().isoformat() + "Z"
            }, timeout=0.1)
        except: 
            pass 

    cv2.imshow("AEGIS Live Spatial Analysis", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()