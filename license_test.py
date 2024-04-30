import cv2
from ultralytics import YOLO
from tracker import *

license_plate_detector = YOLO('best.pt')

cap = cv2.VideoCapture('WhatsApp Video 2023-10-20 at 22.00.54_f0d3634c.mp4')

frame_num = -1
ret = True
while ret:
    frame_num += 1
    ret, frame = cap.read()
    if ret:
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255))

        cv2.imshow("roi", frame)
        cv2.waitKey(1)
