import cv2
from ultralytics import YOLO
from tracker import *
from util import get_car, read_license_plate, write_csv

results = {}

# load models
model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('best.pt')
# this is where we will apply deepsort license_plate_detector

# Creating tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture('../Videos/car_video2.mp4')

vehicles = [2, 3]

frame_num = -1
ret = True
while ret:
    frame_num += 1
    ret, frame = cap.read()
    if ret:
        results[frame_num] = {}
        detections = model(frame)[0]
        license_detections = license_plate_detector(frame)[0]

        detections_list = []  # this list is to save all the bounding boxes that are detected using the yolo

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255))

            if int(class_id) in vehicles:
                detections_list.append([x1, y1, x2, y2])

        # object tracking

        boxes_id = tracker.update(detections_list)

        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255))

            # xcar1, ycar2, xcar2, ycar2, car_id = get_car(license_plate, boxes_id)

            # if car_id != -1:
            #
            #     # cropping the license plate
            #     license_plate_crop = frame[y1:y2, x1: x2, :]
            #
            #     _, license_plate_crop_thresh = cv2.threshold(license_plate_crop, 64, 255, cv2.THRESH_BINARY_INV)
            #
            #     # Reading the license plate
            #     license_plate_text, license_plate_score = read_license_plate(license_plate_crop_thresh)
            #
            #     # Write the results
            #     if license_plate_text is not None:
            #         results[frame_num][car_id] = {'car': {'bbox': [xcar1, ycar2, xcar2, ycar2]},
            #                                       'license_plate': {'bbox': [x1, y1, x2, y2],
            #                                                         'text': license_plate_text,
            #                                                         'bbox_score': score,
            #                                                         'text_score': license_plate_score}}
        cv2.imshow('roi',frame)
        cv2.waitKey(1)

# write results
# write_csv(results, './test3.csv')


