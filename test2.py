import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone


# Load the saved polylines and area names
with open("bike_3", "rb") as f:
   data = pickle.load(f)
   polylines, area_names = data['polylines'], data['area_names']


# Load class names from COCO dataset
with open("coco.txt", "r") as my_file:
   class_list = my_file.read().split("\n")


# Load YOLO model
model = YOLO('yolov8s.pt')


# Load video
cap = cv2.VideoCapture(r"C:\Users\ASUS\Desktop\intern\Data\occ_emp.mp4")


# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


# 🔹 Set up video writer to save output as MP4
output_path = r"C:\Users\ASUS\Desktop\intern\Data\output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Alternative: 'H264', 'mp4v'
out = cv2.VideoWriter(output_path, fourcc, fps, (1020, 500))  # 🔹 Fixed output resolution


count = 0


while True:
   ret, frame = cap.read()
   if not ret:
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       continue


   count += 1
   if count % 3 != 0:  # 🔹 Process every 3rd frame for efficiency
       continue


   frame = cv2.resize(frame, (1020, 500))


   # Run YOLO model
   results = model.predict(frame)
   px = pd.DataFrame(results[0].boxes.data).astype("float")


   list1 = []


   for _, row in px.iterrows():
       x1, y1, x2, y2, _, d = map(int, row[:6])


       c = class_list[d]  # Get class name from COCO dataset
       cx, cy = (x1 + x2) // 2, (y1 + y2) // 2


       # 🔹 Detect only motorcycles
       if 'motorcycle' in c:
           list1.append([cx, cy])


   motorcycle_counter = []
   list2 = []


   for i, polyline in enumerate(polylines):
       list2.append(i)
       cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)  # Green polygon
       cvzone.putTextRect(frame, area_names[i], tuple(polyline[0]), 1, 1)  # Add label


       for i1 in list1:
           cx1, cy1 = i1
           result = cv2.pointPolygonTest(polyline, (cx1, cy1), False)
           if result >= 0:
               cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), -1)  # 🔹 Blue circle for detected motorcycles
               cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)  # 🔹 Highlight occupied spots in red
               motorcycle_counter.append(cx1)


   # 🔹 Ensure only positive values are displayed
   occupied = max(0, len(motorcycle_counter))
   free_space = max(0, len(list2) - occupied)


   cvzone.putTextRect(frame, f'FREE SPACE: {free_space}', (50, 60), 2, 2)
   cvzone.putTextRect(frame, f'OCCUPIED: {occupied}', (50, 120), 2, 2)


   # 🔹 Write frame to video file
   out.write(frame)


   # Show frame
   cv2.imshow('FRAME', frame)
   key = cv2.waitKey(30) & 0xFF
   if key == ord('q'):
       break


cap.release()
out.release()
cv2.destroyAllWindows()
