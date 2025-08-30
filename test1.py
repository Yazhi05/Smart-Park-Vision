import cv2
import cvzone
import numpy as np
import pickle  # For saving and loading drawn regions and their names


# Load the video file
cap = cv2.VideoCapture(r"C:\Users\ASUS\Desktop\intern\Data\occ_emp.mp4")


# Initialize variables
points = []  # Stores the four clicked points for bounding box
area_names = []  # Stores names of marked areas


try:
   # Try loading previously saved bounding boxes and area names
   with open("bike_3", "rb") as f:
       data = pickle.load(f)
       polylines, area_names = data['polylines'], data['area_names']
except:
   # If the file doesn't exist, initialize polylines
   polylines = []




# Function to handle mouse events for selecting 4 points
def select_points(event, x, y, flags, param):
   global points


   if event == cv2.EVENT_LBUTTONUP:  # 🔹 LEFT-click to select a point
       if len(points) < 4:
           points.append((x, y))


       if len(points) == 4:  # When 4 points are selected, create a bounding box
           polyline = np.array(points, np.int32)
           polylines.append(polyline)  # Save bounding box
           area_name = f"Area {len(area_names) + 1}"  # Auto-generate name
           area_names.append(area_name)
           points = []  # Reset points for the next bounding box




# Main loop for processing video frames
while True:
   ret, frame = cap.read()
   if not ret:  # Restart video if it reaches the end
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
       continue


   # Resize frame for display consistency
   frame = cv2.resize(frame, (1020, 500))


   # Draw saved bounding boxes and display names
   for i, polyline in enumerate(polylines):
       cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)  # 🔹 Draw bounding box
       cvzone.putTextRect(frame, area_names[i], tuple(polyline[0]), 1, 1)  # Add label


   # Show frame
   cv2.imshow('FRAME', frame)
   cv2.setMouseCallback('FRAME', select_points)  # 🔹 Set mouse callback for right-click


   # Wait for a key press
   key = cv2.waitKey(30) & 0xFF
   if key == ord('s'):  # 🔹 Save the bounding boxes and names
       with open("bike_3", "wb") as f:
           data = {'polylines': polylines, 'area_names': area_names}
           pickle.dump(data, f)
       print("Data saved successfully")
   elif key == ord('q'):
       break


# Release resources and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
