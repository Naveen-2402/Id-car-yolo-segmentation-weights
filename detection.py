import cv2
import numpy as np
from ultralytics import YOLO

# Load the segmentation model
model = YOLO("id_11_n-seg.pt")  # Ensure this is a *-seg.pt model

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    if results.masks:  # Check if segmentation masks exist
        for box, mask_c in zip(results.boxes, results.masks.xy):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.8:
                # Convert contour points
                contour = mask_c.astype(np.int32).reshape(-1, 1, 2)
                # Draw filled polygon
                cv2.drawContours(frame, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)
                # Optionally outline the polygon
                cv2.drawContours(frame, [contour], -1, (255, 255, 255), thickness=2)
                
                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{results.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Segmentation Polygons", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
