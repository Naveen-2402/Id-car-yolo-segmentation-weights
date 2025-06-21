import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("id_11_n-seg.pt")
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def detect_tilt(points, up_down_threshold=10, left_right_threshold=20, rotation_threshold=5):
    """Detect tilt by comparing distances between opposite points"""
    tl, tr, br, bl = points
    
    # Calculate distances
    top_width = np.linalg.norm(tr - tl)
    bottom_width = np.linalg.norm(br - bl)
    left_height = np.linalg.norm(bl - tl)
    right_height = np.linalg.norm(br - tr)
    
    # Calculate rotation angle
    dx = tr[0] - tl[0]
    dy = tr[1] - tl[1]
    rotation_angle = np.degrees(np.arctan2(dy, dx))
    
    # Determine single combined feedback
    feedback = "CORRECT"
    color = (0, 255, 0)  # Green for correct
    
    # Check rotation first
    if abs(rotation_angle) > rotation_threshold:
        if rotation_angle > 0:
            feedback = "RIGHT TILT"
        else:
            feedback = "LEFT TILT"
        color = (0, 0, 255)  # Red
    else:
        # Check perspective tilt only if rotation is correct
        width_diff = abs(top_width - bottom_width)
        height_diff = abs(left_height - right_height)
        
        # Check up/down tilt with lower threshold for better sensitivity
        if width_diff > up_down_threshold:
            if top_width > bottom_width:
                feedback = "DOWN TILT"
            else:
                feedback = "UP TILT"
            color = (0, 165, 255)  # Orange
        # Check left/right tilt
        elif height_diff > left_right_threshold:
            if left_height > right_height:
                feedback = "LEFT TILT"
            else:
                feedback = "RIGHT TILT"
            color = (0, 165, 255)  # Orange
    
    return feedback, color, rotation_angle

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    if results.masks:
        for box, mask_xy in zip(results.boxes, results.masks.xy):
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.8:  # ID card with high confidence
                # Convert mask to contour and approximate to 4 points
                contour = mask_xy.astype(np.int32).reshape(-1, 1, 2)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, closed=True)

                if len(approx) == 4:
                    # Order points and detect tilt
                    points = order_points(approx.reshape(4, 2))
                    feedback, color, angle = detect_tilt(points)
                    
                    # Draw rectangle and corners
                    cv2.polylines(frame, [approx], True, (255, 255, 0), 2)
                    for i, point in enumerate(points.astype(int)):
                        cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
                    
                    # Display single feedback
                    cv2.putText(frame, feedback, (int(points[0][0]), int(points[0][1]) - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Print to console
                    print(f"Status: {feedback} | Angle: {angle:.1f}°")

    cv2.imshow("ID Card Tilt Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()