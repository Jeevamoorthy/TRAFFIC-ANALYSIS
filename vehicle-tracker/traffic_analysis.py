import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
VIDEO_PATH = '/home/jeeva/Desktop/vehicle-tracker/TRAFFIC1.mp4'  
MODEL_PATH = 'yolov8n.pt'          # YOLOv8 model (n is small and fast, use m, l, or x for more accuracy)
CONFIDENCE_THRESHOLD = 0.4       
VEHICLE_CLASSES = [2, 3, 5, 7]  

# --- COUNTING LINE ---
LINE_Y_COORD = 650

# --- HEATMAP ---
HEATMAP_ALPHA = 0.5 # Transparency of the heatmap overlay
### <<< CHANGE 1: ADD A THRESHOLD FOR "WAITING" VEHICLES >>> ###
# A vehicle is "waiting" if it moves less than this many pixels per frame
WAITING_SPEED_THRESHOLD = 3 

# --- INITIALIZATION ---

# Load YOLO model
model = YOLO(MODEL_PATH)

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
LINE_START = (0, LINE_Y_COORD)
LINE_END = (frame_width, LINE_Y_COORD)

# Data storage
tracked_vehicles = {} # Stores center points of tracked vehicles {id: [cx, cy]}
counted_ids = set()   # Stores IDs of vehicles that have already been counted
vehicle_count = 0

# Heatmap initialization
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)

# --- MAIN PROCESSING LOOP ---

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # 1. OBJECT DETECTION
    results = model(frame, stream=True, verbose=False)
    detections_for_tracker = []
    for res in results:
        boxes = res.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls in VEHICLE_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                if conf > CONFIDENCE_THRESHOLD:
                    detections_for_tracker.append([int(x1), int(y1), int(x2), int(y2), conf])

    detections_np = np.array(detections_for_tracker)

    # 2. OBJECT TRACKING
    if len(detections_np) > 0:
        tracked_objects = tracker.update(detections_np[:, :4])
    else:
        tracked_objects = tracker.update()

    # 3. COUNTING & HEATMAP UPDATE
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # --- COUNTING LOGIC (IMPROVED FOR BOTH DIRECTIONS) ---
        if obj_id in tracked_vehicles:
            prev_cx, prev_cy = tracked_vehicles[obj_id]
            
            ### <<< CHANGE 2: UPDATED COUNTING LOGIC >>> ###
            # Check for crossing in either direction
            crossed_down = prev_cy < LINE_Y_COORD and cy >= LINE_Y_COORD
            crossed_up = prev_cy > LINE_Y_COORD and cy <= LINE_Y_COORD
            
            if (crossed_down or crossed_up) and obj_id not in counted_ids:
                vehicle_count += 1
                counted_ids.add(obj_id)
                # Flash the line green for a moment to indicate a count
                cv2.line(frame, LINE_START, LINE_END, (0, 255, 0), 4)

            # --- HEATMAP UPDATE (FOR WAITING VEHICLES) ---
            ### <<< CHANGE 3: UPDATED HEATMAP LOGIC >>> ###
            # Calculate distance moved since last frame
            distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            
            # If the vehicle moved less than the threshold, it's "waiting"
            if distance < WAITING_SPEED_THRESHOLD:
                # Increment heatmap at this location
                cv2.circle(heatmap, (cx, cy), 15, 1, thickness=-1)

        # Update vehicle's current position for the next frame
        tracked_vehicles[obj_id] = [cx, cy]
        
        # --- VISUALIZATION on frame ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 4. VISUALIZE HEATMAP
    heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1 - HEATMAP_ALPHA, heatmap_colored, HEATMAP_ALPHA, 0)
    
    # 5. VISUALIZE COUNT AND LINE
    cv2.line(overlay, LINE_START, LINE_END, (0, 0, 255), 1)
    cv2.putText(overlay, f'Vehicle Count: {vehicle_count}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # 6. DISPLAY FRAME
    cv2.imshow("Traffic Analysis", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print(f"Final Vehicle Count: {vehicle_count}")