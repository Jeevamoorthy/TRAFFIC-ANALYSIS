# Real-Time Vehicle Traffic Analyzer

This project provides a complete solution for real-time vehicle detection, tracking, counting, and traffic congestion visualization. It processes a video feed (or live footage) to identify vehicles, assign them unique IDs, count them as they cross a designated line, and generate a heatmap to highlight areas of traffic slowdown or congestion.
<img width="1600" height="899" alt="image" src="https://github.com/user-attachments/assets/598ca6f8-e0dd-4c73-ba87-ae6107cf6edb" />


---

## Features

-   **Real-Time Vehicle Detection**: Utilizes the powerful YOLOv8 model to accurately detect various types of vehicles (cars, trucks, buses, motorcycles).
-   **Multi-Object Tracking**: Implements the SORT (Simple Online and Realtime Tracking) algorithm to assign and maintain a unique ID for each detected vehicle as it moves across the frame.
-   **Vehicle Counting**: Counts each vehicle that crosses a user-defined line. The counting logic is bidirectional, capturing traffic moving in both directions.
-   **Congestion Heatmapping**: Generates a dynamic heatmap that visualizes traffic density. The map intensifies in areas where vehicles are stationary or moving slowly, providing an excellent visual indicator of congestion.
-   **Customizable**: Easily configure parameters like the video source, detection confidence, and the position/orientation of the counting line.

---

## Tech Stack

-   **Python 3**
-   **custom YOLO model**: For state-of-the-art object detection.
-   **OpenCV**: For video processing, drawing, and visualization.
-   **SORT Algorithm**: For efficient and real-time object tracking.
-   **NumPy**: For numerical operations.
-   **SciPy & FilterPy**: Required dependencies for the SORT tracker's Kalman Filter implementation.
