
import cv2
import numpy as np
import time


def find_brightest_opencv(gray):
    """
    Finds brightest spot in grayscale image using OpenCV functions.
    """
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    return max_loc  # (x, y)

def find_reddest(image_bgr):
    """
    Define "reddest" = high red value relative to other channels.
    Score = R - max(G, B)
    """
    b, g, r = cv2.split(image_bgr)
    red_score = r.astype(int) - np.maximum(g, b).astype(int)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(red_score)
    return max_loc  # (x, y)

def find_brightest_for_loop(gray):
    """
    Finds the brightest spot using a pixel-by-pixel double loop.
    NOTE: Very slow, but implemented as requested.
    """
    h, w = gray.shape
    max_val = -1
    max_pos = (0, 0)

    for y in range(h):
        for x in range(w):
            val = gray[y, x]
            if val > max_val:
                max_val = val
                max_pos = (x, y)
    return max_pos


# --- Main loop ----------------------------------------------------------------

cap = cv2.VideoCapture(0) #0 for video camera #2 for phone webcam

if not cap.isOpened():
    print("Could not open webcam!")
    exit()

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Frame error!")
        break

    # Convert to grayscale for brightness detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) Brightest spot (OpenCV)
    bright_spot = find_brightest_opencv(gray)
    cv2.circle(frame, bright_spot, 10, (255, 255, 0), 2)

    # 2) Reddest spot
    red_spot = find_reddest(frame)
    cv2.circle(frame, red_spot, 10, (0, 0, 255), 2)

    # 3) Brightest spot (double for-loop)
    loop_spot = find_brightest_for_loop(gray)
    cv2.circle(frame, loop_spot, 10, (0, 255, 0), 2)

    # FPS calculation
    fps = 1.0 / (time.time() - start_time)

    # Draw FPS on screen
    cv2.putText(
        frame, f"FPS: {fps:.2f}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    cv2.imshow("Bright and Red Spot Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


