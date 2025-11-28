import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # ---- PARAMETERS YOU CAN EDIT ----
    t1 = 50          # Canny threshold 1
    t2 = 150         # Canny threshold 2
    hth = 80         # Hough threshold
    minL = 60        # Minimum line length
    maxG = 10        # Maximum line gap
    # ----------------------------------

    win = "Hough Line Detection"
    cv2.namedWindow(win)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Edge detection
        edges = cv2.Canny(gray, t1, t2, apertureSize=3)

        # 3. Hough Transform for lines
        lines = cv2.HoughLinesP(edges,
                                rho=1,
                                theta=np.pi / 180,
                                threshold=hth,
                                minLineLength=minL,
                                maxLineGap=maxG)

        # 4. Select the 4 longest lines only
        four_lines = []
        if lines is not None:
            line_lengths = []
            for l in lines:
                x1, y1, x2, y2 = l[0]
                length = (x2 - x1)**2 + (y2 - y1)**2
                line_lengths.append((length, l[0]))

            line_lengths.sort(reverse=True, key=lambda x: x[0])

            for i in range(min(4, len(line_lengths))):
                four_lines.append(line_lengths[i][1])

        # 5. Draw ONLY those 4 lines
        display = frame.copy()
        for x1, y1, x2, y2 in four_lines:
            cv2.line(display, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 6. Show live output
        cv2.imshow(win, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
