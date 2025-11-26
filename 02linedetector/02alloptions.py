import cv2
import numpy as np
import random
import time


# -----------------------------------------------------
# TOGGLE THESE TO VIEW DIFFERENT OUTPUTS
# -----------------------------------------------------
SHOW_CANNY = False        # If True → displays the Canny image
SHOW_EDGES_ONLY = False   # If True → displays edges only (no RANSAC)
# -----------------------------------------------------


def get_edge_points(gray, low=50, high=150):
    edges = cv2.Canny(gray, low, high)
    ys, xs = np.nonzero(edges)
    return edges, np.column_stack((xs, ys))


def ransac_line(points, iterations=800, dist_thresh=2.5):
    if len(points) < 2:
        return None, None

    best_model = None
    best_inliers = 0
    best_mask = None
    n = len(points)

    rand = random.Random(0)

    for _ in range(iterations):
        i1, i2 = rand.sample(range(n), 2)
        p1, p2 = points[i1], points[i2]

        if np.all(p1 == p2):
            continue

        x1, y1 = p1
        x2, y2 = p2

        a = y1 - y2
        b = x2 - x1
        norm = np.hypot(a, b)
        if norm == 0:
            continue

        a /= norm
        b /= norm
        c = -(a * x1 + b * y1)

        dist = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        mask = dist < dist_thresh
        count = mask.sum()

        if count > best_inliers:
            best_inliers = count
            best_mask = mask
            best_model = (a, b, c)

    return best_model, best_mask


def find_line_endpoints(model, shape):
    a, b, c = model
    h, w = shape[:2]

    pts = []

    if abs(b) > 1e-6:
        y = -c / b
        if 0 <= y <= h:
            pts.append((0, int(y)))

        y = -(a*(w-1) + c) / b
        if 0 <= y <= h:
            pts.append((w-1, int(y)))

    if abs(a) > 1e-6:
        x = -c / a
        if 0 <= x <= w:
            pts.append((int(x), 0))

        x = -(b*(h-1) + c) / a
        if 0 <= x <= w:
            pts.append((int(x), h-1))

    if len(pts) >= 2:
        return pts[0], pts[1]

    return None, None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    prev_time = time.time()   # For FPS counter

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # compute edges
        edges, pts = get_edge_points(gray)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # ---------------------------------------------
        # SHOW ONLY THE CANNY IMAGE
        # ---------------------------------------------
        if SHOW_CANNY:
            fps_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(fps_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Canny Edges", fps_frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # ---------------------------------------------
        # SHOW ONLY EDGE PIXELS
        # ---------------------------------------------
        if SHOW_EDGES_ONLY:
            edge_only = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.putText(edge_only, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.imshow("Edges Only", edge_only)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # ---------------------------------------------
        # FULL RANSAC LINE DETECTION
        # ---------------------------------------------
        model, mask = ransac_line(pts)
        out = frame.copy()

        if model is not None:
            p1, p2 = find_line_endpoints(model, out.shape)
            if p1 and p2:
                cv2.line(out, p1, p2, (0,255,0), 2)

            # Draw inlier points
            for (x, y) in pts[mask]:
                cv2.circle(out, (int(x), int(y)), 1, (0, 0, 255), -1)

        # Draw FPS on video
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Prominent Line Detection", out)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
