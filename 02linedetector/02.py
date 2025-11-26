import cv2
import numpy as np
import random


# ---------------------------------------------
# Canny → edge pixel extraction
# ---------------------------------------------
def get_edge_points(gray, low=50, high=150):
    edges = cv2.Canny(gray, low, high)
    ys, xs = np.nonzero(edges)
    return edges, np.column_stack((xs, ys))  # [(x,y), ...]


# ---------------------------------------------
# RANSAC Line Fitter  (ax + by + c = 0)
# ---------------------------------------------
def ransac_line(points, iterations=800, dist_thresh=2.5):
    if len(points) < 2:
        return None, None

    best_model = None
    best_inliers = 0
    best_mask = None
    n = len(points)

    rand = random.Random(0)

    for _ in range(iterations):
        # Pick 2 unique random points
        i1, i2 = rand.sample(range(n), 2)
        p1, p2 = points[i1], points[i2]

        if np.all(p1 == p2):
            continue

        x1, y1 = p1
        x2, y2 = p2

        # Line normal vector
        a = y1 - y2
        b = x2 - x1
        norm = np.hypot(a, b)
        if norm == 0:
            continue

        a /= norm
        b /= norm
        c = -(a * x1 + b * y1)

        # Distances of all points to this line
        dist = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        mask = dist < dist_thresh
        count = mask.sum()

        if count > best_inliers:
            best_inliers = count
            best_mask = mask
            best_model = (a, b, c)

    return best_model, best_mask


# ------------------------------------------------------
# Convert (a,b,c) into visible segment inside image rect
# ------------------------------------------------------
def find_line_endpoints(model, shape):
    a, b, c = model
    h, w = shape[:2]

    pts = []

    # x = 0
    if abs(b) > 1e-6:
        y = -c / b
        if 0 <= y <= h:
            pts.append((0, int(y)))

    # x = w-1
    if abs(b) > 1e-6:
        y = -(a*(w-1) + c) / b
        if 0 <= y <= h:
            pts.append((w-1, int(y)))

    # y = 0
    if abs(a) > 1e-6:
        x = -c / a
        if 0 <= x <= w:
            pts.append((int(x), 0))

    # y = h-1
    if abs(a) > 1e-6:
        x = -(b*(h-1) + c) / a
        if 0 <= x <= w:
            pts.append((int(x), h-1))

    # Need exactly 2 endpoints
    if len(pts) >= 2:
        return pts[0], pts[1]

    return None, None


# ---------------------------------------------
# Main real-time loop
# ---------------------------------------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges, pts = get_edge_points(gray)

        # Fit line using RANSAC
        model, mask = ransac_line(pts)

        out = frame.copy()

        if model is not None:
            p1, p2 = find_line_endpoints(model, out.shape)
            if p1 and p2:
                cv2.line(out, p1, p2, (0,255,0), 2)

            # Draw inlier points in red
            for (x,y) in pts[mask]:
                cv2.circle(out, (int(x),int(y)), 1, (0,0,255), -1)

        cv2.imshow("Prominent Line Detection", out)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
