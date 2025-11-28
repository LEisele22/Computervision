import cv2
import numpy as np
import math

def extend_line_from_rho_theta(rho, theta, length, img_shape):
    # rho: distance from origin along normal, theta: angle in radians of line direction
    # compute a point on the line using the normal vector
    nx = -math.sin(theta)
    ny = math.cos(theta)
    x0 = nx * rho
    y0 = ny * rho
    dx = math.cos(theta)
    dy = math.sin(theta)
    x1 = int(x0 + dx * length)
    y1 = int(y0 + dy * length)
    x2 = int(x0 - dx * length)
    y2 = int(y0 - dy * length)
    return (x1, y1, x2, y2)


def line_mid_rho_theta(line):
    # line: (x1,y1,x2,y2). Return (rho, theta) in image coordinate system
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    theta = math.atan2(dy, dx)  # direction of the line segment
    # normalize theta to range [0, pi)
    if theta < 0:
        theta += math.pi
    # compute midpoint
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0
    # normal vector (nx, ny)
    nx = -math.sin(theta)
    ny = math.cos(theta)
    rho = mx * nx + my * ny
    return rho, theta


def cluster_lines_by_angle(lines, k=2):
    # lines: list of (x1,y1,x2,y2). Returns labels (0..k-1) for each line and centers (degrees).
    if len(lines) == 0:
        return [], []
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l
        a = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
        if a < 0:
            a += 180.0
        angles.append([a])
    samples = np.array(angles, dtype=np.float32)
    # cv2.kmeans expects float32 samples
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    attempts = 5
    flags = cv2.KMEANS_PP_CENTERS
    if len(samples) < k:
        # fallback: all in one cluster
        labels = [0] * len(samples)
        centers = [float(samples.mean())]
        return labels, centers
    compactness, labels, centers = cv2.kmeans(samples, k, None, criteria, attempts, flags)
    labels = labels.flatten().tolist()
    centers = [float(c[0]) for c in centers]
    return labels, centers


def pick_two_lines_per_orientation(lines, labels, centers):
    # For each label, compute rho for each line, then pick the two lines with extreme rho (min and max)
    label_to_entries = {}
    for idx, l in enumerate(lines):
        rho, theta = line_mid_rho_theta(l)
        lab = labels[idx]
        if lab not in label_to_entries:
            label_to_entries[lab] = []
        label_to_entries[lab].append((rho, theta, l))

    chosen = []
    for lab, entries in label_to_entries.items():
        if len(entries) == 0:
            continue
        # sort by rho
        entries.sort(key=lambda e: e[0])
        if len(entries) == 1:
            chosen.append(entries[0][2])
        else:
            chosen.append(entries[0][2])  # min rho
            chosen.append(entries[-1][2])  # max rho
    return chosen


def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print('Cannot open camera', camera_index)
        return

    win_name = 'Rectangular Boundary Detection (press q to quit)'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Frame capture failed, exiting')
            break
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        # HoughLinesP parameters may need tuning depending on camera / scene
        lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=80, minLineLength=60, maxLineGap=10)

        overlay = frame.copy()

        if lines_p is None:
            # no lines detected: just show edges
            cv2.imshow('Edges', edges)
            cv2.imshow(win_name, overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Convert to simple list of tuples
        lines = [tuple(l[0]) for l in lines_p]

        # cluster by angle into two dominant orientations
        labels, centers = cluster_lines_by_angle(lines, k=2)

        if len(labels) == 0:
            # draw all lines if clustering fails
            for l in lines:
                x1,y1,x2,y2 = l
                cv2.line(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.imshow('Edges', edges)
            cv2.imshow(win_name, overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        chosen = pick_two_lines_per_orientation(lines, labels, centers)

        # If we don't get 4 lines, fallback to drawing the top 4 Hough segments by length
        if len(chosen) < 4:
            lines_sorted = sorted(lines, key=lambda l: math.hypot(l[2]-l[0], l[3]-l[1]), reverse=True)
            for i, l in enumerate(lines_sorted[:4]):
                x1,y1,x2,y2 = l
                cv2.line(overlay, (x1,y1), (x2,y2), (0,255,255), 2)
            note = 'Fallback: drawing top segments'
            cv2.putText(overlay, note, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        else:
            # Convert chosen segments to rho/theta and draw extended lines
            for idx, l in enumerate(chosen[:4]):
                rho, theta = line_mid_rho_theta(l)
                length = int(math.hypot(w, h) * 1.5)
                x1, y1, x2, y2 = extend_line_from_rho_theta(rho, theta, length, frame.shape)

                # Draw extended line
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # ---- ADD NUMERATION HERE ----
                # Midpoint of the extended line
                mx = int((x1 + x2) / 2)
                my = int((y1 + y2) / 2)

                # Draw the line number
                cv2.putText(
                    overlay,
                    f"{idx+1}",
                    (mx, my),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA
                )

            cv2.putText(overlay, 'Detected 4 prominent lines', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # show windows
        cv2.imshow('Edges', edges)
        cv2.imshow(win_name, overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


# -------------------- PART II: RECTIFICATION --------------------
# Added functions and a convenience wrapper to perform corner extraction,
# compute homography with DLT, compare with OpenCV, and warp the image.


def intersect_lines(l1, l2):
    """Return intersection point (x,y) of lines l1 and l2 where each line is (x1,y1,x2,y2).
    Returns None if lines are parallel.
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
    return (px, py)


def order_quad_corners(pts):
    """Order 4 points to consistent order: TL, TR, BR, BL (clockwise starting top-left).
    pts: list/array of 4 (x,y).
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_homography_dlt(src_pts, dst_pts):
    """Compute 3x3 homography H using DLT (Direct Linear Transform) with at least 4 correspondences.
    src_pts and dst_pts are arrays of shape (N,2)
    Returns H normalized so H[2,2] == 1.
    """
    assert src_pts.shape[0] >= 4 and src_pts.shape == dst_pts.shape
    N = src_pts.shape[0]
    A = []
    for i in range(N):
        x, y = src_pts[i,0], src_pts[i,1]
        u, v = dst_pts[i,0], dst_pts[i,1]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A, dtype=np.float64)
    # Solve Ah = 0 via SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    H = h.reshape((3,3))
    if abs(H[2,2]) > 1e-12:
        H = H / H[2,2]
    return H


def rectify_quadrilateral(frame, chosen_lines, out_size=(600,800)):
    """Given frame and four chosen lines (two per orientation), compute corner intersections,
    pick the four quad corners, compute homography with DLT and OpenCV, warp and return results.

    Returns a dict with keys:
      - intersections_all: all 6 pairwise intersections (may include None)
      - quad_corners: 4 ordered corners (TL,TR,BR,BL)
      - H_dlt: 3x3 homography from quad->output (NumPy DLT)
      - H_cv: 3x3 homography from cv2.findHomography
      - warped_dlt: image warped with H_dlt
      - warped_cv: image warped with H_cv
    """
    h_img, w_img = frame.shape[:2]
    # enumerate pairwise intersections (6)
    intersections = []
    for i in range(len(chosen_lines)):
        for j in range(i+1, len(chosen_lines)):
            p = intersect_lines(chosen_lines[i], chosen_lines[j])
            intersections.append((i, j, p))

    # To get 4 quad corners: assume chosen_lines are grouped as [l0,l1] (orientation A) and [l2,l3] (orientation B)
    # then intersections between lines from different groups give 4 corner points
    groupA = chosen_lines[0:2]
    groupB = chosen_lines[2:4]
    quad_pts = []
    for la in groupA:
        for lb in groupB:
            p = intersect_lines(la, lb)
            if p is not None:
                quad_pts.append(p)
    if len(quad_pts) != 4:
        raise RuntimeError('Expected 4 corner intersections, got {}'.format(len(quad_pts)))

    quad_ordered = order_quad_corners(quad_pts)

    # destination corners: map to rectangle of size out_size (h_out, w_out)
    h_out, w_out = out_size
    dst = np.array([[0,0],[w_out-1,0],[w_out-1,h_out-1],[0,h_out-1]], dtype=np.float32)

    src = quad_ordered.astype(np.float32)

    H_dlt = compute_homography_dlt(src, dst)
    # OpenCV homography for comparison
    H_cv, mask = cv2.findHomography(src, dst, method=0)

    warped_dlt = cv2.warpPerspective(frame, H_dlt, (w_out, h_out))
    warped_cv = cv2.warpPerspective(frame, H_cv, (w_out, h_out))

    return {
        'intersections_all': intersections,
        'quad_corners': quad_ordered,
        'H_dlt': H_dlt,
 'H_cv': H_cv,
        'warped_dlt': warped_dlt,
        'warped_cv': warped_cv
    }


# Convenience demo wrapper that integrates Part I detection with Part II rectification.
# Press 'r' while running the Part I script to attempt rectification of the current frame
# (the script will use the most-recent chosen lines to compute corners and warp).

_orig_main = main

def main_with_rectify(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print('Cannot open camera', camera_index)
        return
    win_name = 'Rectify Demo (press q to quit, r to rectify frame)'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=80, minLineLength=60, maxLineGap=10)
        overlay = frame.copy()
        if lines_p is None:
            cv2.imshow('Edges', edges)
            cv2.imshow(win_name, overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        lines = [tuple(l[0]) for l in lines_p]
        labels, centers = cluster_lines_by_angle(lines, k=2)
        if len(labels) == 0:
            for l in lines:
                cv2.line(overlay, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)
            cv2.imshow('Edges', edges)
            cv2.imshow(win_name, overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        chosen = pick_two_lines_per_orientation(lines, labels, centers)
        if len(chosen) >= 4:
            for l in chosen[:4]:
                rho, theta = line_mid_rho_theta(l)
                length = int(math.hypot(w, h) * 1.5)
                x1,y1,x2,y2 = extend_line_from_rho_theta(rho, theta, length, frame.shape)
                cv2.line(overlay, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(overlay, 'Press r to rectify', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            for l in lines[:4]:
                cv2.line(overlay, (l[0],l[1]), (l[2],l[3]), (0,255,255), 2)

        cv2.imshow('Edges', edges)
        cv2.imshow(win_name, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') and len(chosen) >= 4:
            try:
                results = rectify_quadrilateral(frame, chosen, out_size=(600,800))
            except Exception as e:
                print('Rectification failed:', e)
                continue
            cv2.imshow('Warp DLT', results['warped_dlt'])
            cv2.imshow('Warp OpenCV', results['warped_cv'])
            # compute difference between homographies
            Hdiff = np.linalg.norm(results['H_dlt']/results['H_dlt'][2,2] - results['H_cv']/results['H_cv'][2,2])
            print('Homography Frobenius-norm difference (DLT vs OpenCV):', Hdiff)

    cap.release()
    cv2.destroyAllWindows()

# Replace original main with the interactive one
if __name__ == '__main__':
    main_with_rectify()