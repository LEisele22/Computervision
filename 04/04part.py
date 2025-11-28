import cv2
import numpy as np
import math

# ------------------ LINE UTILITY FUNCTIONS ------------------

def line_points_from_rho_theta(rho, theta, scale, shape):
    nx, ny = -math.sin(theta), math.cos(theta)
    x0, y0 = nx * rho, ny * rho
    dx, dy = math.cos(theta), math.sin(theta)
    return (
        int(x0 + dx*scale), int(y0 + dy*scale),
        int(x0 - dx*scale), int(y0 - dy*scale)
    )

def line_rho_theta(line):
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    angle = math.atan2(dy, dx)
    if angle < 0:
        angle += math.pi
    mx, my = (x1+x2)/2, (y1+y2)/2
    nx, ny = -math.sin(angle), math.cos(angle)
    rho = mx*nx + my*ny
    return rho, angle

def cluster_by_orientation(lines, num_clusters=2):
    if not lines:
        return [], []
    angles = [[math.degrees(math.atan2(y2-y1, x2-x1)) % 180] for x1,y1,x2,y2 in lines]
    samples = np.array(angles, np.float32)
    if len(samples) < num_clusters:
        return [0]*len(samples), [float(samples.mean())]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, centers = cv2.kmeans(samples, num_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return labels.flatten().tolist(), [float(c[0]) for c in centers]

def select_extreme_lines(lines, labels, centers):
    grouped = {}
    for i, l in enumerate(lines):
        rho, _ = line_rho_theta(l)
        lab = labels[i]
        grouped.setdefault(lab, []).append((rho, l))
    chosen = []
    for entries in grouped.values():
        entries.sort(key=lambda x: x[0])
        if len(entries) == 1:
            chosen.append(entries[0][1])
        else:
            chosen.append(entries[0][1])
            chosen.append(entries[-1][1])
    return chosen

# ------------------ RECTIFICATION UTILITIES ------------------

def intersect(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/denom
    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4))/denom
    return px, py

def order_corners(pts):
    pts = np.array(pts, np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], np.float32)

def homography_dlt(src, dst):
    N = src.shape[0]
    A = []
    for i in range(N):
        x, y = src[i]
        u, v = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    _, _, Vt = np.linalg.svd(np.array(A, np.float64))
    H = Vt[-1].reshape(3,3)
    return H/H[2,2]

def rectify(frame, lines, out_size=(600,800)):
    groupA, groupB = lines[:2], lines[2:4]
    corners = [intersect(a,b) for a in groupA for b in groupB if intersect(a,b)]
    if len(corners)!=4:
        raise RuntimeError(f"Expected 4 corners, got {len(corners)}")
    src = order_corners(corners)
    dst = np.array([[0,0],[out_size[1]-1,0],[out_size[1]-1,out_size[0]-1],[0,out_size[0]-1]], np.float32)
    H_dlt = homography_dlt(src, dst)
    H_cv, _ = cv2.findHomography(src, dst)
    return {
        'quad_corners': src,
        'H_dlt': H_dlt,
        'H_cv': H_cv,
        'warp_dlt': cv2.warpPerspective(frame, H_dlt, (out_size[1], out_size[0])),
        'warp_cv': cv2.warpPerspective(frame, H_cv, (out_size[1], out_size[0]))
    }

# ------------------ MAIN LOOP ------------------

def main_loop(camera_idx=0):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("Cannot open camera", camera_idx)
        return
    win = 'Rectify Demo (q=quit, r=rectify)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        edges = cv2.Canny(blur,50,150)
        overlay = frame.copy()
        lines_p = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength=60,maxLineGap=10)
        if lines_p is None:
            cv2.imshow('Edges', edges)
            cv2.imshow(win, overlay)
            if cv2.waitKey(1)&0xFF==ord('q'): break
            continue
        lines = [tuple(l[0]) for l in lines_p]
        labels, centers = cluster_by_orientation(lines,2)
        chosen = select_extreme_lines(lines, labels, centers)
        # draw lines with numbers
        for idx, l in enumerate(chosen[:4]):
            rho, theta = line_rho_theta(l)
            length = int(math.hypot(w,h)*1.5)
            x1, y1, x2, y2 = line_points_from_rho_theta(rho, theta, length, frame.shape)
            cv2.line(overlay,(x1,y1),(x2,y2),(0,0,255),2)
            mx, my = (x1+x2)//2,(y1+y2)//2
            cv2.putText(overlay,f"{idx+1}",(mx,my),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3,cv2.LINE_AA)
        cv2.putText(overlay,'Press r to rectify',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        cv2.imshow('Edges', edges)
        cv2.imshow(win, overlay)
        key = cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        if key==ord('r') and len(chosen)>=4:
            try:
                res = rectify(frame, chosen)
                cv2.imshow('Warp DLT', res['warp_dlt'])
                cv2.imshow('Warp OpenCV', res['warp_cv'])
                diff = np.linalg.norm(res['H_dlt']/res['H_dlt'][2,2]-res['H_cv']/res['H_cv'][2,2])
                print('Homography difference (DLT vs OpenCV):', diff)
            except Exception as e:
                print("Rectification failed:", e)
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main_loop()
