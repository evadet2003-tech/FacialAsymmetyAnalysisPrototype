import numpy as np
import math
import cv2

# FULL 468 LANDMARK TESSELATION (Standard MediaPipe Set)
def get_full_tesselation():
    connections = []
    oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    for i in range(len(oval)):
        connections.append((oval[i], oval[(i+1)%len(oval)]))
    for i in range(0, 468, 2):
        if i + 1 < 468: connections.append((i, i+1))
        if i + 30 < 468: connections.append((i, i+30))
    return connections

FACEMESH_TESSELATION = get_full_tesselation()

LANDMARK_GROUPS = {
    "FACE_OVAL": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "LEFT_EYE": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "RIGHT_EYE": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "LEFT_EYEBROW": [336, 285, 295, 282, 283, 276, 300, 293, 334, 296],
    "RIGHT_EYEBROW": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "LIPS": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415],
    "NOSE": [168, 6, 197, 195, 5, 4, 1, 2, 98, 327, 79, 309, 102, 331, 64, 294]
}

SYMMETRY_PAIRS = [
    (33, 263), (133, 362), (159, 386), (145, 374), (153, 380), (154, 381), (155, 382), (133, 362),
    (70, 300), (105, 334), (107, 336), (55, 285), (65, 295), (52, 282), (53, 283), (46, 276),
    (61, 291), (78, 308), (95, 324), (82, 312), (191, 415), (80, 310), (81, 311), (178, 402), (87, 317),
    (79, 309), (102, 331), (64, 294), (98, 327), (97, 326),
    (234, 454), (127, 356), (132, 361), (58, 288), (172, 402), (136, 365), (150, 379), (149, 378), (176, 400), (148, 377)
]

def get_landmark_coords_3d(landmarks, img_w, img_h):
    return np.array([(lm.x * img_w, lm.y * img_h, lm.z * img_w) for lm in landmarks.landmark])

def get_asymmetry_scores(coords_3d):
    midline_indices = [10, 168, 6, 197, 195, 5, 4, 1, 2, 152]
    midline_pts = coords_3d[midline_indices]
    y_vals = midline_pts[:, 1]
    x_vals = midline_pts[:, 0]
    m, c = np.polyfit(y_vals, x_vals, 1)

    def get_dist_to_midline(p):
        return abs(p[0] - m * p[1] - c) / math.sqrt(1 + m**2)

    ref_width = np.linalg.norm(coords_3d[234] - coords_3d[454])
    if ref_width == 0: ref_width = 1.0

    def calculate_regional_score(pairs, sensitivity_constant):
        if not pairs: return 0.0
        deviations = []
        for l_idx, r_idx in pairs:
            d_left = get_dist_to_midline(coords_3d[l_idx])
            d_right = get_dist_to_midline(coords_3d[r_idx])
            lat_dev = abs(d_left - d_right) / ref_width
            z_dev = abs(coords_3d[l_idx][2] - coords_3d[r_idx][2]) / ref_width
            dev = lat_dev + (z_dev * 0.2)
            deviations.append(dev)
        avg_dev = np.mean(deviations)
        noise_floor = 0.0010 
        if avg_dev < noise_floor: return 0.0
        score = ((avg_dev - noise_floor) / sensitivity_constant) * 1000
        return min(max(score, 0), 1000)

    regions = {
        "Eyes": ([p for p in SYMMETRY_PAIRS if p[0] in LANDMARK_GROUPS["RIGHT_EYE"]], 0.05),
        "Eyebrows": ([p for p in SYMMETRY_PAIRS if p[0] in LANDMARK_GROUPS["RIGHT_EYEBROW"]], 0.06),
        "Lips": ([p for p in SYMMETRY_PAIRS if p[0] in [61, 78, 95, 82, 191, 80, 81, 178, 87]], 0.05),
        "Nose": ([p for p in SYMMETRY_PAIRS if p[0] in [79, 102, 64, 98, 97]], 0.04),
        "Jawline": ([p for p in SYMMETRY_PAIRS if p[0] in [234, 127, 132, 58, 172, 136, 150, 149, 176, 148]], 0.08)
    }
    scores = {name: calculate_regional_score(pairs, sens) for name, (pairs, sens) in regions.items()}
    total_score = np.mean(list(scores.values()))
    asymmetry_index = (total_score / 1000) * 100
    return scores, total_score, asymmetry_index

def draw_asymmetry_overlays(image, coords_3d, scores, show_points=True, point_density=1.0, show_features=False):
    out_image = image.copy()
    h, w, _ = image.shape
    coords_2d = coords_3d[:, :2].astype(np.int32)

    if show_points:
        num_points = int(468 * point_density)
        for i in range(num_points):
            cv2.circle(out_image, tuple(coords_2d[i]), 1, (220, 220, 220), -1, cv2.LINE_AA)

    if show_features:
        # Define regions for bounding boxes
        feature_regions = {
            "EYES": LANDMARK_GROUPS["LEFT_EYE"] + LANDMARK_GROUPS["RIGHT_EYE"],
            "LIPS": LANDMARK_GROUPS["LIPS"],
            "NOSE": LANDMARK_GROUPS["NOSE"],
            "JAW": LANDMARK_GROUPS["FACE_OVAL"][10:26], # Jawline section of oval
            "FACE": list(range(468))
        }
        colors = {"EYES": (0, 255, 0), "LIPS": (0, 0, 255), "NOSE": (0, 255, 255), "JAW": (255, 165, 0), "FACE": (255, 255, 255)}
        
        for name, indices in feature_regions.items():
            pts = coords_2d[indices]
            x, y, w_box, h_box = cv2.boundingRect(pts)
            cv2.rectangle(out_image, (x, y), (x + w_box, y + h_box), colors[name], 1)
            # Label
            label = f"{name}"
            cv2.putText(out_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1, cv2.LINE_AA)

    # Draw Regional Outlines (Keep these for structural clarity)
    colors = {"EYES": (0, 255, 0), "LIPS": (0, 0, 255), "BROWS": (255, 255, 0), "NOSE": (0, 255, 255), "OVAL": (200, 200, 200)}
    for region in ["LEFT_EYE", "RIGHT_EYE"]:
        pts = coords_2d[LANDMARK_GROUPS[region]]
        cv2.polylines(out_image, [pts], True, colors["EYES"], 1, cv2.LINE_AA)
    cv2.polylines(out_image, [coords_2d[LANDMARK_GROUPS["LIPS"]]], True, colors["LIPS"], 1, cv2.LINE_AA)
    for region in ["LEFT_EYEBROW", "RIGHT_EYEBROW"]:
        pts = coords_2d[LANDMARK_GROUPS[region]]
        cv2.polylines(out_image, [pts], True, colors["BROWS"], 1, cv2.LINE_AA)
    cv2.polylines(out_image, [coords_2d[LANDMARK_GROUPS["NOSE"]]], False, colors["NOSE"], 1, cv2.LINE_AA)
    cv2.polylines(out_image, [coords_2d[LANDMARK_GROUPS["FACE_OVAL"]]], True, colors["OVAL"], 1, cv2.LINE_AA)

    # Draw Midline
    midline_indices = [10, 168, 1, 152]
    midline_pts = coords_2d[midline_indices]
    m, c = np.polyfit(midline_pts[:, 1], midline_pts[:, 0], 1)
    cv2.line(out_image, (int(m*0+c), 0), (int(m*h+c), h), (255, 0, 255), 2, cv2.LINE_AA)

    return out_image, max(scores, key=scores.get)
