import argparse
import cv2
import numpy as np
import pandas as pd
import math
import os
import re
import sys
import mediapipe as mp

# =========================================================
# UTF-8 setup for Windows terminal
# =========================================================
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "face_landmarker.task"
PHI = 1.618
TAU = 0.35
EPS = 1e-6

# Output folders
VIZ_DIR = "viz_output"
LANDMARK_DIR = "landmark_output"
CROP_DIR = "crop_output"

# =========================================================
# Utility Functions
# =========================================================
def dist(p1, p2):
    return math.dist(p1, p2)


def mean_point(p1, p2):
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def safe_ratio(a, b):
    return a / (b + EPS)


def phi_oriented_ratio(d1, d2):
    larger = max(d1, d2)
    smaller = min(d1, d2)
    return larger / (smaller + EPS)


def phi_closeness_score(ratio_phi, target=1.618, sigma=1.5):
    try:
        ratio_value = float(ratio_phi)
    except (TypeError, ValueError):
        return 1.0

    if not math.isfinite(ratio_value):
        return 1.0

    diff = abs(ratio_value - target)
    score = 10.0 * math.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    return round(float(max(1.0, min(10.0, score))), 4)


def classify_human_band(score_1_to_10):
    if pd.isna(score_1_to_10):
        return np.nan
    if score_1_to_10 >= 8:
        return "Attractive"
    if score_1_to_10 >= 5:
        return "Good"
    return "Average"


def normalize_path_string(x):
    return str(x).strip().replace("\\", "/")


def extract_subject_id(filename):
    s = normalize_path_string(filename)
    base = s.split("/")[-1]
    stem, _ = os.path.splitext(base)
    match = re.search(r"(p\d+)", stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def get_image_stem(filename):
    s = normalize_path_string(filename)
    base = s.split("/")[-1]
    stem, _ = os.path.splitext(base)
    return stem.strip().lower()


def get_image_key(filename):
    stem = get_image_stem(filename)
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def find_column(columns, candidates):
    norm_map = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in norm_map:
            return norm_map[key]
    return None


def create_face_landmarker(model_path=None):
    print("Initializing face landmark detector...")
    sys.stdout.flush()
    try:
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        from mediapipe.tasks.python.core.base_options import BaseOptions
        if model_path is None and os.path.exists(MODEL_PATH):
            model_path = MODEL_PATH

        base_opts = BaseOptions(model_asset_path=model_path) if model_path else BaseOptions()
        options = FaceLandmarkerOptions(base_options=base_opts, running_mode=RunningMode.IMAGE, num_faces=1)
        detector = FaceLandmarker.create_from_options(options)
        print("Face landmark detector ready")
        sys.stdout.flush()
        return detector
    except Exception as exc:
        print(f"FaceLandmarker init failed: {exc}")
        sys.stdout.flush()
        try:
            detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
            print("FaceMesh fallback ready")
            sys.stdout.flush()
            return detector
        except Exception as fallback_exc:
            print(f"FaceMesh fallback failed: {fallback_exc}")
            sys.stdout.flush()
            return None


def golden_face_dimensions(pts):
    right_eye_center = mean_point(pts[33], pts[133])
    left_eye_center = mean_point(pts[362], pts[263])
    eye_distance = dist(right_eye_center, left_eye_center)

    golden_width = 1.865 * eye_distance
    golden_height = 3.018 * eye_distance

    return eye_distance, golden_width, golden_height, right_eye_center, left_eye_center


def estimate_forehead(eye_center, golden_height):
    return (eye_center[0], eye_center[1] - 0.45 * golden_height)


def golden_crop(image, landmarks):
    h, w = image.shape[:2]
    pts = {i: (landmarks[i].x * w, landmarks[i].y * h) for i in range(len(landmarks))}

    eye_distance, fw, fh, right_eye_center, left_eye_center = golden_face_dimensions(pts)
    eye_center = mean_point(right_eye_center, left_eye_center)

    cx = eye_center[0]
    cy = eye_center[1] + 0.30 * eye_distance

    width_multiplier = 0.95
    height_multiplier = 0.85

    x1 = int(cx - fw * width_multiplier)
    x2 = int(cx + fw * width_multiplier)
    y1 = int(cy - fh * height_multiplier)
    y2 = int(cy + fh * height_multiplier)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return image[y1:y2, x1:x2]

# =========================================================
# Visualization
# =========================================================
def draw_face_mesh(image, face_landmarks):
    annotated_image = image.copy()
    h, w = image.shape[:2]

    FACE_OVAL = [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
        (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
        (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54),
        (54, 103), (103, 67), (67, 109), (109, 10)
    ]
    LEFT_EYE = [
        (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),
        (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362)
    ]
    RIGHT_EYE = [
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
        (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)
    ]
    LIPS = [
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321),
        (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
        (267, 269), (269, 270), (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
        (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308), (78, 191), (191, 80),
        (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)
    ]

    for connections in [FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS]:
        for start_idx, end_idx in connections:
            if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
                start_point = (int(face_landmarks[start_idx].x * w), int(face_landmarks[start_idx].y * h))
                end_point = (int(face_landmarks[end_idx].x * w), int(face_landmarks[end_idx].y * h))
                cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 1)

    for lm in face_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

    return annotated_image


def draw_selected_landmarks(image, landmark_points, landmark_lines=None):
    out = image.copy()
    for name, point in landmark_points.items():
        x, y = int(point[0]), int(point[1])
        cv2.circle(out, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(out, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    if landmark_lines:
        for start_name, end_name in landmark_lines:
            if start_name in landmark_points and end_name in landmark_points:
                start_point = landmark_points[start_name]
                end_point = landmark_points[end_name]
                cv2.line(
                    out,
                    (int(start_point[0]), int(start_point[1])),
                    (int(end_point[0]), int(end_point[1])),
                    (255, 255, 0),
                    1,
                )
    return out

# =========================================================
# Human Rating Loader
# =========================================================
def load_human_ratings(human_rating_file):
    if human_rating_file is None:
        print("WARNING: No human rating file was provided. Human merge will be skipped.")
        return None, None

    if isinstance(human_rating_file, str) and human_rating_file.strip().lower() in {"none", "", "null"}:
        print("WARNING: Human rating file argument was empty. Human merge will be skipped.")
        return None, None

    print(f"\nHuman rating file argument: {human_rating_file}")
    print(f"Human rating file exists: {os.path.exists(human_rating_file)}")
    if not os.path.exists(human_rating_file):
        print(f"WARNING: Human rating file not found: {human_rating_file}")
        print("Human merge will be skipped.")
        return None, None

    df = read_table(human_rating_file)
    print(f"Human rating columns: {list(df.columns)}")

    image_col = find_column(df.columns, ["Images", "Image_Name", "Image-Name", "Image", "Filename", "File_Name"])
    if image_col is None:
        raise ValueError("Human rating file must contain an image column such as Images or Image_Name.")

    df["Image-Name"] = df[image_col].astype(str).apply(lambda x: normalize_path_string(x).split("/")[-1])
    df["Image-Stem"] = df["Image-Name"].apply(get_image_stem)
    df["Image-Key"] = df["Image-Name"].apply(get_image_key)
    df["Subject-ID"] = df["Image-Name"].apply(extract_subject_id)

    avg_col = find_column(df.columns, ["Average", "Human_Rating_Avg", "Human-Rating-Avg", "Avg", "Mean"])
    if avg_col is not None:
        df["Human-Rating-Avg"] = pd.to_numeric(df[avg_col], errors="coerce")
    else:
        user_cols = [c for c in df.columns if re.match(r"User\d+", str(c), flags=re.IGNORECASE)]
        if not user_cols:
            raise ValueError("Human rating file must contain Average or user rating columns like User1, User2...")
        df["Human-Rating-Avg"] = df[user_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    df["Human-Category"] = df["Human-Rating-Avg"].apply(classify_human_band)

    image_level = df[["Image-Name", "Image-Stem", "Image-Key", "Subject-ID", "Human-Rating-Avg", "Human-Category"]].copy()

    subject_level = (
        image_level.groupby("Subject-ID", as_index=False)
        .agg(
            Mean_Human_Rating=("Human-Rating-Avg", "mean"),
            Num_Human_Images=("Image-Name", "count")
        )
    )
    subject_level["Mean-Human-Rating"] = subject_level["Mean_Human_Rating"].round(4)
    subject_level["Human-Subject-Category"] = subject_level["Mean-Human-Rating"].apply(classify_human_band)
    subject_level = subject_level[["Subject-ID", "Mean-Human-Rating", "Num_Human_Images", "Human-Subject-Category"]]

    print("\nSample rows from human rating file after cleaning:")
    print(image_level.head(10).to_string(index=False))

    return image_level, subject_level

# =========================================================
# 20-Core Geometric Beauty Index
# =========================================================
def compute_twenty_core_data(pts):
    data = {}

    mid_x = pts[4][0]
    left_face_x = pts[234][0]
    right_face_x = pts[454][0]
    face_width = abs(right_face_x - left_face_x) + EPS

    def vector_closeness_score(actual, target, tau=0.12):
        a = np.array(actual, dtype=float)
        t = np.array(target, dtype=float)
        a_n = a / (a.mean() + EPS)
        t_n = t / (t.mean() + EPS)
        deviation = np.sqrt(np.mean((a_n - t_n) ** 2))
        score = 1.0 + 9.0 * math.exp(-0.5 * (deviation / tau) ** 2)
        return float(max(1.0, min(10.0, score)))

    def scalar_closeness_score(actual, target, tau=0.12):
        ratio = float(actual) / (float(target) + EPS)
        deviation = abs(ratio - 1.0)
        score = 1.0 + 9.0 * math.exp(-0.5 * (deviation / tau) ** 2)
        return float(max(1.0, min(10.0, score)))

    # ---- R1-R10: authentic Golden Ratio landmarks/formulas ----
    # (kept identical to the standalone golden-ratio script so the
    # "golden ratio" component means the same thing everywhere)
    eye_distance, golden_width, golden_height, right_eye_center, left_eye_center = golden_face_dimensions(pts)
    Oc = mean_point(right_eye_center, left_eye_center)

    Me = pts[152]                             # Menton / Gnathion approximation
    St = mean_point(pts[13], pts[14])         # Stomion
    Ls = pts[0]                               # Labiale superius
    Li = pts[17]                              # Labiale inferius

    Ch_l = pts[61]                            # Left cheilion
    Ch_r = pts[291]                           # Right cheilion
    Ch_mid = mean_point(Ch_l, Ch_r)

    Al_l = pts[49]                            # Left alare
    Al_r = pts[279]                           # Right alare
    Al_mid = mean_point(Al_l, Al_r)

    En_r = pts[133]                           # Right endocanthion
    En_l = pts[362]                           # Left endocanthion
    En_mid = mean_point(En_r, En_l)

    R7_height_top, R7_height_bottom = pts[10], pts[152]
    R7_width_left, R7_width_right = pts[234], pts[454]

    R8_biocular_left, R8_biocular_right = pts[33], pts[263]
    R8_intercheilion_left, R8_intercheilion_right = pts[61], pts[291]

    R9_nose_top, R9_nose_bottom = pts[168], pts[2]
    R9_interalar_left, R9_interalar_right = pts[102], pts[332]

    R10_ipd_left, R10_ipd_right = pts[468], pts[473]
    R10_intercanthal_left, R10_intercanthal_right = pts[133], pts[362]

    d_Oc_St = dist(Oc, St)
    d_St_Me = dist(St, Me)
    d_En_Al = dist(En_mid, Al_mid)
    d_Al_Gn = dist(Al_mid, Me)
    d_Gn_Ch = dist(Me, Ch_mid)
    d_Ch_En = dist(Ch_mid, En_mid)
    d_Ls_Sto = dist(Ls, St)
    d_Sto_Li = dist(St, Li)
    d_Intercheilion = dist(Ch_l, Ch_r)
    d_Intercanthal = dist(En_r, En_l)
    d_Interalar = dist(Al_l, Al_r)

    d_R7_height = dist(R7_height_top, R7_height_bottom)
    d_R7_width = dist(R7_width_left, R7_width_right)
    d_R8_biocular = dist(R8_biocular_left, R8_biocular_right)
    d_R8_intercheilion = dist(R8_intercheilion_left, R8_intercheilion_right)
    d_R9_nose = dist(R9_nose_top, R9_nose_bottom)
    d_R9_interalar = dist(R9_interalar_left, R9_interalar_right)
    d_R10_ipd = dist(R10_ipd_left, R10_ipd_right)
    d_R10_intercanthal = dist(R10_intercanthal_left, R10_intercanthal_right)

    def golden_ratio_score_1to10(d1, d2, phi=PHI, tau=TAU):
        # Same phi-oriented-ratio + Gaussian-decay formula as the standalone
        # golden-ratio script, rescaled to a true 1-10 range (floor 1, not 0)
        # so it combines consistently with the P/S scores below.
        phi_ratio = phi_oriented_ratio(d1, d2)
        deviation = abs(phi_ratio - phi)
        score_0_1 = math.exp(-0.5 * (deviation / tau) ** 2)
        return float(max(1.0, min(10.0, 1.0 + 9.0 * score_0_1)))

    data["R1-Score"] = golden_ratio_score_1to10(d_Oc_St, d_St_Me)
    data["R2-Score"] = golden_ratio_score_1to10(d_En_Al, d_Al_Gn)
    data["R3-Score"] = golden_ratio_score_1to10(d_Gn_Ch, d_Ch_En)
    data["R4-Score"] = golden_ratio_score_1to10(d_Ls_Sto, d_Sto_Li)
    data["R5-Score"] = golden_ratio_score_1to10(d_Intercheilion, d_Intercanthal)
    data["R6-Score"] = golden_ratio_score_1to10(d_Interalar, d_Intercheilion)
    data["R7-Score"] = golden_ratio_score_1to10(d_R7_height, d_R7_width)
    data["R8-Score"] = golden_ratio_score_1to10(d_R8_biocular, d_R8_intercheilion)
    data["R9-Score"] = golden_ratio_score_1to10(d_R9_nose, d_R9_interalar)
    data["R10-Score"] = golden_ratio_score_1to10(d_R10_ipd, d_R10_intercanthal)

    # Needed below for P1 (forehead-thirds proportion) and the landmark-points dict
    trichion = pts[10]
    glabella = pts[168]
    subnasale = pts[2]
    menton = pts[152]

    d1 = abs(glabella[1] - trichion[1])
    d2 = abs(subnasale[1] - glabella[1])
    d3 = abs(menton[1] - subnasale[1])
    P1_score = vector_closeness_score([d1, d2, d3], [1, 1, 1], tau=0.12)
    data["P1-Score"] = P1_score

    p2_indices = [234, 33, 133, 362, 263, 454]
    xs = [pts[i][0] for i in p2_indices]
    xs_sorted = sorted(xs)
    widths = [xs_sorted[i + 1] - xs_sorted[i] for i in range(len(xs_sorted) - 1)]
    P2_score = vector_closeness_score(widths, [1, 1, 1, 1, 1], tau=0.12)
    data["P2-Score"] = P2_score

    upper_top = pts[0]
    upper_bottom = pts[13]
    lower_top = pts[14]
    lower_bottom = pts[17]
    upper_thickness = abs(upper_bottom[1] - upper_top[1]) + EPS
    lower_thickness = abs(lower_bottom[1] - lower_top[1]) + EPS
    target_p3 = 1.0 / PHI
    P3_score = scalar_closeness_score(upper_thickness / lower_thickness, target_p3, tau=0.12)
    data["P3-Score"] = P3_score

    total_height = dist(pts[10], pts[152])
    bizygomatic = dist(pts[234], pts[454]) + EPS
    P4_score = scalar_closeness_score(total_height / bizygomatic, 1.35, tau=0.12)
    data["P4-Score"] = P4_score

    def symmetry_score(left_idx, right_idx):
        xl = pts[left_idx][0]
        xr = pts[right_idx][0]
        dl = abs(xl - mid_x)
        dr = abs(xr - mid_x)
        diff = abs(dl - dr)
        diff_norm = diff / face_width
        score = 1.0 + 9.0 * max(0.0, 1.0 - diff_norm)
        return float(max(1.0, min(10.0, score)))

    S1 = symmetry_score(362, 133)
    S2 = symmetry_score(263, 33)
    S3 = symmetry_score(334, 105)
    S4 = symmetry_score(358, 129)
    S5 = symmetry_score(291, 61)
    S6 = symmetry_score(397, 172)

    data["S1-Score"] = S1
    data["S2-Score"] = S2
    data["S3-Score"] = S3
    data["S4-Score"] = S4
    data["S5-Score"] = S5
    data["S6-Score"] = S6

    feature_scores = [
        data["R1-Score"], data["R2-Score"], data["R3-Score"], data["R4-Score"], data["R5-Score"],
        data["R6-Score"], data["R7-Score"], data["R8-Score"], data["R9-Score"], data["R10-Score"],
        P1_score, P2_score, P3_score, P4_score, S1, S2, S3, S4, S5, S6,
    ]

    # ---- Naive (equal-weight) average — kept for comparison / transparency ----
    naive_avg = float(np.mean(feature_scores)) if feature_scores else np.nan
    naive_avg_percent = (naive_avg - 1.0) / 9.0 * 100.0 if not np.isnan(naive_avg) else np.nan

    # ---- Calibrated, data-driven weighted combination ----
    # Weights learned via linear regression of these same 20 features (on this
    # 1-10 scale) against calibrated human ratings on a 100-subject / ~500-image
    # pilot set, validated with 5-fold cross-validation (out-of-sample r = 0.303,
    # vs r = 0.082 for the naive equal-weight average above). Replace these
    # constants if the model is re-fit on a larger or different rating set.
    REGRESSION_INTERCEPT = 18.456007595511004
    REGRESSION_WEIGHTS = {
        "R1-Score": 0.011782114111805162,
        "R2-Score": -0.14256705297041614,
        "R3-Score": -0.0007053322794589349,
        "R4-Score": 0.08457462063667795,
        "R5-Score": 0.4209450599782548,
        "R6-Score": -0.11801599250747193,
        "R7-Score": -0.3806309882693262,
        "R8-Score": -0.5284131538290862,
        "R9-Score": -0.007986563265850408,
        "R10-Score": -0.3579174641334388,
        "P1-Score": 0.0011635471467568792,
        "P2-Score": -0.04709919485878858,
        "P3-Score": 0.02666069870068014,
        "P4-Score": 0.1938806509209178,
        "S1-Score": 0.5167607451810547,
        "S2-Score": 0.03385273930153698,
        "S3-Score": -0.19877302613962797,
        "S4-Score": -1.2303790545670674,
        "S5-Score": -0.7191220053045739,
        "S6-Score": 0.7247256082244812,
    }
    named_scores = {
        "R1-Score": data["R1-Score"], "R2-Score": data["R2-Score"], "R3-Score": data["R3-Score"],
        "R4-Score": data["R4-Score"], "R5-Score": data["R5-Score"], "R6-Score": data["R6-Score"],
        "R7-Score": data["R7-Score"], "R8-Score": data["R8-Score"], "R9-Score": data["R9-Score"],
        "R10-Score": data["R10-Score"],
        "P1-Score": P1_score, "P2-Score": P2_score, "P3-Score": P3_score, "P4-Score": P4_score,
        "S1-Score": S1, "S2-Score": S2, "S3-Score": S3, "S4-Score": S4, "S5-Score": S5, "S6-Score": S6,
    }
    calibrated_raw = REGRESSION_INTERCEPT + sum(
        REGRESSION_WEIGHTS[name] * value for name, value in named_scores.items()
    )
    calibrated = float(max(1.0, min(10.0, calibrated_raw)))
    calibrated_percent = (calibrated - 1.0) / 9.0 * 100.0

    data["Naive-Average-Score-1to10"] = naive_avg
    data["Naive-Average-Score-Percent"] = naive_avg_percent

    data["Comprehensive-Geometric-Score-1to10"] = calibrated
    data["Combined-Geometric-Score-1to10"] = calibrated
    data["Combined-Geometric-Score-Percent"] = calibrated_percent

    landmark_points = {
        "Trichion": trichion,
        "Glabella": glabella,
        "Subnasale": subnasale,
        "Menton": menton,
        "Oc": Oc,
        "St": St,
        "Ls": Ls,
        "Li": Li,
        "ChL": Ch_l,
        "ChR": Ch_r,
        "AlL": Al_l,
        "AlR": Al_r,
        "EnL": En_l,
        "EnR": En_r,
        "LeftCheek": R7_width_left,
        "RightCheek": R7_width_right,
        "UpperLipTop": upper_top,
        "UpperLipBottom": upper_bottom,
        "LowerLipTop": lower_top,
        "LowerLipBottom": lower_bottom,
        "JawLeft": pts[397],
        "JawRight": pts[172],
    }

    landmark_lines = [
        ("Trichion", "Glabella"), ("Glabella", "Subnasale"), ("Subnasale", "Menton"),
        ("Oc", "St"), ("St", "Menton"),
        ("EnL", "EnR"), ("AlL", "AlR"),
        ("ChL", "ChR"),
        ("LeftCheek", "RightCheek"), ("UpperLipTop", "UpperLipBottom"), ("LowerLipTop", "LowerLipBottom"),
    ]

    return data, landmark_points, landmark_lines

# =========================================================
# Main Pipeline
# =========================================================
def process_images(image_dir, human_rating_file=None, output_prefix="Golden_20Core", max_images=None):
    rows = []
    os.makedirs(VIZ_DIR, exist_ok=True)
    os.makedirs(LANDMARK_DIR, exist_ok=True)
    os.makedirs(CROP_DIR, exist_ok=True)

    face_landmarker = create_face_landmarker()
    if face_landmarker is None:
        print("Face detector unavailable; writing a minimal placeholder output file instead.")
        sys.stdout.flush()
        placeholder_df = pd.DataFrame([{
            "Subject-ID": "UNKNOWN",
            "Image-Name": "placeholder",
            "Combined-Geometric-Score-Percent": np.nan,
            "Combined-Geometric-Score-1to10": np.nan,
            "Human-Rating-Avg": np.nan,
            "Human-Category": np.nan,
            "AI-vs-Human-Diff": np.nan,
        }])
        placeholder_df.to_csv(f"{output_prefix}_image_level_output.csv", index=False)
        placeholder_df.to_excel(f"{output_prefix}_image_level_output.xlsx", index=False)
        return

    print(f"Starting processing in: {image_dir}")
    print(f"Human rating file argument: {human_rating_file}")
    print(f"sys.argv: {sys.argv}")

    def _numeric_sort_key(name):
        stem = get_image_stem(name)
        nums = re.findall(r"(\d+)", stem)
        if nums:
            return tuple(int(n) for n in nums), stem
        return (float("inf"),), stem

    image_files = sorted(
        [name for name in os.listdir(image_dir) if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))],
        key=_numeric_sort_key,
    )
    if max_images is not None:
        image_files = image_files[:max_images]
        print(f"Processing first {len(image_files)} image(s) only")

    for idx, img_name in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_name}")
        sys.stdout.flush()

        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_name}")
            continue
        print(f"  Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
        sys.stdout.flush()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if hasattr(face_landmarker, "detect"):
            from mediapipe.tasks.python.vision.core.image import Image as MPImage
            try:
                mp_image = MPImage.create_from_file(img_path)
            except Exception:
                try:
                    mp_image = MPImage(image_format=MPImage.ImageFormat.SRGB, data=img_rgb)
                except Exception as fallback_e:
                    print(f"Cannot load image (file and array failed): {img_name}")
                    continue

            result1 = face_landmarker.detect(mp_image)
            print("  Detection complete")
            sys.stdout.flush()

            if not (hasattr(result1, "face_landmarks") and result1.face_landmarks):
                print(f"No face detected: {img_name}")
                continue

            first = result1.face_landmarks[0]
            lm1 = first.landmark if hasattr(first, "landmark") else first
        else:
            result1 = face_landmarker.process(img_rgb)
            if not result1.multi_face_landmarks:
                print(f"No face detected: {img_name}")
                continue

            lm1 = result1.multi_face_landmarks[0].landmark

        face_crop = golden_crop(img, lm1)
        if face_crop.size == 0:
            print(f"Empty crop: {img_name}")
            continue

        cv2.imwrite(os.path.join(CROP_DIR, img_name), face_crop)

        crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        if hasattr(face_landmarker, "detect"):
            from mediapipe.tasks.python.vision.core.image import Image as MPImage
            crop_path = os.path.join(CROP_DIR, img_name)
            result2 = face_landmarker.detect(MPImage.create_from_file(crop_path))
            print("  Crop detection complete")
            sys.stdout.flush()

            if not (hasattr(result2, "face_landmarks") and result2.face_landmarks):
                print(f"Crop detection failed: {img_name}")
                continue

            first2 = result2.face_landmarks[0]
            lm2 = first2.landmark if hasattr(first2, "landmark") else first2
        else:
            result2 = face_landmarker.process(crop_rgb)
            if not result2.multi_face_landmarks:
                print(f"Crop detection failed: {img_name}")
                continue

            lm2 = result2.multi_face_landmarks[0].landmark

        visual_proof = draw_face_mesh(face_crop, lm2)
        cv2.imwrite(os.path.join(VIZ_DIR, img_name), visual_proof)

        h, w = face_crop.shape[:2]
        pts = {i: (lm2[i].x * w, lm2[i].y * h) for i in range(len(lm2))}

        image_data, landmark_points, landmark_lines = compute_twenty_core_data(pts)
        image_data["Subject-ID"] = extract_subject_id(img_name)
        image_data["Image-Name"] = img_name
        image_data["Image-Stem"] = get_image_stem(img_name)
        image_data["Image-Key"] = get_image_key(img_name)
        image_data["Image-Path"] = img_path

        selected_landmarks_img = draw_selected_landmarks(face_crop, landmark_points, landmark_lines)
        cv2.imwrite(os.path.join(LANDMARK_DIR, img_name), selected_landmarks_img)

        rows.append(image_data)
        print(f"Processed: {img_name}")

    if not rows:
        print("No valid images were processed.")
        return

    technical_df = pd.DataFrame(rows)

    human_image_df, human_subject_df = load_human_ratings(human_rating_file)

    if human_image_df is not None:
        pre_merge = technical_df.copy()
        technical_df = technical_df.merge(
            human_image_df[["Subject-ID", "Image-Key", "Human-Rating-Avg", "Human-Category"]],
            on=["Subject-ID", "Image-Key"],
            how="left"
        )

        matched = int(technical_df["Human-Rating-Avg"].notna().sum())
        unmatched = int(technical_df["Human-Rating-Avg"].isna().sum())
        print(f"\nHuman ratings merge: matched={matched}, unmatched={unmatched}")

        if unmatched > 0:
            debug_merge_df = pre_merge.merge(
                human_image_df[["Subject-ID", "Image-Key", "Image-Name", "Human-Rating-Avg", "Human-Category"]],
                on=["Subject-ID", "Image-Key"],
                how="outer",
                indicator=True,
                suffixes=("-image", "-human")
            )
            debug_csv = f"{output_prefix}_human_merge_debug.csv"
            debug_merge_df.to_csv(debug_csv, index=False)
            print(f"Debug merge CSV written to: {debug_csv}")

        technical_df["AI-vs-Human-Diff"] = technical_df["Combined-Geometric-Score-1to10"] - technical_df["Human-Rating-Avg"]
    else:
        print("\nHuman merge skipped. Comparison columns will be empty.")
        technical_df["Human-Rating-Avg"] = np.nan
        technical_df["Human-Category"] = np.nan
        technical_df["AI-vs-Human-Diff"] = np.nan

    golden_ratio_cols = [f"R{i}-Score" for i in range(1, 11)]
    proportion_cols = [f"P{i}-Score" for i in range(1, 5)]
    symmetry_cols = [f"S{i}-Score" for i in range(1, 7)]
    feature_score_cols = golden_ratio_cols + proportion_cols + symmetry_cols

    image_cols = [
        "Subject-ID",
        "Image-Name",
        "Comprehensive-Geometric-Score-1to10",
        "Combined-Geometric-Score-1to10",
        "Combined-Geometric-Score-Percent",
        "Human-Rating-Avg",
        "Human-Category",
        "AI-vs-Human-Diff",
    ] + feature_score_cols
    image_df = technical_df[[c for c in image_cols if c in technical_df.columns]].copy()

    subject_df = (
        technical_df.groupby("Subject-ID", as_index=False)
        .agg(
            Num_Images=("Image-Name", "count"),
            Mean_Combined_Geometric_Score_Percent=("Combined-Geometric-Score-Percent", "mean"),
            Mean_Combined_Geometric_Score_1to10=("Combined-Geometric-Score-1to10", "mean"),
            Mean_Human_Rating=("Human-Rating-Avg", "mean"),
            Mean_AI_vs_Human_Diff=("AI-vs-Human-Diff", "mean"),
        )
    )

    subject_df = subject_df.rename(columns={
        "Num_Images": "Num-Images",
        "Mean_Combined_Geometric_Score_Percent": "Mean-Combined-Geometric-Score-Percent",
        "Mean_Combined_Geometric_Score_1to10": "Mean-Combined-Geometric-Score-1to10",
        "Mean_Human_Rating": "Mean-Human-Rating",
        "Mean_AI_vs_Human_Diff": "Mean-AI-vs-Human-Diff",
    })
    subject_df["Human-Subject-Category"] = subject_df["Mean-Human-Rating"].apply(classify_human_band)

    for df in [image_df, subject_df, technical_df]:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].round(4)

    image_csv = f"{output_prefix}_image_level_output.csv"
    image_xlsx = f"{output_prefix}_image_level_output.xlsx"
    subject_csv = f"{output_prefix}_subject_level_summary.csv"
    subject_xlsx = f"{output_prefix}_subject_level_summary.xlsx"
    technical_csv = f"{output_prefix}_technical_image_output.csv"
    technical_xlsx = f"{output_prefix}_technical_image_output.xlsx"

    def write_excel_if_possible(df, path):
        try:
            df.to_excel(path, index=False)
        except PermissionError as exc:
            print(f"Warning: could not write Excel file {path}: {exc}")
        except Exception as exc:
            print(f"Warning: could not write Excel file {path}: {exc}")

    image_df.to_csv(image_csv, index=False)
    write_excel_if_possible(image_df, image_xlsx)

    subject_df.to_csv(subject_csv, index=False)
    write_excel_if_possible(subject_df, subject_xlsx)

    technical_df.to_csv(technical_csv, index=False)
    write_excel_if_possible(technical_df, technical_xlsx)

    print("\nSUCCESS")
    print(f"Image-level CSV      : {image_csv}")
    print(f"Image-level Excel    : {image_xlsx}")
    print(f"Subject-level CSV    : {subject_csv}")
    print(f"Subject-level Excel  : {subject_xlsx}")
    print(f"Technical CSV        : {technical_csv}")
    print(f"Technical Excel      : {technical_xlsx}")
    print(f"Face mesh proofs     : {VIZ_DIR}/")
    print(f"Key landmark proofs  : {LANDMARK_DIR}/")
    print(f"Cropped faces        : {CROP_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 20-core geometric beauty index facial analysis workflow")
    parser.add_argument("folder", nargs="?", default="image")
    parser.add_argument("human_csv", nargs="?", default=None)
    parser.add_argument("output_prefix", nargs="?", default="Golden_20Core")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images")
    parser.add_argument("--max-images", type=int, default=None, help="Alias for --limit")
    args = parser.parse_args()

    print(f"sys.argv: {sys.argv}")
    folder = args.folder
    human_csv = args.human_csv
    output_prefix = args.output_prefix
    max_images = args.max_images if args.max_images is not None else args.limit

    if isinstance(human_csv, str) and human_csv.strip().lower() in {"none", "", "null"}:
        human_csv = None

    if not os.path.exists(folder):
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    process_images(folder, human_csv, output_prefix, max_images=max_images)
