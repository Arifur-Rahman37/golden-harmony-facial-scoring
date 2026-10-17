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

# =========================================================
# Data-driven regression weights for the Golden Ratio score
# =========================================================
# Learned via linear regression of the 10 R1-R10 Phi-Closeness scores
# (0-1 scale, as computed by phi_closeness_score() below) against
# calibrated human ratings on the same 100-subject / ~498-image pilot
# set used for the Combined score. Validated with 5-fold cross-validation:
#   naive (equal-weight) score  -> out-of-sample r = 0.082 (not significant)
#   regression-weighted score   -> out-of-sample r = 0.322 (p < 0.0001)
# Replace these constants if the model is re-fit on a larger/different set.
GOLDEN_REGRESSION_INTERCEPT = 8.003267355371463
GOLDEN_REGRESSION_WEIGHTS = {
    "R1-Phi-Closeness-Score": 0.21408329144236513,
    "R2-Phi-Closeness-Score": -1.0633766717073259,
    "R3-Phi-Closeness-Score": -0.15509369786875835,
    "R4-Phi-Closeness-Score": 0.9191918819187049,
    "R5-Phi-Closeness-Score": 3.4805046376691604,
    "R6-Phi-Closeness-Score": -1.0777977048556122,
    "R7-Phi-Closeness-Score": -0.3016530532623649,
    "R8-Phi-Closeness-Score": -4.426251293211048,
    "R9-Phi-Closeness-Score": -0.30417277194842335,
    "R10-Phi-Closeness-Score": -2.9943963433738743,
}

# Output folders
VIZ_DIR = "viz_output"
LANDMARK_DIR = "landmark_output"
CROP_DIR = "crop_output"

# =========================================================
# MediaPipe Init
# =========================================================
def create_face_landmarker(model_path=None):
    print("Initializing face landmark detector...")
    sys.stdout.flush()
    # Prefer the MediaPipe Tasks FaceLandmarker when available.
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
    """
    Converts any pair of distances into a larger-to-smaller ratio.
    This avoids 1.618 vs 0.618 orientation confusion.
    """
    larger = max(d1, d2)
    smaller = min(d1, d2)
    return larger / (smaller + EPS)


def phi_closeness_score(ratio_phi, phi=PHI, tau=TAU):
    """
    Computes facial harmony score using a Gaussian decay function.
    Formula: exp(-0.5 * ((Ratio - 1.618) / tau) ** 2)
    """
    deviation = abs(ratio_phi - phi)
    score = math.exp(-0.5 * (deviation / tau) ** 2)
    return max(0.0, min(1.0, score))


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
    """
    Examples:
    p4_img1.jpg -> P4
    C:\\folder\\p4_img1.jpg -> P4
    """
    s = normalize_path_string(filename)
    base = s.split("/")[-1]
    stem, _ = os.path.splitext(base)
    match = re.search(r"(p\d+)", stem, flags=re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def get_image_stem(filename):
    """Filename without extension, lowercase."""
    s = normalize_path_string(filename)
    base = s.split("/")[-1]
    stem, _ = os.path.splitext(base)
    return stem.strip().lower()


def get_image_key(filename):
    """
    Robust merge key that ignores extension, spaces, underscores, and dashes.
    Example: P4_img1.jpg, p4-img1.png -> p4img1
    """
    stem = get_image_stem(filename)
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def find_column(columns, candidates):
    """Find a column name by comparing normalized forms."""
    norm_map = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if key in norm_map:
            return norm_map[key]
    return None

# =========================================================
# Golden Crop Helpers
# =========================================================
def golden_face_dimensions(pts):
    right_eye_center = mean_point(pts[33], pts[133])
    left_eye_center = mean_point(pts[362], pts[263])
    eye_distance = dist(right_eye_center, left_eye_center)

    golden_width = 1.865 * eye_distance
    golden_height = 3.018 * eye_distance

    return eye_distance, golden_width, golden_height, right_eye_center, left_eye_center


def estimate_forehead(eye_center, golden_height):
    # Approximate trichion/hairline for cropping only.
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
    """Draw full face mesh for visual proof."""
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
    """Draw selected landmarks and optional connecting lines for ratio proof."""
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
    """
    Supports CSV/XLSX formats with one of these image columns:
    Images, Image_Name, Image-Name, Image

    Human score is read from Average/Human_Rating_Avg if present,
    otherwise calculated from User1, User2, ... columns.
    """
    if human_rating_file is None:
        print("WARNING: No human rating file was provided. Human merge will be skipped.")
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
# 10 Core Ratio Computation
# =========================================================
def compute_ten_core_data(pts):
    data = {}

    # Base geometry
    eye_distance, golden_width, golden_height, right_eye_center, left_eye_center = golden_face_dimensions(pts)
    Oc = mean_point(right_eye_center, left_eye_center)

    # Landmark mapping
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

    # New pure golden ratio landmarks
    R7_height_top = pts[10]
    R7_height_bottom = pts[152]
    R7_width_left = pts[234]
    R7_width_right = pts[454]

    R8_biocular_left = pts[33]
    R8_biocular_right = pts[263]
    R8_intercheilion_left = pts[61]
    R8_intercheilion_right = pts[291]

    R9_nose_top = pts[168]
    R9_nose_bottom = pts[2]
    R9_interalar_left = pts[102]
    R9_interalar_right = pts[332]

    R10_ipd_left = pts[468]
    R10_ipd_right = pts[473]
    R10_intercanthal_left = pts[133]
    R10_intercanthal_right = pts[362]

    landmark_points = {
        "Oc": Oc, "St": St, "Me/Gn": Me,
        "EnL": En_l, "EnR": En_r,
        "AlL": Al_l, "AlR": Al_r,
        "ChL": Ch_l, "ChR": Ch_r,
        "Ls": Ls, "Li": Li,
        "R7H1": R7_height_top, "R7H2": R7_height_bottom,
        "R7W1": R7_width_left, "R7W2": R7_width_right,
        "R8B1": R8_biocular_left, "R8B2": R8_biocular_right,
        "R8I1": R8_intercheilion_left, "R8I2": R8_intercheilion_right,
        "R9N1": R9_nose_top, "R9N2": R9_nose_bottom,
        "R9A1": R9_interalar_left, "R9A2": R9_interalar_right,
        "R10P1": R10_ipd_left, "R10P2": R10_ipd_right,
        "R10C1": R10_intercanthal_left, "R10C2": R10_intercanthal_right,
    }

    landmark_lines = [
        ("Oc", "St"), ("St", "Me/Gn"), ("EnL", "EnR"), ("AlL", "AlR"),
        ("ChL", "ChR"), ("Ls", "St"), ("St", "Li"),
        ("R7H1", "R7H2"), ("R7W1", "R7W2"),
        ("R8B1", "R8B2"), ("R8I1", "R8I2"),
        ("R9N1", "R9N2"), ("R9A1", "R9A2"),
        ("R10P1", "R10P2"), ("R10C1", "R10C2"),
    ]

    # Distances needed for the 10 core ratios
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

    core_scores = []

    def add_core_ratio(core_id, name, d1_label, d1, d2_label, d2):
        original_ratio = safe_ratio(d1, d2)
        phi_ratio = phi_oriented_ratio(d1, d2)
        score = phi_closeness_score(phi_ratio)

        data[f"{core_id}-Name"] = name
        data[f"{core_id}-Distance-{d1_label}"] = d1
        data[f"{core_id}-Distance-{d2_label}"] = d2
        data[f"{core_id}-Original-Ratio"] = original_ratio
        data[f"{core_id}-Phi-Oriented-Ratio"] = phi_ratio
        data[f"{core_id}-Target"] = PHI
        data[f"{core_id}-Phi-Closeness-Score"] = score

        core_scores.append(score)

    ratio_definitions = [
        ("R1", "Oc-St / St-Me", "Oc-St", d_Oc_St, "St-Me", d_St_Me),
        ("R2", "En-Al / Al-Gn", "En-Al", d_En_Al, "Al-Gn", d_Al_Gn),
        ("R3", "Gn-Ch / Ch-En", "Gn-Ch", d_Gn_Ch, "Ch-En", d_Ch_En),
        ("R4", "Ls-Sto / Sto-Li", "Ls-Sto", d_Ls_Sto, "Sto-Li", d_Sto_Li),
        ("R5", "Intercheilion / Intercanthal", "Intercheilion", d_Intercheilion, "Intercanthal", d_Intercanthal),
        ("R6", "Interalar / Intercheilion", "Interalar", d_Interalar, "Intercheilion", d_Intercheilion),
        ("R7", "Total Facial Height / Bizygomatic Width", "Total-Facial-Height", d_R7_height, "Bizygomatic-Width", d_R7_width),
        ("R8", "Biocular Width / Intercheilion Width", "Biocular-Width", d_R8_biocular, "Intercheilion-Width", d_R8_intercheilion),
        ("R9", "Nose Height / Interalar Width", "Nose-Height", d_R9_nose, "Interalar-Width", d_R9_interalar),
        ("R10", "Interpupillary Distance / Intercanthal Distance", "Interpupillary-Distance", d_R10_ipd, "Intercanthal-Distance", d_R10_intercanthal),
    ]

    for ratio in ratio_definitions:
        add_core_ratio(*ratio)

    # ---- Naive (equal-weight) average — kept for comparison / transparency ----
    if core_scores:
        golden_raw = float(np.mean(core_scores))
    else:
        golden_raw = np.nan
    naive_rating_1to10 = golden_raw * 10.0 if not np.isnan(golden_raw) else np.nan

    # ---- Calibrated, data-driven weighted combination ----
    # Weights learned via linear regression of these same 10 R1-R10
    # Phi-Closeness-Score features (0-1 scale, as computed above) against
    # calibrated human ratings on the same 100-subject / ~500-image pilot
    # set used for the Combined module, validated with 5-fold
    # cross-validation (out-of-sample r = 0.322, vs r = 0.082 for the naive
    # equal-weight average above). Replace these constants if the model is
    # re-fit on a larger or different rating set.
    REGRESSION_INTERCEPT = 8.003267355371463
    REGRESSION_WEIGHTS = {
        "R1": 0.21408329144236513,
        "R2": -1.0633766717073259,
        "R3": -0.15509369786875835,
        "R4": 0.9191918819187049,
        "R5": 3.4805046376691604,
        "R6": -1.0777977048556122,
        "R7": -0.3016530532623649,
        "R8": -4.426251293211048,
        "R9": -0.30417277194842335,
        "R10": -2.9943963433738743,
    }

    if core_scores:
        named_scores = {
            ratio_definitions[i][0]: core_scores[i] for i in range(len(ratio_definitions))
        }
        calibrated_raw = REGRESSION_INTERCEPT + sum(
            REGRESSION_WEIGHTS[name] * value for name, value in named_scores.items()
        )
        calibrated = float(max(1.0, min(10.0, calibrated_raw)))
    else:
        calibrated = np.nan
    calibrated_percent = (calibrated - 1.0) / 9.0 * 100.0 if not np.isnan(calibrated) else np.nan

    data["Naive-Golden-Raw"] = golden_raw
    data["Naive-Golden-Rating-1to10"] = naive_rating_1to10

    data["Golden-Raw"] = golden_raw
    data["Golden-Rating-1to10"] = calibrated
    data["Golden-Score-Percent"] = calibrated_percent
    data["Core-Ratio-Count"] = len(ratio_definitions)

    return data, landmark_points, landmark_lines

# =========================================================
# Main Pipeline
# =========================================================
def process_images(image_dir, human_rating_file=None, output_prefix="Golden_10Core", max_images=None):
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
            "Golden-Score-Percent": np.nan,
            "Golden-Raw": np.nan,
            "Golden-Rating-1to10": np.nan,
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

    image_files = sorted(
        [name for name in os.listdir(image_dir) if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
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

        # PASS 1: Detection on full image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Support both MediaPipe Tasks (FaceLandmarker.detect) and
        # Solutions FaceMesh (face_landmarker.process) APIs.
        if hasattr(face_landmarker, "detect"):
            from mediapipe.tasks.python.vision.core.image import Image as MPImage
            # Try file-based Image creation first, fallback to array-based if it fails
            try:
                mp_image = MPImage.create_from_file(img_path)
            except Exception as e:
                # Fallback to array-based creation for problematic files
                try:
                    mp_image = MPImage(image_format=MPImage.ImageFormat.SRGB, data=img_rgb)
                except Exception as fallback_e:
                    print(f"Cannot load image (file and array failed): {img_name}")
                    continue
            
            result1 = face_landmarker.detect(mp_image)
            print("  Detection complete")
            sys.stdout.flush()

            # Tasks API returns `face_landmarks` (list) or similar structures.
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

        # Crop
        face_crop = golden_crop(img, lm1)
        if face_crop.size == 0:
            print(f"Empty crop: {img_name}")
            continue

        cv2.imwrite(os.path.join(CROP_DIR, img_name), face_crop)

        # PASS 2: Detection on crop
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

        # Visual proof: full face mesh
        visual_proof = draw_face_mesh(face_crop, lm2)
        cv2.imwrite(os.path.join(VIZ_DIR, img_name), visual_proof)

        # Landmark extraction in pixel coordinates
        h, w = face_crop.shape[:2]
        pts = {i: (lm2[i].x * w, lm2[i].y * h) for i in range(len(lm2))}

        # Compute final 10-ratio data
        image_data, landmark_points, landmark_lines = compute_ten_core_data(pts)
        image_data["Subject-ID"] = extract_subject_id(img_name)
        image_data["Image-Name"] = img_name
        image_data["Image-Stem"] = get_image_stem(img_name)
        image_data["Image-Key"] = get_image_key(img_name)
        image_data["Image-Path"] = img_path

        # Visual proof: selected core landmarks
        selected_landmarks_img = draw_selected_landmarks(face_crop, landmark_points, landmark_lines)
        cv2.imwrite(os.path.join(LANDMARK_DIR, img_name), selected_landmarks_img)

        rows.append(image_data)
        print(f"Processed: {img_name}")

    if not rows:
        print("No valid images were processed.")
        return

    technical_df = pd.DataFrame(rows)

    # Human merge
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

        technical_df["AI-vs-Human-Diff"] = technical_df["Golden-Rating-1to10"] - technical_df["Human-Rating-Avg"]
    else:
        print("\nHuman merge skipped. Comparison columns will be empty.")
        technical_df["Human-Rating-Avg"] = np.nan
        technical_df["Human-Category"] = np.nan
        technical_df["AI-vs-Human-Diff"] = np.nan

    # Main image-level clean output
    ratio_score_cols = [f"R{i}-Phi-Closeness-Score" for i in range(1, 11)]
    ratio_ratio_cols = [f"R{i}-Phi-Oriented-Ratio" for i in range(1, 11)]
    ratio_original_cols = [f"R{i}-Original-Ratio" for i in range(1, 11)]
    image_cols = [
        "Subject-ID",
        "Image-Name",
        "Golden-Score-Percent",
        "Golden-Raw",
        "Golden-Rating-1to10",
        "Naive-Golden-Rating-1to10",
        "Human-Rating-Avg",
        "Human-Category",
        "AI-vs-Human-Diff",
    ] + ratio_score_cols + ratio_ratio_cols + ratio_original_cols
    image_df = technical_df[[c for c in image_cols if c in technical_df.columns]].copy()

    # Subject-level clean output
    subject_df = (
        technical_df.groupby("Subject-ID", as_index=False)
        .agg(
            Num_Images=("Image-Name", "count"),
            Mean_Golden_Score_Percent=("Golden-Score-Percent", "mean"),
            Mean_Golden_Raw=("Golden-Raw", "mean"),
            Mean_Golden_Rating=("Golden-Rating-1to10", "mean"),
            Mean_Naive_Golden_Rating=("Naive-Golden-Rating-1to10", "mean"),
            Mean_Human_Rating=("Human-Rating-Avg", "mean"),
            Mean_AI_vs_Human_Diff=("AI-vs-Human-Diff", "mean"),
        )
    )

    # Humanized column names
    subject_df = subject_df.rename(columns={
        "Num_Images": "Num-Images",
        "Mean_Golden_Score_Percent": "Mean-Golden-Score-Percent",
        "Mean_Golden_Raw": "Mean-Golden-Raw",
        "Mean_Golden_Rating": "Mean-Golden-Rating",
        "Mean_Naive_Golden_Rating": "Mean-Naive-Golden-Rating",
        "Mean_Human_Rating": "Mean-Human-Rating",
        "Mean_AI_vs_Human_Diff": "Mean-AI-vs-Human-Diff",
    })

    if human_subject_df is not None:
        # Keep subject human category if available
        subject_df = subject_df.merge(
            human_subject_df[["Subject-ID", "Human-Subject-Category"]],
            on="Subject-ID",
            how="left"
        )

    # Round numeric columns for readability
    for df in [image_df, subject_df, technical_df]:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].round(4)

    # Save outputs
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

# =========================================================
# Entry
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 10-core golden ratio facial analysis workflow")
    parser.add_argument("folder", nargs="?", default="image")
    parser.add_argument("human_csv", nargs="?", default=None)
    parser.add_argument("output_prefix", nargs="?", default="Golden_10Core")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images")
    parser.add_argument("--max-images", type=int, default=None, help="Alias for --limit")
    args = parser.parse_args()

    print(f"sys.argv: {sys.argv}")
    folder = args.folder
    human_csv = args.human_csv
    output_prefix = args.output_prefix
    max_images = args.max_images if args.max_images is not None else args.limit

    if not os.path.exists(folder):
        print(f"ERROR: Folder not found: {folder}")
        sys.exit(1)

    process_images(folder, human_csv, output_prefix, max_images=max_images)
