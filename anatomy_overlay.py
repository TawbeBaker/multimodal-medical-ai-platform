"""
anatomy_overlay.py
------------------
OpenCV-based anatomy overlay system for chest X-ray visualization.

All coordinates are expressed as fractions of image dimensions so they
scale correctly regardless of input resolution.

Layer pipeline (apply_overlays):
  1. lung_fields      — translucent zone fill + outline
  2. heart            — cardiac silhouette ellipse
  3. carina           — trachea midline + carina bifurcation
  4. ribs             — posterior rib arcs with rib numbers 1-10
  5. diaphragm        — hemidiaphragm dome curves
  6. abnormalities    — AI-prediction zone highlights

Future roadmap:
  - Replace rule-based zones with real segmentation masks
    (e.g. TotalSegmentator, nnU-Net chest model)
  - Per-zone confidence heat strips instead of flat rectangles
  - DICOM window/level integration for pre-processing before overlay
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Color palette  (RGB — matches PIL / Streamlit expectation)
# ---------------------------------------------------------------------------
COLORS = {
    "lung_fields":  (70,  130, 180),   # steel blue
    "ribs":         (255, 215,   0),   # gold
    "diaphragm":    ( 50, 205,  50),   # lime green
    "heart":        (220,  80,  80),   # soft red
    "carina":       (255, 165,   0),   # orange
    "abnormality":  (255,  80,  80),   # warning red
}


def _s(frac: float, dim: int) -> int:
    """Scale a fraction to an integer pixel coordinate."""
    return int(frac * dim)


# ---------------------------------------------------------------------------
# Individual layer functions
# Each function accepts and returns an RGB uint8 numpy array.
# ---------------------------------------------------------------------------

def draw_lung_fields(img: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Semi-transparent fill + outline for right and left lung fields."""
    h, w = img.shape[:2]
    c = COLORS["lung_fields"]
    overlay = img.copy()

    # Approximate polygons — viewer orientation (R = patient right = image left)
    right_lung = np.array([
        [_s(0.07, w), _s(0.17, h)],
        [_s(0.44, w), _s(0.17, h)],
        [_s(0.47, w), _s(0.40, h)],
        [_s(0.46, w), _s(0.71, h)],
        [_s(0.07, w), _s(0.72, h)],
    ], dtype=np.int32)

    left_lung = np.array([
        [_s(0.56, w), _s(0.17, h)],
        [_s(0.93, w), _s(0.17, h)],
        [_s(0.93, w), _s(0.72, h)],
        [_s(0.54, w), _s(0.71, h)],
        [_s(0.53, w), _s(0.40, h)],
    ], dtype=np.int32)

    cv2.fillPoly(overlay, [right_lung], c)
    cv2.fillPoly(overlay, [left_lung], c)
    result = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    cv2.polylines(result, [right_lung], True, c, 1)
    cv2.polylines(result, [left_lung],  True, c, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "R", (_s(0.21, w), _s(0.75, h)), font, 0.45, c, 1)
    cv2.putText(result, "L", (_s(0.71, w), _s(0.75, h)), font, 0.45, c, 1)
    return result


def draw_ribs(img: np.ndarray) -> np.ndarray:
    """
    Draw posterior rib arcs (ribs 1-10) with lateral labels.
    Each rib is a slight downward-curving polyline from spine to lateral chest wall.
    """
    h, w = img.shape[:2]
    c = COLORS["ribs"]
    result = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    rib_y_fracs = [0.17 + i * 0.056 for i in range(10)]

    for i, y_frac in enumerate(rib_y_fracs):
        rib_num = i + 1
        base_y = int(y_frac * h)

        # Right side posterior rib (image-left)
        pts_r = []
        x_start_r, x_end_r = _s(0.07, w), _s(0.44, w)
        for x in range(x_start_r, x_end_r):
            t = (x - x_start_r) / max(x_end_r - x_start_r, 1)
            y_off = int(h * 0.010 * t)
            pts_r.append([x, base_y + y_off])
        cv2.polylines(result, [np.array(pts_r, dtype=np.int32)], False, c, 1)

        # Left side posterior rib (image-right)
        pts_l = []
        x_start_l, x_end_l = _s(0.56, w), _s(0.93, w)
        for x in range(x_start_l, x_end_l):
            t = (x_end_l - x) / max(x_end_l - x_start_l, 1)
            y_off = int(h * 0.010 * t)
            pts_l.append([x, base_y + y_off])
        cv2.polylines(result, [np.array(pts_l, dtype=np.int32)], False, c, 1)

        # Rib number labels on both sides
        cv2.putText(result, str(rib_num), (_s(0.01, w), base_y + 4), font, 0.30, c, 1)
        cv2.putText(result, str(rib_num), (_s(0.94, w), base_y + 4), font, 0.30, c, 1)

    return result


def draw_diaphragm(img: np.ndarray) -> np.ndarray:
    """Draw hemidiaphragm dome curves (right slightly higher than left)."""
    h, w = img.shape[:2]
    c = COLORS["diaphragm"]
    result = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _dome(x_lo: float, x_hi: float, apex_x: float,
               apex_y: float, base_y: float) -> np.ndarray:
        pts = []
        for x in range(_s(x_lo, w), _s(x_hi, w)):
            t = (x - _s(apex_x, w)) / max(
                abs(_s(x_hi, w) - _s(apex_x, w)),
                abs(_s(x_lo, w) - _s(apex_x, w)), 1
            )
            y = int(_s(base_y, h) + (_s(apex_y, h) - _s(base_y, h)) * (1 - t ** 2))
            pts.append([x, max(0, y)])
        return np.array(pts, dtype=np.int32)

    r_pts = _dome(0.07, 0.46, 0.25, 0.67, 0.73)
    l_pts = _dome(0.54, 0.93, 0.75, 0.69, 0.75)

    cv2.polylines(result, [r_pts], False, c, 2)
    cv2.polylines(result, [l_pts], False, c, 2)
    cv2.putText(result, "R Diaphragm", (_s(0.07, w), _s(0.78, h)), font, 0.30, c, 1)
    cv2.putText(result, "L Diaphragm", (_s(0.55, w), _s(0.80, h)), font, 0.30, c, 1)
    return result


def draw_heart(img: np.ndarray, alpha: float = 0.12) -> np.ndarray:
    """Semi-transparent cardiac silhouette ellipse."""
    h, w = img.shape[:2]
    c = COLORS["heart"]
    overlay = img.copy()

    center = (_s(0.44, w), _s(0.49, h))
    axes   = (_s(0.14, w), _s(0.18, h))

    cv2.ellipse(overlay, center, axes, 0, 0, 360, c, -1)
    result = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    cv2.ellipse(result, center, axes, 0, 0, 360, c, 1)
    cv2.putText(result, "Cardiac silhouette",
                (_s(0.29, w), _s(0.71, h)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, c, 1)
    return result


def draw_carina(img: np.ndarray) -> np.ndarray:
    """Draw trachea midline and carina bifurcation."""
    h, w = img.shape[:2]
    c = COLORS["carina"]
    result = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    cx = _s(0.50, w)
    trachea_top = _s(0.10, h)
    carina_y    = _s(0.42, h)

    # Trachea
    cv2.line(result, (cx, trachea_top), (cx, carina_y), c, 2)

    # Carina bifurcation to mainstem bronchi
    cv2.line(result, (cx, carina_y), (_s(0.37, w), _s(0.51, h)), c, 2)
    cv2.line(result, (cx, carina_y), (_s(0.63, w), _s(0.51, h)), c, 2)
    cv2.circle(result, (cx, carina_y), 3, c, -1)

    cv2.putText(result, "Trachea", (cx + 5, _s(0.22, h)),   font, 0.30, c, 1)
    cv2.putText(result, "Carina",  (cx + 5, carina_y + 4), font, 0.30, c, 1)
    cv2.putText(result, "R Bronchus", (_s(0.28, w), _s(0.54, h)), font, 0.28, c, 1)
    cv2.putText(result, "L Bronchus", (_s(0.55, w), _s(0.54, h)), font, 0.28, c, 1)
    return result


# ---------------------------------------------------------------------------
# Abnormality zone map
# Keys match torchxrayvision pathology labels.
# Each entry: list of (zone_label, x1f, y1f, x2f, y2f)
# ---------------------------------------------------------------------------
ZONE_MAP: dict[str, list[tuple]] = {
    "Pneumonia":         [("LZ-R",  0.07, 0.49, 0.45, 0.72),
                          ("LZ-L",  0.55, 0.49, 0.93, 0.72)],
    "Consolidation":     [("LZ-R",  0.07, 0.49, 0.45, 0.72),
                          ("LZ-L",  0.55, 0.49, 0.93, 0.72)],
    "Lung Opacity":      [("MZ-R",  0.07, 0.33, 0.45, 0.55),
                          ("MZ-L",  0.55, 0.33, 0.93, 0.55)],
    "Infiltration":      [("MZ-R",  0.07, 0.33, 0.45, 0.55),
                          ("MZ-L",  0.55, 0.33, 0.93, 0.55)],
    "Effusion":          [("CP-R",  0.07, 0.63, 0.45, 0.74),
                          ("CP-L",  0.55, 0.65, 0.93, 0.74)],
    "Pleural_Thickening":[("Pl-R",  0.04, 0.20, 0.08, 0.70),
                          ("Pl-L",  0.92, 0.20, 0.96, 0.70)],
    "Pneumothorax":      [("UZ-R",  0.07, 0.17, 0.45, 0.37),
                          ("UZ-L",  0.55, 0.17, 0.93, 0.37)],
    "Atelectasis":       [("LZ-R",  0.07, 0.55, 0.45, 0.72),
                          ("LZ-L",  0.55, 0.55, 0.93, 0.72)],
    "Mass":              [("MZ-R",  0.10, 0.28, 0.44, 0.58)],
    "Nodule":            [("MZ-R",  0.18, 0.32, 0.42, 0.54)],
    "Cardiomegaly":      [("Cardiac",0.28, 0.30, 0.62, 0.68)],
    "Fracture":          [("Rib-R", 0.07, 0.17, 0.45, 0.72),
                          ("Rib-L", 0.55, 0.17, 0.93, 0.72)],
    "Edema":             [("BZ-R",  0.07, 0.17, 0.45, 0.72),
                          ("BZ-L",  0.55, 0.17, 0.93, 0.72)],
}


def draw_abnormality_zones(
    img: np.ndarray,
    findings: list[tuple[str, float]],
    alpha: float = 0.28,
) -> np.ndarray:
    """
    Highlight lung zones associated with detected pathologies.

    findings : list of (pathology_name, confidence_score)
    """
    h, w = img.shape[:2]
    c = COLORS["abnormality"]
    overlay = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    drawn_zones: set[tuple] = set()

    for name, score in findings:
        if name not in ZONE_MAP:
            continue
        for zone_label, x1f, y1f, x2f, y2f in ZONE_MAP[name]:
            box = (x1f, y1f, x2f, y2f)
            if box not in drawn_zones:
                x1, y1 = _s(x1f, w), _s(y1f, h)
                x2, y2 = _s(x2f, w), _s(y2f, h)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), c, -1)
                drawn_zones.add(box)

    result = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

    # Draw borders + labels on blended result
    drawn_labels: set[str] = set()
    for name, score in findings:
        if name not in ZONE_MAP:
            continue
        for zone_label, x1f, y1f, x2f, y2f in ZONE_MAP[name]:
            x1, y1 = _s(x1f, w), _s(y1f, h)
            x2, y2 = _s(x2f, w), _s(y2f, h)
            cv2.rectangle(result, (x1, y1), (x2, y2), c, 2)
            label = f"{name[:10]} {score:.2f}"
            label_key = f"{name}_{zone_label}"
            if label_key not in drawn_labels:
                cv2.putText(result, label, (x1 + 2, max(y1 - 3, 10)),
                            font, 0.28, (255, 255, 255), 2)
                cv2.putText(result, label, (x1 + 2, max(y1 - 3, 10)),
                            font, 0.28, c, 1)
                drawn_labels.add(label_key)

    return result


# ---------------------------------------------------------------------------
# Lung segmentation  (threshold-based approximation)
# ---------------------------------------------------------------------------

def approximate_lung_mask(gray_img: np.ndarray) -> np.ndarray:
    """
    Improved threshold-based lung segmentation.

    Improvements over fixed-threshold version:
      1. Otsu's method  — auto-finds optimal cutoff per image
      2. Flood fill from borders  — removes dark background/corners
      3. Larger morphological kernel  — better gap closing
      4. Area filter  — drops tiny artifacts (< 3 % of image area)

    Returns uint8 binary mask at 512 × 512 (255 = lung, 0 = other).
    Roadmap: swap body with MONAI U-Net for production accuracy.
    """
    img = cv2.resize(gray_img, (512, 512))
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Light blur before Otsu reduces sensor noise impact
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Otsu's threshold — bimodal (dark lungs vs bright tissue) works well here
    _, thresh = cv2.threshold(
        img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Flood-fill from image border + midpoints → removes dark background regions
    h, w = thresh.shape
    flood = thresh.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    for pt in [
        (0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1),
        (h // 2, 0), (h // 2, w - 1), (0, w // 2), (h - 1, w // 2),
    ]:
        if flood[pt[0], pt[1]] == 255:
            cv2.floodFill(flood, ff_mask, (pt[1], pt[0]), 0)

    # Morphological clean-up: close small holes, open small specks
    k_close = np.ones((11, 11), np.uint8)
    k_open  = np.ones((5,  5),  np.uint8)
    flood = cv2.morphologyEx(flood, cv2.MORPH_CLOSE, k_close)
    flood = cv2.morphologyEx(flood, cv2.MORPH_OPEN,  k_open)

    # Keep only blobs large enough to be lungs (≥ 3 % of 512×512)
    contours, _ = cv2.findContours(
        flood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    clean = np.zeros_like(flood)
    min_area = 512 * 512 * 0.03
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(clean, [c], -1, 255, -1)

    return clean


def draw_lung_contours(img: np.ndarray, lung_mask: np.ndarray) -> np.ndarray:
    """
    Draw lung region contours derived from approximate_lung_mask.

    Unlike draw_lung_fields (fixed geometry), this traces actual pixel
    intensity boundaries, giving image-specific outlines.
    """
    result = img.copy()
    contours, _ = cv2.findContours(
        lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (0, 255, 80), 2)
    cv2.putText(
        result,
        "Lung segmentation (threshold)",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 80),
        1,
    )
    return result


def build_medical_heatmap(
    base_img_rgb: np.ndarray,
    grayscale_cam: np.ndarray,
    lung_mask: np.ndarray,
    top_findings: list,
) -> np.ndarray:
    """
    Build an anatomically constrained medical heatmap.

    Pipeline
    --------
    1. Resize Grad-CAM to display size
    2. Multiply by lung mask  →  suppress activations outside lungs
    3. Zero weak activations (< 0.45)
    4. Gaussian smoothing
    5. Colorize with COLORMAP_JET
    6. Blend with base image
    7. Contour + label the top finding

    This is *anatomically constrained explainable attention* — far more
    clinically meaningful than raw Grad-CAM alone.

    Returns an RGB uint8 image.
    """
    h, w = base_img_rgb.shape[:2]

    # 1. Resize CAM to display resolution
    cam_resized = cv2.resize(grayscale_cam, (w, h))

    # 2. Constrain to lung region
    lung_float  = lung_mask.astype(np.float32) / 255.0
    medical_cam = cam_resized * lung_float

    # 3. Suppress weak background noise
    medical_cam[medical_cam < 0.45] = 0.0

    # 4. Smooth
    medical_cam = cv2.GaussianBlur(medical_cam, (11, 11), 0)

    # 5. Colorize  (cv2 produces BGR → convert to RGB)
    heatmap_uint8 = (medical_cam * 255).astype(np.uint8)
    heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb   = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # 6. Blend onto base image
    result = cv2.addWeighted(base_img_rgb, 0.65, heatmap_rgb, 0.35, 0)

    # 7. Pathology localization contours
    binary = (medical_cam > 0.55).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (255, 255, 0), 2)

    if top_findings:
        name, score = top_findings[0]
        label = f"{name}  ({score:.2f})"
        # White stroke then colored text
        cv2.putText(result, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (0, 0, 0), 3)
        cv2.putText(result, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 0), 1)

    return result


# ---------------------------------------------------------------------------
# Master compositor
# ---------------------------------------------------------------------------

def apply_overlays(
    img_rgb: np.ndarray,
    layers: dict[str, bool],
    findings: list[tuple[str, float]] | None = None,
) -> np.ndarray:
    """
    Composite selected overlay layers onto an RGB image.

    Parameters
    ----------
    img_rgb  : H x W x 3 uint8 RGB array
    layers   : dict with bool values for each layer key
    findings : list of (name, score) from AI inference

    Returns
    -------
    Composited RGB array of the same shape.
    """
    result = img_rgb.copy()

    if layers.get("lung_fields"):
        result = draw_lung_fields(result)
    if layers.get("heart"):
        result = draw_heart(result)
    if layers.get("carina"):
        result = draw_carina(result)
    if layers.get("ribs"):
        result = draw_ribs(result)
    if layers.get("diaphragm"):
        result = draw_diaphragm(result)
    if layers.get("abnormalities") and findings:
        result = draw_abnormality_zones(result, findings)

    return result


# ---------------------------------------------------------------------------
# Image-derived anatomy  (medically grounded compositor)
# ---------------------------------------------------------------------------

def derive_anatomy_from_mask(lung_mask: np.ndarray) -> dict | None:
    """
    Extract anatomical landmarks from a binary lung mask.

    Strategy
    --------
    * Find the two largest contours → right and left lung fields
    * Bottom boundary per column → hemidiaphragm curves
    * Horizontal gap between the two lungs at mid-thorax → cardiac silhouette

    Returns None when fewer than 2 contours are found, triggering a
    geometric fallback in apply_overlays_medically().
    """
    h, w = lung_mask.shape

    contours, _ = cv2.findContours(
        lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Keep the two largest blobs
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    if len(contours) < 2:
        return None

    def _cx(c):
        M = cv2.moments(c)
        return int(M["m10"] / M["m00"]) if M["m00"] > 0 else 0

    contours.sort(key=_cx)                       # left-to-right in image
    right_lung, left_lung = contours[0], contours[1]   # viewer: left=R, right=L

    def _bottom_boundary(contour):
        """Per-column lowest y inside the filled contour = diaphragm line."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        rx, ry, rw, _ = cv2.boundingRect(contour)
        pts = []
        for x in range(rx, rx + rw):
            ys = np.where(mask[:, x] > 0)[0]
            if len(ys):
                pts.append([x, int(ys[-1])])
        return np.array(pts, dtype=np.int32) if pts else None

    r_diaphragm = _bottom_boundary(right_lung)
    l_diaphragm = _bottom_boundary(left_lung)

    # Heart: bounding gap between the two lung fields at mid-thorax
    r_x, r_y, r_w, r_h = cv2.boundingRect(right_lung)
    l_x, l_y, l_w, l_h = cv2.boundingRect(left_lung)

    heart_x1 = r_x + r_w
    heart_x2 = l_x
    heart_y1 = int(min(r_y, l_y) + min(r_h, l_h) * 0.15)
    heart_y2 = int(min(r_y + r_h, l_y + l_h) - min(r_h, l_h) * 0.05)

    # Reject degenerate box (dense consolidation may cause lungs to merge)
    if heart_x2 <= heart_x1 or heart_y2 <= heart_y1:
        return None

    return {
        "right_lung":  right_lung,
        "left_lung":   left_lung,
        "r_diaphragm": r_diaphragm,
        "l_diaphragm": l_diaphragm,
        "heart_box":   (heart_x1, heart_y1, heart_x2, heart_y2),
    }


def apply_overlays_medically(
    img_rgb: np.ndarray,
    layers: dict[str, bool],
    lung_mask: np.ndarray,
    findings: list[tuple[str, float]] | None = None,
) -> np.ndarray:
    """
    Medically grounded overlay compositor.

    Lung fields, diaphragm, and cardiac silhouette are derived from the
    actual pixel intensity of this specific X-ray.

    Added anatomy:
      - Upper / Middle / Lower lung zone divisions (standard radiology)
      - Hilum markers (approximate medial mid-lung position)
      - Costophrenic angle markers
      - Cardiothoracic ratio estimate
      - All labels use white stroke for legibility on any background

    Trachea/carina and ribs remain geometric — threshold mask cannot provide them.
    Falls back to apply_overlays() if mask yields < 2 contours.
    """
    anatomy = derive_anatomy_from_mask(lung_mask)
    if anatomy is None:
        return apply_overlays(img_rgb, layers, findings)

    result = img_rgb.copy()
    h, w   = result.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX

    c_lung      = COLORS["lung_fields"]
    c_diaphragm = COLORS["diaphragm"]
    c_heart     = COLORS["heart"]
    c_zone      = (160, 210, 255)   # pale blue — lung zone lines
    c_hilum     = (255, 210,  60)   # amber — hilum
    c_cp        = ( 80, 255, 180)   # teal — costophrenic angle

    def _lbl(img, text, pos, scale=0.32, color=(255, 255, 255), thickness=1):
        """Draw text with black stroke so it reads on any background."""
        cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, text, pos, font, scale, color,     thickness)

    # ------------------------------------------------------------------ #
    # LUNG FIELDS  — real mask contours + semi-transparent fill           #
    # ------------------------------------------------------------------ #
    if layers.get("lung_fields"):
        overlay = result.copy()
        cv2.drawContours(overlay, [anatomy["right_lung"]], -1, c_lung, -1)
        cv2.drawContours(overlay, [anatomy["left_lung"]],  -1, c_lung, -1)
        result = cv2.addWeighted(overlay, 0.13, result, 0.87, 0)
        cv2.drawContours(result, [anatomy["right_lung"]], -1, c_lung, 2)
        cv2.drawContours(result, [anatomy["left_lung"]],  -1, c_lung, 2)

        # Per-lung: zones, hilum, costophrenic angle
        for contour, side in [
            (anatomy["right_lung"], "R"),
            (anatomy["left_lung"],  "L"),
        ]:
            rx, ry, rw, rh = cv2.boundingRect(contour)

            # Centroid label
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                _lbl(result, f"{side} LUNG", (cx - 22, cy + 5), 0.40, c_lung, 1)

            # ---- Upper / Middle / Lower zone boundaries (dashed) ----
            zone_h = rh // 3
            # Build per-column lung fill to clip dashes inside the outline
            lung_fill = np.zeros((h, w), np.uint8)
            cv2.drawContours(lung_fill, [contour], -1, 255, -1)

            for zi, (zy, zone_name) in enumerate([
                (ry + zone_h,     "Mid"),
                (ry + zone_h * 2, "Low"),
            ]):
                for x in range(rx, rx + rw, 10):
                    xe = min(x + 5, rx + rw - 1)
                    if 0 <= zy < h and lung_fill[zy, x] > 0:
                        cv2.line(result, (x, zy), (xe, zy), c_zone, 1)

            # Zone labels (left side of each lung)
            lx = rx + 4
            _lbl(result, f"{side} Upper", (lx, ry + zone_h // 2 + 4),          0.28, c_zone)
            _lbl(result, f"{side} Middle",(lx, ry + zone_h + zone_h // 2 + 4), 0.28, c_zone)
            _lbl(result, f"{side} Lower", (lx, ry + zone_h * 2 + zone_h // 2 + 4), 0.28, c_zone)

            # ---- Hilum (pulmonary vessels enter lung at medial mid-zone) ----
            # Viewer left lung (patient R): hilum is on RIGHT side of that lung
            # Viewer right lung (patient L): hilum is on LEFT side
            hilum_x = (rx + int(rw * 0.72)) if side == "R" else (rx + int(rw * 0.28))
            hilum_y  = ry + int(rh * 0.42)
            cv2.circle(result, (hilum_x, hilum_y), 7, c_hilum, 2)
            _lbl(result, "Hilum",
                 (hilum_x + 9, hilum_y + 4) if side == "R" else (hilum_x - 38, hilum_y + 4),
                 0.28, c_hilum)

            # ---- Costophrenic angle (lateral inferior corner of lung) ----
            cp_x = rx + rw - 6 if side == "R" else rx + 6
            cp_y = ry + rh - 6
            cv2.drawMarker(result, (cp_x, cp_y), c_cp,
                           cv2.MARKER_CROSS, 12, 2)
            off = (8, 12) if side == "R" else (-62, 12)
            _lbl(result, "C-P angle", (cp_x + off[0], cp_y + off[1]), 0.28, c_cp)

    # ------------------------------------------------------------------ #
    # DIAPHRAGM  — traced from inferior lung boundary                     #
    # ------------------------------------------------------------------ #
    if layers.get("diaphragm"):
        for dia, label in [
            (anatomy["r_diaphragm"], "R Hemidiaphragm"),
            (anatomy["l_diaphragm"], "L Hemidiaphragm"),
        ]:
            if dia is not None and len(dia) > 0:
                cv2.polylines(result, [dia], False, c_diaphragm, 2)
                mid = dia[len(dia) // 2]
                _lbl(result, label,
                     (int(mid[0]) - 50, int(mid[1]) + 16),
                     0.30, c_diaphragm)

    # ------------------------------------------------------------------ #
    # CARDIAC SILHOUETTE  — sized from mediastinal gap                    #
    # ------------------------------------------------------------------ #
    if layers.get("heart"):
        x1, y1, x2, y2 = anatomy["heart_box"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ax = max((x2 - x1) // 2, 1)
        ay = max((y2 - y1) // 2, 1)
        overlay = result.copy()
        cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, c_heart, -1)
        result = cv2.addWeighted(overlay, 0.12, result, 0.88, 0)
        cv2.ellipse(result, (cx, cy), (ax, ay), 0, 0, 360, c_heart, 2)
        _lbl(result, "Cardiac silhouette", (x1, y2 + 14), 0.30, c_heart)
        # Cardiothoracic ratio  (heart width / thoracic width)
        # Thoracic width ≈ distance between lateral lung edges
        r_x, _, r_w, _ = cv2.boundingRect(anatomy["right_lung"])
        l_x, _, l_w, _ = cv2.boundingRect(anatomy["left_lung"])
        thoracic_w = (l_x + l_w) - r_x
        if thoracic_w > 0:
            ct = (ax * 2) / thoracic_w
            ct_color = (255, 80, 80) if ct > 0.50 else c_heart
            _lbl(result, f"CTR ~{ct:.2f}{'  (enlarged?)' if ct > 0.50 else ''}",
                 (x1, y2 + 28), 0.28, ct_color)

    # ------------------------------------------------------------------ #
    # TRACHEA / CARINA  — geometric (mask cannot provide this)            #
    # ------------------------------------------------------------------ #
    if layers.get("carina"):
        result = draw_carina(result)

    # ------------------------------------------------------------------ #
    # RIBS  — geometric approximation                                     #
    # ------------------------------------------------------------------ #
    if layers.get("ribs"):
        result = draw_ribs(result)

    # ------------------------------------------------------------------ #
    # ABNORMALITY ZONES                                                   #
    # ------------------------------------------------------------------ #
    if layers.get("abnormalities") and findings:
        result = draw_abnormality_zones(result, findings)

    return result
