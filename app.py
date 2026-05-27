import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchxrayvision as xrv
import ollama
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from anatomy_overlay import (
    apply_overlays,
    apply_overlays_medically,
    approximate_lung_mask,
    draw_lung_contours,
    build_medical_heatmap,
)

st.set_page_config(
    page_title="Radiology AI Workstation",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Radiology AI Workstation")

# -------------------------
# DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model = model.to(device)
    model.eval()
    return model

model = load_model()

target_layers = [model.features.denseblock4]

@st.cache_resource
def load_cam(_model):
    return GradCAM(model=_model, target_layers=target_layers)

cam = load_cam(model)

# -------------------------
# CORE LOGIC (IMPORTANT FIX)
# -------------------------

def get_top_findings(preds, threshold=0.6, top_k=2):
    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)

    filtered = [(k, v) for k, v in sorted_preds if v >= threshold]
    return filtered[:top_k]

def is_normal(preds, threshold=0.55):
    return max(preds.values()) < threshold

def group_clinical_pattern(name):
    airspace = ["Pneumonia", "Consolidation", "Lung Opacity", "Infiltration"]
    pleural = ["Effusion", "Pleural_Thickening"]
    mass = ["Mass", "Nodule"]
    collapse = ["Atelectasis"]

    if name in airspace:
        return "Airspace disease"
    if name in pleural:
        return "Pleural abnormality"
    if name in mass:
        return "Mass-like lesion"
    if name in collapse:
        return "Possible collapse"
    return "Other"

def generate_radiology_report(findings_text):
    prompt = f"""You are a senior radiologist.

Write a structured chest X-ray report.

Rules:
- Be medically realistic
- Do NOT over-diagnose
- Use uncertainty language (possible, suggestive of)
- Keep it short and clinical

Findings:
{findings_text}

Output format:

FINDINGS:
...

IMPRESSION:
...

RECOMMENDATION:
...
"""
    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def impression_logic(top_findings):
    if not top_findings:
        return "No acute cardiopulmonary abnormality detected."

    main = top_findings[0][0]

    if main in ["Pneumonia", "Consolidation", "Lung Opacity"]:
        return "Findings suggest possible airspace disease, most consistent with infectious process."
    elif main == "Effusion":
        return "Findings suggest possible pleural effusion."
    elif main in ["Mass", "Nodule"]:
        return "Findings suggest possible nodular or mass-like lesion. Further evaluation recommended."
    else:
        return "Abnormal radiographic finding requiring clinical correlation."

# -------------------------
# SIDEBAR — LAYER CONTROLS
# -------------------------
with st.sidebar:
    st.header("Anatomy Overlays")
    layers = {
        "lung_fields":   st.checkbox("Lung Fields",         value=True),
        "diaphragm":     st.checkbox("Diaphragm",           value=True),
        "heart":         st.checkbox("Cardiac Silhouette",  value=False),
        "carina":        st.checkbox("Trachea / Carina",    value=False),
        "ribs":          st.checkbox("Rib Markers (1-10)",  value=False),
        "abnormalities": st.checkbox("Abnormality Zones",   value=True),
    }
    st.divider()
    st.header("Heatmap Views")
    show_lung_anatomy    = st.checkbox("Lung Segmentation",          value=True)
    show_medical_heatmap = st.checkbox("Medical Heatmap",            value=True)
    show_vision_heatmap  = st.checkbox("Vision Heatmap (Grad-CAM)",  value=False)
    st.divider()

    with st.expander("Inference settings"):
        threshold = st.slider("Confidence threshold", 0.30, 0.90, 0.60, 0.05)
        top_k = st.slider("Max findings shown", 1, 5, 2)

    st.caption("AI-assisted tool — not for clinical use.")

# -------------------------
# UPLOAD
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)

# -------------------------
# PIPELINE
# -------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    # Display size for overlays (keeps overlays crisp at fixed resolution)
    DISPLAY_SIZE = 512
    img_display = np.array(image.resize((DISPLAY_SIZE, DISPLAY_SIZE)))

    # -------------------------
    # CAM IMAGE (224 px for model)
    # -------------------------
    img_for_cam = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    img_for_cam = img_for_cam.mean(axis=2)
    img_for_cam = np.stack([img_for_cam] * 3, axis=-1)

    # -------------------------
    # MODEL INPUT
    # -------------------------
    img = np.array(image)
    img = xrv.datasets.normalize(img, maxval=255)
    img = img.mean(2)[None, :, :]
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)
    input_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    # -------------------------
    # INFERENCE
    # -------------------------
    outputs = model(input_tensor)
    preds = dict(zip(model.pathologies, outputs[0].detach().cpu().numpy()))
    top_findings = get_top_findings(preds, threshold=threshold, top_k=top_k)

    # -------------------------
    # GRAD-CAM
    # -------------------------
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(img_for_cam, grayscale_cam, use_rgb=True)

    # -------------------------
    # LUNG SEGMENTATION
    # -------------------------
    gray_display  = cv2.cvtColor(img_display, cv2.COLOR_RGB2GRAY)
    lung_mask     = approximate_lung_mask(gray_display)
    lung_anatomy_img = draw_lung_contours(img_display.copy(), lung_mask)

    # -------------------------
    # MEDICAL HEATMAP
    # -------------------------
    active_findings = top_findings if not is_normal(preds) else []
    medical_overlay = build_medical_heatmap(
        img_display, grayscale_cam, lung_mask, active_findings
    )

    # -------------------------
    # BUILD REPORT TEXT
    # -------------------------
    if is_normal(preds):
        report = (
            "FINDINGS:\n- No significant radiographic abnormality.\n\n"
            "IMPRESSION:\nNormal chest radiograph."
        )
    else:
        report = "FINDINGS:\n"
        for name, score in top_findings:
            group = group_clinical_pattern(name)
            report += f"- {group}: {name} ({score:.2f})\n"
        report += "\nIMPRESSION:\n" + impression_logic(top_findings)
    report += "\n\nNOTE: AI-assisted output, not a diagnostic tool."

    # -------------------------
    # LAYOUT — two columns
    # -------------------------
    col_img, col_report = st.columns([1.1, 1], gap="large")

    with col_img:
        # --- Anatomy viewer (image-derived: lungs, diaphragm, heart from mask) ---
        overlay_img = apply_overlays_medically(
            img_display, layers, lung_mask, active_findings
        )
        st.subheader("Anatomy Viewer")
        st.image(overlay_img, use_container_width=True)

        # --- Lung segmentation (image-derived contours) ---
        if show_lung_anatomy:
            st.subheader("Lung Segmentation")
            st.image(lung_anatomy_img, use_container_width=True)

        # --- Medical heatmap (lung-constrained Grad-CAM) ---
        if show_medical_heatmap:
            st.subheader("Medical Heatmap (anatomically constrained)")
            st.image(medical_overlay, use_container_width=True)

        # --- Raw vision heatmap ---
        if show_vision_heatmap:
            st.subheader("Vision Heatmap (raw Grad-CAM)")
            st.image(visualization, use_container_width=True)

    with col_report:
        # --- Structured report ---
        st.subheader("Structured Report")
        st.text(report)

        st.divider()

        # --- Prediction bar chart ---
        st.subheader("Top Predictions")
        sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        st.bar_chart({k: float(v) for k, v in sorted_preds[:10]})

        st.divider()

        # --- LLM report ---
        st.subheader("AI Narrative Report (LLM)")
        findings_text = "\n".join(f"{n}: {s:.2f}" for n, s in top_findings) or "No significant findings."
        if st.button("Generate LLM Report"):
            with st.spinner("Querying LLM…"):
                llm_report = generate_radiology_report(findings_text)
            st.text(llm_report)