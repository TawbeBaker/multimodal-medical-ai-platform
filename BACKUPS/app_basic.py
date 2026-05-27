import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchxrayvision as xrv

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

st.title("Radiology AI Assistant (Improved POC)")

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
# UI
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
    st.image(image, use_container_width=True)

    # -------------------------
    # CAM IMAGE
    # -------------------------
    img_for_cam = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    img_for_cam = img_for_cam.mean(axis=2)
    img_for_cam = np.stack([img_for_cam]*3, axis=-1)

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

    preds = dict(zip(
        model.pathologies,
        outputs[0].detach().cpu().numpy()
    ))

    # -------------------------
    # FILTER FINDINGS (FIX FOR OVER-PREDICTION)
    # -------------------------
    top_findings = get_top_findings(preds)

    # -------------------------
    # REPORT
    # -------------------------
    st.subheader("Radiology Report")

    if is_normal(preds):
        report = "FINDINGS:\n- No significant radiographic abnormality.\n\nIMPRESSION:\nNormal chest radiograph."
    else:
        report = "FINDINGS:\n"

        for name, score in top_findings:
            group = group_clinical_pattern(name)
            report += f"- {group}: {name} ({score:.2f})\n"

        report += "\nIMPRESSION:\n"
        report += impression_logic(top_findings)

    report += "\n\nNOTE: AI-assisted output, not a diagnostic tool."
    st.text(report)

    # -------------------------
    # GRAD-CAM
    # -------------------------
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(
        img_for_cam,
        grayscale_cam,
        use_rgb=True
    )

    st.subheader("Heatmap (Model Attention)")
    st.image(visualization, use_container_width=True)

    # -------------------------
    # RAW TOP RESULTS (OPTIONAL)
    # -------------------------
    st.subheader("Top Predictions")

    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    st.bar_chart({k: float(v) for k, v in sorted_preds[:10]})