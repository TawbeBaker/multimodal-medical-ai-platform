import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchxrayvision as xrv

st.title("Radiology AI Assistant")

# Load pretrained radiology model
@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(
        weights="densenet121-res224-all"
    )
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-ray")

    img = np.array(image)

    # Normalize
    img = xrv.datasets.normalize(
        img,
        maxval=255
    )

    # Convert to grayscale
    img = img.mean(2)[None, :, :]

    # Resize/Crop
    transform = xrv.datasets.XRayCenterCrop()
    img = transform(img)

    transform = xrv.datasets.XRayResizer(224)
    img = transform(img)

    # Tensor
    input_tensor = torch.from_numpy(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

    preds = dict(
        zip(
            model.pathologies,
            outputs[0].detach().numpy()
        )
    )

    st.subheader("Predictions")

    sorted_preds = sorted(
        preds.items(),
        key=lambda x: x[1],
        reverse=True
    )
    st.bar_chart(
        {k: float(v) for k, v in sorted_preds[:10]}
    )

    for pathology, score in sorted_preds[:10]:
        st.write(f"{pathology}: {score:.3f}")