import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Load pretrained model
model = models.densenet121(pretrained=True)
model.eval()

# Labels (ImageNet placeholder for now)
labels = ["Normal", "Pneumonia"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("Radiology AI Assistant")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-ray")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    prediction = torch.argmax(output, dim=1).item()

    st.write(f"Prediction: {labels[prediction % 2]}")