import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Banana AI", page_icon="🍌")

# --- MODEL LOADING ---
@st.cache_resource # This keeps the app fast by loading the model only once
def load_model():
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(1280, 4)
    model.load_state_dict(torch.load('bestt1_banana_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# --- UI ELEMENTS ---
st.title("🍌 Banana Freshness Predictor")
with st.expander("📸 Tips for a Perfect Prediction"):
    st.write("""
    1. **Center the Banana:** Try to get the whole fruit in the frame.
    2. **Good Lighting:** Natural light works best (avoid dark rooms).
    3. **Plain Background:** A clear countertop helps the AI focus on the fruit.
    4. **Single Banana:** If you have a bunch, try to focus on just one!
    """)
st.markdown("Is your banana ready for a smoothie or a snack? Let the AI decide!")

with st.sidebar:
    st.header("About")
    st.write("This AI was trained on 13,000 images to identify 4 stages of ripeness.")
    st.info("Created with PyTorch and MobileNetV2")

# File Uploader
file = st.file_uploader("Upload or take a photo of a banana", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Your Banana", use_container_width=True)
    
    # Simple "Thinking" animation
    with st.spinner('Analyzing ripeness...'):
        # 1. Transform
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_t = test_transform(img).unsqueeze(0)
        
        # 2. Predict
        with torch.no_grad():
            out = model(img_t)
            _, pred = torch.max(out, 1)
        
    # 3. Display Result
    classes = ['Overripe (Eat now!)', 'Ripe (Perfect)', 'Rotten (Throw away)', 'Unripe (Wait a bit)']

    st.success(f"**Verdict:** {classes[pred.item()]}")

