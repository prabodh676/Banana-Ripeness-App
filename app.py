import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Banana AI", page_icon="🍌")

# --- MODEL LOADING ---
# This keeps the app fast by loading the model only once
@st.cache_resource
def load_model():
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(1280, 4)
    # BE CAREFUL: Ensure the filename matches your GitHub file EXACTLY (with .pth if it has one)
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
    # 1. Open the raw image
    img = Image.open(file)
    st.image(img, caption="Uploaded Banana", use_container_width=True)
    
    # 2. DEFINE img_t HERE (The Transformation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # This line turns the PIL image into the 'img_t' tensor
    # .unsqueeze(0) adds the batch dimension the model expects
    img_t = test_transform(img).unsqueeze(0)

    # 3. Now you can use it!
    with st.spinner('AI is thinking...'):
        out = model(img_t) # <--- Now this will work!
        _, pred = torch.max(out, 1)
    # 3. Display Result
# Map class index to name and estimated days
# Index order: Overripe, Ripe, Rotten, Unripe
# 1. Define the knowledge base
ripeness_info = {
    0: {"label": "Overripe", "days": "1 day", "advice": "Eat now or freeze for smoothies!"},
    1: {"label": "Ripe", "days": "2-3 days", "advice": "Perfect for a snack!"},
    2: {"label": "Rotten", "days": "0 days", "advice": "Too late! Compost it."},
    3: {"label": "Unripe", "days": "5-7 days", "advice": "Wait for the yellow color."}
}

# 2. Get the index from the AI
with torch.no_grad():
    out = model(img_t)
    _, pred = torch.max(out, 1)
    index = pred.item()

# 3. Pull the specific data
result = ripeness_info[index]

# 4. Show it on the screen
st.success(f"**Verdict:** {result['label']}")
st.metric(label="Days Until Rotten", value=result['days'])
st.info(f"💡 {result['advice']}")

st.divider() # Adds a nice visual line

st.subheader("📢 Share with Friends")

# This creates a URL that pre-fills a message for WhatsApp
share_text = f"My banana is {result['label']}! This AI says it has {result['days']} left. Check yours here:"
app_url = "https://your-app-link.streamlit.app" # Replace with your real link!

# WhatsApp Share Link
whatsapp_url = f"https://wa.me/?text={share_text} {app_url}"

st.column_config.LinkColumn("Share on WhatsApp")
st.link_button("Share on WhatsApp 🟢", whatsapp_url)

# Copy to Clipboard (Built-in Streamlit feature)
st.button("Copy App Link 🔗", on_click=lambda: st.write(f"Link copied: {app_url}"))

