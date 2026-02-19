import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Banana AI", page_icon="🍌")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(1280, 4)
    # Ensure this file exists on GitHub with this exact name!
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
    # 1. Open and show the raw image
    img = Image.open(file)
    st.image(img, caption="Uploaded Banana", use_container_width=True)
    
    # 2. Transform the image for the model
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_t = test_transform(img).unsqueeze(0)

    # 3. Run Inference
    import torch.nn.functional as F

    with st.spinner('Analyzing image...'):
        with torch.no_grad():
            out = model(img_t)
            
            # Convert raw scores to probabilities (0% to 100%)
            probabilities = F.softmax(out, dim=1)
            confidence, pred = torch.max(probabilities, 1)
            
            confidence_score = confidence.item()
            index = pred.item()

    if confidence_score < 0.70:
        st.error("🚨 **Low Confidence!**")
        st.warning(f"The AI is only {confidence_score:.1%} sure about this. This might not be a banana, or the lighting is too poor. Please try again with a clearer photo!")
        
        st.caption(f"Psst... it thought it looked a bit like: {['Overripe', 'Ripe', 'Rotten', 'Unripe'][index]}")

    else:
        ripeness_info = {
            0: {"label": "Overripe", "days": "1 day", "advice": "Eat now or freeze for smoothies!"},
            1: {"label": "Ripe", "days": "2-3 days", "advice": "Perfect for a snack!"},
            2: {"label": "Rotten", "days": "0 days", "advice": "Too late! Compost it."},
            3: {"label": "Unripe", "days": "5-7 days", "advice": "Wait for the yellow color."}
        }

        result = ripeness_info[index]

        # 6. Display Results
        st.success(f"**Verdict: {result['label']}**")
        
        # Create two columns for a clean look
        col1, col2 = st.columns(2)
        col1.metric(label="Days Left", value=result['days'])
        col2.metric(label="AI Confidence", value=f"{confidence_score:.1%}")
        
        st.info(f"💡 **AI Advice:** {result['advice']}")

    # 6. Sharing Section
    st.divider()
    st.subheader("📢 Share with Friends")

    share_text = f"My banana is {result['label']}! This AI says it has {result['days']} left. Check yours here:"
    app_url = "https://banana-ripeness-app.streamlit.app" 

    whatsapp_url = f"https://wa.me/?text={share_text} {app_url}"
    
    st.link_button("Share on WhatsApp 🟢", whatsapp_url)
    
    if st.button("Copy App Link 🔗"):
        st.write(f"Copy this: {app_url}")
        st.toast("Link displayed below!")

else:
    st.info("Waiting for an image to be uploaded...")


