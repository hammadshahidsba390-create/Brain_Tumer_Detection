import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── Professional Light Theme Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 50%, #f8fafc 100%);
    min-height: 100vh;
    color: #0c4a6e !important;
}

.stApp {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 50%, #f8fafc 100%) !important;
    color: #0c4a6e !important;
}

/* Force light theme on all Streamlit containers */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
section.main,
.main .block-container {
    background: transparent !important;
    color: #0c4a6e !important;
}

[data-testid="stHeader"] {
    background: rgba(255,255,255,0.6) !important;
    backdrop-filter: blur(10px);
}

/* All markdown text */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
label, .stFileUploader label, .stFileUploader span, .stFileUploader small {
    color: #0c4a6e !important;
}

/* File uploader dark-mode override */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe) !important;
    border: 2px dashed #0ea5e9 !important;
    color: #0c4a6e !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #0c4a6e !important;
}

/* Buttons */
.stButton > button, [data-testid="stBaseButton-secondary"] {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* Header Card */
.header-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 24px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    border: 1px solid rgba(14, 165, 233, 0.15);
    box-shadow: 0 20px 60px rgba(14, 165, 233, 0.1), 0 4px 12px rgba(0, 0, 0, 0.04);
    backdrop-filter: blur(20px);
}

.header-card h1 {
    color: #0c4a6e;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.header-card .subtitle {
    color: #475569;
    font-size: 1.1rem;
    font-weight: 400;
    margin-top: 0.5rem;
}

.header-card .badge {
    display: inline-block;
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Upload Card */
.upload-card {
    background: rgba(255, 255, 255, 0.92);
    border-radius: 24px;
    padding: 2rem;
    border: 1px solid rgba(14, 165, 233, 0.12);
    box-shadow: 0 12px 40px rgba(14, 165, 233, 0.08), 0 2px 8px rgba(0, 0, 0, 0.03);
    backdrop-filter: blur(20px);
}

/* Custom file uploader */
.stFileUploader {
    background: transparent !important;
}

.stFileUploader > div > div {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe) !important;
    border: 2px dashed #0ea5e9 !important;
    border-radius: 20px !important;
    padding: 2.5rem 1.5rem !important;
    transition: all 0.3s ease;
}

.stFileUploader > div > div:hover {
    border-color: #0284c7 !important;
    background: linear-gradient(135deg, #e0f2fe, #bae6fd) !important;
    box-shadow: 0 8px 30px rgba(14, 165, 233, 0.15) !important;
}

/* Result Card */
.result-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 24px;
    padding: 2rem;
    margin-top: 2rem;
    border: 1px solid rgba(14, 165, 233, 0.12);
    box-shadow: 0 20px 60px rgba(14, 165, 233, 0.1), 0 4px 12px rgba(0, 0, 0, 0.04);
    backdrop-filter: blur(20px);
}

.result-card h3 {
    color: #0c4a6e;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    text-align: center;
}

/* Prediction text */
.prediction-text {
    font-size: 1.6rem;
    font-weight: 700;
    text-align: center;
    margin: 1.5rem 0;
    padding: 1rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    box-shadow: 0 8px 24px rgba(14, 165, 233, 0.3);
}

/* Probability Bars */
.prob-bar-container {
    margin: 0.8rem 0;
}

.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    font-weight: 500;
    color: #334155;
    margin-bottom: 0.35rem;
}

.prob-track {
    height: 12px;
    background: #e2e8f0;
    border-radius: 999px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #0ea5e9, #06b6d4);
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 2px 8px rgba(14, 165, 233, 0.3);
}

/* Confidence indicator */
.confidence-box {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #a7f3d0;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-top: 1.5rem;
}

.confidence-box .conf-label {
    color: #065f46;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.confidence-box .conf-value {
    color: #047857;
    font-size: 1.8rem;
    font-weight: 700;
}

/* Footer */
.footer-note {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(148, 163, 184, 0.2);
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Image styling */
.stImage img {
    border-radius: 16px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid rgba(14, 165, 233, 0.1) !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #0ea5e9 !important;
}

/* Success / Info messages */
.stAlert {
    border-radius: 16px !important;
    border: none !important;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    margin: 2rem 0;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-in {
    animation: fadeInUp 0.6s ease-out;
}

</style>
""", unsafe_allow_html=True)

# ─── Model Loading ───
@st.cache_resource

def get_model():
    import torchvision.models as models
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 4)
    )
    if os.path.exists("brain_tumor_resnet50.pth"):
        model.load_state_dict(torch.load("brain_tumor_resnet50.pth", map_location="cpu"))
    model.eval()
    return model

# ─── Preprocessing ───
@st.cache_data
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ─── Load Classes ───
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# ─── Header ───
st.markdown("""
<div class="header-card animate-in">
    <h1>🧠 Brain Tumor Detection</h1>
    <div class="subtitle">AI-powered MRI classification using deep learning</div>
    <div class="badge">ResNet50 · 4-Class Classifier</div>
</div>
""", unsafe_allow_html=True)

# ─── Main Content ───
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='upload-card animate-in'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload MRI Scan")
    st.markdown("<p style='color:#64748b; font-size:0.95rem; margin-bottom:1.5rem;'>Supported: JPG, PNG, JPEG</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        st.markdown("<div class='result-card animate-in'>", unsafe_allow_html=True)
        st.markdown("<h3>🔬 Analysis Results</h3>", unsafe_allow_html=True)

        with st.spinner("Analyzing scan..."):
            model = get_model()
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item() * 100

        # Prediction display
        pred_class = CLASSES[pred_idx]
        pred_display = pred_class.replace("_", " ").title()

        st.markdown(f"""
        <div class="prediction-text">
            {pred_display}
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("<p style='color:#475569; font-size:0.9rem; font-weight:500; margin-bottom:0.8rem;'>Class Probabilities</p>", unsafe_allow_html=True)

        for i, cls in enumerate(CLASSES):
            prob = probs[i].item() * 100
            cls_display = cls.replace("_", " ").title()
            st.markdown(f"""
            <div class="prob-bar-container">
                <div class="prob-label">
                    <span>{cls_display}</span>
                    <span style="color:#0ea5e9; font-weight:600;">{prob:.1f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {prob}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence box
        st.markdown(f"""
        <div class="confidence-box">
            <div class="conf-label">Confidence Score</div>
            <div class="conf-value">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Empty state
        st.markdown("<div class='result-card animate-in'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">📋</div>
            <h3 style="color:#0c4a6e; margin-bottom:0.5rem;">Ready for Analysis</h3>
            <p style="color:#64748b;">Upload an MRI scan on the left to see the AI classification results.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ───
st.markdown("""
<div class="footer-note">
    ⚠️ This tool is for research and educational purposes. Consult a medical professional for actual diagnosis.
</div>
""", unsafe_allow_html=True)
