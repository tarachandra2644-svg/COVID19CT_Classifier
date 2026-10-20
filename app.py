import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="COVID-19 CT | AI Classifier", page_icon="🩻", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem;}
[data-testid="stAppViewContainer"] {background: radial-gradient(ellipse at 75% 0%, #122940 0%, #0b111b 45%, #090d15 100%);}
[data-testid="stHeader"] {background: transparent;}
h1, h2, h3 {letter-spacing: -0.035em;}
.hero-label {font-size: .77rem; font-weight: 750; color: #67d4e8; letter-spacing: .15em; margin-bottom: .7rem;}
.hero-sub {color: #a5b5c7; font-size: 1.03rem; max-width: 760px; margin-top: -.35rem; margin-bottom: 1.7rem;}
.section-label {color: #9eafc3; font-size: .85rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; margin-bottom: .55rem;}
.prediction-card {border: 1px solid #2a4960; background: linear-gradient(135deg, #102d3b, #132033); border-radius: 14px; padding: 1.1rem 1.25rem; margin: .3rem 0 1rem;}
.prediction-label {color: #a6c8d5; font-size: .78rem; letter-spacing: .09em; text-transform: uppercase;}
.prediction-value {color: #effbff; font-size: 1.5rem; font-weight: 760; margin: .3rem 0;}
.prediction-note {color: #adc2cd; font-size: .83rem;}
[data-testid="stMetric"] {border: 1px solid #243549; background: #101c2b; padding: 1rem; border-radius: 12px;}
[data-testid="stMetricLabel"] {color: #a6b8c9;}
[data-testid="stMetricValue"] {color: #e9faff;}
[data-testid="stFileUploader"] {border-radius: 12px;}
[data-testid="stAlert"] {border-radius: 10px;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

CLASSES = {"NiCT": "Non-informative", "nCT": "Normal", "pCT": "COVID-19 Positive"}

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(960, 1280), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1280, 3))

    def forward(self, x):
        return self.net(x)

@st.cache_resource(show_spinner="Loading trained MobileNetV3-Large model...")
def load_model():
    model_path = Path(__file__).parent / "mobilenet_v3_large.pth"
    if not model_path.exists():
        raise FileNotFoundError("mobilenet_v3_large.pth is missing from the repository.")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = Head()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["class_to_idx"]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.markdown('<div class="hero-label">MEDICAL IMAGING · DEEP LEARNING</div>', unsafe_allow_html=True)
st.title("COVID-19 CT Scan Classifier")
st.markdown('<div class="hero-sub">A three-class chest CT slice classifier powered by <b>MobileNetV3-Large</b>, PyTorch and transfer learning.</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("Test accuracy", "99.59%")
with m2: st.metric("CT slices", "19,685")
with m3: st.metric("Classes", "3")
with m4: st.metric("Training epochs", "25")

st.write("")
st.info("**Research and educational demonstration only.** Not validated for clinical diagnosis, triage or medical decisions. Upload only de-identified images you have permission to use.")
st.write("")

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown('<div class="section-label">01 / Input CT slice</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a chest CT image", type=["png", "jpg", "jpeg"], help="PNG or JPEG. Use de-identified images only.")
    image = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded CT slice", use_container_width=True)
        except Exception:
            st.error("Could not read this image. Try a valid PNG or JPEG file.")
    else:
        st.caption("Upload an image to view the prediction alongside the CT slice.")

with right:
    st.markdown('<div class="section-label">02 / Model prediction</div>', unsafe_allow_html=True)
    if image is None:
        st.markdown('<div class="prediction-card"><div class="prediction-label">Awaiting input</div><div class="prediction-value">No scan uploaded</div><div class="prediction-note">Your classification and output probabilities will appear here.</div></div>', unsafe_allow_html=True)
    else:
        try:
            model, class_to_idx = load_model()
            with torch.inference_mode():
                logits = model(preprocess(image).unsqueeze(0))
                probs = torch.softmax(logits, dim=1)[0]
            scores = {CLASSES[key]: float(probs[idx]) for key, idx in class_to_idx.items()}
            predicted = max(scores, key=scores.get)
            st.markdown(f'<div class="prediction-card"><div class="prediction-label">Predicted category</div><div class="prediction-value">{predicted}</div><div class="prediction-note">Model output · {scores[predicted]:.2%} probability</div></div>', unsafe_allow_html=True)
            st.markdown("**Class probabilities**")
            for label, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
                st.write(f"{label} — **{value:.2%}**")
                st.progress(value)
            st.caption("Softmax probabilities are not calibrated clinical confidence estimates.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
with st.expander("Model details & evaluation", expanded=False):
    st.markdown("**Architecture:** MobileNetV3-Large with a three-class classification head.  ")
    st.markdown("**Training:** Transfer learning / fine-tuning for 25 epochs.  ")
    st.markdown("**Evaluation:** 2,944 correct predictions from 2,956 held-out CT slices (99.59% slice-level test accuracy).  ")
    st.markdown("**Limitations:** Slice-level train/test splits can include correlated images from the same patient, potentially inflating performance. No patient-level external validation has been established.")
    for filename, caption in [
        ("confusion_matrix.png", "Test-set confusion matrix"),
        ("validation_accuracy.png", "Validation accuracy by epoch"),
        ("validation_loss.png", "Validation loss by epoch"),
    ]:
        path = Path(__file__).parent / filename
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)

st.caption("Built with PyTorch · Torchvision · Streamlit | For educational use only")
