import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

st.set_page_config(page_title='COVID-19 CT Scan Classifier', page_icon='🩻', layout='centered')

CLASSES = {'NiCT': 'Non-informative CT', 'nCT': 'Normal CT', 'pCT': 'COVID-19 Positive CT'}

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(960, 1280), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1280, 3))
    def forward(self, x):
        return self.net(x)

@st.cache_resource(show_spinner='Loading trained model...')
def load_model():
    path = Path(__file__).parent / 'mobilenet_v3_large.pth'
    if not path.exists():
        raise FileNotFoundError('Upload mobilenet_v3_large.pth to the repository.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = Head()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint['class_to_idx']

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.title('COVID-19 CT Scan Classifier')
st.caption('MobileNetV3-Large · PyTorch · Transfer learning')
st.write('Upload a chest CT image to classify the slice as **Normal**, **COVID-19 Positive**, or **Non-informative**.')
st.info('Educational demonstration only. Not validated for clinical diagnosis or medical decisions.')

uploaded = st.file_uploader('Upload a CT image', type=['png', 'jpg', 'jpeg'])
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption='Uploaded CT slice', use_container_width=True)
        model, class_to_idx = load_model()
        with torch.inference_mode():
            output = model(preprocess(image).unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)[0]
        scores = {CLASSES[key]: float(probabilities[idx]) for key, idx in class_to_idx.items()}
        winner = max(scores, key=scores.get)
        st.subheader(f'Prediction: {winner}')
        st.write('Model output probabilities (not calibrated clinical confidence):')
        for name, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            st.write(f'**{name}: {value:.2%}**')
            st.progress(value)
    except Exception as exc:
        st.error(f'Unable to process image or load model: {exc}')

st.divider()
st.caption('Held-out slice-level test accuracy: 99.59% (2,944 / 2,956 images). Slice-level splits may include correlated images from the same patient; results do not establish patient-level generalization.')
