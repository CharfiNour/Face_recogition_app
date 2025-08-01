import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("📸 Face Recognition App")

# Paths & Constants
BASE_DIR = Path(__file__).resolve().parent
HAAR_CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = BASE_DIR / "svm_model.pkl"
EMBEDDING_FILE = BASE_DIR / "embedded_faces.npz"
IMG_SIZE = (160, 160)

# Load Resources
@st.cache_resource
def load_models():
    # Load VGG16 base model
    vgg = VGG16(weights="imagenet", include_top=False, input_shape=(160, 160, 3))
    embedder = Model(inputs=vgg.input, outputs=vgg.output)
    # Load classifier and labels
    clf = joblib.load(MODEL_PATH)
    data = np.load(EMBEDDING_FILE, allow_pickle=True)
    labels = data['trainy']
    return embedder, clf, labels

@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))

embedder, clf, label_list = load_models()
face_cascade = load_cascade()


# Helper Functions
def extract_face(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = image_np[y:y+h, x:x+w]
    return cv2.resize(face, IMG_SIZE)

def get_embedding(face):
    face_array = img_to_array(face)
    face_array = np.expand_dims(face_array, axis=0)
    face_array = preprocess_input(face_array)
    features = embedder.predict(face_array, verbose=0)
    return features.flatten()

# Streamlit UI
uploaded = st.file_uploader("Upload an image")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    face = extract_face(image_np)

    if face is None:
        st.warning("No face detected. Try another image.")
    else:
        emb = get_embedding(face)
        pred = clf.predict([emb])[0]
        st.success(f"🧠 Predicted: {pred}")
