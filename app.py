import streamlit as st
import librosa
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
from PIL import Image, ImageDraw

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Emotion Recognition 🎤", layout="centered")

# -------------------------------
# DARK MODE 🌗
# -------------------------------
dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=True)
bg = "#01411C" if dark_mode else "#f5f5f5"
text = "white" if dark_mode else "black"
sidebar_bg = "#013220" if dark_mode else "#e6e6e6"
button_bg = "#228B22"
button_hover = "#00ff7f"
input_bg = "#025927" if dark_mode else "#ffffff"
input_text = "white" if dark_mode else "black"
placeholder_color = "rgba(255,255,255,0.7)" if dark_mode else "rgba(0,0,0,0.7)"

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    color: {text};
    font-family: 'Arial';
}}
[data-testid="stSidebar"] {{
    background: {sidebar_bg};
    color: white !important;
}}
[data-testid="stSidebar"] * {{
    color: white !important;
}}
h1, h2, h3 {{
    text-align: center;
}}
.stButton>button {{
    background: {button_bg};
    color: white;
    border-radius: 15px;
    height: 50px;
    width: 100%;
    font-weight: bold;
    box-shadow: 5px 5px 15px rgba(0,0,0,0.4);
    transition: all 0.3s ease-in-out;
}}
.stButton>button:hover {{
    background: {button_hover};
    color: black;
    transform: scale(1.05);
    box-shadow: 10px 10px 25px rgba(0,0,0,0.6);
}}
.stFileUploader>div>div {{
    background: rgba(255,255,255,0.05) !important;
    color: white !important;
    border-radius: 15px;
    padding: 15px;
}}
.stFileUploader button {{
    color: white !important;
    background-color: {button_bg} !important;
    border-radius: 10px;
}}
.stFileUploader button:hover {{
    background-color: {button_hover} !important;
    color: black !important;
}}
.stSlider>div>div>div {{
    color: {text};
}}
/* Circular Image for About */
.circular-img {{
    border-radius: 50%;
    border: 4px solid #00ff7f;
    width: 200px;
    height: 200px;
    object-fit: cover;
    display: block;
    margin-left: auto;
    margin-right: auto;
}}
/* Input Fields (Login/Signup) */
input {{
    background-color: {input_bg} !important;
    color: {input_text} !important;
}}
input::placeholder {{
    color: {placeholder_color} !important;
}}
label {{
    color: {text} !important;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOGIN SYSTEM 🔐
# -------------------------------
if "users" not in st.session_state:
    st.session_state.users = {"sharifullah7087@gmail.com": "sharifkhan123"}
if "login" not in st.session_state:
    st.session_state.login = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

if not st.session_state.login:
    st.title("🔐 Login System")
    option = st.radio("Select Option", ["Login", "Signup"])
    user = st.text_input("Email", placeholder="Enter your email")
    pwd = st.text_input("Password", type="password", placeholder="Enter your password")
    
    if option == "Login" and st.button("Login"):
        if user in st.session_state.users and st.session_state.users[user] == pwd:
            st.session_state.login = True
            st.session_state.current_user = user
            st.success("Login Successful ✅")
        else:
            st.error("Invalid Credentials ❌")
    elif option == "Signup" and st.button("Create Account"):
        st.session_state.users[user] = pwd
        st.success("Account Created ✅")
    st.stop()

# -------------------------------
# LOGOUT BUTTON
# -------------------------------
if st.sidebar.button("🚪 Logout"):
    st.session_state.login = False
    st.session_state.current_user = None

# -------------------------------
# LOAD MODELS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "../models/emotion_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "../models/scaler.pkl"), "rb"))
le = pickle.load(open(os.path.join(BASE_DIR, "../models/label_encoder.pkl"), "rb"))

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(file):
    audio, sr = librosa.load(file, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    stft = np.abs(librosa.stft(audio))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    spec = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    return np.hstack([mfcc_mean, mfcc_std, chroma, mel, zcr, spec, rms]), audio, sr

# -------------------------------
# NAVIGATION
# -------------------------------
page = st.sidebar.radio("📂 Navigation", ["🏠 Home", "👨‍💻 About"])

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "🏠 Home":
    st.title("🎤 Speech Emotion Recognition")
    st.subheader("📂 Upload Audio")
    uploaded_file = st.file_uploader("", type=["wav"])
    
    st.subheader("🎤 Record Voice")
    duration = st.slider("Duration (seconds)", 1, 5, 3)
    if st.button("Start Recording"):
        fs = 44100
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        write("recorded.wav", fs, recording)
        st.audio("recorded.wav")
        uploaded_file = "recorded.wav"
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        features, audio, sr = extract_features(uploaded_file)
        if st.button("🚀 Predict Emotion"):
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            emotion = le.inverse_transform(prediction)[0]
            st.success(f"🎯 Emotion: {emotion}")

# -------------------------------
# ABOUT PAGE
# -------------------------------
elif page == "👨‍💻 About":
    st.title("👨‍💻 About Me")

    # Circular Image
    img_path = os.path.join(BASE_DIR, "assets", "sharif1.jpg")
    img = Image.open(img_path).convert("RGBA")
    size = (200, 200)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    img = img.resize(size)
    img.putalpha(mask)
    shadow = Image.new("RGBA", (220,220), (0,0,0,0))
    shadow.paste(img, (10,10), img)
    st.image(shadow, width=220)

    st.markdown("""
    ## 👋 Sharif Ullah
    🎓 AI Engineer  
    📚 BS Artificial Intelligence  
    🏫 Hazara University, Mansehra, Pakistan  
    🌍 From Afghanistan 🇦🇫  

    ---
    ### 💡 Skills
    - Artificial Intelligence (AI)
    - Machine Learning (ML)
    - Deep Learning (DL)
    - Data Science
    - Computer Vision (CV)
    - Natural Language Processing (NLP)
    - Digital Image Processing (DIP)

    ---
    🎯 Passionate about AI 🚀
    """)
    st.success("🚀 Open for opportunities")