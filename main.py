import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from gtts import gTTS
import tempfile
import time

# --- Page & CSS ---
st.set_page_config(page_title="Asciende ASL", layout="wide")
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #1f1f1f, #0a0a0a); background-size:400% 400%; animation: gradientShift 15s ease infinite; }
@keyframes gradientShift {0% {background-position:0% 50%;}50% {background-position:100% 50%;}100% {background-position:0% 50%;}}
h1,h2,h3 {background: linear-gradient(135deg,#fffff0,#ffffff,#ff8c42);-webkit-background-clip:text;-webkit-text-fill-color:transparent; filter: drop-shadow(0 0 10px rgba(255,215,0,0.3));}
.card {padding:25px;border-radius:20px;background:linear-gradient(135deg,#2a2a2a 0%,#3d3d3d 50%,#2d1b2b 100%);border:1px solid rgba(255,215,0,0.2);box-shadow:0 8px 32px rgba(0,0,0,0.6);margin-bottom:20px;position:relative;overflow:hidden;}
.stButton>button {background:linear-gradient(135deg,#ff8c42,#ffd700,#ff6b6b);background-size:200% 200%;color:#fff;font-weight:bold;border-radius:12px;padding:10px 20px;border:2px solid rgba(255,215,0,0.3);}
</style>
""", unsafe_allow_html=True)

# --- Load Keras Model ---
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"
model = load_model(MODEL_PATH, compile=False)
with open(LABELS_PATH, "r") as f:
	class_names = [line.strip() for line in f.readlines()]

# --- Helper Functions ---
def preprocess_frame(frame):
	"""Convert OpenCV frame to model-ready array."""
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	img = Image.fromarray(img)
	img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
	img_array = np.asarray(img).astype(np.float32)
	img_array = (img_array / 127.5) - 1  # Normalize [-1,1]
	return np.expand_dims(img_array, axis=0)

def predict_frame(frame, confidence_threshold=0.5):
	data = preprocess_frame(frame)
	prediction = model.predict(data)
	index = np.argmax(prediction)
	confidence = prediction[0][index]
	class_name = class_names[index]
	if confidence >= confidence_threshold:
		return class_name, confidence
	else:
		return None, confidence

# --- Session State ---
for key in ["asl_text", "current_prediction", "prediction_confidence", "last_prediction_time"]:
	if key not in st.session_state:
		st.session_state[key] = "" if "text" in key or "prediction" in key else 0.0

# --- Sidebar Controls ---
st.sidebar.title("Controls")
run = st.sidebar.checkbox("▶️ Start Recognition")
mode = st.sidebar.radio("Mode", ["🎯 Translation", "📚 Training"])
confidence_threshold = st.sidebar.slider("Recognition Sensitivity", 0.3, 0.9, 0.7, 0.1)
auto_audio = st.sidebar.checkbox("Auto-play audio", value=True)

# --- Main Layout ---
col_main, col_right = st.columns([3,1])
FRAME_WINDOW = col_main.empty()
audio_placeholder = col_main.empty()
text_placeholder = col_main.empty()
training_feedback = col_main.empty()

if run:
	if "camera" not in st.session_state:
		st.session_state.camera = cv2.VideoCapture(0)

	while run:
		ret, frame = st.session_state.camera.read()
		if not ret:
			st.error("Cannot access camera")
			break

		FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

		now = time.time()
		if now - st.session_state.last_prediction_time > 1.0:
			prediction, confidence = predict_frame(frame, confidence_threshold)
			if prediction:
				st.session_state.current_prediction = prediction
				st.session_state.prediction_confidence = confidence
				if st.session_state.asl_text:
					if st.session_state.asl_text.split()[-1] != prediction:
						st.session_state.asl_text += " " + prediction
				else:
					st.session_state.asl_text = prediction
			st.session_state.last_prediction_time = now

		# Display hierarchy
		if mode == "🎯 Translation":
			if st.session_state.asl_text and auto_audio:
				try:
					tts = gTTS(st.session_state.asl_text)
					temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
					tts.save(temp_file.name)
					audio_placeholder.audio(temp_file.name, format="audio/mp3")
				except Exception as e:
					audio_placeholder.error(f"Audio error: {e}")

			if st.session_state.asl_text:
				text_placeholder.markdown(f"<div style='padding:25px; border-radius:15px; background:#2a2a2a; color:#fff; font-size:28px; text-align:center;'>{st.session_state.asl_text}</div>", unsafe_allow_html=True)
			else:
				text_placeholder.info("Start signing to see translations!")

		elif mode == "📚 Training":
			if st.session_state.current_prediction:
				conf = st.session_state.prediction_confidence
				if conf >= confidence_threshold:
					training_feedback.success(f"✅ I see: **{st.session_state.current_prediction}**")
				else:
					training_feedback.info(f"🤔 I think I see: **{st.session_state.current_prediction}**")
			else:
				training_feedback.info("Show me an ASL sign!")

	st.session_state.camera.release()
	del st.session_state.camera
else:
	st.info("Webcam is off. Toggle the checkbox to start recognition.")
