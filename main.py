import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from gtts import gTTS
from keras.layers import TFSMLayer

np.set_printoptions(suppress=True)

logo = Image.open("asciendo.ico")
st.set_page_config(page_title="Asciende", page_icon=logo, layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #1f1f1f, #0a0a0a);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

h1, h2, h3 {
    background: linear-gradient(135deg, #fffff0, #ffffff, #ff8c42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.3));
    animation: textGlow 3s ease-in-out infinite alternate;
}

@keyframes textGlow {
    0% { filter: drop-shadow(0 0 10px rgba(255, 215, 0, 0.3)); }
    100% { filter: drop-shadow(0 0 20px rgba(255, 140, 66, 0.5)); }
}

.card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #2a2a2a 0%, #3d3d3d 50%, #2d1b2b 100%);
    border: 1px solid rgba(255, 215, 0, 0.2);
    box-shadow: 
        0 8px 32px rgba(0,0,0,0.6),
        inset 0 1px 0 rgba(255,255,255,0.1),
        0 0 20px rgba(255, 140, 66, 0.1);
    margin-bottom: 20px;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 215, 0, 0.1), transparent);
    transition: left 0.8s;
}

.card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 
        0 20px 40px rgba(0,0,0,0.7),
        inset 0 1px 0 rgba(255,255,255,0.2),
        0 0 30px rgba(255, 140, 66, 0.3);
}

.card:hover::before {
    left: 100%;
}

.stButton>button {
    background: linear-gradient(135deg, #ff8c42, #ffd700, #ff6b6b);
    background-size: 200% 200%;
    color: #ffffff;
    font-weight: bold;
    border-radius: 12px;
    padding: 10px 20px;
    border: 2px solid rgba(255, 215, 0, 0.3);
    transition: all 0.4s ease;
    box-shadow: 
        0 4px 15px rgba(255, 140, 66, 0.3),
        inset 0 1px 0 rgba(255,255,255,0.2);
    position: relative;
    overflow: hidden;
    animation: buttonGradient 3s ease infinite;
}

@keyframes buttonGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stButton>button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.6s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 
        0 8px 25px rgba(255, 140, 66, 0.5),
        inset 0 1px 0 rgba(255,255,255,0.3);
}

.stButton>button:hover::before {
    left: 100%;
}

.stAudio, .stInfo {
    margin-top: 12px;
    border-radius: 6px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    padding: 12px;
    color: #e5e5e5;
}

.webcam-off-container {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 70vh;
    width: 100%;
}

.webcam-off-card {
    padding: 32px;
    border-radius: 8px;
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    text-align: center;
    max-width: 400px;
    width: 100%;
}

.webcam-off-card h3 {
    color: #f5f5f5;
    margin-bottom: 12px;
}

.webcam-off-card p {
    color: #b5b5b5;
    line-height: 1.5;
}

.css-1d391kg {
    background: #2a2a2a;
    border-right: 1px solid #3a3a3a;
}

.css-1v0mbdj {
    background: transparent;
    border-radius: 0;
    margin: 0;
    border: none;
}

.prediction-container {
    background: #2a2a2a;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid rgba(255, 215, 0, 0.2);
}

.prediction-high {
    color: #4CAF50;
    font-weight: bold;
}

.prediction-medium {
    color: #FFC107;
    font-weight: bold;
}

.prediction-low {
    color: #757575;
}

.status-indicator {
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: bold;
    margin: 5px 0;
}

.status-active {
    background: rgba(76, 175, 80, 0.2);
    color: #4CAF50;
}

.status-inactive {
    background: rgba(158, 158, 158, 0.2);
    color: #9E9E9E;
}

.status-error {
    background: rgba(244, 67, 54, 0.2);
    color: #F44336;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
	class_names = None

	try:
		if os.path.exists("labels.txt"):
			with open("labels.txt", "r") as f:
				class_names = f.readlines()
		else:
			return None, None
	except Exception:
		return None, None

	model = TFSMLayer("model.savedmodel", call_endpoint="serving_default")
	return ("savedmodel", model), class_names


model = load_model()


def predict_with_savedmodel(image, model, class_names, confidence_threshold=0.5):
	try:
		image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
		if len(image.shape) == 3 and image.shape[2] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype(np.float32)
		image = np.expand_dims(image, axis=0)
		image = (image / 127.5) - 1
		input_tensor = tf.convert_to_tensor(image)

		if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
			inference_func = model.signatures['serving_default']
		else:
			inference_func = model

		prediction_tf = inference_func
		if isinstance(prediction_tf, dict):
			# Take the first tensor if output is a dict
			prediction_tensor = list(prediction_tf.values())[0]
		else:
			prediction_tensor = prediction_tf

		if isinstance(prediction_tensor, tf.Tensor):
			prediction = prediction_tensor.numpy()
		else:
			prediction = np.array(prediction_tensor)

		if prediction.ndim > 1:
			prediction = prediction[0]

		sorted_indices = np.argsort(prediction)[::-1]
		all_predictions = []
		for i in sorted_indices[:3]:
			class_name = class_names[i].strip()
			all_predictions.append({
				"class": class_name,
				"confidence": float(str(prediction[i]).split(" ")[0])
			})

		high_conf = [p for p in all_predictions if p["confidence"] >= confidence_threshold]
		return high_conf if high_conf else all_predictions[:1]

	except ArithmeticError as e:
		st.error(f"SavedModel prediction error: {str(e)}")
		return []


def predict_with_model(image, model_info, class_names, confidence_threshold=0.5):
	model_type, model = model_info
	return predict_with_savedmodel(image, model, class_names, confidence_threshold)


if "asl_text" not in st.session_state:
	st.session_state.asl_text = ""
if "current_prediction" not in st.session_state:
	st.session_state.current_prediction = ""
if "prediction_confidence" not in st.session_state:
	st.session_state.prediction_confidence = 0.0
if "last_prediction_time" not in st.session_state:
	st.session_state.last_prediction_time = 0

model_info, class_names = load_model()

with st.sidebar:
	st.image(logo, width=80)
	st.title("Controls")

	run = st.toggle("▶️ Start Recognition", help="Turn on webcam and ASL recognition")
	mode = st.radio("Mode", ["🎯 Translation", "📚 Training"], help="Choose your mode")

	st.markdown("---")

	with st.expander("⚙️ Settings", expanded=False):
		confidence_threshold = st.slider(
			"Recognition Sensitivity",
			min_value=0.3,
			max_value=0.9,
			value=0.7,
			step=0.1,
			help="Higher = more accurate, Lower = more responsive"
		)

		auto_audio = st.checkbox(
			"Auto-play audio",
			value=True,
			help="Automatically play audio when new signs are recognized"
		)

	st.subheader("📊 Rate this App")
	rating_input = st.feedback("stars")
	if rating_input:
		with open("rating.txt", "a") as f:
			f.write(str(rating_input + 1) + " ")

try:
	with open("rating.txt", "r") as f:
		ratings_list = [int(i) for i in f.read().split() if i.isnumeric()]
	avg_rating = sum(ratings_list) / len(ratings_list) if ratings_list else 0
except FileNotFoundError:
	avg_rating = 0

col_main, col_right = st.columns([3, 1])

with col_right:
	st.markdown("<div class='card'><h3>🚀 ASL Translator</h3><p>Point your camera at ASL signs and get instant "
	            "translation with audio playback using Keras deep learning.</p></div>",
	            unsafe_allow_html=True)

	if run:
		status_text = "🟢 Active"
		status_color = "#4CAF50"
	else:
		status_text = "🔴 Stopped"
		status_color = "#757575"

	st.markdown(
		f"<div class='card'><h3>Status</h3><p style='color: {status_color}; font-weight: bold;'>{status_text}</p></div>",
		unsafe_allow_html=True)

	if run and st.session_state.current_prediction:
		confidence = st.session_state.prediction_confidence
		if confidence >= confidence_threshold:
			st.markdown(f"""
            <div class='card'>
                <h3>🎯 Detected</h3>
                <p style='font-size: 32px; margin: 10px 0; text-align: center;'>{st.session_state.current_prediction}</p>
                <p style='color: #4CAF50; text-align: center; font-size: 14px;'>Confidence: {confidence:.0%}</p>
            </div>
            """, unsafe_allow_html=True)

	st.markdown(f"<div class='card'><h3>⭐ Rating</h3><p>{avg_rating:.1f} / 5.0</p></div>",
	            unsafe_allow_html=True)

with col_main:
	if run:
		if model is None or class_names is None:
			st.error("❌ Cannot start recognition: Keras model not loaded properly")
			st.info("Please ensure 'keras_Model.h5' and 'labels.txt' files are in the same directory as this script")
		else:
			if "camera" not in st.session_state:
				st.session_state.camera = cv2.VideoCapture(0)
			if "frame_count" not in st.session_state:
				st.session_state.frame_count = 0

			if mode == "🎯 Translation":
				FRAME_WINDOW = st.image([])
				audio_placeholder = st.empty()
				text_placeholder = st.empty()

			elif mode == "📚 Training":
				st.markdown("### 📚 Practice Mode")

				FRAME_WINDOW = st.image([])

				training_feedback = st.empty()

			frame_container = st.empty()

			while run:
				ret, frame = st.session_state.camera.read()
				if not ret:
					st.error("Failed to access camera")
					break

				frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				FRAME_WINDOW.image(frame_rgb, channels="RGB")

				current_time = time.time()
				prediction_interval = 1.0

				if (current_time - st.session_state.last_prediction_time) >= prediction_interval:
					predictions = predict_with_model(frame, model, class_names, confidence_threshold)

					if predictions:
						top_prediction = predictions[0]

						if top_prediction["confidence"] >= confidence_threshold:
							letter = ''.join([c for c in top_prediction["class"] if not c.isdigit()]).strip()
							st.session_state.current_prediction = letter
							st.session_state.prediction_confidence = top_prediction["confidence"]

							if st.session_state.asl_text:
								st.session_state.asl_text += letter
							else:
								st.session_state.asl_text = letter

					st.session_state.last_prediction_time = current_time

				if mode == "🎯 Translation":
					if st.session_state.asl_text and auto_audio:
						try:
							tts = gTTS(st.session_state.asl_text)
							temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
							tts.save(temp_file.name)

							with audio_placeholder.container():
								st.markdown("### 🔊 Audio Translation")
								st.audio(temp_file.name, format="audio/mp3")
						except Exception as e:
							with audio_placeholder.container():
								st.error(f"Audio error: {str(e)}")

					with text_placeholder.container():
						if st.session_state.asl_text:
							st.markdown("### 📝 Translated Text")
							st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #2a2a2a, #3d3d3d); 
                                        padding: 25px; border-radius: 15px; 
                                        border: 2px solid rgba(255, 215, 0, 0.3);
                                        box-shadow: 0 8px 32px rgba(0,0,0,0.6);
                                        margin: 20px 0;'>
                                <h2 style='color: #fff; margin: 0; font-size: 28px; text-align: center;'>
                                    {st.session_state.asl_text}
                                </h2>
                            </div>
                            """, unsafe_allow_html=True)
						else:
							st.markdown("### 📝 Translated Text")
							st.info("👋 Start signing in front of the camera to see translations appear here!")

				elif mode == "📚 Training":
					with training_feedback.container():
						if st.session_state.current_prediction:
							confidence = st.session_state.prediction_confidence
							if confidence >= confidence_threshold:
								st.success(
									f"✅ Great! I can see the sign for: **{st.session_state.current_prediction}** (Confidence: {confidence:.0%})")
							else:
								st.info(
									f"🤔 I think I see: **{st.session_state.current_prediction}** (Confidence: {confidence:.0%} - not confident enough)")
						else:
							st.info("👀 Show me an ASL sign and I'll try to recognize it!")

				st.session_state.frame_count += 1

				if not run:
					break

			if "camera" in st.session_state:
				st.session_state.camera.release()

	else:
		if "camera" in st.session_state:
			st.session_state.camera.release()
			del st.session_state.camera

		st.markdown("""
        <div class='webcam-off-container'>
            <div class='webcam-off-card'>
                <h3 style='color: #f5f5f5; margin-bottom: 12px;'>📷 Webcam is Off</h3>
                <p style='color: #b5b5b5; line-height: 1.5;'>Toggle 'Start Recognition' in the sidebar to begin real-time ASL recognition using your trained Keras model.</p>
                <br>
                <p style='color: #ff8c42; font-weight: bold;'>✨ Features:</p>
                <p style='color: #b5b5b5; font-size: 14px;'>
                • OpenCV video capture<br>
                • Keras deep learning model<br>
                • Real-time ASL sign recognition<br>
                • Confidence-based predictions<br>
                • Audio feedback generation<br>
                • Training mode with detailed feedback<br>
                • Customizable sensitivity settings
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
