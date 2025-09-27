import cv2
import streamlit as st
from gtts import gTTS
import tempfile

st.set_page_config(page_title="Asciende", page_icon="🤟")
st.title("Asciende - Real-time ASL Translation")
st.markdown("Real-time American Sign Language (ASL) Translation using Deep Learning")
st.markdown("This application captures real-time video from your webcam and translates ASL gestures into text using a pre-trained model.")

run = st.toggle("Run Webcam")
FRAME_WINDOW = st.image([])

st.session_state.asl_text = None

if "camera" not in st.session_state:
	st.session_state.camera = cv2.VideoCapture(0)

if "asl_text" not in st.session_state:
	st.session_state.asl_text = ""

i = 1
isIn = False
isIn2 = False
if run:
	while True:
		ret, frame = st.session_state.camera.read()
		if not ret:
			st.warning("Failed to grab frame")
			break
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		FRAME_WINDOW.image(frame)

		# --- ASL input simulation ---
		if not st.session_state.asl_text and not isIn:
			isIn = True
			st.session_state.asl_text = st.text_input(
			"ASL Output (for testing)", st.session_state.asl_text,
				key=i
		)

		if st.session_state.asl_text is not None and st.session_state.asl_text.strip() and not isIn2:
			isIn2 = True
			tts = gTTS(st.session_state.asl_text)
			temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
			tts.save(temp_file.name)

			st.audio(temp_file.name, format="audio/mp3")
			st.info("Playing latest text...")
		i += 1
		if not st.session_state.get("Run", True):
			break
else:
	if "camera" in st.session_state:
		st.session_state.camera.release()
		del st.session_state.camera