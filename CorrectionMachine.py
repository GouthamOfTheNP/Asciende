import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import os
import random
from PIL import Image


st.set_page_config(page_title="Asciende ASL Corrections", layout="wide")
st.title("Asciende ASL Correction Demo")


DATASET_PATH = "asl_dataset"


reference_images = {}
feedback_dict = {
    "A": "Keep your fingers together and curl the thumb across.",
    "B": "Hold your fingers straight and together, thumb across palm.",
    "C": "Curve your hand to form a C shape.",
    "D": "Touch the tip of the middle, ring, and pinky fingers to the thumb, index finger upright.",
    "E": "Curl your fingers down to touch your thumb, forming a closed shape.",
    "F": "Form an 'OK' sign: thumb and index finger touch, other fingers extended.",
    "G": "Hold your index finger straight, thumb out, other fingers curled.",
    "H": "Hold index and middle fingers straight and together, thumb across palm.",
    "I": "Curl all fingers except pinky, which points up.",
    "J": "Draw a 'J' with your pinky in the air.",
    "K": "Thumb between middle and index, other fingers straight.",
    "L": "Make an 'L' shape with thumb and index.",
    "M": "Place thumb under three fingers.",
    "N": "Place thumb under two fingers.",
    "O": "Form an 'O' with all fingers.",
    "P": "Thumb between middle and index, hand upside down.",
    "Q": "Thumb and index hold something, other fingers tucked.",
    "R": "Cross index and middle finger.",
    "S": "Make a fist, thumb in front.",
    "T": "Thumb under index finger.",
    "U": "Index and middle fingers straight together.",
    "V": "Index and middle fingers spread in a V shape.",
    "W": "Index, middle, and ring fingers spread.",
    "X": "Curl index finger, other fingers in a fist.",
    "Y": "Thumb and pinky extended, other fingers folded.",
    "SPACE": "Hold your palm sideways to indicate space."
}


image_extensions = [".jpg", ".jpeg", ".png"]

for letter in os.listdir(DATASET_PATH):
    letter_path = os.path.join(DATASET_PATH, letter)
    if os.path.isdir(letter_path):
        # Filter out only image files
        images = [
            f for f in os.listdir(letter_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        if images:
            img_path = os.path.join(letter_path, images[0])
            reference_images[letter] = Image.open(img_path)
        else:
            st.warning(f"No images found for letter '{letter}'")

letters = list(reference_images.keys())

# ------------------------
# Mock ASL Recognition
# ------------------------
def mock_asl_recognition(frame):
    """Randomly return a letter from the dataset (replace with your real model later)."""
    return random.choice(letters)

# ------------------------
# Video Processor
# ------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Draw a rectangle for fun/feedback area
        cv2.rectangle(img, (50,50), (250,250), (0,255,0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------------
# Webcam Stream
# ------------------------
webrtc_ctx = webrtc_streamer(
    key="asl-corrector",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
)

# ------------------------
# Show Correction Feedback
# ------------------------
if webrtc_ctx.video_processor:
    try:
        # Mock detection (replace with real model)
        predicted_letter = mock_asl_recognition(None)

        # Show text feedback
        st.subheader(f"Feedback for '{predicted_letter}'")
        st.text(feedback_dict[predicted_letter])

        # Show reference image
        st.image(reference_images[predicted_letter], width=200)
    except Exception:
        st.write("Waiting for camera input...")
