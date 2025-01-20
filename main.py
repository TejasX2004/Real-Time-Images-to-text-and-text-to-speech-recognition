import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from gtts import gTTS
import base64
import os


# Text-to-Speech Function using gTTS
def text_to_speech(text):
    """
    Converts text to speech using Google Text-to-Speech API.
    """
    try:
        tts = gTTS(text=text, lang='en')
        tts.save('temp.mp3')
        with open('temp.mp3', 'rb') as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        audio_tag = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_b64}">'
        st.markdown(audio_tag, unsafe_allow_html=True)
        os.remove('temp.mp3')
    except Exception as e:
        st.warning(f"Could not convert text to speech: {text}")


# Text Recognition Function
def recognize_text(frame, reader):
    """
    Detects and recognizes text in the given frame using EasyOCR.
    """
    results = reader.readtext(frame)
    detected_texts = []
    for (bbox, text, prob) in results:
        if prob > 0.5:  # Confidence threshold
            detected_texts.append((bbox, text))
    return detected_texts


# Streamlit App
st.title("Real-Time Text-to-Speech (TTS) System")
st.write("Upload an image or use your webcam to detect text and convert it to speech.")


# Initialize OCR reader
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])


reader = load_ocr()

# Cache to avoid repeating the same speech
if 'spoken_cache' not in st.session_state:
    st.session_state.spoken_cache = set()

# Option for webcam or file upload
source = st.radio("Select Input Source", ("Webcam", "Upload"))

if source == "Webcam":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        frame = np.array(image)

        # Convert frame to RGB if necessary
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Convert frame to grayscale for OCR
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Recognize text
        detected_texts = recognize_text(frame_gray, reader)

        # Process detected text
        for (bbox, text) in detected_texts:
            if text not in st.session_state.spoken_cache:
                st.session_state.spoken_cache.add(text)
                text_to_speech(text)

            # Draw bounding box
            points = np.array(bbox).astype(np.int32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display annotated image
        st.image(frame, caption="Detected Text with Annotations", channels="BGR")

elif source == "Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)

        # Convert frame to RGB if necessary
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Convert frame to grayscale for OCR
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Recognize text
        detected_texts = recognize_text(frame_gray, reader)

        # Process detected text
        for (bbox, text) in detected_texts:
            if text not in st.session_state.spoken_cache:
                st.session_state.spoken_cache.add(text)
                text_to_speech(text)

            # Draw bounding box
            points = np.array(bbox).astype(np.int32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display annotated image
        st.image(frame, caption="Detected Text with Annotations", channels="BGR")
