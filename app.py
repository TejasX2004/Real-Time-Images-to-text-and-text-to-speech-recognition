import streamlit as st
import easyocr
import pyttsx3
import threading
import numpy as np
import cv2
from PIL import Image


# Text-to-Speech Function
def text_to_speech(text):
    """
    Converts text to speech using pyttsx3 (offline and faster).
    """
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.warning(f"Text-to-speech failed: {text}")
        # Optionally store the text in a list to display
        if 'tts_failed_texts' not in st.session_state:
            st.session_state.tts_failed_texts = []
        st.session_state.tts_failed_texts.append(text)


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

reader = easyocr.Reader(['en'])  # Initialize OCR reader
spoken_cache = set()  # Cache to avoid repeating the same speech

# Option for webcam or file upload
source = st.radio("Select Input Source", ("Webcam", "Upload"))

if source == "Webcam":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        frame = np.array(image)

        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Recognize text
        detected_texts = recognize_text(frame_gray, reader)
        for (bbox, text) in detected_texts:
            if text not in spoken_cache:
                spoken_cache.add(text)
                threading.Thread(target=text_to_speech, args=(text,)).start()

            # Draw bounding box on frame
            x_min = int(min([point[0] for point in bbox]))
            y_min = int(min([point[1] for point in bbox]))
            x_max = int(max([point[0] for point in bbox]))
            y_max = int(max([point[1] for point in bbox]))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display annotated image
        st.image(frame, caption="Detected Text with Annotations", channels="BGR")

elif source == "Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)

        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Recognize text
        detected_texts = recognize_text(frame_gray, reader)
        for (bbox, text) in detected_texts:
            if text not in spoken_cache:
                spoken_cache.add(text)
                threading.Thread(target=text_to_speech, args=(text,)).start()

            # Draw bounding box on frame
            x_min = int(min([point[0] for point in bbox]))
            y_min = int(min([point[1] for point in bbox]))
            x_max = int(max([point[0] for point in bbox]))
            y_max = int(max([point[1] for point in bbox]))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display annotated image
        st.image(frame, caption="Detected Text with Annotations", channels="BGR")
