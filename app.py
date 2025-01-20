import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from gtts import gTTS
import base64
import os
import nltk
from nltk.tokenize import sent_tokenize
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def safe_image_open(upload):
    """
    Safely open and process uploaded images, including those from smartphones
    """
    try:
        # Read the file into bytes
        bytes_data = upload.getvalue()
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(bytes_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to numpy array
        frame = np.array(image)
        
        return frame, None
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

def process_text(texts):
    """
    Process and normalize detected texts into proper sentences.
    """
    if not texts:
        return ""
        
    # Combine all texts with spaces
    combined_text = ' '.join(texts)

    # Basic text normalization
    normalized = combined_text.strip()
    normalized = ' '.join(normalized.split())  # Remove extra whitespace

    # Split into sentences and recombine with proper spacing
    try:
        sentences = sent_tokenize(normalized)
        processed_text = ' '.join(sentences)
    except:
        processed_text = normalized

    return processed_text

def create_combined_audio(texts):
    """
    Creates a single audio file from multiple texts with pauses between sentences.
    """
    if not texts:
        return None, ""

    # Process and normalize the text
    processed_text = process_text(texts)

    try:
        # Create audio file
        tts = gTTS(text=processed_text, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read(), processed_text
    except Exception as e:
        st.error(f"Error creating audio: {str(e)}")
        return None, processed_text

def recognize_text(frame, reader):
    """
    Detects and recognizes text in the given frame using EasyOCR.
    """
    try:
        results = reader.readtext(frame)
        detected_texts = []
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Confidence threshold
                detected_texts.append((bbox, text))
        return detected_texts, None
    except Exception as e:
        return None, f"Error detecting text: {str(e)}"

# Streamlit App
st.title("Real-Time-Images-to-Text-Text-to-Speech")
st.write("Upload an image or use your webcam to detect text and convert it to speech.")

# Initialize OCR reader
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

# Store detected texts
if 'detected_texts' not in st.session_state:
    st.session_state.detected_texts = []

# Option for webcam or file upload
source = st.radio("Select Input Source", ("Upload", "Webcam"))

if source == "Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Show loading spinner while processing
        with st.spinner('Processing image...'):
            # Safely open and process the image
            frame, error = safe_image_open(uploaded_file)
            
            if error:
                st.error(error)
            elif frame is not None:
                try:
                    # Convert frame to grayscale for OCR
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                    # Recognize text
                    detected_texts, error = recognize_text(frame_gray, reader)
                    
                    if error:
                        st.error(error)
                    elif detected_texts:
                        # Store unique texts
                        current_texts = [text for _, text in detected_texts]
                        st.session_state.detected_texts.extend(text for text in current_texts
                                                           if text not in st.session_state.detected_texts)

                        # Process detected text
                        frame_with_boxes = frame.copy()
                        for (bbox, text) in detected_texts:
                            # Draw bounding box
                            points = np.array(bbox).astype(np.int32)
                            x_min, y_min = points.min(axis=0)
                            x_max, y_max = points.max(axis=0)
                            cv2.rectangle(frame_with_boxes, 
                                        (int(x_min), int(y_min)), 
                                        (int(x_max), int(y_max)), 
                                        (0, 255, 0), 2)
                            cv2.putText(frame_with_boxes, text, 
                                      (int(x_min), int(y_min) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                        # Display annotated image
                        st.image(frame_with_boxes, caption="Detected Text with Annotations")
                    else:
                        st.warning("No text detected in the image.")
                        st.image(frame, caption="Uploaded Image")
                        
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    st.image(frame, caption="Uploaded Image (Error in Processing)")

elif source == "Webcam":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        # Use the same safe image processing for webcam input
        frame, error = safe_image_open(img_file_buffer)
        
        if error:
            st.error(error)
        elif frame is not None:
            # Use the same processing logic as for uploaded images
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            detected_texts, error = recognize_text(frame_gray, reader)
            
            if error:
                st.error(error)
            elif detected_texts:
                # Process and display results (same as upload section)
                current_texts = [text for _, text in detected_texts]
                st.session_state.detected_texts.extend(text for text in current_texts
                                                   if text not in st.session_state.detected_texts)

                frame_with_boxes = frame.copy()
                for (bbox, text) in detected_texts:
                    points = np.array(bbox).astype(np.int32)
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)
                    cv2.rectangle(frame_with_boxes, 
                                (int(x_min), int(y_min)), 
                                (int(x_max), int(y_max)), 
                                (0, 255, 0), 2)
                    cv2.putText(frame_with_boxes, text, 
                              (int(x_min), int(y_min) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                st.image(frame_with_boxes, caption="Detected Text with Annotations")
            else:
                st.warning("No text detected in the image.")
                st.image(frame, caption="Captured Image")

# Display detected texts and create combined audio
if st.session_state.detected_texts:
    st.write("### Detected Texts:")
    for i, text in enumerate(st.session_state.detected_texts, 1):
        st.write(f"{i}. {text}")

    if st.button("Generate Combined Audio"):
        audio_bytes, processed_text = create_combined_audio(st.session_state.detected_texts)

        if audio_bytes:
            st.write("### Processed Text:")
            st.write(processed_text)

            st.write("### Listen Audio:")
            # Create download button
            st.download_button(
                label="Download Audio",
                data=audio_bytes,
                file_name="combined_text_audio.mp3",
                mime="audio/mp3"
            )

            # Create audio player
            audio_b64 = base64.b64encode(audio_bytes).decode()
            audio_tag = f'<audio controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
            st.markdown(audio_tag, unsafe_allow_html=True)

# Add a button to clear all detected texts
if st.button("Clear All Detected Texts"):
    st.session_state.detected_texts = []
    st.success("All detected texts cleared!")
