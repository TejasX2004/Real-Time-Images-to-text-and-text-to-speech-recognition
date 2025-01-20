import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from gtts import gTTS
import base64
import io
import nltk
from nltk.tokenize import sent_tokenize

# Add debug mode
DEBUG = True

def debug_log(message):
    if DEBUG:
        st.write(f"Debug: {message}")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def safe_image_open(upload):
    """
    Safely open and process uploaded images with extensive error checking
    """
    try:
        debug_log("Starting image processing")
        
        # Read image bytes
        image_bytes = upload.read()
        debug_log(f"Read {len(image_bytes)} bytes")
        
        # Create PIL Image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        debug_log(f"Original image mode: {img.mode}, size: {img.size}")
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
            debug_log("Converted to RGB mode")
        
        # Convert to numpy array
        np_img = np.array(img)
        debug_log(f"Numpy array shape: {np_img.shape}")
        
        return np_img
    except Exception as e:
        st.error(f"Error in image processing: {str(e)}")
        return None

def process_image_for_ocr(frame):
    """
    Process image for better OCR results
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply dilation to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)
        
        return gray
    except Exception as e:
        st.error(f"Error in OCR preprocessing: {str(e)}")
        return None

def recognize_text(frame, reader):
    """
    Detects and recognizes text with error handling
    """
    try:
        debug_log("Starting text recognition")
        results = reader.readtext(frame)
        detected_texts = []
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Confidence threshold
                detected_texts.append((bbox, text))
        debug_log(f"Found {len(detected_texts)} text regions")
        return detected_texts
    except Exception as e:
        st.error(f"Error in text recognition: {str(e)}")
        return []

def create_audio(text):
    """
    Creates audio with error handling
    """
    try:
        tts = gTTS(text=text, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        st.error(f"Error creating audio: {str(e)}")
        return None

# Streamlit App
st.title("Real-Time-Images-to-Text-Text-to-Speech")

# Initialize OCR reader
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

# Initialize session state
if 'detected_texts' not in st.session_state:
    st.session_state.detected_texts = []

# Option for input source
source = st.radio("Select Input Source", ("Upload", "Webcam"))

if source == "Upload":
    st.write("Note: Please wait after uploading while the image is processed.")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        try:
            # Show a spinner while processing
            with st.spinner('Processing image... Please wait.'):
                # Process the image
                frame = safe_image_open(uploaded_file)
                
                if frame is not None:
                    # Display original image
                    st.image(frame, caption="Original Image", use_column_width=True)
                    
                    # Process image for OCR
                    processed_frame = process_image_for_ocr(frame)
                    
                    if processed_frame is not None:
                        # Perform OCR
                        detected_texts = recognize_text(processed_frame, reader)
                        
                        if detected_texts:
                            # Draw boxes on a copy of the original image
                            annotated_frame = frame.copy()
                            for (bbox, text) in detected_texts:
                                points = np.array(bbox).astype(np.int32)
                                cv2.polylines(annotated_frame, [points], True, (255, 0, 0), 2)
                                
                            # Display annotated image
                            st.image(annotated_frame, caption="Detected Text Regions", use_column_width=True)
                            
                            # Store unique texts
                            new_texts = [text for _, text in detected_texts]
                            st.session_state.detected_texts.extend(text for text in new_texts 
                                                               if text not in st.session_state.detected_texts)
                        else:
                            st.warning("No text detected in the image.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif source == "Webcam":
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        frame = safe_image_open(img_file_buffer)
        if frame is not None:
            processed_frame = process_image_for_ocr(frame)
            if processed_frame is not None:
                detected_texts = recognize_text(processed_frame, reader)
                if detected_texts:
                    annotated_frame = frame.copy()
                    for (bbox, text) in detected_texts:
                        points = np.array(bbox).astype(np.int32)
                        cv2.polylines(annotated_frame, [points], True, (255, 0, 0), 2)
                    
                    st.image(annotated_frame, caption="Detected Text Regions", use_column_width=True)
                    
                    new_texts = [text for _, text in detected_texts]
                    st.session_state.detected_texts.extend(text for text in new_texts 
                                                       if text not in st.session_state.detected_texts)
                else:
                    st.warning("No text detected in the image.")

# Display detected texts and create audio
if st.session_state.detected_texts:
    st.write("### Detected Texts:")
    for i, text in enumerate(st.session_state.detected_texts, 1):
        st.write(f"{i}. {text}")
    
    if st.button("Generate Audio"):
        combined_text = " ".join(st.session_state.detected_texts)
        audio_bytes = create_audio(combined_text)
        
        if audio_bytes:
            st.write("### Audio:")
            st.download_button(
                label="Download Audio",
                data=audio_bytes,
                file_name="text_audio.mp3",
                mime="audio/mp3"
            )
            
            audio_b64 = base64.b64encode(audio_bytes).decode()
            st.markdown(f'<audio controls><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
                        unsafe_allow_html=True)

# Clear button
if st.button("Clear All"):
    st.session_state.detected_texts = []
    st.success("All cleared!")
