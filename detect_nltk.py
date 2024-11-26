import cv2
import torch
import pyttsx3
import threading
import numpy as np
from craft_text_detector import Craft
import easyocr
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import download

# Download NLTK resources
download('punkt')
download('stopwords')
stop_words = set(stopwords.words('english'))

# Text-to-Speech Function
def text_to_speech(text):
    """
    Converts text to speech using pyttsx3 (offline and faster).
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Text Cleaning Function
def clean_text(text):
    """
    Cleans and summarizes text using NLP techniques.
    """
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize sentences
    sentences = sent_tokenize(text)
    # Remove stop words and redundant spaces
    cleaned_sentences = []
    for sentence in sentences:
        words = sentence.split()
        words = [word for word in words if word.lower() not in stop_words]
        cleaned_sentences.append(" ".join(words))
    return " ".join(cleaned_sentences)

# Text Detection Function using CRAFT
def detect_text_with_craft(frame, craft):
    """
    Detects text regions in a frame using the CRAFT model.
    """
    prediction_result = craft.detect_text(frame)
    detected_text_regions = prediction_result["boxes"]
    return detected_text_regions

# Text Recognition Function
def recognize_text(frame, bbox, reader):
    """
    Recognizes text within bounding boxes using EasyOCR.
    """
    detected_texts = []
    for box in bbox:
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_region = frame[y_min:y_max, x_min:x_max]
        results = reader.readtext(cropped_region)
        for _, text, prob in results:
            if prob > 0.5:  # Confidence threshold
                detected_texts.append(text)
    return detected_texts

# Real-Time Pipeline Function
def real_time_text_to_speech():
    """
    Captures video from the webcam, detects text in real-time, and converts it to speech.
    """
    # Check if CUDA is available and choose the appropriate device
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use CUDA if available
    else:
        device = torch.device("cpu")  # Fallback to CPU
    print(f"Using device: {device}")

    # Initialize CRAFT for text detection
    craft = Craft(output_dir=None, crop_type="box")
    # Initialize EasyOCR for text recognition
    reader = easyocr.Reader(['en'])

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not found or could not be opened!")
        return

    print("Press 'q' to quit.")
    spoken_cache = set()  # Cache to avoid repeating the same speech
    frame_count = 0
    N = 10  # Process every 10th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not captured!")
            break

        frame_resized = cv2.resize(frame, (640, 480))  # Resize for faster processing

        if frame_count % N == 0:
            # Detect text regions using CRAFT
            text_regions = detect_text_with_craft(frame_resized, craft)

            # Recognize text within detected regions
            detected_texts = recognize_text(frame_resized, text_regions, reader)

            for text in detected_texts:
                # Clean and refine text using NLP
                cleaned_text = clean_text(text)
                if cleaned_text and cleaned_text not in spoken_cache:
                    spoken_cache.add(cleaned_text)
                    threading.Thread(target=text_to_speech, args=(cleaned_text,)).start()

                # Draw bounding boxes and text on the frame
                for box in text_regions:
                    x_min, y_min, x_max, y_max = map(int, box)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        frame_count += 1
        cv2.imshow("Real-Time Text Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the system
if __name__ == "__main__":
    real_time_text_to_speech()
