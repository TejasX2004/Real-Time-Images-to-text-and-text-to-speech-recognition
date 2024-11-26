import cv2
import easyocr
import pyttsx3
import threading
import numpy as np


# Text-to-Speech Function
def text_to_speech(text):
    """
    Converts text to speech using pyttsx3 (offline and faster).
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


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


# Real-Time Pipeline Function
def real_time_text_to_speech():
    """
    Captures video from the webcam, detects text in real-time, and converts it to speech.
    """
    reader = easyocr.Reader(['en'])  # Initialize OCR reader
    cap = cv2.VideoCapture(0)  # Start webcam

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

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frame_resized = cv2.resize(frame_gray, (640, 480))  # Resize for faster processing

        if frame_count % N == 0:
            detected_texts = recognize_text(frame_resized, reader)

            for (bbox, text) in detected_texts:
                if text not in spoken_cache:
                    spoken_cache.add(text)
                    threading.Thread(target=text_to_speech, args=(text,)).start()

                # Draw bounding box and text on the frame
                x_min = int(min([point[0] for point in bbox]))
                y_min = int(min([point[1] for point in bbox]))
                x_max = int(max([point[0] for point in bbox]))
                y_max = int(max([point[1] for point in bbox]))
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
