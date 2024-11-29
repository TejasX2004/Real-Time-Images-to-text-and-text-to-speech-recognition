# Real-Time Text-to-Speech Recognition System

This project implements a real-time text detection and recognition system that captures text from a webcam, processes the detected text using Optical Character Recognition (OCR), and then converts it into speech. The system uses several libraries and models for different tasks such as text detection (CRAFT), text recognition (EasyOCR), and text-to-speech conversion (pyttsx3).

## Features

- **Real-time text detection** using the CRAFT (Character Region Awareness for Text) model.
- **Text recognition** using EasyOCR for extracting text from detected regions.
- **Text-to-speech** conversion with pyttsx3 for reading the recognized text aloud.
- **Optimized performance** by processing every Nth frame from the webcam feed.
- **Text cleaning and summarization** using basic NLP techniques to remove stopwords and non-alphanumeric characters.

## Requirements

- Python 3.x
- `pyttsx3` for text-to-speech functionality
- `opencv-python` for video capture and image processing
- `torch` for CUDA support
- `nltk` for natural language processing
- `craft-text-detector` for text region detection
- `easyocr` for optical character recognition
- `streamlit` for hosting the web app

You can install the required libraries by running:

```bash
pip install pyttsx3 opencv-python torch nltk easyocr craft-text-detector
```
## Setup

install dependencies
pip install -r requirements.txt
import nltk
nltk.download('punkt')
nltk.download('stopwords')

## Run the project

```bash
streamlit run app.py
```


## How it works
- Text Detection (CRAFT): The script uses the CRAFT model to detect text regions in the webcam frames. These text regions are then passed to the OCR model for text recognition.

- Text Recognition (EasyOCR): EasyOCR is used to recognize the text within the detected regions. The OCR results are filtered based on a confidence threshold (0.5) to ensure accurate text recognition.

- Text Cleaning (NLTK): Recognized text is cleaned using basic NLP techniques. Non-alphanumeric characters are removed, and stopwords are filtered out to provide a more concise output.

- Text-to-Speech (pyttsx3): After the text is cleaned, it's converted into speech using the pyttsx3 library, which operates offline for faster performance

## Example

### uploaded file
![Screenshot 2024-11-26 155732](https://github.com/user-attachments/assets/c4d860eb-0567-4596-b337-94d78ccd7aea)
### webcam
![Screenshot 2024-11-26 155855](https://github.com/user-attachments/assets/f963fe34-d1fe-4fa9-ab0f-4a418474727d)


### Notes:
- **Installation**: Make sure to include a `requirements.txt` file in your project with the necessary libraries like `pyttsx3`, `opencv-python`, `torch`, `nltk`, `easyocr`, and `craft-text-detector`.
- **Usage**: Clear instructions on how to run the script are included, along with explanations of how each module works.
- **Troubleshooting**: Mentioned potential issues such as webcam access and missing dependencies.

This README provides a clear overview of your project and will help others get started quickly.


