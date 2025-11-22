# EECS783 Final Project 
Project 2 

## Setup 
Install <a href="https://ollama.com/download">Ollama</a> to support vision language model text extraction
```
python -m venv .venv 

source .venv/Scripts/activate # if on windows 

source .venv/bin/activate # if on linux 

pip install -r requirements.txt
```

## About 
### Text Extraction
The text extraction uses the ollama vision language model quen3-vl 
- Better performance than easyocr!

### Pin Detection 
Uses OpenCV
- Binary Threshold Inversion 
- Morphology 
- Canny edge detection 
- Contours -> bounding boxes  