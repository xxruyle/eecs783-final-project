# EECS783 Final Project 
Project 2 

## Setup 
Install <a href="https://ollama.com/download">Ollama</a> to support vision language model text extraction
- remember to run `ollama pull <model>` 
- example: `ollama pull qwen3-vl:8b`
```
python -m venv .venv 

source .venv/Scripts/activate # if on windows 

source .venv/bin/activate # if on linux 

pip install -r requirements.txt
```

## About 
### Text Extraction
The text extraction uses the ollama vision language model quen3-vl and easyocr 
- Better performance than easyocr!

### Pin Detection 
Depth Anything -> Guassian blurred Grayscale BITWISE_OR IC Mask -> Find Contours -> Bonding Boxes (BB) -> DBSCAN Cluster BB locations
- bounding boxes that are greater than a std deviation from cluster mean width and height = defect 
