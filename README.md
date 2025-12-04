# EECS783 Final Project 
Project 2 

## Setup 
Install python tkinter with `sudo apt-get install python3-tk`
- TODO: add windows instructions

Install <a href="https://ollama.com/download">Ollama</a> to support vision language model text extraction
After installing ollama, run: `ollama pull qwen3-vl:235b-cloud`. 
Then, run `ollama signin` and sign in with your account (you may need to create one, but the account has free starting cloud usage)
- To use other models, install them with `ollama pull <model>` and update `OLLAMA_MODEL` in text/vlm_test.py

Setting up virtual environment and installing requirements:
```
python -m venv .venv 

source .venv/Scripts/activate # if on windows 

source .venv/bin/activate # if on linux 

pip install -r requirements.txt
```

## Running
```
source .venv/Scripts/activate # if on windows 

source .venv/bin/activate # if on linux 

python3 pin_detect.py # for pin detection

python3 text_detect.py # for text detection
```

## About 
### Text Extraction
The text extraction uses the ollama vision language model quen3-vl and easyocr 
- Better performance than easyocr!

### Pin Detection 
Depth Anything -> Guassian blurred Grayscale BITWISE_OR IC Mask -> Find Contours -> Bonding Boxes (BB) -> DBSCAN Cluster BB locations
- bounding boxes that are greater than a std deviation from cluster mean width and height = defect 
