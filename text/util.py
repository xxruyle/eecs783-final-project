import easyocr
import ollama 

def easyocr_extract_text(EASY_OCR_READER, img_filepath='./ic-marking-images/A-J-28SOP-03F-SM.png'): 
    results = EASY_OCR_READER.readtext(img_filepath)
    extracted = ""
    for box, text, score in results:
        extracted += text

    return extracted


def quen_extract_text(quen3_mdl, image_path):
    response = ollama.chat(
        model=quen3_mdl,
        messages=[{
            'role': 'user',
            'content': 'What is the text on this IC packaging? Only give me the exact text (no commentary)',
            'images': [f'{image_path}']  
        }]
    )

    res = response['message']['content']
    return res
