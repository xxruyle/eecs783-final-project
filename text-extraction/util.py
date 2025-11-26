import easyocr
import ollama 

IMAGES_PATH = "../ic-images"


tests = [
    (f'{IMAGES_PATH}/A-J-28SOP-03F-SM.png',  "cy8c27443-24pvxi2001b05cyp603161c"),
    (f'{IMAGES_PATH}/C-T-08DIP-11F-SM.png',  "92aet6g3adc0732ccn"),
    (f'{IMAGES_PATH}/C-T-48QFP-19F-SM.png',  "stm32f103c8t6991rx019umys99008e42"),
    (f'{IMAGES_PATH}/C-T-48QFP-20F-SM.png',  "stm32f103c8t6991uj019umys99009e42")
] # tuple containing image filepath and the visible text on the ic 

def easyocr_extract_text(EASY_OCR_READER, img_filepath='./ic-marking-images/A-J-28SOP-03F-SM.png'): 
    results = EASY_OCR_READER.readtext(img_filepath)
    extracted = []
    for box, text, score in results:
        extracted.append((text, score))

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
