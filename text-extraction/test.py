import cv2
import numpy as np 
import easyocr
from collections import Counter


IMAGES_PATH = "ic-marking-images"

tests = [
    (f'./{IMAGES_PATH}/A-J-28SOP-03F-SM.png',  "cy8c27443-24pvxi2001b05cyp603161c"),
    (f'./{IMAGES_PATH}/C-T-08DIP-11F-SM.png',  "92aet6g3adc0732ccn"),
    (f'./{IMAGES_PATH}/C-T-48QFP-19F-SM.png',  "stm32f103c8t6991rx019umys99008e42"),
    (f'./{IMAGES_PATH}/C-T-48QFP-20F-SM.png',  "stm32f103c8t6991uj019umys99009e42")
] # tuple containing image filepath and the visible text on the ic 

def test_extracted_text(extracted_texts, expected_output):
    true_detection_count = 0 
    false_detection_count = 0 
    expected_output_counter = Counter(expected_output)
    for ch in extracted_texts: 
        if (ch in expected_output_counter and expected_output_counter[ch] > 0): 
            true_detection_count += 1   
            expected_output_counter[ch] -= 1 
        else: 
            false_detection_count += 1 


    print(f"Correct Predictions: {true_detection_count}/{len(expected_output)}\nFalse Detections: {false_detection_count}")



def extract_text(img_filepath='./ic-marking-images/C-T-48QFP-20F-SM.png'): 
    image = cv2.imread(img_filepath)
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)

    extracted = []
    for box, text, score in results:
        print(f"Detected: {text}  (confidence {score:.2f})")
        for t in text: 
            extracted.append(t.lower())

    return extracted


def main(): 
    for img_filepath, expected_output in tests: 
        extracted_texts = extract_text(img_filepath=img_filepath) 
        test_extracted_text(extracted_texts, expected_output)




if __name__ == "__main__": 
    main()