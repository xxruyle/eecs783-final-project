from collections import Counter

import cv2
import easyocr
import numpy as np

IMAGES_PATH = "ic-marking-images"
EASY_OCR_READER = easyocr.Reader(['en'])


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
        if (ch in expected_output[true_detection_count:] and expected_output_counter[ch] > 0): 
            true_detection_count += 1   
            expected_output_counter[ch] -= 1 
        else: 
            false_detection_count += 1 


    print(f"Correct Predictions: {true_detection_count}/{len(expected_output)}\nFalse Detections: {false_detection_count}")


def get_ic_body_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)


    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # skip background

    final_mask = np.zeros_like(mask)
    final_mask[labels == largest_label] = 255
    return final_mask


def extract_text(img_filepath='./ic-marking-images/A-J-28SOP-03F-SM.png'): 
    image = cv2.imread(img_filepath)

    mask = get_ic_body_mask(image)

    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        raise ValueError("No dark IC region found in image!")

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    boxed = image.copy()
    cv2.rectangle(boxed, (x0, y0), (x1, y1), (0, 255, 0), 3)
    cv2.imwrite('./annotated_output.png', boxed)

    results = EASY_OCR_READER.readtext(boxed)

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
