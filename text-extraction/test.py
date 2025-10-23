import cv2
import numpy as np 
import easyocr


def extract_text(img_filepath='./ic-marking-images/C-T-48QFP-20F-SM.png'): 
    image = cv2.imread(img_filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    reader = easyocr.Reader(['en'])
    results = reader.readtext(gray)

    for box, text, score in results:
        print(f"Detected: {text}  (confidence {score:.2f})")

        top_left = tuple(box[0])
        bottom_right = tuple(box[2])
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imwrite("annotated_output.png", image)






def main(): 
    extract_text()


if __name__ == "__main__": 
    main()