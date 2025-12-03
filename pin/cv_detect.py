import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import defect_images as image_paths
import matplotlib
matplotlib.use("tkagg")

def show(img, title="", size=(6,6)):
    print("\tshowing...")
    plt.figure(figsize=size)
    if len(img.shape) == 2:   # grayscale
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def detect_edges(img_path):
    # Load the IC image
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a slight Gaussian blur to reduce noise
    #gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use high-threshold inverted binary to capture shiny pins with shadows
    # _, thresh = cv2.threshold(gray_blur, 200, 255, cv2.THRESH_BINARY_INV)

    _, thresh_bright = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    #show(thresh_bright, f"threshbright {img_path}")
    edges = cv2.Canny(thresh_bright, 220, 255)

    kernel = np.ones((3, 1), np.uint8)
    closing = cv2.morphologyEx(thresh_bright, cv2.MORPH_CLOSE, kernel, iterations=1)
    #show(closing, f"closing {img_path}")
    dilated = cv2.dilate(closing, kernel, iterations=1)

    edges = cv2.Canny(thresh_bright, 220, 255)

    combined = cv2.bitwise_or(dilated, edges)
    return edges, combined

def other_preprocessing(img_path):
    # Load the IC image
    img = cv2.imread(img_path)

    return img


def detect_pins(img_path="../ic-images/C-T-48QFP-19F-SM.png"):
    print(f"{img_path}") 
    img = cv2.imread(img_path)
    edges, combined = detect_edges(img_path)
    #combined = other_preprocessing(img_path)

    # Find contours of pins
    pin_contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours
    pin_contours = [c for c in pin_contours if cv2.contourArea(c) > 50 and cv2.contourArea(c) < 20000]

    # Draw detected pins
    output = img.copy()
    # cv2.drawContours(output, pin_contours, -1, (0, 255, 0), 2)

    for c in pin_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output, (x,y), (x+w,y+h), (0,0,0), 2)

    # Show results
    # show(img, "Original")
    # show(combined, "Combined")
    show(edges, f"Inverted {img_path}")
    show(combined, f"COmbined {img_path}")
    show(output, f"Detected Pins {img_path}")

def run_cv_img_detect():
    for path, _ in image_paths:
        detect_pins(path)

