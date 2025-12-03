from transformers import pipeline
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import cv2 

IMAGES_PATH = '../ic-images'

image_paths = [
    f'{IMAGES_PATH}/A-J-28SOP-03F-SM.png',
    f'{IMAGES_PATH}/C-T-08DIP-11F-SM.png',
    f'{IMAGES_PATH}/C-T-48QFP-19F-SM.png',
    f'{IMAGES_PATH}/C-T-48QFP-20F-SM.png'
]

def show(img, title="", size=(6,6)):
    plt.figure(figsize=size)
    if len(img.shape) == 2:   # grayscale
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()



pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
for img in image_paths:
    image = Image.open(img)
    og_image = np.array(image)
    depth = pipe(image)["depth"]
    depth_np = np.array(depth)

    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    _, ic_packaging_thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    ic_packaging_thresh = cv2.bitwise_not(ic_packaging_thresh)

    _, thresh_bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    ic_packaging_contours, _ = cv2.findContours(ic_packaging_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(ic_packaging_contours)
    largest_contour = max(
        ic_packaging_contours,
        key=lambda c: cv2.contourArea(c) if cv2.contourArea(c) < 0.90 * og_image.shape[0]*og_image.shape[1] else 0
    )
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Expand the bounding box slightly (e.g., 5 pixels)
    pad = 10
    x = max(x - pad, 0)
    y = max(y-pad//2, 0)
    w = min(w + 2*pad, og_image.shape[1] - x)
    h = min(h + 2*pad, og_image.shape[0] - y)
    boxed_img = og_image.copy()
    cv2.rectangle(boxed_img, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # pin_contours = [c for c in contours]
    # output = np.array(image).copy()
    # cv2.drawContours(output, pin_contours, -1, (0, 255, 0), 2)

    # for c in pin_contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     cv2.rectangle(output, (x,y), (x+w,y+h), (0,255,0), 2)



    show(boxed_img, "test")
    # show(thresh_bright, "test")

