from transformers import pipeline
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
import os 
from sklearn.cluster import DBSCAN
import statistics 
import random
from util import defect_images as image_paths

def show(img, title="", save=False, size=(6,6)):
    plt.figure(figsize=size)
    if len(img.shape) == 2:   # grayscale
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if(save):
        plt.savefig(f"./results/{title}")

    plt.title(title)
    plt.axis("off")
    plt.show()

def remove_nested_boxes(bounding_boxes):
    """
    Remove bounding boxes that are completely inside another bounding box.
    bounding_boxes: list of (x, y, w, h)
    """
    keep_boxes = []
    
    for i, (x1, y1, w1, h1) in enumerate(bounding_boxes):
        nested = False
        for j, (x2, y2, w2, h2) in enumerate(bounding_boxes):
            if i == j:
                continue
            # Check if box1 is fully inside box2
            if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                nested = True
                break
        if not nested:
            keep_boxes.append((x1, y1, w1, h1))
    
    return keep_boxes


def run_depth_detect():
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")
    for img_entry in image_paths:
        img = img_entry[0]
        image = Image.open(img)
        og_image = np.array(image)
        depth = pipe(image)["depth"]
        depth_np = np.array(depth)

        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_norm.astype(np.uint8)

        heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        gray =  cv2.GaussianBlur(gray, (3, 3), 0)

        _, ic_packaging_thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        ic_packaging_thresh = cv2.bitwise_not(ic_packaging_thresh)


        ic_packaging_contours, _ = cv2.findContours(ic_packaging_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(
            ic_packaging_contours,
            key=lambda c: cv2.contourArea(c) if cv2.contourArea(c) < 0.95 * og_image.shape[0]*og_image.shape[1] else 0
        )

        x, y, w, h = cv2.boundingRect(largest_contour)
        # Expand the bounding box slightly (e.g., 5 pixels)
        pad = 15
        x = max(x - pad*3, 0)
        y = max(y-pad*3, 0)
        w = min(w + 6*pad, og_image.shape[1] - x)
        h = min(h + 6*pad, og_image.shape[0] - y)
        boxed_img = og_image.copy()
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)

        _, thresh_bright = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_bright, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        pin_contours = [c for c in contours if cv2.contourArea(c) > 500 and cv2.contourArea(c) < 50000]
        output = np.array(image).copy()
        bounding_boxes = []
        for c in pin_contours:
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes.append((x, y, w, h))

        bounding_boxes = remove_nested_boxes(bounding_boxes)

        median_bb_width = statistics.median([w for x,y,w,h in bounding_boxes])
        median_bb_height = statistics.median([h for x,y,w,h in bounding_boxes])

        points = np.array([[x + w/2, y + h/2] for x, y, w, h in bounding_boxes])
        db = DBSCAN(eps=(median_bb_width+median_bb_height)/2 * 2, min_samples=1).fit(points)
        labels = db.labels_

        cluster_data = {}
        for (x, y, w, h), label in zip(bounding_boxes, labels):
            area = w * h
            if label not in cluster_data:
                cluster_data[label] = {"areas": [], "boxes": []}
            cluster_data[label]["areas"].append(area)
            cluster_data[label]["boxes"].append((x, y, w, h))

        # Create output image copy
        output = np.array(image).copy()

        # Draw boxes and detect outliers
        for label, data in cluster_data.items():
            areas = np.array(data["areas"])
            median_area = np.median(areas)
            mad_area = np.median(np.abs(areas - median_area))  # robust deviation
            threshold = 3 * mad_area  # consider boxes >3*MAD as outliers

            for box, area in zip(data["boxes"], areas):
                x, y, w, h = box
                if np.abs(area - median_area) > threshold:
                    color = (0, 0, 255)  # red for outlier
                else:
                    color = (0, 255, 0)  # green bb 


                # Draw rectangle
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                # Draw cluster label
                cv2.putText(output, f"{label}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                # Draw width,height
                cv2.putText(output, f"{area};{median_area}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Show images
        show(og_image, "Original Image")
        show(cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY), "Depth Anything Grayscale")
        show(ic_packaging_thresh, "IC Packaging Detection")
        show(gray, "IC Packaging Mask")
        show(output, os.path.basename(img), True)