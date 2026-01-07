import cv2
import numpy as np
import os


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def detect_edges(img):
    return cv2.Canny(img, 50, 150)


def get_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def classify_shape(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0 or area < 100:
        return "Unknown"

  
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity >= 0.85:
        return "Circle"

    
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        return "Rectangle"

    return "Unknown"


dataset_path = "dataset"

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = preprocess_image(path)
        edges = detect_edges(img)
        contours = get_contours(edges)

        if not contours:
            print(f"{file} -> No contours found")
            continue

        # Largest contour
        largest = max(contours, key=cv2.contourArea)
        prediction = classify_shape(largest)
        print(f"{file} -> Predicted: {prediction}, Actual: {label}")
