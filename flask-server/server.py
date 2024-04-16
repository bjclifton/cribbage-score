from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

def calculate_corners(contour_points):
    contour_points = contour_points.reshape((4, 2))
    corners = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = contour_points.sum(axis=1)
    corners[0] = contour_points[np.argmin(s)]
    corners[2] = contour_points[np.argmax(s)]

    # compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(contour_points, axis=1)
    corners[1] = contour_points[np.argmin(diff)]
    corners[3] = contour_points[np.argmax(diff)]

    return corners

def transform_perspective(input_image, contour_points):
    corners = calculate_corners(contour_points)
    (top_left, top_right, bottom_right, bottom_left) = corners
    width_top = np.linalg.norm(bottom_right - bottom_left)
    width_bottom = np.linalg.norm(top_right - top_left)
    max_width = 200
    height_left = np.linalg.norm(top_right - bottom_right)
    height_right = np.linalg.norm(top_left - bottom_left)
    max_height = 300
    
    average_width = (width_top + width_bottom) / 2
    average_height = (height_left + height_right) / 2
    
    if average_width > average_height:
        # Card is probably sideways
        temp = max_width
        max_width = max_height
        max_height = temp
        
    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
        
    
    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    output_image = cv2.warpPerspective(input_image, matrix, (max_width, max_height))
    # if output image is horizontal, rotate it
    if output_image.shape[0] < output_image.shape[1]:
        output_image = cv2.rotate(output_image, cv2.ROTATE_90_CLOCKWISE)
    return output_image

def process_image(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Image not found")
        return
    
    # If smaller than 600x800, resize
    height, width, _ = img.shape
    new_height = height
    new_width = width
    if height < 600 or width < 800:
        new_height = 600
        new_width = 800
        
    
    # Resize the image
    img = cv2.resize(img, (new_width, new_height))
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur the image 
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Threshold the image just a little bit
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # Apply edge detection
    edged = cv2.Canny(thresh, 200, 400)
    
    # Dilate
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    contours = [c for c in contours if cv2.contourArea(c) > 10000]
    
    cards = []
    
    for i, c in enumerate(contours):
        epsilon = 0.02*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

        if len(approx) == 4:
            output_image = transform_perspective(img, approx)
            # Crop the card to the top left corner to grab rank and suit
            card = output_image[0:80, 0:30]
            # zoom in by a factor of 4
            card = cv2.resize(card, (120, 320))
            gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_not(thresh)
            
            
            # Find contours and split the card between the two contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by the y-coordinate of their bounding rectangle
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

            
            rank = np.zeros_like(thresh)
            suit = np.zeros_like(thresh)
            for j, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if j == 0:
                    rank = thresh[y:y+h, x:x+w]
                elif j == 1:
                    suit = thresh[y:y+h, x:x+w]

            # Resize to 70x125 px
            rank = cv2.resize(rank, (70, 125))
            suit = cv2.resize(suit, (70, 100))
            cards.append((rank, suit))
    return cards