import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return image, edges

def is_square(approx):
    if len(approx) == 4:
        epsilon = 0.02 * cv2.arcLength(approx, True)
        approx_poly = cv2.approxPolyDP(approx, epsilon, True)
        if len(approx_poly) == 4:
            x, y, w, h = cv2.boundingRect(approx_poly)
            aspect_ratio = float(w) / h
            return 0.9 < aspect_ratio < 1.1
    return False

def is_star(approx):
    num_vertices = len(approx)
    if num_vertices >= 10:
        angles = []
        for i in range(num_vertices):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % num_vertices][0]
            p3 = approx[(i + 2) % num_vertices][0]
            v1 = p1 - p2
            v2 = p3 - p2
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            if angle < 0:
                angle += 2 * np.pi
            angles.append(angle)
        angles = np.degrees(angles)
        count_sharp_angles = sum(1 for angle in angles if angle < 90)
        return count_sharp_angles >= 5
    return False

def is_rectangle(approx):
    epsilon = 0.02 * cv2.arcLength(approx, True)
    approx_poly = cv2.approxPolyDP(approx, epsilon, True)
    if len(approx_poly) == 4:
        x, y, w, h = cv2.boundingRect(approx_poly)
        aspect_ratio = float(w) / h
        return 0.5 < aspect_ratio < 2
    return False

def is_ellipse(approx):
    if len(approx) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(approx)
            aspect_ratio = MA / ma
            return 0.5 < aspect_ratio < 2
        except:
            return False
    return False

def is_circle(approx):
    (x, y), radius = cv2.minEnclosingCircle(approx)
    radius = int(radius)
    circle_area = np.pi * (radius ** 2)
    contour_area = cv2.contourArea(approx)
    return abs(circle_area - contour_area) < 0.1 * contour_area

def is_line(approx):
    return len(approx) == 2

def classify_shape(approx):
    if is_star(approx):
        return 'Star', (255, 255, 0)
    elif is_square(approx):
        return 'Square', (255, 0, 255)
    elif is_rectangle(approx):
        return 'Rectangle', (0, 0, 255)
    elif is_circle(approx):
        return 'Circle', (0, 0, 255)
    elif is_ellipse(approx):
        return 'Ellipse', (0, 255, 0)
    elif is_line(approx):
        return 'Line', (0, 255, 255)
    return 'Unknown', (0, 0, 0)

def complete_shape(image, shape_type, contour):
    if shape_type == 'Rectangle' or shape_type == 'Square':
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    elif shape_type == 'Ellipse':
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, (255, 0, 0), 2)
    elif shape_type == 'Circle':
        (x, y), radius = cv2.minEnclosingCircle(contour)
        cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 2)
    elif shape_type == 'Line':
        coords = np.squeeze(contour)
        (vx, vy, x, y) = cv2.fitLine(coords, cv2.DIST_L2, 0, 0.01, 0.01)
        slope = vy / vx
        intercept = y - slope * x
        height, width = image.shape[:2]
        start_point = (0, int(intercept))
        end_point = (width, int(slope * width + intercept))
        cv2.line(image, start_point, end_point, (0, 255, 255), 2)
    elif shape_type == 'Star':
        cv2.drawContours(image, [contour], -1, (255, 255, 0), 2)

def detect_and_complete_shapes(image_path, output_path):
    image, edges = preprocess_image(image_path)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        shape_type, color = classify_shape(approx)
        
        if shape_type != 'Unknown':
            complete_shape(image, shape_type, contour)
    
    cv2.imwrite(output_path, image)

# Paths for input and output images
image_path = r"C:\Users\Jiya Sharma\Dropbox\PC\Downloads\testop-png.png"
output_path = r"C:\Users\Jiya Sharma\Dropbox\PC\Downloads\completed-shapes.png"
detect_and_complete_shapes(image_path, output_path)
