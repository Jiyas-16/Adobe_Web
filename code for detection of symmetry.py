import cv2
import numpy as np

def is_line(approx):
    return len(approx) == 2

def is_circle(approx, contour_area):
    (x, y), radius = cv2.minEnclosingCircle(approx)
    radius = int(radius)
    circle_area = np.pi * (radius ** 2)
    return abs(circle_area - contour_area) < 0.1 * contour_area and len(approx) > 8

def is_ellipse(approx):
    if len(approx) >= 5:
        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(approx)
            aspect_ratio = MA / ma
            return 0.5 < aspect_ratio < 2 and not is_circle(approx, cv2.contourArea(approx))
        except:
            return False
    return False

def is_rectangle(approx):
    epsilon = 0.02 * cv2.arcLength(approx, True)
    approx_poly = cv2.approxPolyDP(approx, epsilon, True)
    if len(approx_poly) == 4:
        x, y, w, h = cv2.boundingRect(approx_poly)
        aspect_ratio = float(w) / h
        return 0.5 < aspect_ratio < 2
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

def is_symmetric_polygon(approx):
    num_vertices = len(approx)
    if num_vertices >= 6:  # Check for polygons with more vertices
        return True
    return False

def classify_shape(approx):
    contour_area = cv2.contourArea(approx)
    if is_star(approx):
        return 'Star', (255, 255, 0)
    elif is_line(approx):
        return 'Line', (0, 255, 255)
    elif is_circle(approx, contour_area):
        return 'Circle', (255, 0, 0)
    elif is_ellipse(approx):
        return 'Ellipse', (0, 255, 0)
    elif is_rectangle(approx):
        return 'Rectangle', (0, 0, 255)
    elif is_symmetric_polygon(approx):
        return 'Symmetric Polygon', (255, 0, 255)
    else:
        return 'Polygon', (255, 165, 0)

def draw_symmetry_lines(canvas, approx, shape):
    if shape == 'Circle':
        (x, y), radius = cv2.minEnclosingCircle(approx)
        center = (int(x), int(y))
        cv2.line(canvas, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 0, 0), 2)
        cv2.line(canvas, (center[0], center[1] - radius), (center[0], center[1] + radius), (0, 0, 0), 2)
    elif shape == 'Ellipse':
        ellipse = cv2.fitEllipse(approx)
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
        angle = ellipse[2]
        box = cv2.boxPoints(ellipse)
        box = np.intp(box)
        cv2.line(canvas, tuple(box[0]), tuple(box[2]), (0, 0, 0), 2)
        cv2.line(canvas, tuple(box[1]), tuple(box[3]), (0, 0, 0), 2)
    elif shape == 'Rectangle':
        x, y, w, h = cv2.boundingRect(approx)
        cv2.line(canvas, (x, y), (x + w, y + h), (0, 0, 0), 2)
        cv2.line(canvas, (x + w, y), (x, y + h), (0, 0, 0), 2)
    elif shape in ['Star', 'Symmetric Polygon', 'Polygon']:
        num_vertices = len(approx)
        for i in range(num_vertices):
            pt1 = tuple(approx[i][0])
            pt2 = tuple(approx[(i + num_vertices // 2) % num_vertices][0])
            cv2.line(canvas, pt1, pt2, (0, 0, 0), 2)

def detect_shapes(image_path, output_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = gray.shape
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        shape, color = classify_shape(approx)
        
        draw_symmetry_lines(canvas, approx, shape)  # Draw symmetry lines through the shape

        if shape == 'Circle':
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(canvas, center, radius, color, 2)
        elif shape == 'Ellipse':
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(canvas, ellipse, color, 2)
        else:
            cv2.drawContours(canvas, [approx], -1, color, 2)
    
    cv2.imwrite(output_path, canvas)

image_path = r"C:\Users\Jiya Sharma\Dropbox\PC\Downloads\test-png.png"
output_path = r"C:\Users\Jiya Sharma\Dropbox\PC\Downloads\output-png.png"
detect_shapes(image_path, output_path)
