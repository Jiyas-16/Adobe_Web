import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimage
from scipy.optimize import minimize
import svgwrite
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import io
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
import pandas as pd

def read_csv(file):
    try:
        np_path_XYs = np.genfromtxt(file, delimiter=',')
        if np_path_XYs.ndim == 1:
            raise ValueError("CSV file seems to be read as a 1D array. Check the delimiter or file format.")
        if np_path_XYs.size == 0:
            raise ValueError("CSV file is empty or invalid.")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return []

    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(path_XYs):
    try:
        colours = ['red', 'green', 'blue', 'orange', 'purple']
        fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
        for i, XYs in enumerate(path_XYs):
            c = colours[i % len(colours)]
            for XY in XYs:
                ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
        ax.set_aspect('equal')
        plt.close(fig)
        return fig
    except Exception as e:
        st.error(f"Error plotting the paths: {e}")
        return None

def polylines2svg(path_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in path_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))

    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ['red', 'green', 'blue', 'orange', 'purple']

    for i, path in enumerate(path_XYs):
        path_data = []
        c = colours[i % len(colours)]

        for XY in path:
            if len(XY.shape) == 1:
                path_data.append(("M", (XY[0], XY[1])))
            else:
                path_data.append(("M", (XY[0, 0], XY[0, 1])))
                for j in range(1, len(XY)):
                    path_data.append(("L", (XY[j, 0], XY[j, 1])))
                if not np.allclose(XY[0], XY[-1]):
                    path_data.append(("Z", None))

        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))

    dwg.add(group)
    dwg.save()

    # Convert SVG to PNG
    png_path = svg_path.replace('.svg', '.png')
    drawing = svg2rlg(svg_path)
    renderPM.drawToFile(drawing, png_path, fmt='png', dpi=300)

    return png_path

def normalize_curve(points):
    points = points.astype(float)
    points -= np.mean(points, axis=0)
    points /= np.max(np.abs(points))
    return points

def gaussian_filter(curve, sigma=2):
    smoothed_curve = ndimage.gaussian_filter(curve, sigma=sigma)
    return smoothed_curve

def detect_edges(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    if gray_image.dtype != np.uint8:
        gray_image = (gray_image * 255).astype(np.uint8)
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
    return edges

def parameterize_curve(curve):
    t = np.linspace(0, 1, len(curve))
    x_spline = ndimage.spline_filter1d(curve[:, 0], order=3)
    y_spline = ndimage.spline_filter1d(curve[:, 1], order=3)
    param_curve = np.stack((x_spline, y_spline), axis=1)
    return param_curve

def match_with_library(curve):
    shapes = ["line", "rectangle", "rounded_rectangle", "star", "circle", "ellipse", "polygon"]
    best_shape = None
    best_error = float('inf')
    for shape in shapes:
        error = calculate_similarity(curve, shape)
        if error < best_error:
            best_error = error
            best_shape = shape
    return best_shape

def calculate_similarity(curve, shape):
    def interpolate_points(points, num_points):
        t = np.linspace(0, 1, len(points))
        t_interp = np.linspace(0, 1, num_points)
        interpolated_points = np.array([
            np.interp(t_interp, t, points[:, i]) for i in range(points.shape[1])
        ]).T
        return interpolated_points

    if shape == "circle":
        center = np.mean(curve, axis=0)
        radius = np.mean(np.sqrt(np.sum((curve - center)**2, axis=1)))
        shape_points = np.array([
            center + radius * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, len(curve))
        ])

    elif shape == "line":
        p1, p2 = curve[0], curve[-1]
        shape_points = np.array([p1 + t * (p2 - p1) for t in np.linspace(0, 1, len(curve))])

    elif shape == "rectangle":
        min_x, min_y = np.min(curve, axis=0)
        max_x, max_y = np.max(curve, axis=0)
        shape_points = np.array([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
            [min_x, min_y]
        ])

    elif shape == "rounded_rectangle":
        min_x, min_y = np.min(curve, axis=0)
        max_x, max_y = np.max(curve, axis=0)
        radius = min(max_x - min_x, max_y - min_y) * 0.1
        shape_points = []
        corners = [
            (min_x + radius, min_y + radius),
            (max_x - radius, min_y + radius),
            (max_x - radius, max_y - radius),
            (min_x + radius, max_y - radius)
        ]
        for i, corner in enumerate(corners):
            theta_start = i * np.pi / 2
            theta_end = (i + 1) * np.pi / 2
            shape_points.extend([
                corner + radius * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(theta_start, theta_end, len(curve) // 4)
            ])
        shape_points = np.array(shape_points)

    elif shape == "star":
        center = np.mean(curve, axis=0)
        radius = np.mean(np.sqrt(np.sum((curve - center)**2, axis=1)))
        inner_radius = radius * 0.5
        outer_radius = radius
        shape_points = []
        for i in range(len(curve)):
            angle = 2 * np.pi * i / len(curve)
            if i % 2 == 0:
                r = outer_radius
            else:
                r = inner_radius
            shape_points.append(center + r * np.array([np.cos(angle), np.sin(angle)]))
        shape_points = np.array(shape_points)

    elif shape == "ellipse":
        center = np.mean(curve, axis=0)
        radii = np.array([np.ptp(curve[:, 0]), np.ptp(curve[:, 1])]) / 2
        shape_points = np.array([
            center + radii * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, len(curve))
        ])

    elif shape == "polygon":
        num_sides = 6
        center = np.mean(curve, axis=0)
        radius = np.mean(np.sqrt(np.sum((curve - center)**2, axis=1)))
        shape_points = np.array([
            center + radius * np.array([np.cos(2 * np.pi * i / num_sides), np.sin(2 * np.pi * i / num_sides)])
            for i in range(num_sides)
        ])
        shape_points = np.concatenate([shape_points, [shape_points[0]]])

    return np.mean(np.sqrt(np.sum((curve - shape_points)**2, axis=1)))

def main():
    st.title('CSV Data Processing and Visualization')

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        path_XYs = read_csv(uploaded_file)

        if path_XYs:
            fig = plot(path_XYs)
            if fig:
                st.pyplot(fig)

            svg_path = 'plot.svg'
            png_path = polylines2svg(path_XYs, svg_path)
            st.image(png_path, caption='Generated SVG and PNG', use_column_width=True)

            with open(png_path, 'rb') as f:
                st.download_button(
                    label="Download PNG",
                    data=f,
                    file_name='plot.png',
                    mime="image/png"
                )

            with open(svg_path, 'rb') as f:
                st.download_button(
                    label="Download SVG",
                    data=f,
                    file_name='plot.svg',
                    mime="image/svg+xml"
                )

if __name__ == "__main__":
    main()
