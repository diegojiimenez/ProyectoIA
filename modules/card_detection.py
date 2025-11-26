"""
Detección y extracción de cartas
"""

import cv2
import numpy as np
from .image_processing import find_contours, four_point_transform

def find_card_contour_from_binary(binary, min_area=10000):
    """Encuentra el contorno principal de una carta"""
    contours, _ = find_contours(binary)
    max_area = 0
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            best = approx
    return best, max_area

def find_all_card_contours_from_binary(binary, min_area=10000):
    """
    Encuentra TODOS los contornos que parezcan cartas
    Devuelve lista de (approx_contour, area)
    """
    contours, _ = find_contours(binary)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) != 4:
            continue
        
        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            continue
        ratio = h / w
        
        if not (0.55 <= ratio <= 1.9):
            continue
        candidates.append((approx, area))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates

def process_card_image(image_path, visualize=False):
    """Procesa una imagen de carta completa"""
    from .image_processing import load_image_rgb
    from .symbol_extraction import extract_top_left_corner, extract_symbols_from_corner
    
    try:
        image_rgb = load_image_rgb(image_path)
    except FileNotFoundError as e:
        print(e)
        return None
    
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    card_contour, area = find_card_contour_from_binary(binary, min_area=10000)
    if card_contour is None:
        print(f"No se encontró contorno de carta en {image_path}.")
        return None
    
    warped = four_point_transform(image_rgb, card_contour, width=300, height=420)
    corner = extract_top_left_corner(warped)
    thresh_corner, symbols = extract_symbols_from_corner(corner)
    
    return {
        "image_path": image_path,
        "original": image_rgb,
        "binary": binary,
        "card_contour": card_contour,
        "warped": warped,
        "corner": corner,
        "thresh_corner": thresh_corner,
        "symbols": symbols
    }