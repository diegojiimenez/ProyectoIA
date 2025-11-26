"""
Extracción de símbolos de esquinas de cartas
"""

import cv2
import numpy as np
from .image_processing import find_contours

def extract_top_left_corner(warped_card, w_ratio=0.28, h_ratio=0.40):
    """Extrae la esquina superior izquierda con proporciones ajustadas"""
    h, w = warped_card.shape[:2]
    
    if w < 250:
        w_ratio = min(w_ratio * 1.2, 0.35)
        h_ratio = min(h_ratio * 1.2, 0.45)
    
    rw = int(w * w_ratio)
    rh = int(h * h_ratio)
    
    rw = max(rw, 60)
    rh = max(rh, 80)
    
    rw = min(rw, w)
    rh = min(rh, h)
    
    print(f"Esquina extraída: {rw}x{rh} de carta {w}x{h}")
    
    return warped_card[0:rh, 0:rw].copy()

def extract_symbols_from_corner(corner_rgb, min_area=50, horizontal_gap=20):
    """
    Extrae símbolos de la esquina con mejor manejo de tamaños variables
    """
    gray = cv2.cvtColor(corner_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    adaptive_min_area = max(min_area, (h * w) // 200)
    
    thresholds = []
    
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresholds.append(thresh1)
    
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    thresholds.append(thresh2)
    
    thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
    thresholds.append(thresh3)
    
    combined_thresh = np.zeros_like(gray)
    for t in thresholds:
        combined_thresh = cv2.bitwise_or(combined_thresh, t)
    
    combined_thresh = cv2.medianBlur(combined_thresh, 3)
    
    kernel_small = np.ones((2, 2), np.uint8)
    combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_small)
    
    contours, _ = find_contours(combined_thresh)
    boxes = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < adaptive_min_area:
            continue
            
        x, y, w_box, h_box = cv2.boundingRect(c)
        
        if w_box < 5 or h_box < 5:
            continue
        if w_box / h_box > 5 or h_box / w_box > 8:
            continue
            
        boxes.append([x, y, x+w_box, y+h_box])
    
    if not boxes:
        return combined_thresh, []
    
    boxes.sort(key=lambda b: (b[1], b[0]))
    
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if not merged:
            merged.append([x1, y1, x2, y2])
        else:
            mx1, my1, mx2, my2 = merged[-1]
            if abs(y1 - my1) < 25 and (x1 - mx2) < horizontal_gap:
                merged[-1] = [min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2)]
            else:
                merged.append([x1, y1, x2, y2])
    
    merged.sort(key=lambda b: (b[1], b[0]))
    
    symbols = []
    for (x1, y1, x2, y2) in merged:
        pad = 3
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = min(w, x2 + pad)
        y2_pad = min(h, y2 + pad)
        
        crop = combined_thresh[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if crop.shape[0] > 5 and crop.shape[1] > 5:
            crop_clean = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel_small)
            symbols.append(crop_clean)
    
    print(f"Símbolos extraídos: {len(symbols)} de esquina {w}x{h}")
    
    return combined_thresh, symbols