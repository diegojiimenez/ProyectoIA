"""
Funciones de procesamiento de imágenes básicas
"""

import cv2
import numpy as np
import math

def load_image_rgb(path):
    """Carga una imagen y la convierte a RGB"""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def find_contours(binary):
    """Encuentra contornos en una imagen binaria"""
    res = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hierarchy = res
    else:
        _, contours, hierarchy = res
    return contours, hierarchy

def order_points(pts):
    """
    Ordena los puntos en el orden: top-left, top-right, bottom-right, bottom-left
    """
    pts = pts.astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    
    center = np.mean(pts, axis=0)
    
    angles = []
    for pt in pts:
        angle = math.atan2(pt[1] - center[1], pt[0] - center[0])
        angles.append((angle, pt))
    
    angles.sort(key=lambda x: x[0])
    sorted_points = [pt for angle, pt in angles]
    
    y_coords = [pt[1] for pt in sorted_points]
    top_indices = sorted(range(4), key=lambda i: y_coords[i])[:2]
    bottom_indices = sorted(range(4), key=lambda i: y_coords[i])[2:]
    
    top_points = [sorted_points[i] for i in top_indices]
    if top_points[0][0] < top_points[1][0]:
        rect[0] = top_points[0]
        rect[1] = top_points[1]
    else:
        rect[0] = top_points[1]
        rect[1] = top_points[0]
    
    bottom_points = [sorted_points[i] for i in bottom_indices]
    if bottom_points[0][0] < bottom_points[1][0]:
        rect[3] = bottom_points[0]
        rect[2] = bottom_points[1]
    else:
        rect[3] = bottom_points[1]
        rect[2] = bottom_points[0]
    
    return rect

def four_point_transform(image_rgb, pts, width=300, height=420):
    """
    Aplica transformación perspectiva asegurando orientación vertical correcta
    """
    src = pts.reshape(4, 2)
    src_ord = order_points(src)
    
    top_width = np.linalg.norm(src_ord[1] - src_ord[0])
    left_height = np.linalg.norm(src_ord[3] - src_ord[0])
    bottom_width = np.linalg.norm(src_ord[2] - src_ord[3])
    right_height = np.linalg.norm(src_ord[2] - src_ord[1])
    
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    
    print(f"Dimensiones detectadas: ancho={avg_width:.1f}, alto={avg_height:.1f}, ratio={avg_height/avg_width:.2f}")
    
    card_is_horizontal = avg_width > avg_height
    
    if card_is_horizontal:
        print("Carta detectada en orientación horizontal - rotando a vertical")
        actual_card_width = avg_height
        actual_card_height = avg_width
        
        src_ord_rotated = np.array([
            src_ord[1], src_ord[2], src_ord[3], src_ord[0]
        ], dtype="float32")
        src_ord = src_ord_rotated
    else:
        print("Carta detectada en orientación vertical - manteniendo orientación")
        actual_card_width = avg_width
        actual_card_height = avg_height
    
    target_aspect_ratio = 1.4
    detected_ratio = actual_card_height / actual_card_width
    
    if 1.2 <= detected_ratio <= 1.8:
        if actual_card_width > actual_card_height:
            actual_card_width, actual_card_height = actual_card_height, actual_card_width
        
        scale_factor = min(width / actual_card_width, height / actual_card_height)
        final_width = int(actual_card_width * scale_factor)
        final_height = int(actual_card_height * scale_factor)
        
        final_width = min(final_width, width)
        final_height = min(final_height, height)
    else:
        print(f"Ratio detectado {detected_ratio:.2f} fuera de rango esperado, usando dimensiones por defecto")
        final_width = width
        final_height = height
    
    print(f"Dimensiones finales: {final_width}x{final_height}, ratio={final_height/final_width:.2f}")
    
    dst = np.array([
        [0, 0],
        [final_width-1, 0],
        [final_width-1, final_height-1],
        [0, final_height-1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_ord, dst)
    warped = cv2.warpPerspective(image_rgb, M, (final_width, final_height))
    
    return warped