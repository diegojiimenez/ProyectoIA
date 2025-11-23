import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import math

try:
    import scipy.ndimage as ndi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Aviso: scipy no está instalado. Instala con: pip install scipy para mejor detección de lóbulos de corazones.")

# =========================================================
# Mostrar imagen
# =========================================================
def show_img(img, title="", figsize=(6, 6), cmap=None, mode="block"):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if len(img.shape) == 2 or cmap is not None:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
    if mode == "block":
        plt.show()
    elif mode == "enter":
        plt.show(block=False)
        try:
            input("Presiona Enter para continuar...")
        except EOFError:
            pass
        plt.close(fig)
    elif mode == "auto":
        plt.show(block=False)
        plt.pause(0.25)
        plt.close(fig)
    else:
        plt.show()

def wait_enter(enabled=True, message="Presiona Enter para continuar..."):
    if enabled:
        try:
            input(message)
        except EOFError:
            pass

# =========================================================
# Utilidades básicas
# =========================================================
def load_image_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def find_contours(binary):
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
    
    # Calcular el centro
    center = np.mean(pts, axis=0)
    
    # Calcular ángulos desde el centro para cada punto
    angles = []
    for pt in pts:
        angle = math.atan2(pt[1] - center[1], pt[0] - center[0])
        angles.append((angle, pt))
    
    # Ordenar por ángulo (comenzando desde arriba-izquierda, sentido horario)
    angles.sort(key=lambda x: x[0])
    
    # Asignar puntos en orden: top-left, top-right, bottom-right, bottom-left
    # Pero necesitamos determinar cuál es cuál basado en las posiciones relativas
    sorted_points = [pt for angle, pt in angles]
    
    # Encontrar los puntos más arriba y más abajo
    y_coords = [pt[1] for pt in sorted_points]
    top_indices = sorted(range(4), key=lambda i: y_coords[i])[:2]  # Los 2 puntos más arriba
    bottom_indices = sorted(range(4), key=lambda i: y_coords[i])[2:]  # Los 2 puntos más abajo
    
    # Entre los puntos de arriba, el de la izquierda es top-left, el de la derecha es top-right
    top_points = [sorted_points[i] for i in top_indices]
    if top_points[0][0] < top_points[1][0]:  # Comparar coordenada x
        rect[0] = top_points[0]  # top-left
        rect[1] = top_points[1]  # top-right
    else:
        rect[0] = top_points[1]  # top-left
        rect[1] = top_points[0]  # top-right
    
    # Entre los puntos de abajo, el de la derecha es bottom-right, el de la izquierda es bottom-left
    bottom_points = [sorted_points[i] for i in bottom_indices]
    if bottom_points[0][0] < bottom_points[1][0]:  # Comparar coordenada x
        rect[3] = bottom_points[0]  # bottom-left
        rect[2] = bottom_points[1]  # bottom-right
    else:
        rect[3] = bottom_points[1]  # bottom-left
        rect[2] = bottom_points[0]  # bottom-right
    
    return rect

def find_card_contour_from_binary(binary, min_area=10000):
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

def four_point_transform(image_rgb, pts, width=300, height=420):
    """
    Aplica transformación perspectiva asegurando que la carta quede en orientación vertical
    con proporciones correctas
    """
    src = pts.reshape(4, 2)
    src_ord = order_points(src)
    
    # Calcular las dimensiones de los lados de la carta detectada
    # Lado superior
    top_width = np.linalg.norm(src_ord[1] - src_ord[0])
    # Lado izquierdo  
    left_height = np.linalg.norm(src_ord[3] - src_ord[0])
    # Lado inferior
    bottom_width = np.linalg.norm(src_ord[2] - src_ord[3])
    # Lado derecho
    right_height = np.linalg.norm(src_ord[2] - src_ord[1])
    
    # Calcular las dimensiones promedio
    avg_width = (top_width + bottom_width) / 2
    avg_height = (left_height + right_height) / 2
    
    print(f"Dimensiones detectadas: ancho={avg_width:.1f}, alto={avg_height:.1f}, ratio={avg_height/avg_width:.2f}")
    
    # Determinar si la carta está en orientación horizontal (más ancha que alta)
    # Las cartas de póker tienen ratio height/width ≈ 1.4
    card_is_horizontal = avg_width > avg_height
    
    if card_is_horizontal:
        print("Carta detectada en orientación horizontal - rotando a vertical")
        # Cuando está horizontal, intercambiamos width y height en el cálculo de dimensiones
        actual_card_width = avg_height   # lo que era altura se convierte en ancho
        actual_card_height = avg_width   # lo que era ancho se convierte en altura
        
        # Rotar los puntos fuente 90 grados en sentido antihorario
        # El punto que era top-right se convierte en top-left, etc.
        src_ord_rotated = np.array([
            src_ord[1],  # top-right → top-left
            src_ord[2],  # bottom-right → top-right  
            src_ord[3],  # bottom-left → bottom-right
            src_ord[0]   # top-left → bottom-left
        ], dtype="float32")
        src_ord = src_ord_rotated
    else:
        print("Carta detectada en orientación vertical - manteniendo orientación")
        actual_card_width = avg_width
        actual_card_height = avg_height
    
    # Calcular las proporciones finales manteniendo el aspect ratio correcto
    target_aspect_ratio = 1.4  # Ratio estándar de cartas de póker (altura/ancho)
    
    # Si las dimensiones detectadas están muy lejos del ratio estándar, usar valores por defecto
    detected_ratio = actual_card_height / actual_card_width
    
    if 1.2 <= detected_ratio <= 1.8:  # Ratio razonable para una carta
        # Usar las dimensiones detectadas pero ajustar para mantener proporciones
        if actual_card_width > actual_card_height:
            # Caso raro, forzar orientación correcta
            actual_card_width, actual_card_height = actual_card_height, actual_card_width
        
        # Escalar manteniendo el aspect ratio objetivo
        scale_factor = min(width / actual_card_width, height / actual_card_height)
        final_width = int(actual_card_width * scale_factor)
        final_height = int(actual_card_height * scale_factor)
        
        # Asegurar que no exceda las dimensiones máximas
        final_width = min(final_width, width)
        final_height = min(final_height, height)
        
    else:
        # Usar dimensiones por defecto si la detección es dudosa
        print(f"Ratio detectado {detected_ratio:.2f} fuera de rango esperado, usando dimensiones por defecto")
        final_width = width
        final_height = height
    
    print(f"Dimensiones finales: {final_width}x{final_height}, ratio={final_height/final_width:.2f}")
    
    # Definir puntos de destino (siempre en orientación vertical)
    dst = np.array([
        [0, 0],                         # top-left
        [final_width-1, 0],             # top-right
        [final_width-1, final_height-1], # bottom-right
        [0, final_height-1]             # bottom-left
    ], dtype="float32")
    
    # Aplicar transformación perspectiva
    M = cv2.getPerspectiveTransform(src_ord, dst)
    warped = cv2.warpPerspective(image_rgb, M, (final_width, final_height))
    
    return warped

def extract_top_left_corner(warped_card, w_ratio=0.28, h_ratio=0.40):
    """
    Extrae la esquina superior izquierda con proporciones ajustadas
    """
    h, w = warped_card.shape[:2]
    
    # Ajustar las proporciones basándose en el tamaño real de la carta warpada
    # Para cartas más pequeñas, usar proporciones ligeramente mayores
    if w < 250:  # Carta pequeña, usar proporciones mayores
        w_ratio = min(w_ratio * 1.2, 0.35)
        h_ratio = min(h_ratio * 1.2, 0.45)
    
    rw = int(w * w_ratio)
    rh = int(h * h_ratio)
    
    # Asegurar dimensiones mínimas para la esquina
    rw = max(rw, 60)  # Mínimo 60 píxeles de ancho
    rh = max(rh, 80)  # Mínimo 80 píxeles de alto
    
    # No exceder las dimensiones de la carta
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
    
    # Ajustar el área mínima basándose en el tamaño de la esquina
    adaptive_min_area = max(min_area, (h * w) // 200)  # Área mínima adaptativa
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Aplicar un ligero filtro de mediana para reducir ruido
    thresh = cv2.medianBlur(thresh, 3)
    
    contours, _ = find_contours(thresh)
    boxes = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < adaptive_min_area:
            continue
            
        x, y, w_box, h_box = cv2.boundingRect(c)
        
        # Filtrar cajas demasiado pequeñas o con proporciones extrañas
        if w_box < 5 or h_box < 5:
            continue
        if w_box / h_box > 5 or h_box / w_box > 8:  # Filtrar proporciones muy extremas
            continue
            
        boxes.append([x, y, x+w_box, y+h_box])
    
    if not boxes:
        return thresh, []
    
    # Ordenar por posición (arriba a abajo, izquierda a derecha)
    boxes.sort(key=lambda b: (b[1], b[0]))
    
    # Fusionar cajas que estén muy cerca horizontalmente
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if not merged:
            merged.append([x1, y1, x2, y2])
        else:
            mx1, my1, mx2, my2 = merged[-1]
            # Si están en la misma fila horizontal y cerca
            if abs(y1 - my1) < 25 and (x1 - mx2) < horizontal_gap:
                merged[-1] = [min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2)]
            else:
                merged.append([x1, y1, x2, y2])
    
    # Ordenar las cajas fusionadas
    merged.sort(key=lambda b: (b[1], b[0]))
    
    # Extraer los símbolos con un poco de padding
    symbols = []
    for (x1, y1, x2, y2) in merged:
        # Añadir padding pero sin salir de la imagen
        pad = 2
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = min(w, x2 + pad)
        y2_pad = min(h, y2 + pad)
        
        crop = thresh[y1_pad:y2_pad, x1_pad:x2_pad]
        
        # Solo añadir si tiene tamaño suficiente
        if crop.shape[0] > 5 and crop.shape[1] > 5:
            symbols.append(crop)
    
    print(f"Símbolos extraídos: {len(symbols)} de esquina {w}x{h}")
    
    return thresh, symbols

# =========================================================
# Multi-template scoring
# =========================================================
def multi_template_scores(symbol_img, templates_list):
    if symbol_img is None or len(templates_list) == 0:
        return 0.0, {}
    H0,W0 = 32,32
    symbol_norm = cv2.resize(symbol_img, (W0,W0))
    symbol_edges = cv2.Canny(symbol_norm, 50,150)
    symbol_inv = cv2.bitwise_not(symbol_norm)
    dist_symbol = cv2.distanceTransform(symbol_inv, cv2.DIST_L2, 3)
    vec_symbol = symbol_norm.flatten().astype(np.float32)
    vec_symbol /= (np.linalg.norm(vec_symbol) + 1e-6)
    best = 0.0
    best_detail = None
    for tmpl in templates_list:
        tmpl_norm = cv2.resize(tmpl, (W0,W0))
        res = cv2.matchTemplate(symbol_norm, tmpl_norm, cv2.TM_CCOEFF_NORMED)
        _, corr_score, _, _ = cv2.minMaxLoc(res)
        tmpl_edges = cv2.Canny(tmpl_norm, 50,150)
        res_e = cv2.matchTemplate(symbol_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
        _, edge_score, _, _ = cv2.minMaxLoc(res_e)
        tmpl_inv = cv2.bitwise_not(tmpl_norm)
        dist_tmpl = cv2.distanceTransform(tmpl_inv, cv2.DIST_L2, 3)
        sym_pts = np.where(symbol_edges > 0)
        tmpl_pts = np.where(tmpl_edges > 0)
        ch1 = dist_tmpl[sym_pts].mean() if len(sym_pts[0])>0 else 50.0
        ch2 = dist_symbol[tmpl_pts].mean() if len(tmpl_pts[0])>0 else 50.0
        chamfer_score = np.exp(-0.5*(ch1+ch2)/10.0)
        vec_tmpl = tmpl_norm.flatten().astype(np.float32)
        vec_tmpl /= (np.linalg.norm(vec_tmpl)+1e-6)
        cosine = float(np.dot(vec_symbol, vec_tmpl))
        if cosine < 0: cosine = 0.0
        combined = (0.35*corr_score + 0.20*edge_score + 0.25*chamfer_score + 0.20*cosine)
        if combined > best:
            best = combined
            best_detail = {
                "corr": corr_score,
                "edge": edge_score,
                "chamfer": chamfer_score,
                "cosine": cosine,
                "combined": combined
            }
    return best, (best_detail if best_detail else {})

def enhanced_match_symbol_v2(symbol_img, templates_dict, symbol_type="rank"):
    if symbol_img is None or len(templates_dict) == 0:
        return "Unknown", -1.0, {}
    best_name = "Unknown"
    best_score = -1.0
    best_detail = {}
    for name, tmpl_list in templates_dict.items():
        if not isinstance(tmpl_list, list):
            tmpl_list = [tmpl_list]
        score, detail = multi_template_scores(symbol_img, tmpl_list)
        if score > best_score:
            best_score = score
            best_name = name
            best_detail = detail
    return best_name, float(best_score), best_detail

# =========================================================
# Rank
# =========================================================
def enhanced_rank_classification(rank_symbol, rank_templates):
    name, score, _ = enhanced_match_symbol_v2(rank_symbol, rank_templates, "rank")
    if name in ["Q","J","9"] or score < 0.6:
        best_name, best_score = name, score
        for scale in [0.85,0.9,1.0,1.1]:
            h,w = rank_symbol.shape
            nh,nw = int(h*scale), int(w*scale)
            if nh<=0 or nw<=0: continue
            scaled = cv2.resize(rank_symbol, (nw,nh))
            n_name, n_score, _ = enhanced_match_symbol_v2(scaled, rank_templates, "rank")
            if n_score > best_score:
                best_score = n_score
                best_name = n_name
        return best_name, best_score
    return name, score

# =========================================================
# Color stats (v3) con fallback rojo
# =========================================================
def extract_symbol_color_stats_v3(symbol_binary, corner_rgb, template_name=None):
    contours, _ = find_contours(symbol_binary)
    if not contours:
        return {
            "red_pct_hsv":0.0,"lab_a_mean":0.0,"rg_diff_mean":0.0,"cr_mean":0.0,
            "lab_a_norm":0.0,"cr_norm":0.0,"red_confidence":0.0,"is_red":False,
            "fallback_color_red":False
        }
    largest = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    roi = corner_rgb[y:y+h, x:x+w]
    mask = np.zeros((h,w), dtype=np.uint8)
    shifted = largest - [x,y]
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)

    R = roi[:,:,0].astype(np.float32)
    G = roi[:,:,1].astype(np.float32)
    B = roi[:,:,2].astype(np.float32)
    denom = (R+G+B+1e-6)
    rg_diff = (R - G)/denom

    symbol_mask = (mask>0)

    lower_sets = [
        ((0,25,25),(12,255,255)),
        ((8,25,25),(25,255,255)),
        ((165,25,25),(180,255,255))
    ]
    red_hsv_mask = np.zeros_like(mask)
    for (l,u) in lower_sets:
        l_arr = np.array(l, dtype=np.uint8)
        u_arr = np.array(u, dtype=np.uint8)
        temp = cv2.inRange(hsv, l_arr, u_arr)
        red_hsv_mask = cv2.bitwise_or(red_hsv_mask, temp)

    red_pct_hsv = np.sum((red_hsv_mask>0) & symbol_mask) / (np.sum(symbol_mask)+1e-6)
    a_lab = lab[:,:,1]
    Cr = ycrcb[:,:,1]
    lab_a_mean = np.mean(a_lab[symbol_mask]) if np.sum(symbol_mask)>0 else 0.0
    rg_diff_mean = np.mean(rg_diff[symbol_mask]) if np.sum(symbol_mask)>0 else 0.0
    cr_mean = np.mean(Cr[symbol_mask]) if np.sum(symbol_mask)>0 else 0.0
    lab_a_norm = (lab_a_mean - 128)/64.0
    cr_norm = (cr_mean - 128)/64.0

    raw_score = (0.35*red_pct_hsv + 0.25*max(0,lab_a_norm) + 0.25*max(0,rg_diff_mean) + 0.15*max(0,cr_norm))
    red_confidence = 1.0 / (1.0 + math.exp(-8*(raw_score - 0.18)))
    is_red = red_confidence > 0.30

    fallback_color_red = False
    if (template_name in ["Corazones","Diamantes"]) and not is_red:
        kernel = np.ones((3,3), np.uint8)
        dil_mask = cv2.dilate(mask, kernel, iterations=1)
        dil_symbol = (dil_mask>0)
        R_mean = np.mean(R[dil_symbol]) if np.sum(dil_symbol)>0 else 0
        G_mean = np.mean(G[dil_symbol]) if np.sum(dil_symbol)>0 else 0
        B_mean = np.mean(B[dil_symbol]) if np.sum(dil_symbol)>0 else 0
        if (R_mean - G_mean > 10) and (R_mean - B_mean > 10):
            red_confidence = max(red_confidence, 0.42)
            is_red = True
            fallback_color_red = True

    return {
        "red_pct_hsv": red_pct_hsv,
        "lab_a_mean": lab_a_mean,
        "rg_diff_mean": rg_diff_mean,
        "cr_mean": cr_mean,
        "lab_a_norm": lab_a_norm,
        "cr_norm": cr_norm,
        "red_confidence": red_confidence,
        "is_red": is_red,
        "fallback_color_red": fallback_color_red
    }

# =========================================================
# Shape metrics
# =========================================================
def compute_shape_metrics(symbol_binary):
    contours,_ = find_contours(symbol_binary)
    if not contours:
        return {
            "area":0,"perimeter":0,"circularity":0,"solidity":0,
            "aspect_ratio":1,"vertices":0,"defects":0,"hu":np.zeros(7)
        }
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4*math.pi*area/(perimeter*perimeter)) if perimeter>0 else 0
    x,y,w,h = cv2.boundingRect(c)
    aspect_ratio = h/w if w>0 else 1
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area>0 else 0
    eps = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    vertices = len(approx)
    defects_count = 0
    if len(c)>=3:
        hull_indices = cv2.convexHull(c, returnPoints=False)
        if hull_indices is not None and len(hull_indices)>3:
            defects = cv2.convexityDefects(c, hull_indices)
            if defects is not None:
                for d in defects:
                    depth = d[0][3]/256.0
                    if depth > 4:
                        defects_count += 1
    m = cv2.moments(symbol_binary)
    hu = cv2.HuMoments(m).flatten()
    for i in range(len(hu)):
        if hu[i]!=0:
            hu[i] = -math.copysign(1.0, hu[i]) * math.log10(abs(hu[i]))
    return {
        "area":area,"perimeter":perimeter,"circularity":circularity,
        "solidity":solidity,"aspect_ratio":aspect_ratio,"vertices":vertices,
        "defects":defects_count,"hu":hu
    }

# =========================================================
# Heart features
# =========================================================
def compute_heart_features(symbol_binary):
    h,w = symbol_binary.shape
    if h<8 or w<8:
        return {
            "peak_count":0,
            "top_bottom_ratio":1.0,
            "symmetry":0.0,
            "avg_roundness":0.0,
            "heart_lobes_score":0.0
        }
    H0,W0 = 64,64
    norm = cv2.resize(symbol_binary, (W0,W0), interpolation=cv2.INTER_NEAREST)
    top_h = H0//3
    top_region = norm[:top_h,:]
    col_sum = top_region.sum(axis=0).astype(np.float32)
    if SCIPY_AVAILABLE:
        col_smooth = ndi.gaussian_filter1d(col_sum, sigma=2)
    else:
        kernel = np.ones(5)/5.0
        col_smooth = np.convolve(col_sum, kernel, mode='same')
    peaks = []
    for i in range(1,len(col_smooth)-1):
        if col_smooth[i]>col_smooth[i-1] and col_smooth[i]>col_smooth[i+1]:
            peaks.append(i)
    threshold_peak = 0.25*col_smooth.max() if col_smooth.max()>0 else 0
    peaks = [p for p in peaks if col_smooth[p]>=threshold_peak]
    active_cols = np.where(col_smooth > 0.20*(col_smooth.max()+1e-6))[0]
    top_width = active_cols[-1]-active_cols[0]+1 if len(active_cols)>0 else W0
    bottom_region = norm[-top_h:,:]
    bottom_sum = bottom_region.sum(axis=0)
    bottom_active = np.where(bottom_sum > 0.20*(bottom_sum.max()+1e-6))[0]
    bottom_width = bottom_active[-1]-bottom_active[0]+1 if len(bottom_active)>0 else W0
    top_bottom_ratio = top_width / bottom_width if bottom_width>0 else 1.0
    left_half = top_region[:, :W0//2]
    right_half = top_region[:, W0//2:]
    right_reflect = np.flip(right_half, axis=1)
    l_vec = left_half.flatten().astype(np.float32)
    r_vec = right_reflect.flatten().astype(np.float32)
    l_vec /= (np.linalg.norm(l_vec)+1e-6)
    r_vec /= (np.linalg.norm(r_vec)+1e-6)
    symmetry = float(np.dot(l_vec, r_vec))
    gauss_x = np.linspace(-1,1,W0//2)
    gauss_y = np.linspace(-1,1,top_h)
    gx,gy = np.meshgrid(gauss_x, gauss_y)
    gauss_mask = np.exp(-(gx*2 + gy*2)*3.0)
    gauss_mask = (gauss_mask - gauss_mask.min())/(gauss_mask.max()-gauss_mask.min()+1e-6)
    def roundness_score(lobe):
        if lobe.sum()==0:
            return 0.0
        ln = (lobe - lobe.min())/(lobe.max()-lobe.min()+1e-6)
        return float(np.sum(ln * gauss_mask)/np.sum(gauss_mask))
    left_roundness = roundness_score(left_half.astype(np.float32))
    right_roundness = roundness_score(right_half.astype(np.float32))
    avg_roundness = (left_roundness + right_roundness)/2.0
    peaks_count = len(peaks)
    lobes_score = 0.0
    if peaks_count >= 2: lobes_score += 0.4
    elif peaks_count == 1: lobes_score += 0.15
    if top_bottom_ratio > 1.05: lobes_score += 0.25
    if avg_roundness > 0.45: lobes_score += 0.2
    if symmetry > 0.85: lobes_score += 0.15
    lobes_score = min(lobes_score, 1.0)
    return {
        "peak_count": peaks_count,
        "top_bottom_ratio": top_bottom_ratio,
        "symmetry": symmetry,
        "avg_roundness": avg_roundness,
        "heart_lobes_score": lobes_score
    }

# =========================================================
# Diamond features (nuevo)
# =========================================================
def compute_diamond_features(symbol_binary):
    """
    Rasgos distintivos de diamante:
    - 4 vértices dominantes (approx poligonal estable)
    - Aspect ratio ~ 1 (0.8 - 1.25)
    - Uniformidad radial (distancias a centro similares)
    - Ángulos internos ~60-120 grados
    - Orientación cercana a 45° (rotado) o vertical (0°) dependiendo diseño.
    Devuelve dict + diamond_feature_score (0–1).
    """
    h, w = symbol_binary.shape
    if h < 8 or w < 8:
        return {
            "approx_vertices":0,
            "aspect_ratio_ok":False,
            "radial_uniformity":0.0,
            "angle_uniformity":0.0,
            "orientation_score":0.0,
            "diamond_feature_score":0.0
        }
    contours,_ = find_contours(symbol_binary)
    if not contours:
        return {
            "approx_vertices":0,
            "aspect_ratio_ok":False,
            "radial_uniformity":0.0,
            "angle_uniformity":0.0,
            "orientation_score":0.0,
            "diamond_feature_score":0.0
        }
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 30:
        return {
            "approx_vertices":0,
            "aspect_ratio_ok":False,
            "radial_uniformity":0.0,
            "angle_uniformity":0.0,
            "orientation_score":0.0,
            "diamond_feature_score":0.0
        }
    x,y,wc,hc = cv2.boundingRect(c)
    aspect_ratio = hc/wc if wc>0 else 1
    aspect_ratio_ok = 0.8 <= aspect_ratio <= 1.25

    # Aproximaciones múltiples
    peri = cv2.arcLength(c, True)
    approx_list = []
    for factor in [0.015, 0.02, 0.03]:
        eps = factor * peri
        ap = cv2.approxPolyDP(c, eps, True)
        approx_list.append(ap)
    # Buscar la aproximación con 4-6 vértices más cercana a 4
    chosen = min(approx_list, key=lambda ap: abs(len(ap)-4))
    approx_vertices = len(chosen)

    # Radial uniformity
    M = cv2.moments(c)
    cx = M["m10"]/M["m00"] if M["m00"]>0 else wc/2
    cy = M["m01"]/M["m00"] if M["m00"]>0 else hc/2
    pts = chosen.reshape(-1,2)
    dists = [math.hypot(px-cx, py-cy) for px,py in pts]
    if dists:
        mean_d = np.mean(dists)
        std_d = np.std(dists)
        radial_uniformity = 1.0 - min(std_d/(mean_d+1e-6), 1.0)
    else:
        radial_uniformity = 0.0

    # Ángulos internos
    def angle(a,b,c):
        # ángulo en b formado por a-b-c
        ab = a-b
        cb = c-b
        dot = ab.dot(cb)
        nab = np.linalg.norm(ab)
        ncb = np.linalg.norm(cb)
        if nab*ncb == 0:
            return 0
        cosang = np.clip(dot/(nab*ncb), -1, 1)
        return math.degrees(math.acos(cosang))
    angles = []
    pts_cycle = np.vstack([pts, pts[0], pts[1]]) if len(pts)>=3 else pts
    for i in range(len(pts)):
        a = pts_cycle[i]
        b = pts_cycle[i+1]
        c2 = pts_cycle[i+2]
        ang = angle(a,b,c2)
        angles.append(ang)
    if angles:
        # Queremos 4 ángulos relativamente uniformes
        mean_ang = np.mean(angles)
        std_ang = np.std(angles)
        angle_uniformity = 1.0 - min(std_ang/(mean_ang+1e-6),1.0)
    else:
        angle_uniformity = 0.0

    # Orientación (PCA)
    pts_all = c.reshape(-1,2).astype(np.float32)
    pts_norm = pts_all - np.mean(pts_all, axis=0)
    cov = np.cov(pts_norm.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    idx = np.argmax(eigvals)
    principal = eigvecs[:,idx]
    angle_deg = abs(math.degrees(math.atan2(principal[1], principal[0])))
    # Diamante puede estar cerca de 45° o cerca de 0°
    # Calculamos distancia mínima a 45° o 0°
    dist_to_45 = min(abs(angle_deg-45), abs(angle_deg-135))
    dist_to_0 = min(abs(angle_deg-0), abs(angle_deg-180))
    orientation_score = 1.0 - min(dist_to_45, dist_to_0)/90.0  # normalizado

    # Score compuesto
    score = 0.0
    # Vértices cerca de 4
    if approx_vertices == 4:
        score += 0.25
    elif approx_vertices == 5:
        score += 0.18
    elif approx_vertices == 6:
        score += 0.10
    if aspect_ratio_ok:
        score += 0.20
    score += 0.20 * radial_uniformity
    score += 0.20 * angle_uniformity
    score += 0.15 * orientation_score
    # Normalizar a máximo teórico ≈ 1.0
    diamond_feature_score = min(score, 1.0)

    return {
        "approx_vertices": approx_vertices,
        "aspect_ratio_ok": aspect_ratio_ok,
        "radial_uniformity": radial_uniformity,
        "angle_uniformity": angle_uniformity,
        "orientation_score": orientation_score,
        "diamond_feature_score": diamond_feature_score
    }

# =========================================================
# Degenerate re-segmentation (mejorado para diamantes)
# =========================================================
def resegment_symbol_if_degenerate(symbol_binary, corner_rgb, template_name=None):
    h,w = symbol_binary.shape
    if h < 10 or w < 10:
        return symbol_binary, False
    contours,_ = find_contours(symbol_binary)
    if not contours:
        return symbol_binary, False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    circularity = (4*math.pi*area/(peri*peri)) if peri>0 else 0
    x,y,wc,hc = cv2.boundingRect(c)
    aspect_ratio = hc / wc if wc>0 else 1
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area>0 else 0
    degenerate = (circularity < 0.05) or (solidity < 0.2) or (aspect_ratio > 3.0)
    if not degenerate:
        return symbol_binary, False

    gray_corner = cv2.cvtColor(corner_rgb, cv2.COLOR_RGB2GRAY)
    thr = cv2.adaptiveThreshold(gray_corner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 5)
    # Si template Diamantes, aplicar cierre para unir rombo
    if template_name == "Diamantes":
        kernel = np.ones((3,3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    thr_dil = cv2.dilate(thr, np.ones((3,3), np.uint8), iterations=1)
    contours2,_ = find_contours(thr_dil)
    if not contours2:
        return symbol_binary, False
    h2, w2 = thr_dil.shape
    candidates = []
    for c2 in contours2:
        area2 = cv2.contourArea(c2)
        if area2 < 30:
            continue
        x2,y2,w2c,h2c = cv2.boundingRect(c2)
        if y2 < h2 * 0.85:
            candidates.append((area2, c2))
    if not candidates:
        return symbol_binary, False
    best_c = max(candidates, key=lambda x: x[0])[1]
    x2,y2,w2c,h2c = cv2.boundingRect(best_c)
    new_symbol = thr_dil[y2:y2+h2c, x2:x2+w2c].copy()
    return new_symbol, True

# =========================================================
# Prototipos de color
# =========================================================
def extract_symbol_color_vector(symbol_binary, corner_rgb):
    contours,_ = find_contours(symbol_binary)
    if not contours:
        return np.array([0,0,0], dtype=np.float32)
    largest = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    roi = corner_rgb[y:y+h, x:x+w]
    mask = np.zeros((h,w), dtype=np.uint8)
    shifted = largest - [x,y]
    cv2.drawContours(mask, [shifted], -1, 255, -1)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    rgb = roi.astype(np.float32)
    R = rgb[:,:,0]; G = rgb[:,:,1]; B = rgb[:,:,2]
    denom = (R+G+B+1e-6)
    rg_diff = (R - G)/denom
    symbol_mask = (mask>0)
    if np.sum(symbol_mask)==0:
        return np.array([0,0,0], dtype=np.float32)
    mean_H = np.mean(hsv[:,:,0][symbol_mask])
    mean_a = np.mean(lab[:,:,1][symbol_mask])
    mean_rg = np.mean(rg_diff[symbol_mask])
    return np.array([mean_H, mean_a, mean_rg], dtype=np.float32)

def build_templates_from_directory(template_base_dir="template/"):
    rank_templates = {}
    suit_templates = {}
    suit_color_prototypes = {}
    suit_mapping = {
        'corazones': 'Corazones',
        'diamantes': 'Diamantes',
        'picas': 'Picas',
        'treboles': 'Treboles'
    }
    for suit_dir in ['corazones','diamantes','picas','treboles']:
        suit_path = os.path.join(template_base_dir, suit_dir)
        if not os.path.exists(suit_path):
            continue
        print(f"Procesando templates de: {suit_dir}...")
        color_vectors = []
        for filename in os.listdir(suit_path):
            if not filename.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            rank = filename.split('_')[0]
            image_path = os.path.join(suit_path, filename)
            result = process_card_image(image_path, visualize=False)
            if result is None or len(result['symbols']) < 2:
                print(f"  Omitiendo {filename} - símbolos insuficientes")
                continue
            rank_symbol = result['symbols'][0]
            suit_symbol = result['symbols'][1]
            rank_resized = cv2.resize(rank_symbol, (40,60))
            suit_resized = cv2.resize(suit_symbol, (32,32))
            rank_templates.setdefault(rank, []).append(rank_resized)
            suit_name = suit_mapping[suit_dir]
            suit_templates.setdefault(suit_name, []).append(suit_resized)
            vec = extract_symbol_color_vector(suit_symbol, result["corner"])
            color_vectors.append(vec)
        if color_vectors:
            suit_name = suit_mapping[suit_dir]
            suit_color_prototypes[suit_name] = np.mean(np.stack(color_vectors, axis=0), axis=0)
    print("Resumen templates:")
    print("  Ranks:", {k: len(v) for k,v in rank_templates.items()})
    print("  Suits:", {k: len(v) for k,v in suit_templates.items()})
    print("Prototipos de color (H, a_lab, rg_diff):")
    for s, vec in suit_color_prototypes.items():
        print(f"  {s}: {vec}")
    return rank_templates, suit_templates, suit_color_prototypes

# =========================================================
# Procesamiento carta
# =========================================================
def process_card_image(image_path, visualize=False):
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

# =========================================================
# Clasificación de palos v7 (diamantes mejorados)
# =========================================================
def classify_suit_v7(suit_symbol_binary, corner_rgb, suit_templates, suit_color_prototypes):
    best_name_template, template_score, template_detail = enhanced_match_symbol_v2(
        suit_symbol_binary, suit_templates, "suit"
    )

    color_stats = extract_symbol_color_stats_v3(suit_symbol_binary, corner_rgb, template_name=best_name_template)
    shape = compute_shape_metrics(suit_symbol_binary)
    heart_feats = compute_heart_features(suit_symbol_binary)
    diamond_feats = compute_diamond_features(suit_symbol_binary)

    # Re-segmentación si degenerado
    degenerate_fix_applied = False
    if (shape["circularity"] < 0.05) or (shape["solidity"] < 0.2) or (shape["aspect_ratio"] > 3.0):
        new_symbol, applied = resegment_symbol_if_degenerate(suit_symbol_binary, corner_rgb, template_name=best_name_template)
        if applied:
            degenerate_fix_applied = True
            suit_symbol_binary = new_symbol
            color_stats = extract_symbol_color_stats_v3(suit_symbol_binary, corner_rgb, template_name=best_name_template)
            shape = compute_shape_metrics(suit_symbol_binary)
            heart_feats = compute_heart_features(suit_symbol_binary)
            diamond_feats = compute_diamond_features(suit_symbol_binary)

    red_flag = color_stats["is_red"]
    color_group = "red" if red_flag else "black"

    candidate_suits = ["Diamantes", "Corazones"] if color_group == "red" else ["Picas", "Treboles"]

    # Similitud prototipos rojos
    proto_dist = {}
    if color_group == "red" and ("Corazones" in suit_color_prototypes) and ("Diamantes" in suit_color_prototypes):
        vec_current = extract_symbol_color_vector(suit_symbol_binary, corner_rgb)
        for s in ["Corazones","Diamantes"]:
            proto = suit_color_prototypes.get(s)
            if proto is not None:
                proto_dist[s] = np.linalg.norm(vec_current - proto)
        if proto_dist:
            max_d = max(proto_dist.values())
            min_d = min(proto_dist.values())
            if max_d - min_d < 1e-6:
                for k in proto_dist:
                    proto_dist[k] = 0.5
            else:
                for k in proto_dist:
                    proto_dist[k] = 1 - (proto_dist[k] - min_d)/(max_d - min_d)

    per_suit_scores = {}
    strong_template_diamond = (best_name_template == "Diamantes" and template_score >= 0.88)
    strong_template_heart   = (best_name_template == "Corazones" and template_score >= 0.88)

    for suit in candidate_suits:
        base = template_score if best_name_template == suit else template_score * 0.85
        heur = 0.0

        if color_group == "red":
            # Diamantes
            if suit == "Diamantes":
                df = diamond_feats["diamond_feature_score"]
                heur += df * 0.70
                if diamond_feats["approx_vertices"] in [4,5]: heur += 0.08
                if diamond_feats["aspect_ratio_ok"]: heur += 0.05
                if diamond_feats["radial_uniformity"] > 0.6: heur += 0.04
                if diamond_feats["angle_uniformity"] > 0.6: heur += 0.04
                if diamond_feats["orientation_score"] > 0.5: heur += 0.04
                # penalización si corazón fuerte
                if heart_feats["heart_lobes_score"] > 0.55:
                    heur -= 0.10
                if suit in proto_dist:
                    heur += 0.10 * proto_dist[suit]
                if strong_template_diamond:
                    heur += 0.06
            # Corazones
            if suit == "Corazones":
                lobes = heart_feats["heart_lobes_score"]
                heur += lobes * 0.80
                if heart_feats["peak_count"] >= 2: heur += 0.10
                if heart_feats["top_bottom_ratio"] > 1.05: heur += 0.06
                if heart_feats["symmetry"] > 0.85: heur += 0.06
                if diamond_feats["diamond_feature_score"] > 0.50:
                    heur -= 0.08  # diamante fuerte reduce corazón
                if suit in proto_dist:
                    heur += 0.12 * proto_dist[suit]
                if strong_template_heart:
                    heur += 0.06
        else:
            # Negro (sin cambios mayores)
            if suit == "Picas":
                if shape["aspect_ratio"] > 1.05: heur += 0.25
                if shape["defects"] <= 2: heur += 0.15
                if shape["circularity"] < 0.70: heur += 0.08
                if shape["solidity"] < 0.90: heur += 0.06
            if suit == "Treboles":
                if shape["defects"] >= 2: heur += 0.25
                if 0.85 <= shape["aspect_ratio"] <= 1.15: heur += 0.12
                if shape["circularity"] >= 0.65: heur += 0.12
                if shape["solidity"] >= 0.90: heur += 0.08

        # Ajuste si degenerado pero template Diamantes fuerte
        if degenerate_fix_applied and strong_template_diamond and suit == "Diamantes":
            heur += 0.10
            base += 0.05

        final_score = base + heur
        per_suit_scores[suit] = {"base": base, "heur": heur, "final": final_score}

    chosen = max(per_suit_scores.items(), key=lambda x: x[1]["final"])[0]
    final_score = per_suit_scores[chosen]["final"]
    if final_score > 1.0: final_score = 1.0

    debug_info = {
        "template_name": best_name_template,
        "template_score": template_score,
        "template_detail": template_detail,
        "color_stats": color_stats,
        "color_group": color_group,
        "shape": shape,
        "heart_features": heart_feats,
        "diamond_features": diamond_feats,
        "per_suit_scores": per_suit_scores,
        "proto_similarity": proto_dist,
        "degenerate_fix_applied": degenerate_fix_applied
    }

    # Overrides para diamantes (template fuerte + diamond_feature_score)
    if strong_template_diamond and diamond_feats["diamond_feature_score"] >= 0.45 and chosen != "Diamantes":
        if heart_feats["heart_lobes_score"] < 0.70:  # no corazón extremadamente fuerte
            debug_info["override"] = "Forzado Diamantes por template + diamond_feature_score"
            chosen = "Diamantes"
            final_score = max(final_score, 0.80)

    return chosen, final_score, debug_info

# =========================================================
# Interactivo
# =========================================================
def interactive_process_and_classify(image_path, rank_templates, suit_templates, suit_color_prototypes, interactive=True):
    print(f"\n=== Procesando carta: {os.path.basename(image_path)} ===")
    wait_enter(interactive, "Inicio - Enter para procesar...")

    result = process_card_image(image_path, visualize=False)
    if result is None:
        print("  ERROR: No se pudo procesar.")
        return {
            "filename": os.path.basename(image_path),
            "detected_rank": "Failed",
            "detected_suit": "Failed",
            "rank_score": 0.0,
            "suit_score": 0.0,
            "status": "failed"
        }

    mode = "enter" if interactive else "auto"
    show_img(result["original"], "Original", figsize=(5,4), mode=mode)
    show_img(result["binary"], "Binaria", cmap='gray', figsize=(5,4), mode=mode)
    debug = result["original"].copy()
    cv2.drawContours(debug, [result["card_contour"]], -1, (0,255,0), 3)
    show_img(debug, "Contorno", figsize=(5,4), mode=mode)
    show_img(result["warped"], "Warped", figsize=(4,6), mode=mode)
    show_img(result["corner"], "Esquina", figsize=(3,4), mode=mode)
    show_img(result["thresh_corner"], "Esquina thresh", cmap='gray', figsize=(3,4), mode=mode)
    for i, sym in enumerate(result["symbols"]):
        show_img(sym, f"Símbolo {i}", cmap='gray', figsize=(2,2), mode=mode)

    wait_enter(interactive, "Enter para clasificar...")

    symbols_main = result["symbols"]
    potential_rank = symbols_main[0] if len(symbols_main) > 0 else None
    potential_suit = symbols_main[1] if len(symbols_main) > 1 else None

    if potential_rank is not None:
        rank_match, rank_score = enhanced_rank_classification(potential_rank, rank_templates)
    else:
        rank_match, rank_score = "Unknown", 0.0

    if potential_suit is not None:
        suit_match, suit_score, suit_debug = classify_suit_v7(
            potential_suit, result["corner"], suit_templates, suit_color_prototypes
        )
    else:
        suit_match, suit_score = "Unknown", 0.0
        suit_debug = {}

    final_rank = rank_match if rank_score > 0.30 else "Unknown"
    final_suit = suit_match if suit_score > 0.30 else "Unknown"

    warped_with_text = result["warped"].copy()
    text_color = (255,0,0)
    cv2.putText(warped_with_text, f"Rank: {final_rank}", (10,350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    cv2.putText(warped_with_text, f"Palo: {final_suit}", (10,375),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    cv2.putText(warped_with_text, f"Scores: {rank_score:.2f}, {suit_score:.2f}",
                (10,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    show_img(warped_with_text, f"Reconocido: {final_rank} de {final_suit}", figsize=(4,6), mode=mode)

    if potential_suit is not None:
        shape = suit_debug.get("shape", {})
        heartf = suit_debug.get("heart_features", {})
        diamondf = suit_debug.get("diamond_features", {})
        color_stats = suit_debug.get("color_stats", {})
        proto_sim = suit_debug.get("proto_similarity", {})
        print("DEBUG SUIT → Template:", suit_debug.get("template_name"),
              f"(score {suit_debug.get('template_score'):.3f}) | Grupo:", suit_debug.get("color_group"))
        print(f"  Color: red_conf={color_stats.get('red_confidence',0):.3f} hsv_pct={color_stats.get('red_pct_hsv',0):.3f} "
              f"lab_a={color_stats.get('lab_a_mean',0):.1f} rg_diff={color_stats.get('rg_diff_mean',0):.3f} "
              f"fallback={color_stats.get('fallback_color_red',False)}")
        print(f"  Forma: vertices={shape.get('vertices')} defects={shape.get('defects')} "
              f"circ={shape.get('circularity',0):.3f} sol={shape.get('solidity',0):.3f} aspect h/w={shape.get('aspect_ratio',0):.3f} "
              f"deg_fix={suit_debug.get('degenerate_fix_applied',False)}")
        print(f"  Corazón: lobes_score={heartf.get('heart_lobes_score',0):.3f}")
        print(f"  Diamante: diamond_score={diamondf.get('diamond_feature_score',0):.3f} "
              f"approx_vertices={diamondf.get('approx_vertices')} radial_uni={diamondf.get('radial_uniformity',0):.3f} "
              f"angle_uni={diamondf.get('angle_uniformity',0):.3f} orient_score={diamondf.get('orientation_score',0):.3f}")
        if proto_sim:
            print("  Similitud prototipos:", proto_sim)
        if "override" in suit_debug:
            print("  OVERRIDE:", suit_debug["override"])
        print("  Scores por palo:")
        for k,v in suit_debug.get("per_suit_scores", {}).items():
            print(f"    {k}: base={v['base']:.3f} heur={v['heur']:.3f} final={v['final']:.3f}")

    wait_enter(interactive, "Enter para siguiente carta...")

    return {
        "filename": os.path.basename(image_path),
        "detected_rank": final_rank,
        "detected_suit": final_suit,
        "rank_score": rank_score,
        "suit_score": suit_score,
        "status": "ok"
    }

# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Reconocimiento de cartas (v7 - diamantes mejorados).")
    parser.add_argument("--content-dir", default="content/", help="Directorio con imágenes de cartas.")
    parser.add_argument("--template-dir", default="template/", help="Directorio con templates.")
    parser.add_argument("--no-interactive", action="store_true", help="Modo automático (sin Enter).")
    parser.add_argument("--save-csv", action="store_true", help="Guardar resultados en card_recognition_results.csv")
    args = parser.parse_args()

    content_dir = args.content_dir
    template_dir = args.template_dir
    interactive = not args.no_interactive

    print("Construyendo templates y prototipos de color...")
    rank_templates, suit_templates, suit_color_prototypes = build_templates_from_directory(template_dir)
    print(f"Ranks disponibles: {list(rank_templates.keys())}")
    print(f"Suits disponibles: {list(suit_templates.keys())}")

    if not os.path.exists(content_dir):
        print(f"ERROR: No existe el directorio {content_dir}")
        return
    image_files = [f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not image_files:
        print(f"No hay imágenes en {content_dir}")
        return

    print(f"Procesando {len(image_files)} cartas...\n")
    all_results = []
    for filename in sorted(image_files):
        image_path = os.path.join(content_dir, filename)
        res = interactive_process_and_classify(image_path, rank_templates, suit_templates, suit_color_prototypes, interactive=interactive)
        all_results.append(res)

    print("\n=== RESUMEN FINAL ===")
    total_ok = sum(1 for r in all_results if r["status"]=="ok")
    print(f"Cartas procesadas correctamente: {total_ok}/{len(all_results)}")
    for r in all_results:
        print(f"{r['filename']}: {r['detected_rank']} de {r['detected_suit']} (Scores: {r['rank_score']:.2f}, {r['suit_score']:.2f})")

    if args.save_csv:
        with open('card_recognition_results.csv','w',newline='') as csvfile:
            fieldnames = ['filename','detected_rank','detected_suit','rank_score','suit_score','status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print("Resultados guardados en card_recognition_results.csv")

if __name__ == "__main__":
    main()


