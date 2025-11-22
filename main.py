import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Helper: safe imshow
# -----------------------
def show_img(img, title="", figsize=(6, 6), cmap=None):
    plt.figure(figsize=figsize)
    if len(img.shape) == 2 or cmap is not None:
        plt.imshow(img, cmap=cmap)
    else:
        # assume image is in RGB
        plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# -----------------------
# Load image (BGR -> RGB) with check
# -----------------------
def load_image_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

# -----------------------
# Find contours wrapper (compatible with different OpenCV versions)
# -----------------------
def find_contours(binary):
    # cv2.findContours returns either (contours, hierarchy) or (image, contours, hierarchy)
    res = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hierarchy = res
    else:
        _, contours, hierarchy = res
    return contours, hierarchy

# -----------------------
# Order points for perspective transform
# -----------------------
def order_points(pts):
    pts = pts.astype("float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

# -----------------------
# Find the largest quadrilateral contour (candidate card)
# -----------------------
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

# -----------------------
# Four point perspective transform wrapper
# -----------------------
def four_point_transform(image_rgb, pts, width=300, height=420):
    src = pts.reshape(4, 2)
    src_ord = order_points(src)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(src_ord, dst)
    warped = cv2.warpPerspective(image_rgb, M, (width, height))
    return warped

# -----------------------
# Extract rank + suit top-left corner from warped card
# -----------------------
def extract_top_left_corner(warped_card, w_ratio=0.25, h_ratio=0.35):
    h, w = warped_card.shape[:2]
    rw = int(w * w_ratio)
    rh = int(h * h_ratio)
    corner = warped_card[0:rh, 0:rw].copy()
    return corner

# -----------------------
# Extract symbol contours from corner (returns binary image, list of symbol images)
# -----------------------
def extract_symbols_from_corner(corner_rgb, min_area=50, horizontal_gap=20):
    """
    Extrae los símbolos (rank y suit) desde la esquina superior izquierda de la carta.
    Combina automáticamente contornos horizontales cercanos para obtener '10' correctamente.
    Retorna: thresh_binary, lista_de_simbolos (ordenados: RANK primero, SUIT después)
    """

    gray = cv2.cvtColor(corner_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = find_contours(thresh)

    # Obtener bounding boxes
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])

    if not boxes:
        return thresh, []

    # Orden preliminar por coordenada Y (arriba→abajo)
    boxes.sort(key=lambda b: (b[1], b[0]))

    # === FUSIÓN DE CONTORNOS HORIZONTALES (para formar "10") ===
    merged = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if not merged:
            merged.append([x1, y1, x2, y2])
        else:
            mx1, my1, mx2, my2 = merged[-1]

            # Condición: está en la misma línea vertical y muy cerca horizontalmente
            if abs(y1 - my1) < 25 and (x1 - mx2) < horizontal_gap:
                # fusionar
                merged[-1] = [
                    min(mx1, x1),
                    min(my1, y1),
                    max(mx2, x2),
                    max(my2, y2)
                ]
            else:
                merged.append([x1, y1, x2, y2])

    # Orden final: RANK primero (más arriba), luego SUIT
    merged.sort(key=lambda b: (b[1], b[0]))

    # Recortar los símbolos fusionados
    symbols = []
    for (x1, y1, x2, y2) in merged:
        crop = thresh[max(0, y1-2):y2+2, max(0, x1-2):x2+2]
        symbols.append(crop)

    return thresh, symbols


# -----------------------
# Template matching helper
# -----------------------
def match_symbol(symbol_img, templates_dict, method=cv2.TM_CCOEFF_NORMED):
    # symbol_img and templates should be single-channel (grayscale or binary)
    if symbol_img is None or len(templates_dict) == 0:
        return "Unknown", -1.0

    best_name = "Unknown"
    best_score = -1.0
    for name, tmpl in templates_dict.items():
        # ensure same size for matching
        if tmpl.shape != symbol_img.shape:
            tmpl_resized = cv2.resize(tmpl, (symbol_img.shape[1], symbol_img.shape[0]))
        else:
            tmpl_resized = tmpl

        res = cv2.matchTemplate(symbol_img, tmpl_resized, method)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, float(best_score)

# -----------------------
# High-level pipeline for a single image path
# -----------------------
def process_card_image(image_path, visualize=False):
    try:
        image_rgb = load_image_rgb(image_path)
    except FileNotFoundError as e:
        print(e)
        return None

    # show original
    if visualize:
        show_img(image_rgb, "Original (RGB)", figsize=(8,6))

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if visualize:
        show_img(binary, "Binary (Otsu)", cmap='gray', figsize=(6,6))

    card_contour, area = find_card_contour_from_binary(binary, min_area=10000)
    if card_contour is None:
        print(f"No card contour found in {image_path} (area threshold may be too high).")
        return None

    # draw contour for debugging
    debug = image_rgb.copy()
    cv2.drawContours(debug, [card_contour], -1, (0,255,0), 3)
    if visualize:
        show_img(debug, "Detected card contour")

    # warp
    warped = four_point_transform(image_rgb, card_contour, width=300, height=420)
    if visualize:
        show_img(warped, "Warped card", figsize=(4,6))

    # extract corner symbols
    corner = extract_top_left_corner(warped)
    if visualize:
        show_img(corner, "Top-left corner (rank & suit)", figsize=(3,4))

    thresh_corner, symbols = extract_symbols_from_corner(corner)
    if visualize:
        show_img(thresh_corner, "Thresh corner", cmap='gray', figsize=(3,4))
        # show extracted symbol crops
        for i, s in enumerate(symbols):
            show_img(s, f"Symbol {i}", cmap='gray', figsize=(2,2))

    # Return results
    return {
        "image_path": image_path,
        "original": image_rgb,
        "warped": warped,
        "corner": corner,
        "thresh_corner": thresh_corner,
        "symbols": symbols
    }

def build_templates_from_directory(template_base_dir="template/"):
    """
    Construye diccionarios de templates para ranks y suits desde el directorio template/
    que contiene subdirectorios por cada palo (corazones, diamantes, picas, treboles)
    """
    rank_templates = {}
    suit_templates = {}
    
    # Mapeo de nombres de palos en español
    suit_mapping = {
        'corazones': 'Corazones',
        'diamantes': 'Diamantes', 
        'picas': 'Picas',
        'treboles': 'Treboles'
    }
    
    for suit_dir in ['corazones', 'diamantes', 'picas', 'treboles']:
        suit_path = os.path.join(template_base_dir, suit_dir)
        if not os.path.exists(suit_path):
            continue
            
        print(f"Processing {suit_dir} templates...")
        
        for filename in os.listdir(suit_path):
            if not filename.lower().endswith('.jpg'):
                continue
                
            # Extraer el rank del nombre del archivo (ej: "A_corazones.jpg" -> "A")
            rank = filename.split('_')[0]
            
            image_path = os.path.join(suit_path, filename)
            result = process_card_image(image_path, visualize=False)
            
            if result is None or len(result['symbols']) < 2:
                print(f"  Skipping {filename} - insufficient symbols detected")
                continue
                
            # Primer símbolo = rank, segundo símbolo = suit
            rank_symbol = result['symbols'][0] 
            suit_symbol = result['symbols'][1]
            
            # Redimensionar para consistencia
            rank_resized = cv2.resize(rank_symbol, (40, 60))
            suit_resized = cv2.resize(suit_symbol, (30, 40))
            
            # Almacenar template de rank (usar el primer ejemplo encontrado de cada rank)
            if rank not in rank_templates:
                rank_templates[rank] = rank_resized
                print(f"  Added rank template: {rank}")
            
            # Almacenar template de suit (usar el primer ejemplo encontrado de cada suit)
            suit_name = suit_mapping[suit_dir]
            if suit_name not in suit_templates:
                suit_templates[suit_name] = suit_resized
                print(f"  Added suit template: {suit_name}")
    
    return rank_templates, suit_templates

def enhanced_match_symbol(symbol_img, templates_dict, method=cv2.TM_CCOEFF_NORMED):
    """
    Versión mejorada del template matching con normalización y múltiples métricas
    """
    if symbol_img is None or len(templates_dict) == 0:
        return "Unknown", -1.0
    
    # Normalizar el símbolo de entrada
    symbol_normalized = cv2.resize(symbol_img, (40, 60))
    
    best_name = "Unknown"
    best_score = -1.0
    
    # También probar con diferentes métodos de matching para mayor robustez
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    
    for name, template in templates_dict.items():
        total_score = 0
        
        for method in methods:
            # Asegurar que ambas imágenes tengan el mismo tamaño
            template_resized = cv2.resize(template, symbol_normalized.shape[::-1])
            
            res = cv2.matchTemplate(symbol_normalized, template_resized, method)
            _, score, _, _ = cv2.minMaxLoc(res)
            total_score += score
        
        # Promedio de scores de diferentes métodos
        avg_score = total_score / len(methods)
        
        if avg_score > best_score:
            best_score = avg_score
            best_name = name
    
    return best_name, float(best_score)

def analyze_suit_color_and_shape(suit_symbol, corner_rgb_region):
    """
    Analiza el color y forma del símbolo del palo para mejorar la clasificación
    """
    # Obtener región de color del símbolo del palo
    h, w = corner_rgb_region.shape[:2]
    
    # Convertir a HSV para análisis de color
    hsv = cv2.cvtColor(corner_rgb_region, cv2.COLOR_RGB2HSV)
    
    # Definir rangos de color rojo más amplios para capturar variaciones
    lower_red1 = np.array([0, 30, 30])  # Rangos más permisivos
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    
    # Crear máscaras para rojo
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Calcular porcentaje de píxeles rojos
    red_pixels = np.sum(red_mask > 0)
    total_pixels = red_mask.size
    red_percentage = red_pixels / total_pixels
    
    # Análisis de forma del símbolo
    contours, _ = find_contours(suit_symbol)
    if not contours:
        return {"is_red": False, "aspect_ratio": 1.0, "solidity": 0.0, "red_percentage": red_percentage}
    
    # Tomar el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calcular características de forma
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = w / h
    
    # Calcular solidez (área del contorno / área del rectángulo delimitador)
    contour_area = cv2.contourArea(largest_contour)
    rect_area = w * h
    solidity = contour_area / rect_area if rect_area > 0 else 0
    
    # Calcular compacidad (perímetro²/área) - los diamantes tienen valores específicos
    perimeter = cv2.arcLength(largest_contour, True)
    compactness = (perimeter * perimeter) / contour_area if contour_area > 0 else 0
    
    return {
        "is_red": red_percentage > 0.05,  # Umbral más bajo para capturar rojos tenues
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "red_percentage": red_percentage,
        "compactness": compactness
    }

def enhanced_suit_classification(suit_symbol, corner_rgb, templates_dict):
    """
    Clasificación mejorada de palos usando template matching + análisis de color y forma
    """
    # Template matching básico con múltiples métricas
    basic_match, basic_score = enhanced_match_symbol_v2(suit_symbol, templates_dict, "suit")
    
    # Análisis de color y forma
    shape_analysis = analyze_suit_color_and_shape(suit_symbol, corner_rgb)
    
    print(f"  Template match: {basic_match} (score: {basic_score:.3f})")
    print(f"  Shape analysis: Red={shape_analysis['is_red']} ({shape_analysis['red_percentage']:.3f}), AR={shape_analysis['aspect_ratio']:.2f}, Solidity={shape_analysis['solidity']:.2f}, Compactness={shape_analysis['compactness']:.2f}")
    
    # Lógica de decisión mejorada con más heurísticas
    if shape_analysis["is_red"] or shape_analysis["red_percentage"] > 0.02:  # Es rojo o tiene trazas de rojo
        # Clasificación entre corazones y diamantes
        
        # Los diamantes tienen características específicas:
        # - Aspect ratio más cercano a 1 (más cuadrado)
        # - Compacidad específica
        # - Solidez moderada
        
        diamond_score = 0
        heart_score = 0
        
        # Análisis de aspect ratio
        if 0.7 <= shape_analysis["aspect_ratio"] <= 1.3:
            diamond_score += 2
        else:
            heart_score += 1
        
        # Análisis de compacidad (los diamantes tienen valores específicos)
        if 15 <= shape_analysis["compactness"] <= 25:
            diamond_score += 2
        elif shape_analysis["compactness"] > 25:
            heart_score += 2
        
        # Análisis de solidez
        if 0.5 <= shape_analysis["solidity"] <= 0.8:
            diamond_score += 1
        elif shape_analysis["solidity"] > 0.8:
            heart_score += 1
        
        # Si el template matching ya dio una respuesta confiable
        if basic_match in ["Diamantes", "Corazones"] and basic_score > 0.6:
            if basic_match == "Diamantes":
                diamond_score += 3
            else:
                heart_score += 3
        
        # Decisión final para palos rojos
        if diamond_score > heart_score:
            final_score = max(basic_score, 0.7) if diamond_score >= 4 else basic_score
            return "Diamantes", final_score
        else:
            final_score = max(basic_score, 0.7) if heart_score >= 4 else basic_score
            return "Corazones", final_score
            
    else:
        # Es un palo negro (picas o tréboles)
        if basic_match in ["Picas", "Treboles"] and basic_score > 0.5:
            return basic_match, basic_score
        else:
            # Análisis para palos negros
            # Los tréboles tienen más "complejidad" y menor solidez
            if shape_analysis["solidity"] < 0.6 or shape_analysis["compactness"] > 20:
                return "Treboles", max(basic_score, 0.6)
            else:
                return "Picas", max(basic_score, 0.6)

def enhanced_rank_classification(rank_symbol, templates_dict):
    """
    Clasificación específica mejorada para ranks problemáticos como 9, Q, J
    """
    # Template matching básico
    basic_match, basic_score = enhanced_match_symbol_v2(rank_symbol, templates_dict, "rank")
    
    # Para casos específicos de Q, J, 9 que pueden ser problemáticos
    if basic_match in ["Q", "J", "9"] or basic_score < 0.6:
        
        # Probar con diferentes escalas y rotaciones ligeras para Q, J, 9
        best_match = basic_match
        best_score = basic_score
        
        # Probar con diferentes tamaños
        for scale in [0.9, 1.0, 1.1]:
            h, w = rank_symbol.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            if new_h > 0 and new_w > 0:
                scaled_symbol = cv2.resize(rank_symbol, (new_w, new_h))
                
                # Template matching con el símbolo escalado
                for name, template in templates_dict.items():
                    if name in ["Q", "J", "9"]:  # Solo para los problemáticos
                        try:
                            # Redimensionar template al símbolo escalado
                            template_scaled = cv2.resize(template, (new_w, new_h))
                            
                            # Usar múltiples métodos
                            scores = []
                            for method in [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]:
                                res = cv2.matchTemplate(scaled_symbol, template_scaled, method)
                                _, score, _, _ = cv2.minMaxLoc(res)
                                scores.append(score)
                            
                            avg_score = np.mean(scores)
                            
                            if avg_score > best_score:
                                best_score = avg_score
                                best_match = name
                        except:
                            continue
        
        return best_match, best_score
    
    return basic_match, basic_score

def enhanced_match_symbol_v2(symbol_img, templates_dict, symbol_type="rank"):
    """
    Versión mejorada del template matching con mejor normalización
    """
    if symbol_img is None or len(templates_dict) == 0:
        return "Unknown", -1.0
    
    best_name = "Unknown"
    best_score = -1.0
    
    for name, template in templates_dict.items():
        # Redimensionar el símbolo al tamaño del template
        symbol_resized = cv2.resize(symbol_img, (template.shape[1], template.shape[0]))
        
        # Usar múltiples métodos de matching con pesos ajustados
        methods_and_weights = [
            (cv2.TM_CCOEFF_NORMED, 0.5),
            (cv2.TM_CCORR_NORMED, 0.3),
            (cv2.TM_SQDIFF_NORMED, 0.2)
        ]
        
        total_weighted_score = 0
        for method, weight in methods_and_weights:
            res = cv2.matchTemplate(symbol_resized, template, method)
            _, score, _, _ = cv2.minMaxLoc(res)
            
            # Para TM_SQDIFF_NORMED, invertir el score
            if method == cv2.TM_SQDIFF_NORMED:
                score = 1.0 - score
            
            total_weighted_score += score * weight
        
        if total_weighted_score > best_score:
            best_score = total_weighted_score
            best_name = name
    
    return best_name, float(best_score)

# -----------------------
# Example usage: process a single card and then attempt recognition using templates in a folder
# -----------------------
if __name__ == "__main__":
    # Input file to analyze
    target_file = "content/Q_picas.jpg"  # Cambia esto para probar diferentes cartas

    result = process_card_image(target_file, visualize=True)
    if result is None:
        raise SystemExit("Processing failed - check input path and image quality.")

    # Build templates from the template directory
    print("Building templates from template directory...")
    rank_templates, suit_templates = build_templates_from_directory("template/")

    print(f"Rank templates loaded: {list(rank_templates.keys())}")
    print(f"Suit templates loaded: {list(suit_templates.keys())}")

    # Now attempt recognition on the main processed card
    symbols_main = result["symbols"]
    potential_rank = symbols_main[0] if len(symbols_main) > 0 else None
    potential_suit = symbols_main[1] if len(symbols_main) > 1 else None

    # Initialize default values
    rank_match, rank_score = "Unknown", 0.0
    suit_match, suit_score = "Unknown", 0.0

    if potential_rank is not None:
        # Usar clasificación mejorada para ranks
        rank_match, rank_score = enhanced_rank_classification(potential_rank, rank_templates)
        print(f"Rank match: {rank_match} (score: {rank_score:.3f})")
    else:
        print("No potential rank symbol detected.")

    if potential_suit is not None:
        # Usar la función mejorada para palos
        print("Analyzing suit...")
        suit_match, suit_score = enhanced_suit_classification(
            potential_suit, result["corner"], suit_templates
        )
        print(f"Final suit match: {suit_match} (score: {suit_score:.3f})")
    else:
        print("No potential suit symbol detected.")

    # Visualize final annotation on warped card
    final_rank = rank_match if rank_score > 0.4 else "Unknown"  # Umbral más bajo
    final_suit = suit_match if suit_score > 0.4 else "Unknown"

    warped_with_text = result["warped"].copy()
    text_color = (255, 0, 0)  # Texto en rojo para mejor visibilidad
    cv2.putText(warped_with_text, f"Rank: {final_rank}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    cv2.putText(warped_with_text, f"Suit: {final_suit}", (10, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    cv2.putText(warped_with_text, f"Scores: {rank_score:.2f}, {suit_score:.2f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    show_img(warped_with_text, f"Recognized: {final_rank} of {final_suit}", figsize=(4,6))

def test_all_cards(content_dir="content/", template_base_dir="template/"):
    """
    Prueba todas las cartas en el directorio content/ y genera un reporte de precisión
    """
    print("Building templates from template directory...")
    rank_templates, suit_templates = build_templates_from_directory(template_base_dir)
    print(f"Rank templates loaded: {list(rank_templates.keys())}")
    print(f"Suit templates loaded: {list(suit_templates.keys())}")
    
    # Obtener todas las imágenes del directorio content
    if not os.path.exists(content_dir):
        print(f"Directory {content_dir} does not exist!")
        return
    
    image_files = [f for f in os.listdir(content_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {content_dir}")
        return
    
    print(f"\nTesting {len(image_files)} cards...\n")
    
    correct_rank = 0
    correct_suit = 0
    correct_both = 0
    total_cards = 0
    failed_processing = 0
    
    results = []
    
    for filename in sorted(image_files):
        print(f"Processing: {filename}")
        
        # Extraer el rank y suit esperados del nombre del archivo
        try:
            if '_' in filename:
                expected_rank, expected_suit_with_ext = filename.split('_', 1)
                expected_suit = expected_suit_with_ext.split('.')[0]
                
                # Mapear nombres de archivos a nombres de templates
                suit_mapping = {
                    'corazones': 'Corazones',
                    'diamantes': 'Diamantes',
                    'picas': 'Picas',
                    'treboles': 'Treboles'
                }
                expected_suit = suit_mapping.get(expected_suit, expected_suit)
            else:
                print(f"  Skipping {filename} - invalid filename format")
                continue
        except:
            print(f"  Skipping {filename} - cannot parse filename")
            continue
        
        # Procesar la imagen
        image_path = os.path.join(content_dir, filename)
        result = process_card_image(image_path, visualize=False)
        
        if result is None:
            print(f"  FAILED: Could not process image")
            failed_processing += 1
            continue
        
        # Detectar símbolos
        symbols_main = result["symbols"]
        potential_rank = symbols_main[0] if len(symbols_main) > 0 else None
        potential_suit = symbols_main[1] if len(symbols_main) > 1 else None
        
        # Clasificar rank
        if potential_rank is not None:
            rank_match, rank_score = enhanced_rank_classification(potential_rank, rank_templates)
        else:
            rank_match, rank_score = "Unknown", 0.0
        
        # Clasificar suit
        if potential_suit is not None:
            suit_match, suit_score = enhanced_suit_classification(
                potential_suit, result["corner"], suit_templates
            )
        else:
            suit_match, suit_score = "Unknown", 0.0
        
        # Aplicar umbrales
        final_rank = rank_match if rank_score > 0.4 else "Unknown"
        final_suit = suit_match if suit_score > 0.4 else "Unknown"
        
        # Evaluar resultados
        rank_correct = (final_rank == expected_rank)
        suit_correct = (final_suit == expected_suit)
        both_correct = rank_correct and suit_correct
        
        if rank_correct:
            correct_rank += 1
        if suit_correct:
            correct_suit += 1
        if both_correct:
            correct_both += 1
        
        total_cards += 1
        
        # Guardar resultado
        results.append({
            'filename': filename,
            'expected_rank': expected_rank,
            'expected_suit': expected_suit,
            'detected_rank': final_rank,
            'detected_suit': final_suit,
            'rank_score': rank_score,
            'suit_score': suit_score,
            'rank_correct': rank_correct,
            'suit_correct': suit_correct,
            'both_correct': both_correct
        })
        
        # Mostrar resultado
        status = "✓" if both_correct else "✗"
        print(f"  {status} Expected: {expected_rank} of {expected_suit} | "
              f"Detected: {final_rank} of {final_suit} | "
              f"Scores: {rank_score:.2f}, {suit_score:.2f}")
        
        if not both_correct:
            if not rank_correct:
                print(f"    RANK ERROR: Expected {expected_rank}, got {final_rank}")
            if not suit_correct:
                print(f"    SUIT ERROR: Expected {expected_suit}, got {final_suit}")
    
    # Mostrar resumen final
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total cards processed: {total_cards}")
    print(f"Failed to process: {failed_processing}")
    print(f"Rank accuracy: {correct_rank}/{total_cards} ({100*correct_rank/total_cards:.1f}%)")
    print(f"Suit accuracy: {correct_suit}/{total_cards} ({100*correct_suit/total_cards:.1f}%)")
    print(f"Both correct: {correct_both}/{total_cards} ({100*correct_both/total_cards:.1f}%)")
    
    # Mostrar errores
    print("\nERRORS:")
    for result in results:
        if not result['both_correct']:
            print(f"  {result['filename']}: Expected {result['expected_rank']} of {result['expected_suit']} | "
                  f"Got {result['detected_rank']} of {result['detected_suit']}")
    
    return results

def test_single_card(target_file):
    """
    Función para probar una sola carta (funcionalidad original)
    """
    result = process_card_image(target_file, visualize=True)
    if result is None:
        raise SystemExit("Processing failed - check input path and image quality.")

    # Build templates from the template directory
    print("Building templates from template directory...")
    rank_templates, suit_templates = build_templates_from_directory("template/")

    print(f"Rank templates loaded: {list(rank_templates.keys())}")
    print(f"Suit templates loaded: {list(suit_templates.keys())}")

    # Now attempt recognition on the main processed card
    symbols_main = result["symbols"]
    potential_rank = symbols_main[0] if len(symbols_main) > 0 else None
    potential_suit = symbols_main[1] if len(symbols_main) > 1 else None

    # Initialize default values
    rank_match, rank_score = "Unknown", 0.0
    suit_match, suit_score = "Unknown", 0.0

    if potential_rank is not None:
        # Usar clasificación mejorada para ranks
        rank_match, rank_score = enhanced_rank_classification(potential_rank, rank_templates)
        print(f"Rank match: {rank_match} (score: {rank_score:.3f})")
    else:
        print("No potential rank symbol detected.")

    if potential_suit is not None:
        # Usar la función mejorada para palos
        print("Analyzing suit...")
        suit_match, suit_score = enhanced_suit_classification(
            potential_suit, result["corner"], suit_templates
        )
        print(f"Final suit match: {suit_match} (score: {suit_score:.3f})")
    else:
        print("No potential suit symbol detected.")

    # Visualize final annotation on warped card
    final_rank = rank_match if rank_score > 0.4 else "Unknown"
    final_suit = suit_match if suit_score > 0.4 else "Unknown"

    warped_with_text = result["warped"].copy()
    text_color = (255, 0, 0)
    cv2.putText(warped_with_text, f"Rank: {final_rank}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    cv2.putText(warped_with_text, f"Suit: {final_suit}", (10, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    cv2.putText(warped_with_text, f"Scores: {rank_score:.2f}, {suit_score:.2f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    show_img(warped_with_text, f"Recognized: {final_rank} of {final_suit}", figsize=(4,6))

# -----------------------
# Example usage: process a single card and then attempt recognition using templates in a folder
# -----------------------
if __name__ == "__main__":
    # Elegir el modo de operación
    MODE = "single"  # Cambiar a "single" para probar una sola carta
    
    if MODE == "single":
        # Probar una sola carta
        target_file = "content/5_corazones.jpg"
        test_single_card(target_file)
    
    elif MODE == "test_all":
        # Probar todas las cartas
        results = test_all_cards("content/", "template/")
        
        # Opcional: guardar resultados en archivo CSV
        import csv
        with open('card_recognition_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['filename', 'expected_rank', 'expected_suit', 'detected_rank', 'detected_suit', 
                         'rank_score', 'suit_score', 'rank_correct', 'suit_correct', 'both_correct']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerows([result])
        print("\nResults saved to card_recognition_results.csv")


