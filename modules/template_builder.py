"""
Construcción de templates desde directorio
"""

import os
import cv2
import numpy as np

def build_templates_from_directory(template_base_dir="template/"):
    """
    Construye templates de ranks y suits desde directorio
    """
    from .card_detection import process_card_image
    from .suit_classifier import extract_symbol_color_vector
    
    rank_templates = {}
    suit_templates = {}
    suit_color_prototypes = {}
    
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
        
        print(f"Procesando templates de: {suit_dir}...")
        color_vectors = []
        
        for filename in os.listdir(suit_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            rank = filename.split('_')[0]
            image_path = os.path.join(suit_path, filename)
            result = process_card_image(image_path, visualize=False)
            
            if result is None or len(result['symbols']) < 2:
                print(f"  Omitiendo {filename} - símbolos insuficientes")
                continue
            
            rank_symbol = result['symbols'][0]
            suit_symbol = result['symbols'][1]
            
            rank_resized = cv2.resize(rank_symbol, (40, 60))
            suit_resized = cv2.resize(suit_symbol, (32, 32))
            
            rank_templates.setdefault(rank, []).append(rank_resized)
            
            suit_name = suit_mapping[suit_dir]
            suit_templates.setdefault(suit_name, []).append(suit_resized)
            
            vec = extract_symbol_color_vector(suit_symbol, result["corner"])
            color_vectors.append(vec)
        
        if color_vectors:
            suit_name = suit_mapping[suit_dir]
            suit_color_prototypes[suit_name] = np.mean(np.stack(color_vectors, axis=0), axis=0)
    
    print("Resumen templates:")
    print("  Ranks:", {k: len(v) for k, v in rank_templates.items()})
    print("  Suits:", {k: len(v) for k, v in suit_templates.items()})
    print("Prototipos de color (H, a_lab, rg_diff):")
    for s, vec in suit_color_prototypes.items():
        print(f"  {s}: {vec}")
    
    return rank_templates, suit_templates, suit_color_prototypes