"""
Clasificación de palos con análisis de color y forma
"""

import cv2
import numpy as np
import math

from .suit_features import (
    compute_shape_metrics,
    compute_heart_features,
    compute_diamond_features,
    compute_clover_features,
    compute_spade_features
)
from .image_processing import find_contours

def extract_symbol_color_stats_v3(symbol_binary, corner_rgb, template_name=None):
    """
    Mejora en detección de color para condiciones de cámara en vivo
    """
    contours, _ = find_contours(symbol_binary)
    if not contours:
        return {
            "red_pct_hsv": 0.0, "lab_a_mean": 0.0, "rg_diff_mean": 0.0, "cr_mean": 0.0,
            "lab_a_norm": 0.0, "cr_norm": 0.0, "red_confidence": 0.0, "is_red": False,
            "fallback_color_red": False
        }
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    roi = corner_rgb[y:y+h, x:x+w]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = largest - [x, y]
    cv2.drawContours(mask, [shifted], -1, 255, -1)
    
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    l, a, b_channel = cv2.split(roi_lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_equalized = clahe.apply(l)
    roi_lab_enhanced = cv2.merge([l_equalized, a, b_channel])
    roi_enhanced = cv2.cvtColor(roi_lab_enhanced, cv2.COLOR_LAB2RGB)
    
    hsv = cv2.cvtColor(roi_enhanced, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi_enhanced, cv2.COLOR_RGB2LAB)
    ycrcb = cv2.cvtColor(roi_enhanced, cv2.COLOR_RGB2YCrCb)
    
    R = roi_enhanced[:, :, 0].astype(np.float32)
    G = roi_enhanced[:, :, 1].astype(np.float32)
    B = roi_enhanced[:, :, 2].astype(np.float32)
    denom = (R + G + B + 1e-6)
    rg_diff = (R - G) / denom
    
    symbol_mask = (mask > 0)
    
    lower_sets = [
        ((0, 20, 20), (15, 255, 255)),
        ((5, 20, 20), (25, 255, 255)),
        ((160, 20, 20), (180, 255, 255)),
        ((170, 20, 20), (180, 255, 255))
    ]
    
    red_hsv_mask = np.zeros_like(mask)
    for (l_bound, u_bound) in lower_sets:
        l_arr = np.array(l_bound, dtype=np.uint8)
        u_arr = np.array(u_bound, dtype=np.uint8)
        temp = cv2.inRange(hsv, l_arr, u_arr)
        red_hsv_mask = cv2.bitwise_or(red_hsv_mask, temp)
    
    red_pct_hsv = np.sum((red_hsv_mask > 0) & symbol_mask) / (np.sum(symbol_mask) + 1e-6)
    
    a_lab = lab[:, :, 1]
    Cr = ycrcb[:, :, 1]
    
    lab_a_mean = np.mean(a_lab[symbol_mask]) if np.sum(symbol_mask) > 0 else 0.0
    rg_diff_mean = np.mean(rg_diff[symbol_mask]) if np.sum(symbol_mask) > 0 else 0.0
    cr_mean = np.mean(Cr[symbol_mask]) if np.sum(symbol_mask) > 0 else 0.0
    
    lab_a_norm = (lab_a_mean - 128) / 64.0
    cr_norm = (cr_mean - 128) / 64.0
    
    raw_score = (0.40 * red_pct_hsv + 
                 0.25 * max(0, lab_a_norm) + 
                 0.20 * max(0, rg_diff_mean) + 
                 0.15 * max(0, cr_norm))
    
    red_confidence = 1.0 / (1.0 + math.exp(-10 * (raw_score - 0.15)))
    is_red = red_confidence > 0.25
    
    fallback_color_red = False
    if (template_name in ["Corazones", "Diamantes"]) and not is_red:
        kernel = np.ones((3, 3), np.uint8)
        dil_mask = cv2.dilate(mask, kernel, iterations=2)
        dil_symbol = (dil_mask > 0)
        
        R_mean = np.mean(R[dil_symbol]) if np.sum(dil_symbol) > 0 else 0
        G_mean = np.mean(G[dil_symbol]) if np.sum(dil_symbol) > 0 else 0
        B_mean = np.mean(B[dil_symbol]) if np.sum(dil_symbol) > 0 else 0
        
        if (R_mean - G_mean > 5) and (R_mean - B_mean > 5):
            red_confidence = max(red_confidence, 0.40)
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

def extract_symbol_color_vector(symbol_binary, corner_rgb):
    """Extrae vector de color para comparación con prototipos"""
    contours, _ = find_contours(symbol_binary)
    if not contours:
        return np.array([0, 0, 0], dtype=np.float32)
    
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    roi = corner_rgb[y:y+h, x:x+w]
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = largest - [x, y]
    cv2.drawContours(mask, [shifted], -1, 255, -1)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    rgb = roi.astype(np.float32)
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]
    denom = (R + G + B + 1e-6)
    rg_diff = (R - G) / denom
    
    symbol_mask = (mask > 0)
    if np.sum(symbol_mask) == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    
    mean_H = np.mean(hsv[:, :, 0][symbol_mask])
    mean_a = np.mean(lab[:, :, 1][symbol_mask])
    mean_rg = np.mean(rg_diff[symbol_mask])
    
    return np.array([mean_H, mean_a, mean_rg], dtype=np.float32)

def resegment_symbol_if_degenerate(symbol_binary, corner_rgb, template_name=None):
    """Re-segmenta el símbolo si está degenerado"""
    h, w = symbol_binary.shape
    if h < 10 or w < 10:
        return symbol_binary, False
    
    contours, _ = find_contours(symbol_binary)
    if not contours:
        return symbol_binary, False
    
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    circularity = (4*math.pi*area/(peri*peri)) if peri > 0 else 0
    x, y, wc, hc = cv2.boundingRect(c)
    aspect_ratio = hc / wc if wc > 0 else 1
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area > 0 else 0
    
    degenerate = (circularity < 0.05) or (solidity < 0.2) or (aspect_ratio > 3.0)
    if not degenerate:
        return symbol_binary, False
    
    gray_corner = cv2.cvtColor(corner_rgb, cv2.COLOR_RGB2GRAY)
    thr = cv2.adaptiveThreshold(gray_corner, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 5)
    
    if template_name == "Diamantes":
        kernel = np.ones((3, 3), np.uint8)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    thr_dil = cv2.dilate(thr, np.ones((3, 3), np.uint8), iterations=1)
    contours2, _ = find_contours(thr_dil)
    if not contours2:
        return symbol_binary, False
    
    h2, w2 = thr_dil.shape
    candidates = []
    for c2 in contours2:
        area2 = cv2.contourArea(c2)
        if area2 < 30:
            continue
        x2, y2, w2c, h2c = cv2.boundingRect(c2)
        if y2 < h2 * 0.85:
            candidates.append((area2, c2))
    
    if not candidates:
        return symbol_binary, False
    
    best_c = max(candidates, key=lambda x: x[0])[1]
    x2, y2, w2c, h2c = cv2.boundingRect(best_c)
    new_symbol = thr_dil[y2:y2+h2c, x2:x2+w2c].copy()
    
    return new_symbol, True

def classify_suit_v7(suit_symbol_binary, corner_rgb, suit_templates, suit_color_prototypes):
    """
    Clasificación de palos v7 con detección mejorada
    """
    from .rank_classifier import enhanced_match_symbol_v2
    
    best_name_template, template_score, template_detail = enhanced_match_symbol_v2(
        suit_symbol_binary, suit_templates, "suit"
    )
    
    color_stats = extract_symbol_color_stats_v3(suit_symbol_binary, corner_rgb, template_name=best_name_template)
    shape = compute_shape_metrics(suit_symbol_binary)
    heart_feats = compute_heart_features(suit_symbol_binary)
    diamond_feats = compute_diamond_features(suit_symbol_binary)
    clover_feats = compute_clover_features(suit_symbol_binary)
    spade_feats = compute_spade_features(suit_symbol_binary)
    
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
            clover_feats = compute_clover_features(suit_symbol_binary)
            spade_feats = compute_spade_features(suit_symbol_binary)
    
    red_flag = color_stats["is_red"]
    color_group = "red" if red_flag else "black"
    
    candidate_suits = ["Diamantes", "Corazones"] if color_group == "red" else ["Picas", "Treboles"]
    
    proto_dist = {}
    if color_group == "red" and ("Corazones" in suit_color_prototypes) and ("Diamantes" in suit_color_prototypes):
        vec_current = extract_symbol_color_vector(suit_symbol_binary, corner_rgb)
        for s in ["Corazones", "Diamantes"]:
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
    strong_template_heart = (best_name_template == "Corazones" and template_score >= 0.88)
    strong_template_clover = (best_name_template == "Treboles" and template_score >= 0.85)
    strong_template_spade = (best_name_template == "Picas" and template_score >= 0.85)
    
    for suit in candidate_suits:
        base = template_score if best_name_template == suit else template_score * 0.85
        heur = 0.0
        
        if color_group == "red":
            if suit == "Diamantes":
                df = diamond_feats["diamond_feature_score"]
                heur += df * 0.70
                if diamond_feats["approx_vertices"] in [4, 5]:
                    heur += 0.08
                if diamond_feats["aspect_ratio_ok"]:
                    heur += 0.05
                if diamond_feats["radial_uniformity"] > 0.6:
                    heur += 0.04
                if diamond_feats["angle_uniformity"] > 0.6:
                    heur += 0.04
                if diamond_feats["orientation_score"] > 0.5:
                    heur += 0.04
                
                if heart_feats["heart_lobes_score"] > 0.55:
                    heur -= 0.10
                
                if suit in proto_dist:
                    heur += 0.10 * proto_dist[suit]
                if strong_template_diamond:
                    heur += 0.06
            
            elif suit == "Corazones":
                lobes = heart_feats["heart_lobes_score"]
                heur += lobes * 0.80
                if heart_feats["peak_count"] >= 2:
                    heur += 0.10
                if heart_feats["top_bottom_ratio"] > 1.05:
                    heur += 0.06
                if heart_feats["symmetry"] > 0.85:
                    heur += 0.06
                
                if diamond_feats["diamond_feature_score"] > 0.50:
                    heur -= 0.08
                
                if suit in proto_dist:
                    heur += 0.12 * proto_dist[suit]
                if strong_template_heart:
                    heur += 0.06
        
        else:  # Negro
            if suit == "Picas":
                sp = spade_feats["spade_feature_score"]
                heur += sp * 0.75
                
                if spade_feats["peak_sharpness"] > 0.5:
                    heur += 0.10
                if spade_feats["vertical_aspect"] > 0.5:
                    heur += 0.08
                if spade_feats["lateral_symmetry"] > 0.8:
                    heur += 0.06
                if spade_feats["base_narrowness"] > 0.3:
                    heur += 0.05
                
                if clover_feats["clover_feature_score"] > 0.45:
                    heur -= 0.15
                if clover_feats["lobe_count"] == 3:
                    heur -= 0.10
                if clover_feats["base_narrowness"] > 0.3:
                    heur -= 0.08
                
                if strong_template_spade:
                    heur += 0.08
            
            elif suit == "Treboles":
                cf = clover_feats["clover_feature_score"]
                heur += cf * 0.75
                
                if clover_feats["lobe_count"] == 3:
                    heur += 0.12
                elif clover_feats["lobe_count"] == 2:
                    heur += 0.06
                
                if clover_feats["base_narrowness"] > 0.25:
                    heur += 0.08
                
                if clover_feats["top_bottom_complexity"] > 0.4:
                    heur += 0.06
                
                if clover_feats["convexity_defects_score"] > 0.6:
                    heur += 0.06
                
                if 0.85 <= shape["aspect_ratio"] <= 1.15:
                    heur += 0.08
                if shape["defects"] >= 2:
                    heur += 0.10
                if shape["circularity"] >= 0.60:
                    heur += 0.06
                if shape["solidity"] >= 0.85:
                    heur += 0.05
                
                if diamond_feats["diamond_feature_score"] > 0.50:
                    heur -= 0.15
                if diamond_feats["approx_vertices"] == 4:
                    heur -= 0.10
                if diamond_feats["radial_uniformity"] > 0.7:
                    heur -= 0.08
                
                if strong_template_clover:
                    heur += 0.10
        
        if degenerate_fix_applied:
            if strong_template_diamond and suit == "Diamantes":
                heur += 0.10
                base += 0.05
            elif strong_template_clover and suit == "Treboles":
                heur += 0.12
                base += 0.06
        
        final_score = base + heur
        per_suit_scores[suit] = {"base": base, "heur": heur, "final": final_score}
    
    chosen = max(per_suit_scores.items(), key=lambda x: x[1]["final"])[0]
    final_score = per_suit_scores[chosen]["final"]
    if final_score > 1.0:
        final_score = 1.0
    
    debug_info = {
        "template_name": best_name_template,
        "template_score": template_score,
        "template_detail": template_detail,
        "color_stats": color_stats,
        "color_group": color_group,
        "shape": shape,
        "heart_features": heart_feats,
        "diamond_features": diamond_feats,
        "clover_features": clover_feats,
        "spade_features": spade_feats,
        "per_suit_scores": per_suit_scores,
        "proto_similarity": proto_dist,
        "degenerate_fix_applied": degenerate_fix_applied
    }
    
    # Overrides
    if strong_template_diamond and diamond_feats["diamond_feature_score"] >= 0.45 and chosen != "Diamantes":
        if heart_feats["heart_lobes_score"] < 0.70:
            debug_info["override"] = "Forzado Diamantes por template + diamond_feature_score"
            chosen = "Diamantes"
            final_score = max(final_score, 0.80)
    
    if color_group == "black":
        if strong_template_clover and clover_feats["clover_feature_score"] >= 0.40 and chosen != "Treboles":
            if diamond_feats["diamond_feature_score"] < 0.60:
                debug_info["override"] = "Forzado Treboles por template + clover_feature_score"
                chosen = "Treboles"
                final_score = max(final_score, 0.75)
        
        if clover_feats["clover_feature_score"] >= 0.65 and clover_feats["lobe_count"] == 3:
            if chosen != "Treboles":
                debug_info["override"] = "Forzado Treboles por características muy fuertes (3 lóbulos)"
                chosen = "Treboles"
                final_score = max(final_score, 0.80)
    
    return chosen, final_score, debug_info