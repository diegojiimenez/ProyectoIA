"""
Clasificación de números y letras (ranks)
"""

import cv2
import numpy as np

def multi_template_scores(symbol_img, templates_list):
    """
    Mejorado con más robustez para condiciones de cámara en vivo
    """
    if symbol_img is None or len(templates_list) == 0:
        return 0.0, {}
    
    H0, W0 = 32, 32
    
    symbol_norm = cv2.resize(symbol_img, (W0, W0))
    symbol_norm = cv2.equalizeHist(symbol_norm)
    
    symbol_edges = cv2.Canny(symbol_norm, 30, 150)
    symbol_inv = cv2.bitwise_not(symbol_norm)
    dist_symbol = cv2.distanceTransform(symbol_inv, cv2.DIST_L2, 3)
    
    vec_symbol = symbol_norm.flatten().astype(np.float32)
    vec_symbol /= (np.linalg.norm(vec_symbol) + 1e-6)
    
    best = 0.0
    best_detail = None
    
    for tmpl in templates_list:
        scales = [0.9, 1.0, 1.1]
        scale_scores = []
        
        for scale in scales:
            h_tmpl, w_tmpl = tmpl.shape
            new_h, new_w = int(h_tmpl * scale), int(w_tmpl * scale)
            if new_h <= 0 or new_w <= 0 or new_h > W0*2 or new_w > W0*2:
                continue
                
            tmpl_scaled = cv2.resize(tmpl, (new_w, new_h))
            tmpl_norm = cv2.resize(tmpl_scaled, (W0, W0))
            tmpl_norm = cv2.equalizeHist(tmpl_norm)
            
            res = cv2.matchTemplate(symbol_norm, tmpl_norm, cv2.TM_CCOEFF_NORMED)
            _, corr_score, _, _ = cv2.minMaxLoc(res)
            
            tmpl_edges = cv2.Canny(tmpl_norm, 30, 150)
            res_e = cv2.matchTemplate(symbol_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
            _, edge_score, _, _ = cv2.minMaxLoc(res_e)
            
            tmpl_inv = cv2.bitwise_not(tmpl_norm)
            dist_tmpl = cv2.distanceTransform(tmpl_inv, cv2.DIST_L2, 3)
            
            sym_pts = np.where(symbol_edges > 0)
            tmpl_pts = np.where(tmpl_edges > 0)
            
            ch1 = dist_tmpl[sym_pts].mean() if len(sym_pts[0]) > 0 else 50.0
            ch2 = dist_symbol[tmpl_pts].mean() if len(tmpl_pts[0]) > 0 else 50.0
            chamfer_score = np.exp(-0.5 * (ch1 + ch2) / 10.0)
            
            vec_tmpl = tmpl_norm.flatten().astype(np.float32)
            vec_tmpl /= (np.linalg.norm(vec_tmpl) + 1e-6)
            cosine = float(np.dot(vec_symbol, vec_tmpl))
            if cosine < 0:
                cosine = 0.0
            
            mean_s = np.mean(symbol_norm)
            mean_t = np.mean(tmpl_norm)
            std_s = np.std(symbol_norm)
            std_t = np.std(tmpl_norm)
            
            cov = np.mean((symbol_norm - mean_s) * (tmpl_norm - mean_t))
            ssim_score = (2 * mean_s * mean_t + 1e-6) / (mean_s**2 + mean_t**2 + 1e-6) * \
                        (2 * cov + 1e-6) / (std_s**2 + std_t**2 + 1e-6)
            ssim_score = max(0, min(1, ssim_score))
            
            combined = (0.30 * corr_score + 
                       0.20 * edge_score + 
                       0.20 * chamfer_score + 
                       0.15 * cosine +
                       0.15 * ssim_score)
            
            scale_scores.append({
                "corr": corr_score,
                "edge": edge_score,
                "chamfer": chamfer_score,
                "cosine": cosine,
                "ssim": ssim_score,
                "combined": combined,
                "scale": scale
            })
        
        if scale_scores:
            best_scale = max(scale_scores, key=lambda x: x["combined"])
            
            if best_scale["combined"] > best:
                best = best_scale["combined"]
                best_detail = best_scale
    
    return best, (best_detail if best_detail else {})

def enhanced_match_symbol_v2(symbol_img, templates_dict, symbol_type="rank"):
    """Encuentra el mejor match entre templates"""
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

def enhanced_rank_classification(rank_symbol, rank_templates):
    """
    Clasificación mejorada de ranks con detección específica para números problemáticos
    """
    h, w = rank_symbol.shape
    
    rank_symbol_enhanced = cv2.equalizeHist(rank_symbol)
    rank_symbol_denoised = cv2.bilateralFilter(rank_symbol_enhanced, 5, 50, 50)
    
    name, score, detail = enhanced_match_symbol_v2(rank_symbol_denoised, rank_templates, "rank")
    
    problematic_pairs = {
        '8': ['5', '6', '3'],
        '5': ['8', '6'],
        '10': ['6', '8'],
        '3': ['8', '6', '5'],
        '6': ['8', '5', '3']
    }
    
    if name in problematic_pairs and score < 0.65:
        print(f"  [Validación extra para '{name}' con score {score:.3f}]")
        
        alternative_versions = []
        alternative_versions.append(('original_enhanced', rank_symbol_denoised))
        
        _, otsu = cv2.threshold(rank_symbol_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        alternative_versions.append(('otsu', otsu))
        
        adaptive_gauss = cv2.adaptiveThreshold(
            rank_symbol_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 3
        )
        alternative_versions.append(('adaptive_gauss', adaptive_gauss))
        
        kernel_thick = np.ones((2, 2), np.uint8)
        thickened = cv2.erode(rank_symbol_denoised, kernel_thick, iterations=1)
        alternative_versions.append(('thickened', thickened))
        
        thinned = cv2.dilate(rank_symbol_denoised, kernel_thick, iterations=1)
        alternative_versions.append(('thinned', thinned))
        
        best_candidates = {}
        
        for version_name, version_img in alternative_versions:
            for scale in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]:
                nh, nw = int(h * scale), int(w * scale)
                if nh <= 0 or nw <= 0:
                    continue
                
                scaled = cv2.resize(version_img, (nw, nh))
                test_name, test_score, _ = enhanced_match_symbol_v2(scaled, rank_templates, "rank")
                
                if test_name not in best_candidates or test_score > best_candidates[test_name]['score']:
                    best_candidates[test_name] = {
                        'score': test_score,
                        'version': version_name,
                        'scale': scale
                    }
        
        if best_candidates:
            sorted_candidates = sorted(best_candidates.items(), key=lambda x: x[1]['score'], reverse=True)
            
            print(f"  Top candidatos:")
            for rank_name, info in sorted_candidates[:3]:
                print(f"    {rank_name}: {info['score']:.3f} ({info['version']}, scale={info['scale']:.2f})")
            
            top_rank, top_info = sorted_candidates[0]
            second_rank, second_info = sorted_candidates[1] if len(sorted_candidates) > 1 else (None, {'score': 0})
            
            score_diff = top_info['score'] - second_info['score']
            
            # Validación para 8 vs 5
            if name == '8' and '5' in [r for r, _ in sorted_candidates[:2]]:
                _, binary = cv2.threshold(rank_symbol_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                inverted = cv2.bitwise_not(binary)
                num_labels, labels = cv2.connectedComponents(inverted)
                
                if num_labels >= 3 and top_rank == '8':
                    print(f"  → Confirmado '8' por componentes cerrados ({num_labels})")
                    name, score = '8', max(top_info['score'], 0.70)
                elif num_labels <= 2 and top_rank == '5':
                    print(f"  → Confirmado '5' por componentes cerrados ({num_labels})")
                    name, score = '5', max(top_info['score'], 0.70)
                elif score_diff > 0.15:
                    name, score = top_rank, top_info['score']
            
            # Validación para 10 vs 6
            elif name == '10' and '6' in [r for r, _ in sorted_candidates[:2]]:
                _, binary = cv2.threshold(rank_symbol_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                horizontal_projection = np.sum(binary, axis=0)
                threshold_valley = 0.2 * horizontal_projection.max()
                valleys = horizontal_projection < threshold_valley
                
                transitions = 0
                for i in range(1, len(valleys)):
                    if valleys[i] != valleys[i-1]:
                        transitions += 1
                
                if transitions >= 4 and top_rank == '10':
                    print(f"  → Confirmado '10' por separación de dígitos ({transitions} transiciones)")
                    name, score = '10', max(top_info['score'], 0.70)
                elif transitions <= 3 and top_rank == '6':
                    print(f"  → Confirmado '6' por dígito único ({transitions} transiciones)")
                    name, score = '6', max(top_info['score'], 0.70)
                elif score_diff > 0.15:
                    name, score = top_rank, top_info['score']
            
            # Validación para 3 vs 6 vs 8
            elif name == '3' and any(r in ['6', '8'] for r, _ in sorted_candidates[:2]):
                _, binary = cv2.threshold(rank_symbol_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                h_mid = h // 2
                w_mid = w // 2
                
                top_left = np.sum(binary[:h_mid, :w_mid])
                top_right = np.sum(binary[:h_mid, w_mid:])
                bottom_left = np.sum(binary[h_mid:, :w_mid])
                bottom_right = np.sum(binary[h_mid:, w_mid:])
                
                right_ratio = (top_right + bottom_right) / (top_left + bottom_left + 1e-6)
                
                if right_ratio > 1.2 and top_rank == '3':
                    print(f"  → Confirmado '3' por distribución derecha ({right_ratio:.2f})")
                    name, score = '3', max(top_info['score'], 0.70)
                elif right_ratio <= 1.2 and top_rank in ['6', '8']:
                    print(f"  → Confirmado '{top_rank}' por distribución ({right_ratio:.2f})")
                    name, score = top_rank, max(top_info['score'], 0.70)
                elif score_diff > 0.15:
                    name, score = top_rank, top_info['score']
            
            elif score_diff > 0.20:
                name, score = top_rank, top_info['score']
            elif top_info['score'] > 0.70:
                name, score = top_rank, top_info['score']
    
    elif score < 0.50:
        best_name, best_score = name, score
        
        for scale in [0.85, 0.90, 0.95, 1.0, 1.05, 1.10]:
            nh, nw = int(h * scale), int(w * scale)
            if nh <= 0 or nw <= 0:
                continue
            
            scaled = cv2.resize(rank_symbol_denoised, (nw, nh))
            n_name, n_score, _ = enhanced_match_symbol_v2(scaled, rank_templates, "rank")
            
            if n_score > best_score:
                best_score = n_score
                best_name = n_name
        
        return best_name, best_score
    
    return name, score