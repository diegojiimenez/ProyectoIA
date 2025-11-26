
"""
Sistema de Reconocimiento de Cartas - Main
Versión modular y optimizada
"""

import os
import argparse
import csv
import cv2

from modules.template_builder import build_templates_from_directory
from modules.card_detection import process_card_image
from modules.rank_classifier import enhanced_rank_classification
from modules.suit_classifier import classify_suit_v7
from modules.visualization import show_img, wait_enter

def interactive_process_and_classify(image_path, rank_templates, suit_templates, 
                                     suit_color_prototypes, interactive=True):
    """
    Procesa y clasifica una carta interactivamente
    """
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
    show_img(result["original"], "Original", figsize=(5, 4), mode=mode)
    show_img(result["binary"], "Binaria", cmap='gray', figsize=(5, 4), mode=mode)
    
    debug = result["original"].copy()
    cv2.drawContours(debug, [result["card_contour"]], -1, (0, 255, 0), 3)
    show_img(debug, "Contorno", figsize=(5, 4), mode=mode)
    show_img(result["warped"], "Warped", figsize=(4, 6), mode=mode)
    show_img(result["corner"], "Esquina", figsize=(3, 4), mode=mode)
    show_img(result["thresh_corner"], "Esquina thresh", cmap='gray', figsize=(3, 4), mode=mode)
    
    for i, sym in enumerate(result["symbols"]):
        show_img(sym, f"Símbolo {i}", cmap='gray', figsize=(2, 2), mode=mode)
    
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
    text_color = (255, 0, 0)
    cv2.putText(warped_with_text, f"Rank: {final_rank}", (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    cv2.putText(warped_with_text, f"Palo: {final_suit}", (10, 375),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    cv2.putText(warped_with_text, f"Scores: {rank_score:.2f}, {suit_score:.2f}",
                (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    show_img(warped_with_text, f"Reconocido: {final_rank} de {final_suit}", figsize=(4, 6), mode=mode)
    
    # Debug info
    if potential_suit is not None:
        print_debug_info(suit_debug)
    
    wait_enter(interactive, "Enter para siguiente carta...")
    
    return {
        "filename": os.path.basename(image_path),
        "detected_rank": final_rank,
        "detected_suit": final_suit,
        "rank_score": rank_score,
        "suit_score": suit_score,
        "status": "ok"
    }

def print_debug_info(suit_debug):
    """Imprime información de debug del palo"""
    shape = suit_debug.get("shape", {})
    heartf = suit_debug.get("heart_features", {})
    diamondf = suit_debug.get("diamond_features", {})
    cloverf = suit_debug.get("clover_features", {})
    spadef = suit_debug.get("spade_features", {})
    color_stats = suit_debug.get("color_stats", {})
    proto_sim = suit_debug.get("proto_similarity", {})
    
    print("DEBUG SUIT → Template:", suit_debug.get("template_name"),
          f"(score {suit_debug.get('template_score'):.3f}) | Grupo:", suit_debug.get("color_group"))
    print(f"  Color: red_conf={color_stats.get('red_confidence',0):.3f} "
          f"hsv_pct={color_stats.get('red_pct_hsv',0):.3f} "
          f"lab_a={color_stats.get('lab_a_mean',0):.1f} "
          f"rg_diff={color_stats.get('rg_diff_mean',0):.3f} "
          f"fallback={color_stats.get('fallback_color_red',False)}")
    print(f"  Forma: vertices={shape.get('vertices')} defects={shape.get('defects')} "
          f"circ={shape.get('circularity',0):.3f} sol={shape.get('solidity',0):.3f} "
          f"aspect h/w={shape.get('aspect_ratio',0):.3f} "
          f"deg_fix={suit_debug.get('degenerate_fix_applied',False)}")
    print(f"  Corazón: lobes_score={heartf.get('heart_lobes_score',0):.3f}")
    print(f"  Diamante: diamond_score={diamondf.get('diamond_feature_score',0):.3f} "
          f"approx_vertices={diamondf.get('approx_vertices')} "
          f"radial_uni={diamondf.get('radial_uniformity',0):.3f} "
          f"angle_uni={diamondf.get('angle_uniformity',0):.3f} "
          f"orient_score={diamondf.get('orientation_score',0):.3f}")
    print(f"  Trébol: clover_score={cloverf.get('clover_feature_score',0):.3f} "
          f"lobe_count={cloverf.get('lobe_count')} "
          f"base_narrow={cloverf.get('base_narrowness',0):.3f} "
          f"complexity={cloverf.get('top_bottom_complexity',0):.3f} "
          f"defects_score={cloverf.get('convexity_defects_score',0):.3f}")
    print(f"  Pica: spade_score={spadef.get('spade_feature_score',0):.3f} "
          f"peak_sharpness={spadef.get('peak_sharpness',0):.3f} "
          f"vertical_aspect={spadef.get('vertical_aspect',0):.3f} "
          f"lateral_symmetry={spadef.get('lateral_symmetry',0):.3f} "
          f"base_narrowness={spadef.get('base_narrowness',0):.3f} "
          f"top_concentration={spadef.get('top_concentration',0):.3f}")
    
    if proto_sim:
        print("  Similitud prototipos:", proto_sim)
    if "override" in suit_debug:
        print("  OVERRIDE:", suit_debug["override"])
    print("  Scores por palo:")
    for k, v in suit_debug.get("per_suit_scores", {}).items():
        print(f"    {k}: base={v['base']:.3f} heur={v['heur']:.3f} final={v['final']:.3f}")

def run_live_mode(camera_url=None):
    """Ejecuta el modo de detección en vivo"""
    from live_detector import LiveCardDetector
    
    print("Construyendo templates...")
    rank_templates, suit_templates, suit_color_prototypes = build_templates_from_directory("template/")
    print(f"Ranks disponibles: {list(rank_templates.keys())}")
    print(f"Suits disponibles: {list(suit_templates.keys())}")
    
    try:
        detector = LiveCardDetector(
            rank_templates,
            suit_templates,
            suit_color_prototypes,
            camera_source=camera_url
        )
        detector.run()
    except Exception as e:
        print(f"\nError al iniciar detector: {e}")
        print("\nPosibles soluciones:")
        print("1. Verifica que DroidCam esté corriendo en tu teléfono")
        print("2. Verifica que estés en la misma red WiFi")
        print("3. Verifica la URL (debe terminar en /video)")
        print("4. Ejemplo: http://192.168.1.100:4747/video")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Sistema de Reconocimiento de Cartas (v2.0 - Modular)"
    )
    parser.add_argument("--content-dir", default="content/", 
                       help="Directorio con imágenes de cartas")
    parser.add_argument("--template-dir", default="template/", 
                       help="Directorio con templates")
    parser.add_argument("--no-interactive", action="store_true", 
                       help="Modo automático (sin Enter)")
    parser.add_argument("--save-csv", action="store_true", 
                       help="Guardar resultados en CSV")
    parser.add_argument("--live", action="store_true", 
                       help="Modo detección en vivo con DroidCam")
    parser.add_argument("--camera-url", type=str, default=None,
                       help="URL de DroidCam (ej: http://192.168.1.100:4747/video)")
    
    args = parser.parse_args()
    
    # Modo detección en vivo
    if args.live:
        run_live_mode(args.camera_url)
        return
    
    # Modo procesamiento de imágenes
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
    
    image_files = [f for f in os.listdir(content_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"No hay imágenes en {content_dir}")
        return
    
    print(f"Procesando {len(image_files)} cartas...\n")
    all_results = []
    
    for filename in sorted(image_files):
        image_path = os.path.join(content_dir, filename)
        res = interactive_process_and_classify(
            image_path, rank_templates, suit_templates, 
            suit_color_prototypes, interactive=interactive
        )
        all_results.append(res)
    
    print("\n=== RESUMEN FINAL ===")
    total_ok = sum(1 for r in all_results if r["status"] == "ok")
    print(f"Cartas procesadas correctamente: {total_ok}/{len(all_results)}")
    for r in all_results:
        print(f"{r['filename']}: {r['detected_rank']} de {r['detected_suit']} "
              f"(Scores: {r['rank_score']:.2f}, {r['suit_score']:.2f})")
    
    if args.save_csv:
        with open('card_recognition_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['filename', 'detected_rank', 'detected_suit', 
                         'rank_score', 'suit_score', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print("Resultados guardados en card_recognition_results.csv")

if __name__ == "__main__":
    main()