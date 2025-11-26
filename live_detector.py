"""
Detector de cartas en vivo con soporte multi-carta
Versión con interfaz visual limpia y elegante
"""

import cv2
import numpy as np
import time
import os
from collections import deque, Counter

from modules.card_detection import (
    find_card_contour_from_binary,
    find_all_card_contours_from_binary
)
from modules.image_processing import four_point_transform
from modules.symbol_extraction import (
    extract_top_left_corner,
    extract_symbols_from_corner
)
from modules.rank_classifier import enhanced_rank_classification
from modules.suit_classifier import classify_suit_v7

class LiveCardDetector:
    def __init__(self, rank_templates, suit_templates, suit_color_prototypes, camera_source=None):
        self.rank_templates = rank_templates
        self.suit_templates = suit_templates
        self.suit_color_prototypes = suit_color_prototypes

        if camera_source is None:
            camera_source = 0
        print(f"Intentando conectar a cámara: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)
        if not self.cap.isOpened():
            raise Exception(f"No se pudo abrir la cámara: {camera_source}\nVerifica DroidCam / URL.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.min_card_area = 25000
        self.paused = False
        self.last_frame = None

        # Historial por "celda" (posición) para estabilizar cada carta
        self.per_card_histories = {}
        self.history_len = 6
        self.stable_threshold = 3

        self.last_warps = {}
        self.detection_cooldown = 0.8
        self.last_announce_time = 0
        self.last_announced_cards = set()
        
        # Paleta de colores minimalista
        self.colors = {
            'primary': (0, 180, 255),      # Azul cielo
            'success': (0, 200, 100),      # Verde esmeralda
            'warning': (255, 180, 0),      # Ámbar
            'text': (240, 240, 240),       # Blanco suave
            'bg': (30, 30, 35),            # Gris oscuro
            'hearts': (220, 60, 100),      # Rojo corazones
            'diamonds': (255, 100, 50),    # Naranja diamantes
            'clubs': (100, 200, 120),      # Verde tréboles
            'spades': (100, 120, 200)      # Azul picas
        }

    def _cell_key(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx // 40, cy // 40)

    def _update_history(self, key, rank, suit):
        if key is None:
            return
        dq = self.per_card_histories.get(key)
        if dq is None:
            dq = deque(maxlen=self.history_len)
            self.per_card_histories[key] = dq
        dq.append((rank, suit))

    def _stable_vote(self, key):
        dq = self.per_card_histories.get(key)
        if dq is None or len(dq) < self.stable_threshold:
            return None, None
        ranks = [r for r,_ in dq]
        suits = [s for _,s in dq]
        r_cnt = Counter(ranks).most_common(1)[0]
        s_cnt = Counter(suits).most_common(1)[0]
        if r_cnt[1] >= self.stable_threshold and s_cnt[1] >= self.stable_threshold:
            return r_cnt[0], s_cnt[0]
        return None, None

    def _get_suit_color(self, suit):
        """Retorna el color según el palo de la carta"""
        suit_colors = {
            'Corazones': self.colors['hearts'],
            'Diamantes': self.colors['diamonds'],
            'Treboles': self.colors['clubs'],
            'Picas': self.colors['spades']
        }
        return suit_colors.get(suit, self.colors['text'])

    def process_frame_multi(self, frame):
        """Procesa el frame y detecta múltiples cartas"""
        annotated = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        candidates = find_all_card_contours_from_binary(binary, min_area=self.min_card_area)
        detections = []

        for approx, area in candidates:
            key = self._cell_key(approx)
            warped = four_point_transform(frame_rgb, approx, width=300, height=420)
            corner = extract_top_left_corner(warped)
            _, symbols = extract_symbols_from_corner(corner)

            if len(symbols) < 2:
                # Contorno punteado para cartas no reconocidas
                pts = approx.reshape(-1, 2)
                for i in range(len(pts)):
                    if i % 8 < 4:
                        cv2.line(annotated, tuple(pts[i]), tuple(pts[(i+1)%len(pts)]), 
                                self.colors['warning'], 2)
                continue

            rank_sym = symbols[0]
            suit_sym = symbols[1]

            rank_match, rank_score = enhanced_rank_classification(rank_sym, self.rank_templates)
            suit_match, suit_score, _ = classify_suit_v7(
                suit_sym, corner, self.suit_templates, self.suit_color_prototypes
            )

            if rank_score < 0.25 or suit_score < 0.25:
                pts = approx.reshape(-1, 2)
                for i in range(len(pts)):
                    if i % 8 < 4:
                        cv2.line(annotated, tuple(pts[i]), tuple(pts[(i+1)%len(pts)]), 
                                self.colors['warning'], 2)
                continue

            self._update_history(key, rank_match, suit_match)
            stable_rank, stable_suit = self._stable_vote(key)

            stable = (stable_rank is not None and stable_suit is not None)
            final_rank = stable_rank if stable else rank_match
            final_suit = stable_suit if stable else suit_match

            # Contorno limpio
            x, y, wc, hc = cv2.boundingRect(approx)
            
            if stable:
                # Contorno verde sólido para cartas estables
                cv2.drawContours(annotated, [approx], -1, self.colors['success'], 3)
            else:
                # Contorno azul para cartas en proceso
                cv2.drawContours(annotated, [approx], -1, self.colors['primary'], 2)

            # Etiqueta simple y elegante
            suit_color = self._get_suit_color(final_suit)
            label = f"{final_rank} {final_suit}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Fondo semi-transparente
            label_h = text_h + 20
            label_y = max(5, y - label_h - 5)
            
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x, label_y), (x + text_w + 20, label_y + label_h),
                         self.colors['bg'], -1)
            cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)
            
            # Borde de color según el palo
            cv2.rectangle(annotated, (x, label_y), (x + text_w + 20, label_y + label_h),
                         suit_color, 2)
            
            # Texto de la carta
            text_x = x + 10
            text_y = label_y + text_h + 8
            cv2.putText(annotated, label, (text_x, text_y),
                       font, font_scale, self.colors['text'], thickness, cv2.LINE_AA)

            if key is not None:
                self.last_warps[key] = warped

            detections.append({
                "key": key,
                "rank": final_rank,
                "suit": final_suit,
                "stable": stable,
                "scores": (rank_score, suit_score),
                "contour": approx
            })

        return annotated, detections

    def draw_info_panel_mini(self, frame, detections, fps):
        """Panel de información minimalista"""
        h, w = frame.shape[:2]
        
        # === BARRA SUPERIOR SIMPLE ===
        bar_h = 45
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), self.colors['bg'], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Línea de acento
        cv2.line(frame, (0, bar_h-2), (w, bar_h-2), self.colors['primary'], 2)
        
        # Título
        cv2.putText(frame, "CARD DETECTOR", (15, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2, cv2.LINE_AA)
        
        # Contador de cartas
        count_text = f"{len(detections)} {'CARTA' if len(detections) == 1 else 'CARTAS'}"
        cv2.putText(frame, count_text, (w - 200, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['primary'], 2, cv2.LINE_AA)
        
        # FPS discreto
        fps_text = f"{fps:.0f} FPS"
        fps_color = self.colors['success'] if fps > 20 else self.colors['warning']
        cv2.putText(frame, fps_text, (w - 90, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)

        # === BARRA INFERIOR CON CONTROLES ===
        footer_h = 35
        footer_y = h - footer_h
        
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, footer_y), (w, h), self.colors['bg'], -1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)
        
        cv2.line(frame, (0, footer_y), (w, footer_y), self.colors['primary'], 2)
        
        # Controles simplificados
        controls_text = "Q: Salir  |  C: Capturar  |  R: Reset  |  SPACE: Pausa  |  A/Z: Area"
        cv2.putText(frame, controls_text, (20, footer_y + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1, cv2.LINE_AA)

    def run(self):
        """Ejecuta el detector con la interfaz limpia"""
        print("\n" + "="*60)
        print("  CARD DETECTOR - Modo Multi-Carta")
        print("="*60)
        print("\n  Q: Salir | C: Capturar | R: Reset | SPACE: Pausa | A/Z: Área")
        print(f"\n  Configuración: Área mínima = {self.min_card_area}")
        print("="*60 + "\n")

        frame_count = 0
        fps_time = time.time()
        fps = 0

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame inválido, revisa la cámara...")
                    time.sleep(0.5)
                    continue
                self.last_frame = frame.copy()

                annotated, detections = self.process_frame_multi(frame)
                frame_count += 1
                if frame_count % 30 == 0:
                    now = time.time()
                    fps = 30 / (now - fps_time)
                    fps_time = now

                # Anunciar nuevas cartas estables
                now = time.time()
                for det in detections:
                    if det["stable"]:
                        key = det["key"]
                        if key is None:
                            continue
                        card_id = (det["rank"], det["suit"], key)
                        if (now - self.last_announce_time > self.detection_cooldown and
                            card_id not in self.last_announced_cards):
                            print(f"  ✓ {det['rank']} de {det['suit']} "
                                  f"(R:{det['scores'][0]:.2f} S:{det['scores'][1]:.2f})")
                            self.last_announced_cards.add(card_id)
                            self.last_announce_time = now

                self.draw_info_panel_mini(annotated, detections, fps)
                display = annotated
            else:
                display = self.last_frame.copy() if self.last_frame is not None else \
                         np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Overlay de pausa minimalista
                overlay = display.copy()
                h, w = display.shape[:2]
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
                
                # Símbolo de pausa simple
                pause_w = 25
                pause_h = 60
                center_x = w // 2
                center_y = h // 2
                
                cv2.rectangle(display, 
                            (center_x - 35, center_y - pause_h//2),
                            (center_x - 10, center_y + pause_h//2),
                            self.colors['primary'], -1)
                cv2.rectangle(display,
                            (center_x + 10, center_y - pause_h//2),
                            (center_x + 35, center_y + pause_h//2),
                            self.colors['primary'], -1)
                
                # Texto
                cv2.putText(display, "PAUSADO",
                           (center_x - 70, center_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['text'], 2, cv2.LINE_AA)

            cv2.imshow("Card Detector", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                print("\n  Saliendo...\n")
                break
            elif key in (ord(' '),):
                self.paused = not self.paused
                status = "Pausado" if self.paused else "Reanudado"
                print(f"  {status}")
            elif key in (ord('r'), ord('R')):
                self.per_card_histories.clear()
                self.last_announced_cards.clear()
                print("  Historial reiniciado")
            elif key in (ord('a'), ord('A')):
                self.min_card_area += 5000
                print(f"  Área mínima: {self.min_card_area} (+)")
            elif key in (ord('z'), ord('Z')):
                self.min_card_area = max(8000, self.min_card_area - 5000)
                print(f"  Área mínima: {self.min_card_area} (-)")
            elif key in (ord('c'), ord('C')):
                stable_warps = [(k, self.last_warps[k]) for k in self.last_warps 
                               if self._stable_vote(k)[0]]
                if not stable_warps:
                    print("  No hay cartas estables para capturar")
                else:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    os.makedirs("captures", exist_ok=True)
                    print(f"\n  📸 Capturando {len(stable_warps)} carta(s)...")
                    for idx, (k, warp) in enumerate(stable_warps, start=1):
                        rank, suit = self._stable_vote(k)
                        fname = f"captures/card_{rank}_{suit}_{ts}_{idx}.jpg"
                        cv2.imwrite(fname, cv2.cvtColor(warp, cv2.COLOR_RGB2BGR))
                        print(f"    ✓ {fname}")
                    print(f"  Captura completada\n")

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("  Detector finalizado")
        print("="*60 + "\n")