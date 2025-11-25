import cv2
import numpy as np
import time
import os
from collections import deque, Counter

from main import (
    find_card_contour_from_binary,
    four_point_transform,
    extract_top_left_corner,
    extract_symbols_from_corner,
    enhanced_rank_classification,
    classify_suit_v7,
    find_all_card_contours_from_binary  # NUEVO
)

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

        # Historial por “celda” (posición) para estabilizar cada carta
        # key -> deque([(rank,suit)])
        self.per_card_histories = {}
        self.history_len = 6
        self.stable_threshold = 3  # mínimo apariciones para considerar estable

        # Guardar último warp por key para captura individual si quieres
        self.last_warps = {}

        self.detection_cooldown = 0.8
        self.last_announce_time = 0
        self.last_announced_cards = set()  # {(rank,suit,key)}

    def _cell_key(self, contour):
        """
        Genera una clave de celda para agrupar detecciones de la misma carta
        basado en el centro aproximado del contorno.
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Cuantizar para evitar variaciones pequeñas
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

    def process_frame_multi(self, frame):
        """
        Procesa el frame y detecta múltiples cartas.
        Devuelve:
          frame_annotated,
          detecciones = [ { 'key':key, 'rank':r, 'suit':s, 'stable':bool,
                            'scores':(rank_score,suit_score), 'contour':approx } ]
        """
        annotated = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Obtener TODOS los contornos candidatos
        candidates = find_all_card_contours_from_binary(binary, min_area=self.min_card_area)
        detections = []

        for approx, area in candidates:
            # Obtener key para estabilización
            key = self._cell_key(approx)
            # Warp
            warped = four_point_transform(frame_rgb, approx, width=300, height=420)
            corner = extract_top_left_corner(warped)
            _, symbols = extract_symbols_from_corner(corner)

            if len(symbols) < 2:
                # Dibujar contorno en amarillo (insuficientes símbolos)
                cv2.drawContours(annotated, [approx], -1, (0, 255, 255), 2)
                continue

            rank_sym = symbols[0]
            suit_sym = symbols[1]

            rank_match, rank_score = enhanced_rank_classification(rank_sym, self.rank_templates)
            suit_match, suit_score, _ = classify_suit_v7(
                suit_sym, corner, self.suit_templates, self.suit_color_prototypes
            )

            # Filtro mínimo
            if rank_score < 0.25 or suit_score < 0.25:
                cv2.drawContours(annotated, [approx], -1, (0, 255, 255), 2)
                continue

            # Actualizar historial
            self._update_history(key, rank_match, suit_match)
            stable_rank, stable_suit = self._stable_vote(key)

            stable = (stable_rank is not None and stable_suit is not None)
            final_rank = stable_rank if stable else rank_match
            final_suit = stable_suit if stable else suit_match

            # Color del contorno
            color = (0, 255, 0) if stable else (255, 255, 0)
            cv2.drawContours(annotated, [approx], -1, color, 3)

            # Texto
            x,y,wc,hc = cv2.boundingRect(approx)
            label = f"{final_rank} {final_suit}"
            cv2.rectangle(annotated, (x, y-25), (x+wc, y), (0,0,0), -1)
            cv2.putText(annotated, label, (x+5, y-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if stable else (0,255,255), 1)

            # Guardar warp para posible captura
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

    def draw_info_panel_multi(self, frame, detections, fps):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(frame, f"Cartas detectadas: {len(detections)}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Panel lateral (opcional)
        panel_w = 260
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (w - panel_w, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.25, frame, 0.75, 0, frame)

        y0 = 80
        cv2.putText(frame, "DETALLES:", (w - panel_w + 10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        for i, det in enumerate(detections[:12]):  # limitar listado
            txt = f"{i+1}. {det['rank']} {det['suit']} {'(S)' if det['stable'] else ''}"
            cv2.putText(frame, txt, (w - panel_w + 10, y0 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0) if det["stable"] else (0,255,255), 1)

        # Instrucciones abajo
        instructions = [
            "Q: Salir | C: Capturar todas | R: Reiniciar historial",
            "ESPACIO: Pausa | A/Z: Ajustar área mínima"
        ]
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (10, h - 40 + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    def run(self):
        print("\n=== MODO MULTI-CARTA EN VIVO ===")
        print("Controles: Q=Salir  C=Capturar  R=Reset  ESPACIO=Pausa  A/Z=Area +/-")
        print(f"Área mínima inicial: {self.min_card_area}")

        frame_count = 0
        fps_time = time.time()
        fps = 0

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame inválido, revisa la cámara")
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
                            print(f"✓ Carta estable: {det['rank']} de {det['suit']} (scores R={det['scores'][0]:.2f}, S={det['scores'][1]:.2f})")
                            self.last_announced_cards.add(card_id)
                            self.last_announce_time = now

                self.draw_info_panel_multi(annotated, detections, fps)
                display = annotated
            else:
                display = self.last_frame.copy() if self.last_frame is not None else np.zeros((480,640,3), dtype=np.uint8)
                cv2.putText(display, "PAUSADO - ESPACIO para continuar",
                            (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

            cv2.imshow("Deteccion Multi-Carta", display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                print("Saliendo...")
                break
            elif key in (ord(' '),):
                self.paused = not self.paused
                print("Pausado" if self.paused else "Reanudado")
            elif key in (ord('r'), ord('R')):
                self.per_card_histories.clear()
                self.last_announced_cards.clear()
                print("Historial reiniciado.")
            elif key in (ord('a'), ord('A')):
                self.min_card_area += 5000
                print(f"Área mínima ahora: {self.min_card_area}")
            elif key in (ord('z'), ord('Z')):
                self.min_card_area = max(8000, self.min_card_area - 5000)
                print(f"Área mínima ahora: {self.min_card_area}")
            elif key in (ord('c'), ord('C')):
                # Capturar warps de cartas estables
                stable_warps = [ (k, self.last_warps[k]) for k in self.last_warps if self._stable_vote(k)[0] ]
                if not stable_warps:
                    print("No hay cartas estables para capturar.")
                else:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    os.makedirs("captures", exist_ok=True)
                    for idx,(k,warp) in enumerate(stable_warps, start=1):
                        rank, suit = self._stable_vote(k)
                        fname = f"captures/card_{rank}_{suit}_{ts}_{idx}.jpg"
                        cv2.imwrite(fname, cv2.cvtColor(warp, cv2.COLOR_RGB2BGR))
                        print(f"Capturada: {fname}")

        self.cap.release()
        cv2.destroyAllWindows()
        print("Fin detección multi-carta.")