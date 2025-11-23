import cv2
import numpy as np
import time
import os
from collections import deque, Counter

# Importar funciones del main
from main import (
    find_card_contour_from_binary,
    four_point_transform,
    extract_top_left_corner,
    extract_symbols_from_corner,
    enhanced_rank_classification,
    classify_suit_v7
)

class LiveCardDetector:
    """
    Detector de cartas en tiempo real usando cámara IP (DroidCam)
    """
    def __init__(self, rank_templates, suit_templates, suit_color_prototypes, camera_source=None):
        """
        Args:
            rank_templates: Diccionario con templates de números
            suit_templates: Diccionario con templates de palos
            suit_color_prototypes: Prototipos de color para palos
            camera_source: URL de DroidCam o ID de cámara local
                          Ejemplo: "http://192.168.1.100:4747/video"
                          o 0 para cámara local
        """
        self.rank_templates = rank_templates
        self.suit_templates = suit_templates
        self.suit_color_prototypes = suit_color_prototypes
        
        # Configurar fuente de video
        if camera_source is None:
            camera_source = 0  # Cámara por defecto
        
        print(f"Intentando conectar a cámara: {camera_source}")
        self.cap = cv2.VideoCapture(camera_source)
        
        if not self.cap.isOpened():
            raise Exception(f"No se pudo abrir la cámara: {camera_source}\n"
                          "Verifica que DroidCam esté corriendo y la URL sea correcta.")
        
        # Configurar resolución (DroidCam soporta hasta 1920x1080)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Obtener resolución real
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolución de cámara: {actual_width}x{actual_height}")
        
        # Variables de estado para estabilización
        self.last_detection = None
        self.detection_history = deque(maxlen=5)  # Últimas 5 detecciones
        self.last_detection_time = 0
        
        # Configuración mejorada para detección en vivo
        self.detection_cooldown = 1.0  # Reducir cooldown
        self.min_card_area = 25000  # Área mínima más pequeña para permitir cartas más lejanas
        
        # Buffer de frames para mejor estabilización
        self.frame_buffer = deque(maxlen=10)
        
        # Estado de pausa
        self.paused = False
        self.last_frame = None
        
    def stabilize_detection(self, rank, suit):
        """
        Estabiliza la detección usando historial para evitar falsos positivos
        Solo reporta una detección si aparece al menos 3 veces en las últimas 5 frames
        """
        self.detection_history.append((rank, suit))
        
        if len(self.detection_history) < 3:
            return None, None
        
        # Contar ocurrencias
        ranks = [d[0] for d in self.detection_history]
        suits = [d[1] for d in self.detection_history]
        
        rank_counter = Counter(ranks)
        suit_counter = Counter(suits)
        
        most_common_rank = rank_counter.most_common(1)[0][0]
        most_common_suit = suit_counter.most_common(1)[0][0]
        
        # Solo retornar si hay consenso (al menos 3/5)
        if rank_counter[most_common_rank] >= 3 and suit_counter[most_common_suit] >= 3:
            return most_common_rank, most_common_suit
        
        return None, None
    
    def process_frame(self, frame):
        """
        Procesa un frame - VERSIÓN OPTIMIZADA sin delay
        """
        try:
            # Convertir BGR a RGB directamente (sin denoising que causa delay)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detectar carta con procesamiento mínimo
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            
            # Umbralización simple y rápida
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            card_contour, area = find_card_contour_from_binary(binary, min_area=self.min_card_area)
            
            if card_contour is None:
                return frame, None, None, None
            
            # Transformar perspectiva
            warped = four_point_transform(frame_rgb, card_contour, width=300, height=420)
            corner = extract_top_left_corner(warped)
            thresh_corner, symbols = extract_symbols_from_corner(corner)
            
            if len(symbols) < 2:
                cv2.drawContours(frame, [card_contour], -1, (0, 255, 255), 3)
                return frame, None, None, None
            
            # Clasificar rank y suit
            potential_rank = symbols[0]
            potential_suit = symbols[1]
            
            rank_match, rank_score = enhanced_rank_classification(potential_rank, self.rank_templates)
            suit_match, suit_score, _ = classify_suit_v7(
                potential_suit, corner, self.suit_templates, self.suit_color_prototypes
            )
            
            # Umbrales para cámara en vivo
            if rank_score < 0.25 or suit_score < 0.25:
                cv2.drawContours(frame, [card_contour], -1, (0, 255, 255), 3)
                return frame, None, None, None
            
            # Estabilizar detección
            stable_rank, stable_suit = self.stabilize_detection(rank_match, suit_match)
            
            # Dibujar contorno
            if stable_rank and stable_suit:
                cv2.drawContours(frame, [card_contour], -1, (0, 255, 0), 3)
            else:
                cv2.drawContours(frame, [card_contour], -1, (255, 255, 0), 3)
            
            return frame, stable_rank, stable_suit, (rank_score, suit_score)
            
        except Exception as e:
            print(f"Error procesando frame: {e}")
            return frame, None, None, None
    
    def draw_info_panel(self, frame, rank, suit, scores, fps):
        """
        Dibuja panel de información en el frame
        """
        h, w = frame.shape[:2]
        
        # Panel semi-transparente en la parte superior
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Estado
        status_color = (0, 255, 0) if rank and suit else (0, 0, 255)
        status_text = "DETECTANDO..." if rank and suit else "Buscando carta..."
        cv2.putText(frame, status_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Detección actual
        if rank and suit and scores:
            text = f"{rank} de {suit}"
            cv2.putText(frame, text, (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Confianza: R={scores[0]:.2f} S={scores[1]:.2f}",
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instrucciones en la parte inferior
        instructions = [
            "Q: Salir | C: Capturar | R: Reiniciar | ESPACIO: Pausa/Reanudar",
            "A: Ajustar umbral area (+) | Z: Ajustar umbral area (-)"
        ]
        
        y_offset = h - 60
        for i, inst in enumerate(instructions):
            cv2.putText(frame, inst, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """
        Loop principal de detección en vivo
        """
        print("\n" + "="*60)
        print("MODO DETECCIÓN EN VIVO - DroidCam")
        print("="*60)
        print("\nControles:")
        print("  Q - Salir")
        print("  C - Capturar y guardar carta detectada")
        print("  R - Reiniciar historial de detección")
        print("  ESPACIO - Pausar/Reanudar")
        print("  A - Aumentar umbral mínimo de área")
        print("  Z - Disminuir umbral mínimo de área")
        print(f"\nÁrea mínima actual: {self.min_card_area}")
        print("\n¡Apunta la cámara a una carta!\n")
        
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: No se pudo leer el frame de la cámara")
                    print("Verifica que DroidCam siga corriendo")
                    time.sleep(1)
                    continue
                
                self.last_frame = frame.copy()
                
                # Procesar frame
                processed_frame, rank, suit, scores = self.process_frame(frame)
                
                # Calcular FPS cada 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - fps_time)
                    fps_time = current_time
                
                # Dibujar información
                self.draw_info_panel(processed_frame, rank, suit, scores, fps)
                
                # Anunciar detección nueva
                if rank and suit and scores:
                    current_time = time.time()
                    
                    # Verificar cooldown
                    if current_time - self.last_detection_time > self.detection_cooldown:
                        if (rank, suit) != self.last_detection:
                            print(f"\n✓ CARTA DETECTADA: {rank} de {suit}")
                            print(f"  Confianza: Rank={scores[0]:.2f}, Suit={scores[1]:.2f}")
                            self.last_detection = (rank, suit)
                            self.last_detection_time = current_time
                
                display_frame = processed_frame
            else:
                # Modo pausado
                display_frame = self.last_frame.copy() if self.last_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(display_frame, "PAUSADO - Presiona ESPACIO para continuar", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Mostrar frame
            cv2.imshow('Deteccion de Cartas - DroidCam', display_frame)
            
            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("\nSaliendo...")
                break
            elif key == ord('c') or key == ord('C'):
                if rank and suit:
                    # Guardar captura
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"captured_{rank}_{suit}_{timestamp}.jpg"
                    filepath = os.path.join("content", filename)
                    cv2.imwrite(filepath, self.last_frame)
                    print(f"✓ Captura guardada: {filename}")
                else:
                    print("⚠ No hay carta detectada para capturar")
            elif key == ord('r') or key == ord('R'):
                # Reiniciar historial
                self.detection_history.clear()
                self.last_detection = None
                print("✓ Historial de detección reiniciado")
            elif key == ord(' '):
                # Pausar/Reanudar
                self.paused = not self.paused
                print("⏸ PAUSADO" if self.paused else "▶ REANUDADO")
            elif key == ord('a') or key == ord('A'):
                # Aumentar umbral de área
                self.min_card_area += 5000
                print(f"✓ Área mínima: {self.min_card_area}")
            elif key == ord('z') or key == ord('Z'):
                # Disminuir umbral de área
                self.min_card_area = max(10000, self.min_card_area - 5000)
                print(f"✓ Área mínima: {self.min_card_area}")
        
        # Limpiar
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Detección en vivo finalizada")