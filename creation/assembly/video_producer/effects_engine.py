import os
import logging
import json
import random
import time
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, TextClip
from moviepy.editor import vfx, concatenate_videoclips, clips_array, CompositeAudioClip
import moviepy.video.fx.all as vfx

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EffectsEngine")

class EffectsEngine:
    """
    Motor de efectos para videos que proporciona transformaciones visuales y de audio
    """
    
    def __init__(self, config_path: str = None, temp_dir: str = "temp/effects"):
        """
        Inicializa el motor de efectos
        
        Args:
            config_path: Ruta al archivo de configuración
            temp_dir: Directorio temporal para archivos intermedios
        """
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
        # Cargar configuración
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        
        # Configuración por defecto
        self.default_transition_duration = self.config.get("default_transition_duration", 0.5)
        self.default_effect_intensity = self.config.get("default_effect_intensity", 0.5)
        
        logger.info(f"EffectsEngine inicializado con directorio temporal: {self.temp_dir}")
    
    def apply_effect(self, 
                    video_path: str, 
                    effect_name: str,
                    intensity: float = None,
                    start_time: float = 0,
                    end_time: float = None,
                    output_path: str = None,
                    **effect_params) -> str:
        """
        Aplica un efecto visual a un video
        
        Args:
            video_path: Ruta al video
            effect_name: Nombre del efecto a aplicar
            intensity: Intensidad del efecto (0-1)
            start_time: Tiempo de inicio del efecto
            end_time: Tiempo de fin del efecto
            output_path: Ruta de salida para el video
            effect_params: Parámetros adicionales específicos del efecto
            
        Returns:
            Ruta al video con efecto aplicado
        """
        try:
            # Verificar que existe el video
            if not os.path.exists(video_path):
                logger.error(f"No se encontró el video: {video_path}")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                base_name = os.path.basename(video_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(os.path.dirname(video_path), f"{name}_{effect_name}{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Establecer intensidad por defecto si no se proporciona
            if intensity is None:
                intensity = self.default_effect_intensity
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Establecer tiempo de fin si no se proporciona
            if end_time is None:
                end_time = video.duration
            
            # Aplicar efecto según el nombre
            if effect_name == "blur":
                radius = effect_params.get("radius", int(20 * intensity))
                processed_video = self._apply_blur(video, radius, start_time, end_time)
            
            elif effect_name == "brightness":
                factor = 1.0 + intensity
                processed_video = self._apply_brightness(video, factor, start_time, end_time)
            
            elif effect_name == "contrast":
                factor = 1.0 + intensity * 2
                processed_video = self._apply_contrast(video, factor, start_time, end_time)
            
            elif effect_name == "grayscale":
                processed_video = self._apply_grayscale(video, start_time, end_time)
            
            elif effect_name == "sepia":
                processed_video = self._apply_sepia(video, start_time, end_time)
            
            elif effect_name == "mirror":
                axis = effect_params.get("axis", "x")
                processed_video = self._apply_mirror(video, axis, start_time, end_time)
            
            elif effect_name == "reverse":
                processed_video = self._apply_reverse(video, start_time, end_time)
            
            elif effect_name == "slow_motion":
                factor = 1.0 / (1.0 + intensity * 3)  # 0.25x - 1x
                processed_video = self._apply_slow_motion(video, factor, start_time, end_time)
            
            elif effect_name == "fast_motion":
                factor = 1.0 + intensity * 3  # 1x - 4x
                processed_video = self._apply_fast_motion(video, factor, start_time, end_time)
            
            elif effect_name == "zoom":
                zoom_factor = 1.0 + intensity
                processed_video = self._apply_zoom(video, zoom_factor, start_time, end_time)
            
            elif effect_name == "rotate":
                angle = effect_params.get("angle", 90 * intensity)
                processed_video = self._apply_rotate(video, angle, start_time, end_time)
            
            elif effect_name == "shake":
                intensity_px = int(10 * intensity)
                processed_video = self._apply_shake(video, intensity_px, start_time, end_time)
            
            elif effect_name == "vignette":
                processed_video = self._apply_vignette(video, intensity, start_time, end_time)
            
            elif effect_name == "fade_in":
                duration = effect_params.get("duration", self.default_transition_duration)
                processed_video = self._apply_fade_in(video, duration)
            
            elif effect_name == "fade_out":
                duration = effect_params.get("duration", self.default_transition_duration)
                processed_video = self._apply_fade_out(video, duration)
            
            elif effect_name == "color_filter":
                color = effect_params.get("color", (0, 0, 255))  # Default: rojo
                processed_video = self._apply_color_filter(video, color, intensity, start_time, end_time)
            
            else:
                logger.warning(f"Efecto desconocido: {effect_name}, devolviendo video original")
                processed_video = video
            
            # Guardar video
            processed_video.write_videofile(output_path, fps=video.fps, codec="libx264", 
                                          audio_codec="aac", threads=4)
            
            logger.info(f"Efecto {effect_name} aplicado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al aplicar efecto {effect_name}: {str(e)}")
            return None
    
    def apply_transition(self, 
                        video_path1: str, 
                        video_path2: str,
                        transition_name: str = "fade",
                        duration: float = None,
                        output_path: str = None,
                        **transition_params) -> str:
        """
        Aplica una transición entre dos videos
        
        Args:
            video_path1: Ruta al primer video
            video_path2: Ruta al segundo video
            transition_name: Nombre de la transición
            duration: Duración de la transición en segundos
            output_path: Ruta de salida para el video
            transition_params: Parámetros adicionales específicos de la transición
            
        Returns:
            Ruta al video con transición
        """
        try:
            # Verificar que existen los videos
            if not os.path.exists(video_path1):
                logger.error(f"No se encontró el primer video: {video_path1}")
                return None
            
            if not os.path.exists(video_path2):
                logger.error(f"No se encontró el segundo video: {video_path2}")
                return None
            
            # Establecer duración por defecto si no se proporciona
            if duration is None:
                duration = self.default_transition_duration
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                base_name1 = os.path.basename(video_path1)
                base_name2 = os.path.basename(video_path2)
                name1, ext = os.path.splitext(base_name1)
                name2, _ = os.path.splitext(base_name2)
                output_path = os.path.join(os.path.dirname(video_path1), f"{name1}_{transition_name}_{name2}{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar videos
            video1 = VideoFileClip(video_path1)
            video2 = VideoFileClip(video_path2)
            
            # Aplicar transición según el nombre
            if transition_name == "fade":
                processed_video = self._apply_fade_transition(video1, video2, duration)
            
            elif transition_name == "wipe_left":
                processed_video = self._apply_wipe_transition(video1, video2, duration, direction="left")
            
            elif transition_name == "wipe_right":
                processed_video = self._apply_wipe_transition(video1, video2, duration, direction="right")
            
            elif transition_name == "wipe_up":
                processed_video = self._apply_wipe_transition(video1, video2, duration, direction="up")
            
            elif transition_name == "wipe_down":
                processed_video = self._apply_wipe_transition(video1, video2, duration, direction="down")
            
            elif transition_name == "zoom_in":
                processed_video = self._apply_zoom_transition(video1, video2, duration, zoom_in=True)
            
            elif transition_name == "zoom_out":
                processed_video = self._apply_zoom_transition(video1, video2, duration, zoom_in=False)
            
            elif transition_name == "rotate":
                angle = transition_params.get("angle", 90)
                processed_video = self._apply_rotate_transition(video1, video2, duration, angle)
            
            elif transition_name == "dissolve":
                processed_video = self._apply_dissolve_transition(video1, video2, duration)
            
            elif transition_name == "slide_left":
                processed_video = self._apply_slide_transition(video1, video2, duration, direction="left")
            
            elif transition_name == "slide_right":
                processed_video = self._apply_slide_transition(video1, video2, duration, direction="right")
            
            elif transition_name == "slide_up":
                processed_video = self._apply_slide_transition(video1, video2, duration, direction="up")
            
            elif transition_name == "slide_down":
                processed_video = self._apply_slide_transition(video1, video2, duration, direction="down")
            
            elif transition_name == "none":
                # Simplemente concatenar los videos sin transición
                processed_video = concatenate_videoclips([video1, video2])
            
            else:
                logger.warning(f"Transición desconocida: {transition_name}, usando fade por defecto")
                processed_video = self._apply_fade_transition(video1, video2, duration)
            
            # Guardar video
            processed_video.write_videofile(output_path, fps=max(video1.fps, video2.fps), 
                                          codec="libx264", audio_codec="aac", threads=4)
            
            logger.info(f"Transición {transition_name} aplicada exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al aplicar transición {transition_name}: {str(e)}")
            return None
    
    def apply_audio_effect(self, 
                          video_path: str, 
                          effect_name: str,
                          intensity: float = None,
                          start_time: float = 0,
                          end_time: float = None,
                          output_path: str = None,
                          **effect_params) -> str:
        """
        Aplica un efecto de audio a un video
        
        Args:
            video_path: Ruta al video
            effect_name: Nombre del efecto a aplicar
            intensity: Intensidad del efecto (0-1)
            start_time: Tiempo de inicio del efecto
            end_time: Tiempo de fin del efecto
            output_path: Ruta de salida para el video
            effect_params: Parámetros adicionales específicos del efecto
            
        Returns:
            Ruta al video con efecto de audio aplicado
        """
        try:
            # Verificar que existe el video
            if not os.path.exists(video_path):
                logger.error(f"No se encontró el video: {video_path}")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                base_name = os.path.basename(video_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(os.path.dirname(video_path), f"{name}_{effect_name}_audio{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Establecer intensidad por defecto si no se proporciona
            if intensity is None:
                intensity = self.default_effect_intensity
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Establecer tiempo de fin si no se proporciona
            if end_time is None:
                end_time = video.duration
            
            # Verificar que el video tiene audio
            if video.audio is None:
                logger.warning(f"El video no tiene audio: {video_path}")
                return video_path
            
            # Aplicar efecto según el nombre
            if effect_name == "volume":
                factor = 1.0 + intensity
                processed_video = self._apply_volume_effect(video, factor, start_time, end_time)
            
            elif effect_name == "fade_in_audio":
                duration = effect_params.get("duration", self.default_transition_duration)
                processed_video = self._apply_fade_in_audio(video, duration)
            
            elif effect_name == "fade_out_audio":
                duration = effect_params.get("duration", self.default_transition_duration)
                processed_video = self._apply_fade_out_audio(video, duration)
            
            elif effect_name == "echo":
                delay = effect_params.get("delay", 0.5)
                processed_video = self._apply_echo_effect(video, intensity, delay, start_time, end_time)
            
            elif effect_name == "mute":
                processed_video = self._apply_mute_effect(video, start_time, end_time)
            
            else:
                logger.warning(f"Efecto de audio desconocido: {effect_name}, devolviendo video original")
                processed_video = video
            
            # Guardar video
            processed_video.write_videofile(output_path, fps=video.fps, codec="libx264", 
                                          audio_codec="aac", threads=4)
            
            logger.info(f"Efecto de audio {effect_name} aplicado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al aplicar efecto de audio {effect_name}: {str(e)}")
            return None
    
    def apply_multiple_effects(self, 
                              video_path: str, 
                              effects: List[Dict[str, Any]],
                              output_path: str = None) -> str:
        """
        Aplica múltiples efectos a un video en secuencia
        
        Args:
            video_path: Ruta al video
            effects: Lista de diccionarios con efectos a aplicar
            output_path: Ruta de salida para el video
            
        Returns:
            Ruta al video con efectos aplicados
        """
        try:
            # Verificar que existe el video
            if not os.path.exists(video_path):
                logger.error(f"No se encontró el video: {video_path}")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                base_name = os.path.basename(video_path)
                name, ext = os.path.splitext(base_name)
                timestamp = int(time.time())
                output_path = os.path.join(os.path.dirname(video_path), f"{name}_multi_effects_{timestamp}{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Verificar que hay efectos
            if not effects:
                logger.warning("No se proporcionaron efectos, devolviendo video original")
                return video_path
            
            # Aplicar efectos en secuencia
            current_video_path = video_path
            
            for i, effect in enumerate(effects):
                effect_type = effect.get("type", "visual")
                effect_name = effect.get("name", "")
                intensity = effect.get("intensity", self.default_effect_intensity)
                start_time = effect.get("start_time", 0)
                end_time = effect.get("end_time", None)
                params = effect.get("params", {})
                
                # Generar ruta temporal para este efecto
                temp_output = os.path.join(self.temp_dir, f"temp_effect_{i}_{int(time.time())}.mp4")
                
                if effect_type == "visual":
                    # Aplicar efecto visual
                    result = self.apply_effect(
                        current_video_path, 
                        effect_name, 
                        intensity, 
                        start_time, 
                        end_time, 
                        temp_output, 
                        **params
                    )
                elif effect_type == "audio":
                    # Aplicar efecto de audio
                    result = self.apply_audio_effect(
                        current_video_path, 
                        effect_name, 
                        intensity, 
                        start_time, 
                        end_time, 
                        temp_output, 
                        **params
                    )
                else:
                    logger.warning(f"Tipo de efecto desconocido: {effect_type}, omitiendo")
                    continue
                
                # Verificar que se aplicó correctamente
                if result:
                    current_video_path = result
                else:
                    logger.warning(f"Error al aplicar efecto {effect_name}, continuando con el siguiente")
            
            # Renombrar el último resultado al nombre de salida final
            if current_video_path != video_path:
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(current_video_path, output_path)
                
                # Limpiar archivos temporales
                for file in os.listdir(self.temp_dir):
                    if file.startswith("temp_effect_"):
                        try:
                            os.remove(os.path.join(self.temp_dir, file))
                        except:
                            pass
                
                logger.info(f"Múltiples efectos aplicados exitosamente: {output_path}")
                return output_path
            else:
                logger.warning("No se aplicaron efectos correctamente")
                return video_path
            
        except Exception as e:
            logger.error(f"Error al aplicar múltiples efectos: {str(e)}")
            return None
    
    def create_split_screen(self, 
                           video_paths: List[str], 
                           layout: str = "horizontal",
                           output_path: str = None) -> str:
        """
        Crea una pantalla dividida con múltiples videos
        
        Args:
            video_paths: Lista de rutas a videos
            layout: Disposición de los videos (horizontal, vertical, grid)
            output_path: Ruta de salida para el video
            
        Returns:
            Ruta al video con pantalla dividida
        """
        try:
            # Verificar que hay videos
            if not video_paths:
                logger.error("No se proporcionaron videos para la pantalla dividida")
                return None
            
            # Verificar que existen los videos
            for path in video_paths:
                if not os.path.exists(path):
                    logger.error(f"No se encontró el video: {path}")
                    return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                output_path = os.path.join(os.path.dirname(video_paths[0]), f"split_screen_{timestamp}.mp4")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar videos
            video_clips = [VideoFileClip(path) for path in video_paths]
            
            # Encontrar la duración máxima
            max_duration = max(clip.duration for clip in video_clips)
            
            # Ajustar duración de todos los clips a la máxima
            for i in range(len(video_clips)):
                if video_clips[i].duration < max_duration:
                    # Crear un clip negro para rellenar
                    black_clip = ImageClip(np.zeros((video_clips[i].h, video_clips[i].w, 3), dtype=np.uint8))
                    black_clip = black_clip.set_duration(max_duration - video_clips[i].duration)
                    
                    # Concatenar con el clip original
                    video_clips[i] = concatenate_videoclips([video_clips[i], black_clip])
            
            # Crear pantalla dividida según el layout
            if layout == "horizontal":
                # Ajustar altura de todos los clips a la misma
                height = min(clip.h for clip in video_clips)
                resized_clips = [clip.resize(height=height) for clip in video_clips]
                
                # Crear pantalla dividida horizontal
                final_clip = clips_array([[clip] for clip in resized_clips])
            
            elif layout == "vertical":
                # Ajustar ancho de todos los clips al mismo
                width = min(clip.w for clip in video_clips)
                resized_clips = [clip.resize(width=width) for clip in video_clips]
                
                # Crear pantalla dividida vertical
                final_clip = clips_array([[clip] for clip in resized_clips])
            
            elif layout == "grid":
                # Determinar número de filas y columnas
                n = len(video_clips)
                cols = int(np.ceil(np.sqrt(n)))
                rows = int(np.ceil(n / cols))
                
                # Rellenar con clips negros si es necesario
                while len(video_clips) < rows * cols:
                    # Crear un clip negro del mismo tamaño que el último
                    black_clip = ImageClip(np.zeros((video_clips[-1].h, video_clips[-1].w, 3), dtype=np.uint8))
                    black_clip = black_clip.set_duration(max_duration)
                    video_clips.append(black_clip)
                
                # Reorganizar clips en una matriz
                grid = []
                for i in range(rows):
                    row = []
                    for j in range(cols):
                        idx = i * cols + j
                        if idx < len(video_clips):
                            row.append(video_clips[idx])
                    grid.append(row)
                
                # Crear pantalla dividida en cuadrícula
                final_clip = clips_array(grid)
            
            else:
                logger.warning(f"Layout desconocido: {layout}, usando horizontal por defecto")
                # Ajustar altura de todos los clips a la misma
                height = min(clip.h for clip in video_clips)
                resized_clips = [clip.resize(height=height) for clip in video_clips]
                
                # Crear pantalla dividida horizontal
                final_clip = clips_array([[clip] for clip in resized_clips])
            
            # Guardar video
            final_clip.write_videofile(output_path, fps=video_clips[0].fps, 
                                     codec="libx264", audio_codec="aac", threads=4)
            
            logger.info(f"Pantalla dividida creada exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al crear pantalla dividida: {str(e)}")
            return None
    
    # Métodos privados para implementar efectos específicos
    
    def _apply_blur(self, clip, radius, start_time, end_time):
        """Aplica efecto de desenfoque"""
        def blur_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                return cv2.GaussianBlur(frame, (radius, radius), 0)
            return frame
        
        return clip.fl(blur_func)
    
    def _apply_brightness(self, clip, factor, start_time, end_time):
        """Aplica efecto de brillo"""
        def brightness_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                return np.clip(frame * factor, 0, 255).astype('uint8')
            return frame
        
        return clip.fl(brightness_func)
    
    def _apply_contrast(self, clip, factor, start_time, end_time):
        """Aplica efecto de contraste"""
        def contrast_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                mean = np.mean(frame, axis=(0, 1))
                return np.clip((frame - mean) * factor + mean, 0, 255).astype('uint8')
            return frame
        
        return clip.fl(contrast_func)
    
    def _apply_grayscale(self, clip, start_time, end_time):
        """Aplica efecto de escala de grises"""
        def grayscale_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                return np.stack([gray, gray, gray], axis=2)
            return frame
        
        return clip.fl(grayscale_func)
    
    def _apply_sepia(self, clip, start_time, end_time):
        """Aplica efecto sepia"""
        def sepia_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                normalized = gray.astype(float) / 255.0
                sepia = np.zeros_like(frame)
                sepia[:, :, 0] = np.clip(normalized * 255 * 1.07, 0, 255)  # B
                sepia[:, :, 1] = np.clip(normalized * 255 * 0.74, 0, 255)  # G
                sepia[:, :, 2] = np.clip(normalized * 255 * 0.43, 0, 255)  # R
                return sepia.astype('uint8')
            return frame
        
        return clip.fl(sepia_func)
    
    def _apply_mirror(self, clip, axis, start_time, end_time):
        """Aplica efecto espejo"""
        def mirror_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                if axis == "x":
                    return frame[:, ::-1]
                elif axis == "y":
                    return frame[::-1, :]
                else:
                    return frame[::-1, ::-1]  # Ambos ejes
            return frame
        
        return clip.fl(mirror_func)
    
    def _apply_reverse(self, clip, start_time, end_time):
        """Aplica efecto de reproducción inversa"""
        if start_time > 0 or end_time < clip.duration:
            # Recortar el clip para aplicar el efecto solo a una parte
            subclip = clip.subclip(start_time, end_time)
            reversed_subclip = subclip.fx(vfx.time_mirror)
            
            # Crear clips para las partes antes y después
            if start_time > 0:
                before_clip = clip.subclip(0, start_time)
            else:
                before_clip = None
                
            if end_time < clip.duration:
                after_clip = clip.subclip(end_time, clip.duration)
            else:
                after_clip = None
            
            # Concatenar las partes
            clips_to_concat = []
            if before_clip:
                clips_to_concat.append(before_clip)
            clips_to_concat.append(reversed_subclip)
            if after_clip:
                clips_to_concat.append(after_clip)
            
            return concatenate_videoclips(clips_to_concat)
        else:
            # Aplicar a todo el clip
            return clip.fx(vfx.time_mirror)
    
    def _apply_slow_motion(self, clip, factor, start_time, end_time):
        """Aplica efecto de cámara lenta"""
        if start_time > 0 or end_time < clip.duration:
            # Recortar el clip para aplicar el efecto solo a una parte
            subclip = clip.subclip(start_time, end_time)
            slow_subclip = subclip.fx(vfx.speedx, factor)
            
            # Calcular la nueva duración del subclip
            new_duration = subclip.duration / factor
            
                        # Crear clips para las partes antes y después
            if start_time > 0:
                before_clip = clip.subclip(0, start_time)
            else:
                before_clip = None
                
            if end_time < clip.duration:
                after_clip = clip.subclip(end_time, clip.duration)
            else:
                after_clip = None
            
            # Concatenar las partes
            clips_to_concat = []
            if before_clip:
                clips_to_concat.append(before_clip)
            clips_to_concat.append(slow_subclip)
            if after_clip:
                clips_to_concat.append(after_clip)
            
            return concatenate_videoclips(clips_to_concat)
        else:
            # Aplicar a todo el clip
            return clip.fx(vfx.speedx, factor)
    
    def _apply_fast_motion(self, clip, factor, start_time, end_time):
        """Aplica efecto de cámara rápida"""
        # Mismo código que slow_motion pero con factor > 1
        return self._apply_slow_motion(clip, factor, start_time, end_time)
    
    def _apply_zoom(self, clip, zoom_factor, start_time, end_time):
        """Aplica efecto de zoom"""
        def zoom_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                h, w = frame.shape[:2]
                
                # Calcular nuevas dimensiones
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                
                # Calcular coordenadas para recortar
                y1 = max(0, (new_h - h) // 2)
                y2 = y1 + h
                x1 = max(0, (new_w - w) // 2)
                x2 = x1 + w
                
                # Redimensionar y recortar
                zoomed = cv2.resize(frame, (new_w, new_h))
                if y2 <= zoomed.shape[0] and x2 <= zoomed.shape[1]:
                    return zoomed[y1:y2, x1:x2]
            return frame
        
        return clip.fl(zoom_func)
    
    def _apply_rotate(self, clip, angle, start_time, end_time):
        """Aplica efecto de rotación"""
        def rotate_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                
                # Crear matriz de rotación
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Aplicar rotación
                rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR)
                return rotated
            return frame
        
        return clip.fl(rotate_func)
    
    def _apply_shake(self, clip, intensity, start_time, end_time):
        """Aplica efecto de temblor/sacudida"""
        def shake_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                h, w = frame.shape[:2]
                
                # Generar desplazamiento aleatorio
                dx = random.randint(-intensity, intensity)
                dy = random.randint(-intensity, intensity)
                
                # Crear matriz de transformación
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                
                # Aplicar transformación
                shaken = cv2.warpAffine(frame, M, (w, h))
                return shaken
            return frame
        
        return clip.fl(shake_func)
    
    def _apply_vignette(self, clip, intensity, start_time, end_time):
        """Aplica efecto de viñeta"""
        def vignette_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                h, w = frame.shape[:2]
                
                # Crear máscara radial
                y, x = np.ogrid[0:h, 0:w]
                center_y, center_x = h // 2, w // 2
                mask = (x - center_x)**2 + (y - center_y)**2
                
                # Normalizar máscara
                mask = mask / (mask.max() * (1.0 - intensity))
                mask = np.clip(mask, 0, 1)
                
                # Invertir máscara
                mask = 1 - mask
                
                # Aplicar máscara
                for c in range(3):
                    frame[:, :, c] = frame[:, :, c] * mask
                
                return frame.astype('uint8')
            return frame
        
        return clip.fl(vignette_func)
    
    def _apply_fade_in(self, clip, duration):
        """Aplica efecto de fade in"""
        return clip.fadein(duration)
    
    def _apply_fade_out(self, clip, duration):
        """Aplica efecto de fade out"""
        return clip.fadeout(duration)
    
    def _apply_color_filter(self, clip, color, intensity, start_time, end_time):
        """Aplica filtro de color"""
        def color_filter_func(get_frame, t):
            frame = get_frame(t)
            if start_time <= t <= end_time:
                # Convertir color a array numpy
                color_array = np.array(color, dtype=np.uint8)
                
                # Crear máscara de color
                color_mask = np.ones_like(frame) * color_array
                
                # Mezclar con intensidad
                filtered = cv2.addWeighted(frame, 1 - intensity, color_mask, intensity, 0)
                return filtered
            return frame
        
        return clip.fl(color_filter_func)
    
    def _apply_fade_transition(self, clip1, clip2, duration):
        """Aplica transición de fundido"""
        # Recortar el final del primer clip
        clip1 = clip1.subclip(0, clip1.duration)
        
        # Recortar el inicio del segundo clip
        clip2 = clip2.subclip(0, clip2.duration)
        
        # Aplicar fade out al final del primer clip
        clip1 = clip1.fadeout(duration)
        
        # Aplicar fade in al inicio del segundo clip
        clip2 = clip2.fadein(duration)
        
        # Calcular tiempo de superposición
        overlap = duration
        
        # Concatenar con superposición
        return concatenate_videoclips([clip1, clip2], method="compose", padding=-overlap)
    
    def _apply_wipe_transition(self, clip1, clip2, duration, direction="left"):
        """Aplica transición de barrido"""
        # Implementación básica de transición de barrido
        w, h = clip1.size
        
        def make_frame(t):
            # Normalizar tiempo para la transición
            if t < clip1.duration - duration:
                return clip1.get_frame(t)
            elif t >= clip1.duration:
                return clip2.get_frame(t - clip1.duration + duration)
            else:
                # Tiempo dentro de la transición
                progress = (t - (clip1.duration - duration)) / duration
                
                frame1 = clip1.get_frame(t)
                frame2 = clip2.get_frame(t - clip1.duration + duration)
                
                if direction == "left":
                    wipe_width = int(w * progress)
                    result = frame1.copy()
                    result[:, :wipe_width] = frame2[:, :wipe_width]
                elif direction == "right":
                    wipe_width = int(w * progress)
                    result = frame1.copy()
                    result[:, w-wipe_width:] = frame2[:, w-wipe_width:]
                elif direction == "up":
                    wipe_height = int(h * progress)
                    result = frame1.copy()
                    result[:wipe_height, :] = frame2[:wipe_height, :]
                elif direction == "down":
                    wipe_height = int(h * progress)
                    result = frame1.copy()
                    result[h-wipe_height:, :] = frame2[h-wipe_height:, :]
                else:
                    # Por defecto, barrido desde la izquierda
                    wipe_width = int(w * progress)
                    result = frame1.copy()
                    result[:, :wipe_width] = frame2[:, :wipe_width]
                
                return result
        
        # Crear clip compuesto
        final_duration = clip1.duration + clip2.duration - duration
        new_clip = VideoFileClip(None, ismask=False)
        new_clip.size = clip1.size
        new_clip.fps = max(clip1.fps, clip2.fps)
        new_clip.make_frame = make_frame
        new_clip.duration = final_duration
        
        # Manejar audio
        if clip1.audio is not None and clip2.audio is not None:
            audio1 = clip1.audio
            audio2 = clip2.audio
            new_audio = CompositeAudioClip([
                audio1.set_end(clip1.duration),
                audio2.set_start(clip1.duration - duration)
            ])
            new_clip.audio = new_audio
        elif clip1.audio is not None:
            new_clip.audio = clip1.audio
        elif clip2.audio is not None:
            new_clip.audio = clip2.audio.set_start(clip1.duration - duration)
        
        return new_clip
    
    def _apply_zoom_transition(self, clip1, clip2, duration, zoom_in=True):
        """Aplica transición de zoom"""
        # Implementación básica de transición de zoom
        w, h = clip1.size
        
        def make_frame(t):
            # Normalizar tiempo para la transición
            if t < clip1.duration - duration:
                return clip1.get_frame(t)
            elif t >= clip1.duration:
                return clip2.get_frame(t - clip1.duration + duration)
            else:
                # Tiempo dentro de la transición
                progress = (t - (clip1.duration - duration)) / duration
                
                frame1 = clip1.get_frame(t)
                frame2 = clip2.get_frame(t - clip1.duration + duration)
                
                if zoom_in:
                    # Zoom in: clip1 se hace más grande, clip2 aparece detrás
                    zoom_factor = 1 + progress
                    
                    # Redimensionar clip1
                    h1, w1 = frame1.shape[:2]
                    new_h, new_w = int(h1 * zoom_factor), int(w1 * zoom_factor)
                    
                    # Centrar y recortar
                    zoomed = cv2.resize(frame1, (new_w, new_h))
                    y1 = max(0, (new_h - h) // 2)
                    y2 = min(new_h, y1 + h)
                    x1 = max(0, (new_w - w) // 2)
                    x2 = min(new_w, x1 + w)
                    
                    # Mezclar frames con alpha
                    result = frame2.copy()
                    alpha = 1 - progress
                    
                    # Aplicar zoom y mezcla
                    if y2 - y1 == h and x2 - x1 == w:
                        zoomed_crop = zoomed[y1:y2, x1:x2]
                        result = cv2.addWeighted(zoomed_crop, alpha, result, 1-alpha, 0)
                else:
                    # Zoom out: clip1 se hace más pequeño, clip2 aparece detrás
                    zoom_factor = 1 - progress * 0.5  # No llegar a 0
                    
                    # Redimensionar clip1
                    h1, w1 = frame1.shape[:2]
                    new_h, new_w = int(h1 * zoom_factor), int(w1 * zoom_factor)
                    
                    # Crear frame resultado
                    result = frame2.copy()
                    
                    # Redimensionar y posicionar clip1
                    if new_h > 0 and new_w > 0:
                        resized = cv2.resize(frame1, (new_w, new_h))
                        y_offset = (h - new_h) // 2
                        x_offset = (w - new_w) // 2
                        
                        # Mezclar con alpha
                        alpha = 1 - progress
                        
                        # Colocar clip1 redimensionado sobre clip2
                        if 0 <= y_offset < h and 0 <= x_offset < w:
                            y_end = min(y_offset + new_h, h)
                            x_end = min(x_offset + new_w, w)
                            y_src_end = y_end - y_offset
                            x_src_end = x_end - x_offset
                            
                            # Región de interés
                            roi = result[y_offset:y_end, x_offset:x_end]
                            src = resized[:y_src_end, :x_src_end]
                            
                            # Mezclar
                            if roi.shape == src.shape:
                                result[y_offset:y_end, x_offset:x_end] = cv2.addWeighted(
                                    src, alpha, roi, 1-alpha, 0
                                )
                
                return result
        
        # Crear clip compuesto
        final_duration = clip1.duration + clip2.duration - duration
        new_clip = VideoFileClip(None, ismask=False)
        new_clip.size = clip1.size
        new_clip.fps = max(clip1.fps, clip2.fps)
        new_clip.make_frame = make_frame
        new_clip.duration = final_duration
        
        # Manejar audio
        if clip1.audio is not None and clip2.audio is not None:
            audio1 = clip1.audio
            audio2 = clip2.audio
            new_audio = CompositeAudioClip([
                audio1.set_end(clip1.duration),
                audio2.set_start(clip1.duration - duration)
            ])
            new_clip.audio = new_audio
        elif clip1.audio is not None:
            new_clip.audio = clip1.audio
        elif clip2.audio is not None:
            new_clip.audio = clip2.audio.set_start(clip1.duration - duration)
        
        return new_clip
    
    def _apply_rotate_transition(self, clip1, clip2, duration, angle=90):
        """Aplica transición de rotación"""
        # Implementación básica de transición de rotación
        w, h = clip1.size
        
        def make_frame(t):
            # Normalizar tiempo para la transición
            if t < clip1.duration - duration:
                return clip1.get_frame(t)
            elif t >= clip1.duration:
                return clip2.get_frame(t - clip1.duration + duration)
            else:
                # Tiempo dentro de la transición
                progress = (t - (clip1.duration - duration)) / duration
                
                frame1 = clip1.get_frame(t)
                frame2 = clip2.get_frame(t - clip1.duration + duration)
                
                # Calcular ángulo actual
                current_angle = angle * progress
                
                # Rotar frame1
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, current_angle, 1.0 - progress * 0.5)
                rotated1 = cv2.warpAffine(frame1, M, (w, h))
                
                # Mezclar frames con alpha
                alpha = 1 - progress
                result = cv2.addWeighted(rotated1, alpha, frame2, 1-alpha, 0)
                
                return result
        
        # Crear clip compuesto
        final_duration = clip1.duration + clip2.duration - duration
        new_clip = VideoFileClip(None, ismask=False)
        new_clip.size = clip1.size
        new_clip.fps = max(clip1.fps, clip2.fps)
        new_clip.make_frame = make_frame
        new_clip.duration = final_duration
        
        # Manejar audio
        if clip1.audio is not None and clip2.audio is not None:
            audio1 = clip1.audio
            audio2 = clip2.audio
            new_audio = CompositeAudioClip([
                audio1.set_end(clip1.duration),
                audio2.set_start(clip1.duration - duration)
            ])
            new_clip.audio = new_audio
        elif clip1.audio is not None:
            new_clip.audio = clip1.audio
        elif clip2.audio is not None:
            new_clip.audio = clip2.audio.set_start(clip1.duration - duration)
        
        return new_clip
    
    def _apply_dissolve_transition(self, clip1, clip2, duration):
        """Aplica transición de disolución"""
        # Similar a fade pero con diferentes parámetros
        return self._apply_fade_transition(clip1, clip2, duration)
    
    def _apply_slide_transition(self, clip1, clip2, duration, direction="left"):
        """Aplica transición de deslizamiento"""
        # Implementación básica de transición de deslizamiento
        w, h = clip1.size
        
        def make_frame(t):
            # Normalizar tiempo para la transición
            if t < clip1.duration - duration:
                return clip1.get_frame(t)
            elif t >= clip1.duration:
                return clip2.get_frame(t - clip1.duration + duration)
            else:
                # Tiempo dentro de la transición
                progress = (t - (clip1.duration - duration)) / duration
                
                frame1 = clip1.get_frame(t)
                frame2 = clip2.get_frame(t - clip1.duration + duration)
                
                result = np.zeros_like(frame1)
                
                if direction == "left":
                    # Clip1 sale por la izquierda, clip2 entra por la derecha
                    offset = int(w * progress)
                    # Parte visible de clip1
                    if offset < w:
                        result[:, :w-offset] = frame1[:, offset:]
                    # Parte visible de clip2
                    if offset > 0:
                        result[:, w-offset:] = frame2[:, :offset]
                
                elif direction == "right":
                    # Clip1 sale por la derecha, clip2 entra por la izquierda
                    offset = int(w * progress)
                    # Parte visible de clip1
                    if offset < w:
                        result[:, offset:] = frame1[:, :w-offset]
                    # Parte visible de clip2
                    if offset > 0:
                        result[:, :offset] = frame2[:, w-offset:]
                
                elif direction == "up":
                    # Clip1 sale por arriba, clip2 entra por abajo
                    offset = int(h * progress)
                    # Parte visible de clip1
                    if offset < h:
                        result[:h-offset, :] = frame1[offset:, :]
                    # Parte visible de clip2
                    if offset > 0:
                        result[h-offset:, :] = frame2[:offset, :]
                
                elif direction == "down":
                    # Clip1 sale por abajo, clip2 entra por arriba
                    offset = int(h * progress)
                    # Parte visible de clip1
                    if offset < h:
                        result[offset:, :] = frame1[:h-offset, :]
                    # Parte visible de clip2
                    if offset > 0:
                        result[:offset, :] = frame2[h-offset:, :]
                
                return result
        
        # Crear clip compuesto
        final_duration = clip1.duration + clip2.duration - duration
        new_clip = VideoFileClip(None, ismask=False)
        new_clip.size = clip1.size
        new_clip.fps = max(clip1.fps, clip2.fps)
        new_clip.make_frame = make_frame
        new_clip.duration = final_duration
        
        # Manejar audio
        if clip1.audio is not None and clip2.audio is not None:
            audio1 = clip1.audio
            audio2 = clip2.audio
            new_audio = CompositeAudioClip([
                audio1.set_end(clip1.duration),
                audio2.set_start(clip1.duration - duration)
            ])
            new_clip.audio = new_audio
        elif clip1.audio is not None:
            new_clip.audio = clip1.audio
        elif clip2.audio is not None:
            new_clip.audio = clip2.audio.set_start(clip1.duration - duration)
        
        return new_clip
    
    def _apply_volume_effect(self, clip, factor, start_time, end_time):
        """Aplica efecto de volumen"""
        # Verificar que el clip tiene audio
        if clip.audio is None:
            return clip
        
        # Función para modificar el volumen
        def change_volume(t, audio_t):
            if start_time <= t <= end_time:
                return audio_t * factor
            return audio_t
        
        # Aplicar efecto al audio
        new_audio = clip.audio.fl(change_volume, keep_duration=True)
        new_clip = clip.set_audio(new_audio)
        
        return new_clip
    
    def _apply_fade_in_audio(self, clip, duration):
        """Aplica fade in al audio"""
        # Verificar que el clip tiene audio
        if clip.audio is None:
            return clip
        
        # Aplicar fade in al audio
        new_audio = clip.audio.audio_fadein(duration)
        new_clip = clip.set_audio(new_audio)
        
        return new_clip
    
    def _apply_fade_out_audio(self, clip, duration):
        """Aplica fade out al audio"""
        # Verificar que el clip tiene audio
        if clip.audio is None:
            return clip
        
        # Aplicar fade out al audio
        new_audio = clip.audio.audio_fadeout(duration)
        new_clip = clip.set_audio(new_audio)
        
        return new_clip
    
    def _apply_echo_effect(self, clip, intensity, delay, start_time, end_time):
        """Aplica efecto de eco al audio"""
        # Verificar que el clip tiene audio
        if clip.audio is None:
            return clip
        
        # Extraer audio original
        original_audio = clip.audio
        
        # Crear versión retrasada del audio
        delayed_audio = original_audio.set_start(delay)
        
        # Ajustar volumen del eco
        delayed_audio = delayed_audio.volumex(intensity)
        
        # Combinar audio original con eco
        new_audio = CompositeAudioClip([original_audio, delayed_audio])
        
        # Recortar al mismo tamaño que el original
        new_audio = new_audio.set_duration(original_audio.duration)
        
        # Aplicar solo en el rango especificado
        if start_time > 0 or end_time < clip.duration:
            audio_before = None
            audio_after = None
            
            if start_time > 0:
                audio_before = original_audio.subclip(0, start_time)
            
            audio_effect = new_audio.subclip(start_time, end_time)
            
            if end_time < clip.duration:
                audio_after = original_audio.subclip(end_time, clip.duration)
            
            # Combinar segmentos
            segments = []
            if audio_before:
                segments.append(audio_before)
            segments.append(audio_effect)
            if audio_after:
                segments.append(audio_after)
            
            new_audio = concatenate_audioclips(segments)
        
        # Aplicar al clip
        new_clip = clip.set_audio(new_audio)
        
        return new_clip
    
    def _apply_mute_effect(self, clip, start_time, end_time):
        """Silencia el audio en un rango específico"""
        # Verificar que el clip tiene audio
        if clip.audio is None:
            return clip
        
        # Extraer audio original
        original_audio = clip.audio
        
        # Función para silenciar
        def mute_audio(t, audio_t):
            if start_time <= t <= end_time:
                return 0
            return audio_t
        
        # Aplicar silencio
        new_audio = original_audio.fl(mute_audio, keep_duration=True)
        new_clip = clip.set_audio(new_audio)
        
        return new_clip