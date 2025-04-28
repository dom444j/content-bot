import os
import logging
import json
import random
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, TextClip, concatenate_videoclips

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VideoComposer")

class VideoComposer:
    """
    Clase para componer videos a partir de clips, imágenes, audio y efectos
    """
    
    def __init__(self, config_path: str = None, output_dir: str = "uploads/videos"):
        """
        Inicializa el compositor de videos
        
        Args:
            config_path: Ruta al archivo de configuración
            output_dir: Directorio de salida para los videos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar configuración
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        
        # Configuración por defecto
        self.default_resolution = self.config.get("default_resolution", (1080, 1920))  # Vertical por defecto
        self.default_fps = self.config.get("default_fps", 30)
        self.default_format = self.config.get("default_format", "mp4")
        self.temp_dir = self.config.get("temp_dir", "temp")
        
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"VideoComposer inicializado con resolución {self.default_resolution}")
    
    def create_video(self, 
                    storyboard: Dict[str, Any], 
                    output_path: str = None, 
                    add_cta: bool = True,
                    cta_timing: Tuple[float, float] = (4, 8)) -> str:
        """
        Crea un video completo a partir de un storyboard
        
        Args:
            storyboard: Diccionario con la estructura del video
            output_path: Ruta de salida para el video
            add_cta: Si se debe añadir CTA
            cta_timing: Tiempo de inicio y fin del CTA (en segundos)
            
        Returns:
            Ruta al video generado
        """
        try:
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"video_{timestamp}_{random_id}.{self.default_format}"
                output_path = os.path.join(self.output_dir, filename)
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Extraer información del storyboard
            clips_info = storyboard.get("clips", [])
            audio_path = storyboard.get("audio")
            background_music = storyboard.get("background_music")
            cta_info = storyboard.get("cta")
            resolution = storyboard.get("resolution", self.default_resolution)
            fps = storyboard.get("fps", self.default_fps)
            duration = storyboard.get("duration", 0)
            
            # Procesar clips
            video_clips = []
            total_duration = 0
            
            for clip_info in clips_info:
                clip_type = clip_info.get("type", "video")
                clip_path = clip_info.get("path")
                clip_duration = clip_info.get("duration", 0)
                clip_start = clip_info.get("start_time", 0)
                clip_end = clip_info.get("end_time", None)
                position = clip_info.get("position", ("center", "center"))
                
                if clip_type == "video" and clip_path and os.path.exists(clip_path):
                    # Cargar clip de video
                    video = VideoFileClip(clip_path)
                    
                    # Recortar si se especifica
                    if clip_start > 0 or clip_end:
                        video = video.subclip(clip_start, clip_end)
                    
                    # Ajustar duración si se especifica
                    if clip_duration > 0:
                        video = video.set_duration(clip_duration)
                    
                    # Redimensionar si es necesario
                    if video.size != resolution:
                        video = video.resize(resolution)
                    
                    video_clips.append(video)
                    total_duration += video.duration
                
                elif clip_type == "image" and clip_path and os.path.exists(clip_path):
                    # Cargar imagen
                    image = ImageClip(clip_path)
                    
                    # Establecer duración
                    if clip_duration > 0:
                        image = image.set_duration(clip_duration)
                    else:
                        image = image.set_duration(3)  # Duración por defecto: 3 segundos
                    
                    # Redimensionar si es necesario
                    if image.size != resolution:
                        image = image.resize(resolution)
                    
                    video_clips.append(image)
                    total_duration += image.duration
            
            # Si no hay clips, crear un clip en blanco
            if not video_clips:
                blank_clip = ImageClip(np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8))
                blank_clip = blank_clip.set_duration(duration or 10)
                video_clips.append(blank_clip)
                total_duration = blank_clip.duration
            
            # Concatenar clips
            final_clip = concatenate_videoclips(video_clips) if len(video_clips) > 1 else video_clips[0]
            
            # Añadir audio principal si se proporciona
            if audio_path and os.path.exists(audio_path):
                audio = AudioFileClip(audio_path)
                
                # Ajustar duración del audio a la del video
                if audio.duration > final_clip.duration:
                    audio = audio.subclip(0, final_clip.duration)
                
                final_clip = final_clip.set_audio(audio)
            
            # Añadir música de fondo si se proporciona
            if background_music and os.path.exists(background_music):
                bg_music = AudioFileClip(background_music)
                
                # Ajustar duración y volumen
                if bg_music.duration < final_clip.duration:
                    # Repetir música si es necesario
                    repeats = int(final_clip.duration / bg_music.duration) + 1
                    bg_music = concatenate_videoclips([bg_music] * repeats).subclip(0, final_clip.duration)
                else:
                    bg_music = bg_music.subclip(0, final_clip.duration)
                
                # Reducir volumen de la música de fondo
                bg_music = bg_music.volumex(0.3)
                
                # Mezclar con el audio existente o establecer como audio principal
                if final_clip.audio:
                    final_audio = final_clip.audio.set_duration(final_clip.duration)
                    mixed_audio = final_audio.audio_fadeout(1)
                    final_clip = final_clip.set_audio(mixed_audio)
                else:
                    final_clip = final_clip.set_audio(bg_music)
            
            # Añadir CTA si se solicita
            if add_cta and cta_info:
                cta_text = cta_info.get("text", "¡Sigue para más!")
                cta_font = cta_info.get("font", "Arial")
                cta_font_size = cta_info.get("font_size", 70)
                cta_color = cta_info.get("color", "white")
                cta_bg_color = cta_info.get("bg_color", "black")
                cta_position = cta_info.get("position", ("center", "bottom"))
                
                # Crear clip de texto para CTA
                cta_clip = TextClip(cta_text, fontsize=cta_font_size, font=cta_font, 
                                   color=cta_color, bg_color=cta_bg_color, 
                                   method="caption", align="center")
                
                # Posicionar CTA
                if cta_position[1] == "bottom":
                    y_pos = resolution[1] - cta_clip.h - 50
                elif cta_position[1] == "top":
                    y_pos = 50
                else:
                    y_pos = resolution[1] // 2 - cta_clip.h // 2
                
                if cta_position[0] == "left":
                    x_pos = 50
                elif cta_position[0] == "right":
                    x_pos = resolution[0] - cta_clip.w - 50
                else:
                    x_pos = resolution[0] // 2 - cta_clip.w // 2
                
                # Establecer duración y posición del CTA
                cta_start, cta_end = cta_timing
                if cta_end > final_clip.duration:
                    cta_end = final_clip.duration
                
                cta_clip = cta_clip.set_position((x_pos, y_pos)).set_start(cta_start).set_end(cta_end)
                
                # Añadir CTA al video
                final_clip = CompositeVideoClip([final_clip, cta_clip])
            
            # Guardar video final
            final_clip.write_videofile(output_path, fps=fps, codec="libx264", 
                                      audio_codec="aac", threads=4)
            
            logger.info(f"Video creado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al crear video: {str(e)}")
            return None
    
    def create_slideshow(self, 
                        image_paths: List[str], 
                        audio_path: str = None,
                        durations: List[float] = None,
                        transitions: List[str] = None,
                        output_path: str = None,
                        add_cta: bool = True,
                        cta_info: Dict[str, Any] = None) -> str:
        """
        Crea un slideshow a partir de imágenes
        
        Args:
            image_paths: Lista de rutas a imágenes
            audio_path: Ruta al archivo de audio
            durations: Lista de duraciones para cada imagen
            transitions: Lista de transiciones entre imágenes
            output_path: Ruta de salida para el video
            add_cta: Si se debe añadir CTA
            cta_info: Información del CTA
            
        Returns:
            Ruta al video generado
        """
        try:
            # Verificar que hay imágenes
            if not image_paths:
                logger.error("No se proporcionaron imágenes para el slideshow")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"slideshow_{timestamp}_{random_id}.{self.default_format}"
                output_path = os.path.join(self.output_dir, filename)
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Establecer duraciones por defecto si no se proporcionan
            if not durations:
                durations = [3.0] * len(image_paths)
            elif len(durations) < len(image_paths):
                durations.extend([3.0] * (len(image_paths) - len(durations)))
            
            # Crear clips de imágenes
            image_clips = []
            for i, (img_path, duration) in enumerate(zip(image_paths, durations)):
                if os.path.exists(img_path):
                    img_clip = ImageClip(img_path).set_duration(duration)
                    
                    # Redimensionar si es necesario
                    if img_clip.size != self.default_resolution:
                        img_clip = img_clip.resize(self.default_resolution)
                    
                    image_clips.append(img_clip)
            
            # Concatenar clips
            slideshow = concatenate_videoclips(image_clips, method="compose")
            
            # Añadir audio si se proporciona
            if audio_path and os.path.exists(audio_path):
                audio = AudioFileClip(audio_path)
                
                # Ajustar duración del audio a la del slideshow
                if audio.duration > slideshow.duration:
                    audio = audio.subclip(0, slideshow.duration)
                elif audio.duration < slideshow.duration:
                    # Repetir audio si es necesario
                    repeats = int(slideshow.duration / audio.duration) + 1
                    audio = concatenate_videoclips([audio] * repeats).subclip(0, slideshow.duration)
                
                slideshow = slideshow.set_audio(audio)
            
            # Añadir CTA si se solicita
            if add_cta and cta_info:
                cta_text = cta_info.get("text", "¡Sigue para más!")
                cta_font = cta_info.get("font", "Arial")
                cta_font_size = cta_info.get("font_size", 70)
                cta_color = cta_info.get("color", "white")
                cta_bg_color = cta_info.get("bg_color", "black")
                cta_position = cta_info.get("position", ("center", "bottom"))
                cta_timing = cta_info.get("timing", (4, 8))
                
                # Crear clip de texto para CTA
                cta_clip = TextClip(cta_text, fontsize=cta_font_size, font=cta_font, 
                                   color=cta_color, bg_color=cta_bg_color, 
                                   method="caption", align="center")
                
                # Posicionar CTA
                if cta_position[1] == "bottom":
                    y_pos = self.default_resolution[1] - cta_clip.h - 50
                elif cta_position[1] == "top":
                    y_pos = 50
                else:
                    y_pos = self.default_resolution[1] // 2 - cta_clip.h // 2
                
                if cta_position[0] == "left":
                    x_pos = 50
                elif cta_position[0] == "right":
                    x_pos = self.default_resolution[0] - cta_clip.w - 50
                else:
                    x_pos = self.default_resolution[0] // 2 - cta_clip.w // 2
                
                # Establecer duración y posición del CTA
                cta_start, cta_end = cta_timing
                if cta_end > slideshow.duration:
                    cta_end = slideshow.duration
                
                cta_clip = cta_clip.set_position((x_pos, y_pos)).set_start(cta_start).set_end(cta_end)
                
                # Añadir CTA al slideshow
                slideshow = CompositeVideoClip([slideshow, cta_clip])
            
            # Guardar slideshow
            slideshow.write_videofile(output_path, fps=self.default_fps, codec="libx264", 
                                     audio_codec="aac", threads=4)
            
            logger.info(f"Slideshow creado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al crear slideshow: {str(e)}")
            return None
    
    def add_overlay(self, 
                   video_path: str, 
                   overlay_path: str, 
                   position: Tuple[str, str] = ("center", "center"),
                   start_time: float = 0, 
                   end_time: float = None,
                   opacity: float = 1.0,
                   output_path: str = None) -> str:
        """
        Añade una superposición (imagen o video) a un video
        
        Args:
            video_path: Ruta al video base
            overlay_path: Ruta a la imagen o video a superponer
            position: Posición de la superposición (x, y)
            start_time: Tiempo de inicio de la superposición
            end_time: Tiempo de fin de la superposición
            opacity: Opacidad de la superposición (0-1)
            output_path: Ruta de salida para el video
            
        Returns:
            Ruta al video con superposición
        """
        try:
            # Verificar que existen los archivos
            if not os.path.exists(video_path):
                logger.error(f"No se encontró el video base: {video_path}")
                return None
            
            if not os.path.exists(overlay_path):
                logger.error(f"No se encontró la superposición: {overlay_path}")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                base_name = os.path.basename(video_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(self.output_dir, f"{name}_overlay{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar video base
            video = VideoFileClip(video_path)
            
            # Determinar tipo de superposición
            overlay_ext = os.path.splitext(overlay_path)[1].lower()
            is_video = overlay_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
            
            # Cargar superposición
            if is_video:
                overlay = VideoFileClip(overlay_path)
            else:
                overlay = ImageClip(overlay_path)
            
            # Establecer duración de la superposición
            if end_time is None:
                end_time = video.duration
            
            overlay = overlay.set_start(start_time).set_end(end_time)
            
            # Establecer opacidad
            if opacity < 1.0:
                overlay = overlay.set_opacity(opacity)
            
            # Calcular posición
            video_width, video_height = video.size
            overlay_width, overlay_height = overlay.size
            
            if position[0] == "left":
                x_pos = 50
            elif position[0] == "right":
                x_pos = video_width - overlay_width - 50
            else:  # center
                x_pos = (video_width - overlay_width) // 2
            
            if position[1] == "top":
                y_pos = 50
            elif position[1] == "bottom":
                y_pos = video_height - overlay_height - 50
            else:  # center
                y_pos = (video_height - overlay_height) // 2
            
            # Posicionar superposición
            overlay = overlay.set_position((x_pos, y_pos))
            
            # Componer video final
            final_video = CompositeVideoClip([video, overlay])
            
            # Guardar video
            final_video.write_videofile(output_path, fps=video.fps, codec="libx264", 
                                       audio_codec="aac", threads=4)
            
            logger.info(f"Video con superposición creado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al añadir superposición: {str(e)}")
            return None
    
    def add_text(self, 
                video_path: str, 
                text: str,
                font: str = "Arial",
                font_size: int = 50,
                color: str = "white",
                bg_color: str = None,
                position: Tuple[str, str] = ("center", "bottom"),
                start_time: float = 0,
                end_time: float = None,
                output_path: str = None) -> str:
        """
        Añade texto a un video
        
        Args:
            video_path: Ruta al video
            text: Texto a añadir
            font: Fuente del texto
            font_size: Tamaño de la fuente
            color: Color del texto
            bg_color: Color de fondo del texto
            position: Posición del texto (x, y)
            start_time: Tiempo de inicio del texto
            end_time: Tiempo de fin del texto
            output_path: Ruta de salida para el video
            
        Returns:
            Ruta al video con texto
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
                output_path = os.path.join(self.output_dir, f"{name}_text{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Crear clip de texto
            text_clip = TextClip(text, fontsize=font_size, font=font, 
                               color=color, bg_color=bg_color, 
                               method="caption", align="center")
            
            # Establecer duración del texto
            if end_time is None:
                end_time = video.duration
            
            text_clip = text_clip.set_start(start_time).set_end(end_time)
            
            # Calcular posición
            video_width, video_height = video.size
            text_width, text_height = text_clip.size
            
            if position[0] == "left":
                x_pos = 50
            elif position[0] == "right":
                x_pos = video_width - text_width - 50
            else:  # center
                x_pos = (video_width - text_width) // 2
            
            if position[1] == "top":
                y_pos = 50
            elif position[1] == "bottom":
                y_pos = video_height - text_height - 50
            else:  # center
                y_pos = (video_height - text_height) // 2
            
            # Posicionar texto
            text_clip = text_clip.set_position((x_pos, y_pos))
            
            # Componer video final
            final_video = CompositeVideoClip([video, text_clip])
            
            # Guardar video
            final_video.write_videofile(output_path, fps=video.fps, codec="libx264", 
                                       audio_codec="aac", threads=4)
            
            logger.info(f"Video con texto creado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al añadir texto: {str(e)}")
            return None
    
    def add_subtitles(self, 
                     video_path: str, 
                     subtitles: List[Dict[str, Any]],
                     font: str = "Arial",
                     font_size: int = 40,
                     color: str = "white",
                     bg_color: str = "black",
                     position: Tuple[str, str] = ("center", "bottom"),
                     output_path: str = None) -> str:
        """
        Añade subtítulos a un video
        
        Args:
            video_path: Ruta al video
            subtitles: Lista de diccionarios con texto, tiempo de inicio y fin
            font: Fuente de los subtítulos
            font_size: Tamaño de la fuente
            color: Color de los subtítulos
            bg_color: Color de fondo de los subtítulos
            position: Posición de los subtítulos (x, y)
            output_path: Ruta de salida para el video
            
        Returns:
            Ruta al video con subtítulos
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
                output_path = os.path.join(self.output_dir, f"{name}_subs{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Crear clips de subtítulos
            subtitle_clips = []
            
            for sub in subtitles:
                text = sub.get("text", "")
                start_time = sub.get("start", 0)
                end_time = sub.get("end", start_time + 2)
                
                # Crear clip de texto
                text_clip = TextClip(text, fontsize=font_size, font=font, 
                                   color=color, bg_color=bg_color, 
                                   method="caption", align="center")
                
                # Establecer duración
                text_clip = text_clip.set_start(start_time).set_end(end_time)
                
                # Calcular posición
                video_width, video_height = video.size
                text_width, text_height = text_clip.size
                
                if position[0] == "left":
                    x_pos = 50
                elif position[0] == "right":
                    x_pos = video_width - text_width - 50
                else:  # center
                    x_pos = (video_width - text_width) // 2
                
                if position[1] == "top":
                    y_pos = 50
                elif position[1] == "bottom":
                    y_pos = video_height - text_height - 50
                else:  # center
                    y_pos = (video_height - text_height) // 2
                
                # Posicionar texto
                text_clip = text_clip.set_position((x_pos, y_pos))
                
                subtitle_clips.append(text_clip)
            
            # Componer video final
            final_video = CompositeVideoClip([video] + subtitle_clips)
            
            # Guardar video
            final_video.write_videofile(output_path, fps=video.fps, codec="libx264", 
                                       audio_codec="aac", threads=4)
            
            logger.info(f"Video con subtítulos creado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al añadir subtítulos: {str(e)}")
            return None
    
    def add_cta(self, 
               video_path: str, 
               cta_text: str = "¡Sigue para más!",
               font: str = "Arial",
               font_size: int = 70,
               color: str = "white",
               bg_color: str = "black",
               position: Tuple[str, str] = ("center", "bottom"),
               start_time: float = 4,
               end_time: float = 8,
               output_path: str = None) -> str:
        """
        Añade un CTA (Call to Action) a un video
        
        Args:
            video_path: Ruta al video
            cta_text: Texto del CTA
            font: Fuente del texto
            font_size: Tamaño de la fuente
            color: Color del texto
            bg_color: Color de fondo del texto
            position: Posición del texto (x, y)
            start_time: Tiempo de inicio del CTA
            end_time: Tiempo de fin del CTA
            output_path: Ruta de salida para el video
            
        Returns:
            Ruta al video con CTA
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
                output_path = os.path.join(self.output_dir, f"{name}_cta{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Crear clip de texto para CTA
            cta_clip = TextClip(cta_text, fontsize=font_size, font=font, 
                               color=color, bg_color=bg_color, 
                               method="caption", align="center")
            
            # Establecer duración del CTA
            if end_time > video.duration:
                end_time = video.duration
            
            cta_clip = cta_clip.set_start(start_time).set_end(end_time)
            
            # Calcular posición
            video_width, video_height = video.size
            cta_width, cta_height = cta_clip.size
            
            if position[0] == "left":
                x_pos = 50
            elif position[0] == "right":
                x_pos = video_width - cta_width - 50
            else:  # center
                x_pos = (video_width - cta_width) // 2
            
                        if position[1] == "top":
                y_pos = 50
            elif position[1] == "bottom":
                y_pos = video_height - cta_height - 50
            else:  # center
                y_pos = (video_height - cta_height) // 2
            
            # Posicionar texto
            cta_clip = cta_clip.set_position((x_pos, y_pos))
            
            # Componer video final
            final_video = CompositeVideoClip([video, cta_clip])
            
            # Guardar video
            final_video.write_videofile(output_path, fps=video.fps, codec="libx264", 
                                       audio_codec="aac", threads=4)
            
            logger.info(f"Video con CTA creado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al añadir CTA: {str(e)}")
            return None
    
    def merge_videos(self, 
                    video_paths: List[str], 
                    output_path: str = None,
                    transition_type: str = None,
                    transition_duration: float = 1.0) -> str:
        """
        Combina varios videos en uno solo
        
        Args:
            video_paths: Lista de rutas a videos
            output_path: Ruta de salida para el video
            transition_type: Tipo de transición entre videos (None, fade, wipe)
            transition_duration: Duración de la transición en segundos
            
        Returns:
            Ruta al video combinado
        """
        try:
            # Verificar que hay videos
            if not video_paths:
                logger.error("No se proporcionaron videos para combinar")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"merged_{timestamp}_{random_id}.{self.default_format}"
                output_path = os.path.join(self.output_dir, filename)
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar videos
            video_clips = []
            for video_path in video_paths:
                if os.path.exists(video_path):
                    clip = VideoFileClip(video_path)
                    video_clips.append(clip)
            
            # Verificar que hay clips
            if not video_clips:
                logger.error("No se pudieron cargar los videos")
                return None
            
            # Aplicar transiciones si se solicita
            if transition_type and len(video_clips) > 1:
                # Implementar transiciones (fade, wipe, etc.)
                # Esto requeriría una implementación más compleja
                # Por ahora, simplemente concatenamos los clips
                pass
            
            # Concatenar clips
            final_video = concatenate_videoclips(video_clips)
            
            # Guardar video
            final_video.write_videofile(output_path, fps=self.default_fps, codec="libx264", 
                                       audio_codec="aac", threads=4)
            
            logger.info(f"Videos combinados exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al combinar videos: {str(e)}")
            return None
    
    def trim_video(self, 
                  video_path: str, 
                  start_time: float = 0, 
                  end_time: float = None,
                  output_path: str = None) -> str:
        """
        Recorta un video
        
        Args:
            video_path: Ruta al video
            start_time: Tiempo de inicio del recorte (en segundos)
            end_time: Tiempo de fin del recorte (en segundos)
            output_path: Ruta de salida para el video
            
        Returns:
            Ruta al video recortado
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
                output_path = os.path.join(self.output_dir, f"{name}_trimmed{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Verificar tiempos
            if end_time is None:
                end_time = video.duration
            
            if start_time >= end_time:
                logger.error("El tiempo de inicio debe ser menor que el tiempo de fin")
                return None
            
            if start_time < 0:
                start_time = 0
            
            if end_time > video.duration:
                end_time = video.duration
            
            # Recortar video
            trimmed_video = video.subclip(start_time, end_time)
            
            # Guardar video
            trimmed_video.write_videofile(output_path, fps=video.fps, codec="libx264", 
                                         audio_codec="aac", threads=4)
            
            logger.info(f"Video recortado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al recortar video: {str(e)}")
            return None
    
    def change_speed(self, 
                    video_path: str, 
                    speed_factor: float = 1.0,
                    maintain_audio_pitch: bool = True,
                    output_path: str = None) -> str:
        """
        Cambia la velocidad de un video
        
        Args:
            video_path: Ruta al video
            speed_factor: Factor de velocidad (>1 más rápido, <1 más lento)
            maintain_audio_pitch: Mantener el tono del audio al cambiar la velocidad
            output_path: Ruta de salida para el video
            
        Returns:
            Ruta al video con velocidad modificada
        """
        try:
            # Verificar que existe el video
            if not os.path.exists(video_path):
                logger.error(f"No se encontró el video: {video_path}")
                return None
            
            # Verificar factor de velocidad
            if speed_factor <= 0:
                logger.error("El factor de velocidad debe ser mayor que 0")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                base_name = os.path.basename(video_path)
                name, ext = os.path.splitext(base_name)
                speed_str = f"{speed_factor:.1f}".replace(".", "_")
                output_path = os.path.join(self.output_dir, f"{name}_speed{speed_str}x{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Cambiar velocidad
            modified_video = video.fx(lambda clip: clip.speedx(factor=speed_factor))
            
            # Guardar video
            modified_video.write_videofile(output_path, fps=video.fps, codec="libx264", 
                                          audio_codec="aac", threads=4)
            
            logger.info(f"Velocidad de video modificada exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al cambiar velocidad del video: {str(e)}")
            return None
    
    def add_watermark(self, 
                     video_path: str, 
                     watermark_path: str,
                     position: Tuple[str, str] = ("right", "bottom"),
                     opacity: float = 0.5,
                     size_percent: float = 0.2,
                     output_path: str = None) -> str:
        """
        Añade una marca de agua a un video
        
        Args:
            video_path: Ruta al video
            watermark_path: Ruta a la imagen de marca de agua
            position: Posición de la marca de agua (x, y)
            opacity: Opacidad de la marca de agua (0-1)
            size_percent: Tamaño de la marca de agua como porcentaje del video
            output_path: Ruta de salida para el video
            
        Returns:
            Ruta al video con marca de agua
        """
        try:
            # Verificar que existen los archivos
            if not os.path.exists(video_path):
                logger.error(f"No se encontró el video: {video_path}")
                return None
            
            if not os.path.exists(watermark_path):
                logger.error(f"No se encontró la marca de agua: {watermark_path}")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                base_name = os.path.basename(video_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(self.output_dir, f"{name}_watermark{ext}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Cargar marca de agua
            watermark = ImageClip(watermark_path)
            
            # Redimensionar marca de agua
            video_width, video_height = video.size
            watermark_width = int(video_width * size_percent)
            watermark_height = int(watermark_width * watermark.h / watermark.w)
            watermark = watermark.resize(width=watermark_width)
            
            # Establecer opacidad
            watermark = watermark.set_opacity(opacity)
            
            # Calcular posición
            if position[0] == "left":
                x_pos = 20
            elif position[0] == "right":
                x_pos = video_width - watermark_width - 20
            else:  # center
                x_pos = (video_width - watermark_width) // 2
            
            if position[1] == "top":
                y_pos = 20
            elif position[1] == "bottom":
                y_pos = video_height - watermark_height - 20
            else:  # center
                y_pos = (video_height - watermark_height) // 2
            
            # Posicionar marca de agua
            watermark = watermark.set_position((x_pos, y_pos))
            
            # Establecer duración
            watermark = watermark.set_duration(video.duration)
            
            # Componer video final
            final_video = CompositeVideoClip([video, watermark])
            
            # Guardar video
            final_video.write_videofile(output_path, fps=video.fps, codec="libx264", 
                                       audio_codec="aac", threads=4)
            
            logger.info(f"Video con marca de agua creado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al añadir marca de agua: {str(e)}")
            return None