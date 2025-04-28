"""
Content Recycler Module

Este módulo se encarga de reciclar y reutilizar contenido existente para maximizar
el ROI y reducir costos de producción. Permite transformar videos entre plataformas,
crear clips de contenido más largo, y reutilizar assets y personajes.
"""

import os
import json
import logging
import random
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips, TextClip
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/recycling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ContentRecycler")

class ContentRecycler:
    """
    Sistema de reciclaje y reutilización de contenido para maximizar ROI
    y reducir costos de producción.
    """
    
    def __init__(self, data_path: str = "data", uploads_path: str = "uploads"):
        """
        Inicializa el reciclador de contenido.
        
        Args:
            data_path: Ruta a la carpeta de datos
            uploads_path: Ruta a la carpeta de uploads
        """
        self.data_path = data_path
        self.uploads_path = uploads_path
        self.content_db_path = os.path.join(data_path, "content_database.json")
        self.recycling_stats_path = os.path.join(data_path, "recycling_stats.json")
        
        # Crear directorios si no existen
        os.makedirs(os.path.join(uploads_path, "recycled"), exist_ok=True)
        os.makedirs(os.path.join(uploads_path, "clips"), exist_ok=True)
        os.makedirs(os.path.join(uploads_path, "transformed"), exist_ok=True)
        
        # Cargar base de datos de contenido
        self.content_db = self._load_content_database()
        
        # Cargar estadísticas de reciclaje
        self.recycling_stats = self._load_recycling_stats()
        
        # Vectorizador para análisis de similitud
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        logger.info("ContentRecycler inicializado correctamente")
    
    def _load_content_database(self) -> Dict:
        """Carga la base de datos de contenido o crea una nueva si no existe."""
        if os.path.exists(self.content_db_path):
            try:
                with open(self.content_db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error al cargar base de datos de contenido: {str(e)}")
                return {"videos": [], "assets": [], "characters": [], "scripts": []}
        else:
            return {"videos": [], "assets": [], "characters": [], "scripts": []}
    
    def _load_recycling_stats(self) -> Dict:
        """Carga las estadísticas de reciclaje o crea nuevas si no existen."""
        if os.path.exists(self.recycling_stats_path):
            try:
                with open(self.recycling_stats_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error al cargar estadísticas de reciclaje: {str(e)}")
                return {"clips_created": 0, "transformations": 0, "assets_reused": 0, "cost_savings": 0.0}
        else:
            return {"clips_created": 0, "transformations": 0, "assets_reused": 0, "cost_savings": 0.0}
    
    def _save_content_database(self) -> None:
        """Guarda la base de datos de contenido."""
        try:
            with open(self.content_db_path, 'w', encoding='utf-8') as f:
                json.dump(self.content_db, f, indent=4)
        except Exception as e:
            logger.error(f"Error al guardar base de datos de contenido: {str(e)}")
    
    def _save_recycling_stats(self) -> None:
        """Guarda las estadísticas de reciclaje."""
        try:
            with open(self.recycling_stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.recycling_stats, f, indent=4)
        except Exception as e:
            logger.error(f"Error al guardar estadísticas de reciclaje: {str(e)}")
    
    def register_content(self, content_type: str, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Registra nuevo contenido en la base de datos para su posible reciclaje.
        
        Args:
            content_type: Tipo de contenido ('video', 'asset', 'character', 'script')
            content_data: Datos del contenido
            
        Returns:
            Resultado del registro
        """
        try:
            # Validar tipo de contenido
            if content_type not in ["video", "asset", "character", "script"]:
                return {
                    "status": "error",
                    "message": f"Tipo de contenido no válido: {content_type}"
                }
            
            # Añadir ID único y timestamp
            content_data["id"] = f"{content_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
            content_data["registered_at"] = datetime.now().isoformat()
            content_data["recycled_count"] = 0
            
            # Añadir a la base de datos correspondiente
            if content_type == "video":
                self.content_db["videos"].append(content_data)
            elif content_type == "asset":
                self.content_db["assets"].append(content_data)
            elif content_type == "character":
                self.content_db["characters"].append(content_data)
            elif content_type == "script":
                self.content_db["scripts"].append(content_data)
            
            # Guardar base de datos actualizada
            self._save_content_database()
            
            logger.info(f"Contenido registrado: {content_type} - {content_data['id']}")
            
            return {
                "status": "success",
                "message": f"Contenido registrado correctamente",
                "content_id": content_data["id"]
            }
            
        except Exception as e:
            logger.error(f"Error al registrar contenido: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al registrar contenido: {str(e)}"
            }
    
    def find_recyclable_content(self, 
                               content_type: str = None, 
                               platform: str = None,
                               topic: str = None,
                               min_performance: float = None,
                               max_recycled_count: int = None,
                               limit: int = 10) -> Dict[str, Any]:
        """
        Busca contenido que puede ser reciclado según criterios específicos.
        
        Args:
            content_type: Filtrar por tipo ('video', 'asset', 'character', 'script')
            platform: Filtrar por plataforma
            topic: Filtrar por tema o categoría
            min_performance: Rendimiento mínimo (engagement, views, etc.)
            max_recycled_count: Número máximo de veces que ya ha sido reciclado
            limit: Límite de resultados
            
        Returns:
            Contenido reciclable que cumple los criterios
        """
        try:
            results = []
            
            # Determinar qué colecciones buscar
            collections = []
            if content_type:
                if content_type == "video":
                    collections = ["videos"]
                elif content_type == "asset":
                    collections = ["assets"]
                elif content_type == "character":
                    collections = ["characters"]
                elif content_type == "script":
                    collections = ["scripts"]
                else:
                    return {
                        "status": "error",
                        "message": f"Tipo de contenido no válido: {content_type}"
                    }
            else:
                collections = ["videos", "assets", "characters", "scripts"]
            
            # Buscar en cada colección
            for collection in collections:
                for item in self.content_db[collection]:
                    # Aplicar filtros
                    if platform and item.get("platform") != platform:
                        continue
                    
                    if topic and topic.lower() not in item.get("topic", "").lower() and topic.lower() not in item.get("tags", []):
                        continue
                    
                    if min_performance is not None:
                        performance = item.get("performance", 0)
                        if performance < min_performance:
                            continue
                    
                    if max_recycled_count is not None:
                        if item.get("recycled_count", 0) > max_recycled_count:
                            continue
                    
                    # Añadir a resultados
                    results.append({
                        "id": item["id"],
                        "type": collection[:-1],  # Quitar 's' final
                        "title": item.get("title", "Sin título"),
                        "platform": item.get("platform", "Desconocida"),
                        "topic": item.get("topic", ""),
                        "tags": item.get("tags", []),
                        "performance": item.get("performance", 0),
                        "recycled_count": item.get("recycled_count", 0),
                        "file_path": item.get("file_path", ""),
                        "duration": item.get("duration", 0) if collection == "videos" else None,
                        "registered_at": item.get("registered_at", "")
                    })
            
            # Ordenar por rendimiento (descendente) y número de reciclajes (ascendente)
            results.sort(key=lambda x: (-x["performance"], x["recycled_count"]))
            
            # Limitar resultados
            results = results[:limit]
            
            return {
                "status": "success",
                "count": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error al buscar contenido reciclable: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al buscar contenido reciclable: {str(e)}"
            }
    
    def create_clips_from_video(self, 
                               video_id: str, 
                               clip_strategy: str = "highlights",
                               num_clips: int = 3,
                               min_duration: int = 15,
                               max_duration: int = 60,
                               custom_timestamps: List[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Crea clips cortos a partir de un video más largo.
        
        Args:
            video_id: ID del video original
            clip_strategy: Estrategia de clipping ('highlights', 'equal_parts', 'custom')
            num_clips: Número de clips a crear
            min_duration: Duración mínima de cada clip (segundos)
            max_duration: Duración máxima de cada clip (segundos)
            custom_timestamps: Lista de tuplas (inicio, fin) para clips personalizados
            
        Returns:
            Información sobre los clips creados
        """
        try:
            # Buscar video en la base de datos
            video_data = None
            for video in self.content_db["videos"]:
                if video["id"] == video_id:
                    video_data = video
                    break
            
            if not video_data:
                return {
                    "status": "error",
                    "message": f"Video no encontrado: {video_id}"
                }
            
            video_path = video_data.get("file_path")
            if not video_path or not os.path.exists(video_path):
                return {
                    "status": "error",
                    "message": f"Archivo de video no encontrado: {video_path}"
                }
            
            # Cargar video
            video_clip = VideoFileClip(video_path)
            video_duration = video_clip.duration
            
            # Determinar timestamps para los clips
            timestamps = []
            
            if clip_strategy == "custom" and custom_timestamps:
                # Usar timestamps personalizados
                timestamps = custom_timestamps
                
            elif clip_strategy == "equal_parts":
                # Dividir en partes iguales
                part_duration = video_duration / num_clips
                for i in range(num_clips):
                    start = i * part_duration
                    end = min((i + 1) * part_duration, video_duration)
                    # Ajustar duración si es necesario
                    if end - start > max_duration:
                        end = start + max_duration
                    if end - start < min_duration:
                        continue
                    timestamps.append((start, end))
                
            elif clip_strategy == "highlights":
                # Intentar detectar momentos destacados (cambios de escena, audio alto, etc.)
                # Implementación básica: dividir en partes y añadir variación
                base_duration = min(max_duration, video_duration / num_clips)
                
                # Detectar cambios de escena (implementación simple)
                scene_changes = self._detect_scene_changes(video_path)
                
                if scene_changes:
                    # Usar cambios de escena como puntos de inicio
                    for i, scene_start in enumerate(scene_changes[:num_clips]):
                        start = scene_start
                        end = min(start + base_duration, video_duration)
                        if end - start < min_duration:
                            continue
                        timestamps.append((start, end))
                else:
                    # Fallback: dividir con algo de aleatoriedad
                    for i in range(num_clips):
                        # Añadir algo de variación aleatoria
                        variation = random.uniform(-0.1, 0.1) * base_duration
                        start = max(0, i * (video_duration / num_clips) + variation)
                        end = min(start + base_duration, video_duration)
                        if end - start < min_duration:
                            continue
                        timestamps.append((start, end))
            
            # Limitar al número solicitado
            timestamps = timestamps[:num_clips]
            
            if not timestamps:
                return {
                    "status": "error",
                    "message": "No se pudieron generar timestamps válidos para los clips"
                }
            
            # Crear clips
            clips_info = []
            clips_dir = os.path.join(self.uploads_path, "clips")
            
            for i, (start, end) in enumerate(timestamps):
                # Crear subclip
                subclip = video_clip.subclip(start, end)
                
                # Generar nombre de archivo
                clip_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_clip_{i+1}.mp4"
                clip_path = os.path.join(clips_dir, clip_filename)
                
                # Guardar clip
                subclip.write_videofile(clip_path, codec="libx264", audio_codec="aac")
                
                # Registrar clip en la base de datos
                clip_data = {
                    "title": f"{video_data.get('title', 'Video')} - Clip {i+1}",
                    "description": f"Clip extraído de {video_data.get('title', 'Video original')}",
                    "platform": video_data.get("platform", ""),
                    "topic": video_data.get("topic", ""),
                    "tags": video_data.get("tags", []),
                    "file_path": clip_path,
                    "original_video_id": video_id,
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                    "performance": 0,  # Inicialmente sin datos de rendimiento
                    "parent_content": {
                        "id": video_id,
                        "type": "video",
                        "title": video_data.get("title", "")
                    }
                }
                
                clip_result = self.register_content("video", clip_data)
                
                if clip_result["status"] == "success":
                    clips_info.append({
                        "clip_id": clip_result["content_id"],
                        "title": clip_data["title"],
                        "duration": clip_data["duration"],
                        "file_path": clip_path,
                        "start_time": start,
                        "end_time": end
                    })
            
            # Cerrar video
            video_clip.close()
            
            # Actualizar estadísticas
            self.recycling_stats["clips_created"] += len(clips_info)
            # Estimar ahorro de costos (asumiendo $1 por minuto de contenido nuevo)
            total_clip_duration = sum(clip["duration"] for clip in clips_info) / 60  # en minutos
            self.recycling_stats["cost_savings"] += total_clip_duration * 1.0
            self._save_recycling_stats()
            
            # Actualizar contador de reciclaje del video original
            for video in self.content_db["videos"]:
                if video["id"] == video_id:
                    video["recycled_count"] = video.get("recycled_count", 0) + 1
                    break
            self._save_content_database()
            
            return {
                "status": "success",
                "message": f"Se crearon {len(clips_info)} clips exitosamente",
                "clips": clips_info,
                "original_video": {
                    "id": video_id,
                    "title": video_data.get("title", ""),
                    "duration": video_duration
                }
            }
            
        except Exception as e:
            logger.error(f"Error al crear clips: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al crear clips: {str(e)}"
            }
    
    def _detect_scene_changes(self, video_path: str, threshold: float = 30.0) -> List[float]:
        """
        Detecta cambios de escena en un video.
        
        Args:
            video_path: Ruta al archivo de video
            threshold: Umbral para detectar cambios
            
        Returns:
            Lista de timestamps (segundos) donde hay cambios de escena
        """
        try:
            # Abrir video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Leer primer frame
            ret, prev_frame = cap.read()
            if not ret:
                return []
            
            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            
            scene_changes = []
            frame_count = 1
            
            # Procesar frames
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                # Convertir a escala de grises
                curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Calcular diferencia
                diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)
                mean_diff = np.mean(diff)
                
                # Detectar cambio de escena
                if mean_diff > threshold:
                    timestamp = frame_count / fps
                    scene_changes.append(timestamp)
                
                # Actualizar frame anterior
                prev_frame_gray = curr_frame_gray
                frame_count += 1
            
            # Liberar recursos
            cap.release()
            
            return scene_changes
            
        except Exception as e:
            logger.error(f"Error al detectar cambios de escena: {str(e)}")
            return []
    
    def transform_for_platform(self, 
                              content_id: str, 
                              target_platform: str,
                              transformation_type: str = "auto",
                              new_aspect_ratio: str = None,
                              new_duration: int = None,
                              add_cta: bool = False,
                              cta_text: str = None) -> Dict[str, Any]:
        """
        Transforma contenido existente para otra plataforma.
        
        Args:
            content_id: ID del contenido a transformar
            target_platform: Plataforma destino ('youtube', 'tiktok', 'instagram', etc.)
            transformation_type: Tipo de transformación ('auto', 'crop', 'resize', 'trim')
            new_aspect_ratio: Nueva relación de aspecto ('9:16', '16:9', '1:1', etc.)
            new_duration: Nueva duración en segundos
            add_cta: Añadir llamada a la acción
            cta_text: Texto de la llamada a la acción
            
        Returns:
            Información sobre el contenido transformado
        """
        try:
            # Buscar contenido en la base de datos
            content_data = None
            content_type = None
            
            # Determinar tipo de contenido por el prefijo del ID
            if content_id.startswith("video_"):
                for item in self.content_db["videos"]:
                    if item["id"] == content_id:
                        content_data = item
                        content_type = "video"
                        break
            else:
                return {
                    "status": "error",
                    "message": "Solo se pueden transformar videos actualmente"
                }
            
            if not content_data:
                return {
                    "status": "error",
                    "message": f"Contenido no encontrado: {content_id}"
                }
            
            # Verificar archivo
            file_path = content_data.get("file_path")
            if not file_path or not os.path.exists(file_path):
                return {
                    "status": "error",
                    "message": f"Archivo no encontrado: {file_path}"
                }
            
            # Determinar configuración según plataforma destino
            platform_configs = {
                "youtube": {
                    "aspect_ratio": "16:9",
                    "max_duration": 600,  # 10 minutos
                    "min_duration": 60,   # 1 minuto
                    "cta_position": "end"  # CTA al final
                },
                "youtube_shorts": {
                    "aspect_ratio": "9:16",
                    "max_duration": 60,   # 1 minuto
                    "min_duration": 15,   # 15 segundos
                    "cta_position": "middle"  # CTA en medio
                },
                "tiktok": {
                    "aspect_ratio": "9:16",
                    "max_duration": 60,   # 1 minuto
                    "min_duration": 15,   # 15 segundos
                    "cta_position": "middle"  # CTA en medio
                },
                "instagram": {
                    "aspect_ratio": "9:16",
                    "max_duration": 60,   # 1 minuto
                    "min_duration": 15,   # 15 segundos
                    "cta_position": "middle"  # CTA en medio
                },
                "instagram_post": {
                    "aspect_ratio": "1:1",
                    "max_duration": 60,   # 1 minuto
                    "min_duration": 3,    # 3 segundos
                    "cta_position": "end"  # CTA al final
                },
                "threads": {
                    "aspect_ratio": "9:16",
                    "max_duration": 300,  # 5 minutos
                    "min_duration": 5,    # 5 segundos
                    "cta_position": "middle"  # CTA en medio
                }
            }
            
            # Obtener configuración de la plataforma destino
            if target_platform not in platform_configs:
                return {
                    "status": "error",
                    "message": f"Plataforma no soportada: {target_platform}"
                }
            
            platform_config = platform_configs[target_platform]
            
            # Determinar parámetros de transformación
            target_aspect_ratio = new_aspect_ratio or platform_config["aspect_ratio"]
            target_duration = new_duration or platform_config["max_duration"]
            cta_position = platform_config["cta_position"]
            
            # Cargar video
            video = VideoFileClip(file_path)
            original_duration = video.duration
            original_width = video.w
            original_height = video.h
            original_aspect_ratio = f"{original_width}:{original_height}"
            
            # Transformar video
            transformed_video = video
            
            # 1. Ajustar relación de aspecto
            if target_aspect_ratio != original_aspect_ratio:
                # Parsear relaciones de aspecto
                target_w, target_h = map(int, target_aspect_ratio.split(':'))
                target_ratio = target_w / target_h
                original_ratio = original_width / original_height
                
                if transformation_type == "auto" or transformation_type == "crop":
                    # Recortar para ajustar a la nueva relación de aspecto
                    if target_ratio > original_ratio:
                        # El target es más ancho, recortar altura
                        new_h = original_width / target_ratio
                        crop_top = (original_height - new_h) / 2
                        transformed_video = video.crop(y1=crop_top, height=new_h)
                    else:
                        # El target es más alto, recortar ancho
                        new_w = original_height * target_ratio
                        crop_left = (original_width - new_w) / 2
                        transformed_video = video.crop(x1=crop_left, width=new_w)
                
                elif transformation_type == "resize":
                    # Redimensionar (puede distorsionar)
                    new_w = original_width
                    new_h = original_width / target_ratio
                    if new_h > original_height:
                        new_h = original_height
                        new_w = original_height * target_ratio
                    
                    transformed_video = video.resize(newsize=(new_w, new_h))
            
            # 2. Ajustar duración
            if target_duration < original_duration:
                if transformation_type == "auto" or transformation_type == "trim":
                    # Recortar duración
                    middle_point = original_duration / 2
                    start_time = max(0, middle_point - target_duration / 2)
                    end_time = min(original_duration, middle_point + target_duration / 2)
                    transformed_video = transformed_video.subclip(start_time, end_time)
            
            # 3. Añadir CTA si se solicita
            if add_cta and cta_text:
                # Crear clip de texto
                txt_clip = TextClip(cta_text, fontsize=30, color='white', bg_color='black',
                                   size=(transformed_video.w, transformed_video.h // 5))
                txt_clip = txt_clip.set_duration(3)  # 3 segundos de duración
                
                # Posicionar CTA según configuración
                if cta_position == "start":
                    # Al inicio
                    txt_clip = txt_clip.set_start(0)
                    transformed_video = CompositeVideoClip([transformed_video, txt_clip])
                
                elif cta_position == "end":
                    # Al final
                    txt_clip = txt_clip.set_start(transformed_video.duration - 3)
                    transformed_video = CompositeVideoClip([transformed_video, txt_clip])
                
                elif cta_position == "middle":
                    # En medio
                    middle_time = transformed_video.duration / 2
                    txt_clip = txt_clip.set_start(middle_time - 1.5)
                    transformed_video = CompositeVideoClip([transformed_video, txt_clip])
            
            # Generar nombre de archivo para el video transformado
            transformed_dir = os.path.join(self.uploads_path, "transformed")
            transformed_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_{target_platform}.mp4"
            transformed_path = os.path.join(transformed_dir, transformed_filename)
            
            # Guardar video transformado
            transformed_video.write_videofile(transformed_path, codec="libx264", audio_codec="aac")
            
            # Registrar video transformado en la base de datos
            transformed_data = {
                "title": f"{content_data.get('title', 'Video')} - {target_platform}",
                "description": f"Transformado de {content_data.get('title', 'Video original')} para {target_platform}",
                "platform": target_platform,
                "topic": content_data.get("topic", ""),
                "tags": content_data.get("tags", []),
                "file_path": transformed_path,
                "original_content_id": content_id,
                "transformation_type": transformation_type,
                "aspect_ratio": target_aspect_ratio,
                "duration": transformed_video.duration,
                "has_cta": add_cta,
                "cta_text": cta_text if add_cta else None,
                "performance": 0,  # Inicialmente sin datos de rendimiento
                "parent_content": {
                    "id": content_id,
                    "type": content_type,
                    "title": content_data.get("title", "")
                }
            }
            
            transformed_result = self.register_content("video", transformed_data)
            
            # Cerrar clips
            video.close()
            transformed_video.close()
            
                        # Actualizar estadísticas
            self.recycling_stats["transformations"] += 1
            # Estimar ahorro de costos (asumiendo $1 por minuto de contenido nuevo)
            self.recycling_stats["cost_savings"] += transformed_video.duration / 60 * 1.0
            self._save_recycling_stats()
            
            # Actualizar contador de reciclaje del contenido original
            for item in self.content_db[content_type + "s"]:
                if item["id"] == content_id:
                    item["recycled_count"] = item.get("recycled_count", 0) + 1
                    break
            self._save_content_database()
            
            return {
                "status": "success",
                "message": f"Contenido transformado exitosamente para {target_platform}",
                "transformed_content": {
                    "id": transformed_result.get("content_id", ""),
                    "title": transformed_data["title"],
                    "platform": target_platform,
                    "file_path": transformed_path,
                    "duration": transformed_video.duration,
                    "aspect_ratio": target_aspect_ratio
                },
                "original_content": {
                    "id": content_id,
                    "title": content_data.get("title", ""),
                    "platform": content_data.get("platform", "")
                }
            }
            
        except Exception as e:
            logger.error(f"Error al transformar contenido: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al transformar contenido: {str(e)}"
            }
    
    def reuse_assets(self,
                    asset_ids: List[str],
                    target_content_id: str = None,
                    new_content_type: str = None,
                    new_content_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Reutiliza assets existentes en contenido nuevo o existente.
        
        Args:
            asset_ids: Lista de IDs de assets a reutilizar
            target_content_id: ID del contenido donde reutilizar los assets (opcional)
            new_content_type: Tipo de contenido nuevo a crear (si no se especifica target_content_id)
            new_content_data: Datos para el nuevo contenido (si no se especifica target_content_id)
            
        Returns:
            Información sobre la reutilización de assets
        """
        try:
            # Verificar que se proporcionaron asset_ids
            if not asset_ids:
                return {
                    "status": "error",
                    "message": "No se proporcionaron IDs de assets para reutilizar"
                }
            
            # Buscar assets en la base de datos
            assets_data = []
            for asset_id in asset_ids:
                asset_found = False
                for asset in self.content_db["assets"]:
                    if asset["id"] == asset_id:
                        assets_data.append(asset)
                        asset_found = True
                        break
                
                if not asset_found:
                    return {
                        "status": "error",
                        "message": f"Asset no encontrado: {asset_id}"
                    }
            
            # Determinar si se reutilizará en contenido existente o nuevo
            if target_content_id:
                # Buscar contenido existente
                target_content = None
                target_content_type = None
                
                # Determinar tipo de contenido por el prefijo del ID
                if target_content_id.startswith("video_"):
                    for item in self.content_db["videos"]:
                        if item["id"] == target_content_id:
                            target_content = item
                            target_content_type = "video"
                            break
                elif target_content_id.startswith("script_"):
                    for item in self.content_db["scripts"]:
                        if item["id"] == target_content_id:
                            target_content = item
                            target_content_type = "script"
                            break
                
                if not target_content:
                    return {
                        "status": "error",
                        "message": f"Contenido destino no encontrado: {target_content_id}"
                    }
                
                # Añadir assets al contenido existente
                if "assets_used" not in target_content:
                    target_content["assets_used"] = []
                
                for asset in assets_data:
                    # Evitar duplicados
                    asset_already_used = False
                    for used_asset in target_content["assets_used"]:
                        if used_asset["id"] == asset["id"]:
                            asset_already_used = True
                            break
                    
                    if not asset_already_used:
                        target_content["assets_used"].append({
                            "id": asset["id"],
                            "title": asset.get("title", ""),
                            "type": asset.get("type", ""),
                            "file_path": asset.get("file_path", "")
                        })
                
                # Guardar cambios
                self._save_content_database()
                
                # Actualizar estadísticas
                self.recycling_stats["assets_reused"] += len(assets_data)
                self.recycling_stats["cost_savings"] += len(assets_data) * 5.0  # Asumiendo $5 por asset reutilizado
                self._save_recycling_stats()
                
                # Actualizar contador de reciclaje de los assets
                for asset_id in asset_ids:
                    for asset in self.content_db["assets"]:
                        if asset["id"] == asset_id:
                            asset["recycled_count"] = asset.get("recycled_count", 0) + 1
                            break
                self._save_content_database()
                
                return {
                    "status": "success",
                    "message": f"Se reutilizaron {len(assets_data)} assets en el contenido existente",
                    "target_content": {
                        "id": target_content_id,
                        "title": target_content.get("title", ""),
                        "type": target_content_type
                    },
                    "assets_used": [
                        {
                            "id": asset["id"],
                            "title": asset.get("title", "")
                        } for asset in assets_data
                    ]
                }
                
            elif new_content_type and new_content_data:
                # Crear nuevo contenido con los assets
                if new_content_type not in ["video", "script"]:
                    return {
                        "status": "error",
                        "message": f"Tipo de contenido no válido para reutilización de assets: {new_content_type}"
                    }
                
                # Añadir assets al nuevo contenido
                new_content_data["assets_used"] = [
                    {
                        "id": asset["id"],
                        "title": asset.get("title", ""),
                        "type": asset.get("type", ""),
                        "file_path": asset.get("file_path", "")
                    } for asset in assets_data
                ]
                
                # Registrar nuevo contenido
                new_content_result = self.register_content(new_content_type, new_content_data)
                
                if new_content_result["status"] != "success":
                    return new_content_result
                
                # Actualizar estadísticas
                self.recycling_stats["assets_reused"] += len(assets_data)
                self.recycling_stats["cost_savings"] += len(assets_data) * 5.0  # Asumiendo $5 por asset reutilizado
                self._save_recycling_stats()
                
                # Actualizar contador de reciclaje de los assets
                for asset_id in asset_ids:
                    for asset in self.content_db["assets"]:
                        if asset["id"] == asset_id:
                            asset["recycled_count"] = asset.get("recycled_count", 0) + 1
                            break
                self._save_content_database()
                
                return {
                    "status": "success",
                    "message": f"Se creó nuevo contenido reutilizando {len(assets_data)} assets",
                    "new_content": {
                        "id": new_content_result.get("content_id", ""),
                        "title": new_content_data.get("title", ""),
                        "type": new_content_type
                    },
                    "assets_used": [
                        {
                            "id": asset["id"],
                            "title": asset.get("title", "")
                        } for asset in assets_data
                    ]
                }
                
            else:
                return {
                    "status": "error",
                    "message": "Debe proporcionar un ID de contenido existente o datos para crear nuevo contenido"
                }
                
        except Exception as e:
            logger.error(f"Error al reutilizar assets: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al reutilizar assets: {str(e)}"
            }
    
    def get_recycling_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de reciclaje de contenido.
        
        Returns:
            Estadísticas de reciclaje
        """
        try:
            # Calcular estadísticas adicionales
            total_videos = len(self.content_db["videos"])
            total_assets = len(self.content_db["assets"])
            total_characters = len(self.content_db["characters"])
            total_scripts = len(self.content_db["scripts"])
            
            # Calcular porcentaje de contenido reciclado
            recycled_videos = sum(1 for v in self.content_db["videos"] if v.get("recycled_count", 0) > 0)
            recycled_assets = sum(1 for a in self.content_db["assets"] if a.get("recycled_count", 0) > 0)
            
            recycled_percentage = 0
            if total_videos + total_assets > 0:
                recycled_percentage = (recycled_videos + recycled_assets) / (total_videos + total_assets) * 100
            
            return {
                "status": "success",
                "basic_stats": self.recycling_stats,
                "detailed_stats": {
                    "total_content": {
                        "videos": total_videos,
                        "assets": total_assets,
                        "characters": total_characters,
                        "scripts": total_scripts,
                        "total": total_videos + total_assets + total_characters + total_scripts
                    },
                    "recycled_content": {
                        "videos": recycled_videos,
                        "assets": recycled_assets,
                        "recycled_percentage": round(recycled_percentage, 2)
                    },
                    "estimated_savings": {
                        "cost_savings": round(self.recycling_stats["cost_savings"], 2),
                        "time_savings": round(self.recycling_stats["cost_savings"] * 0.5, 2)  # Estimando 0.5 horas por cada $1 ahorrado
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas de reciclaje: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al obtener estadísticas de reciclaje: {str(e)}"
            }