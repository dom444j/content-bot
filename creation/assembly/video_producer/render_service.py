import os
import logging
import json
import time
import random
import shutil
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RenderService")

class RenderService:
    """
    Servicio de renderizado para videos que gestiona la cola de procesamiento
    y optimiza la salida final
    """
    
    def __init__(self, 
                config_path: str = None, 
                output_dir: str = "output/renders",
                temp_dir: str = "temp/renders",
                max_workers: int = 2):
        """
        Inicializa el servicio de renderizado
        
        Args:
            config_path: Ruta al archivo de configuración
            output_dir: Directorio de salida para los videos renderizados
            temp_dir: Directorio temporal para archivos intermedios
            max_workers: Número máximo de trabajos de renderizado simultáneos
        """
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        self.max_workers = max_workers
        
        # Crear directorios
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Cargar configuración
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        
        # Configuración por defecto
        self.default_format = self.config.get("default_format", "mp4")
        self.default_resolution = self.config.get("default_resolution", (1080, 1920))  # Vertical por defecto
        self.default_fps = self.config.get("default_fps", 30)
        self.default_bitrate = self.config.get("default_bitrate", "8000k")
        self.ffmpeg_path = self.config.get("ffmpeg_path", "ffmpeg")
        
        # Cola de renderizado
        self.render_queue = []
        self.queue_lock = threading.Lock()
        self.active_renders = {}
        self.render_results = {}
        
        # Executor para procesamiento paralelo
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"RenderService inicializado con {max_workers} workers")
    
    def add_to_queue(self, 
                    video_path: str, 
                    output_path: str = None,
                    format: str = None,
                    resolution: Tuple[int, int] = None,
                    fps: int = None,
                    bitrate: str = None,
                    watermark_path: str = None,
                    optimize_size: bool = True,
                    priority: int = 1,
                    callback: callable = None) -> str:
        """
        Añade un video a la cola de renderizado
        
        Args:
            video_path: Ruta al video a renderizar
            output_path: Ruta de salida para el video renderizado
            format: Formato de salida (mp4, mov, etc.)
            resolution: Resolución de salida (altura, anchura)
            fps: Frames por segundo
            bitrate: Tasa de bits
            watermark_path: Ruta a la imagen de marca de agua
            optimize_size: Optimizar tamaño del archivo
            priority: Prioridad en la cola (mayor número = mayor prioridad)
            callback: Función a llamar cuando se complete el renderizado
            
        Returns:
            ID del trabajo de renderizado
        """
        try:
            # Verificar que existe el video
            if not os.path.exists(video_path):
                logger.error(f"No se encontró el video: {video_path}")
                return None
            
            # Generar ID único para este trabajo
            job_id = f"render_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                base_name = os.path.basename(video_path)
                name, ext = os.path.splitext(base_name)
                format = format or self.default_format
                output_path = os.path.join(self.output_dir, f"{name}_rendered.{format}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Configurar parámetros de renderizado
            render_params = {
                "video_path": video_path,
                "output_path": output_path,
                "format": format or self.default_format,
                "resolution": resolution or self.default_resolution,
                "fps": fps or self.default_fps,
                "bitrate": bitrate or self.default_bitrate,
                "watermark_path": watermark_path,
                "optimize_size": optimize_size,
                "priority": priority,
                "callback": callback,
                "status": "pending",
                                "progress": 0,
                "start_time": None,
                "end_time": None,
                "error": None
            }
            
            # Añadir a la cola con bloqueo para evitar problemas de concurrencia
            with self.queue_lock:
                self.render_queue.append({
                    "id": job_id,
                    "params": render_params
                })
                # Ordenar cola por prioridad (mayor primero)
                self.render_queue.sort(key=lambda x: x["params"]["priority"], reverse=True)
                self.render_results[job_id] = {
                    "status": "pending",
                    "output_path": None,
                    "error": None
                }
            
            logger.info(f"Video añadido a la cola de renderizado: {job_id}")
            
            # Iniciar procesamiento si no hay suficientes trabajos activos
            self._process_queue()
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error al añadir video a la cola: {str(e)}")
            return None
    
    def get_queue_status(self) -> List[Dict[str, Any]]:
        """
        Obtiene el estado actual de la cola de renderizado
        
        Returns:
            Lista de trabajos en la cola con su estado
        """
        with self.queue_lock:
            # Combinar información de la cola y trabajos activos
            status = []
            
            # Añadir trabajos en cola
            for job in self.render_queue:
                job_id = job["id"]
                params = job["params"]
                status.append({
                    "id": job_id,
                    "video_path": params["video_path"],
                    "output_path": params["output_path"],
                    "status": "pending",
                    "progress": 0,
                    "priority": params["priority"]
                })
            
            # Añadir trabajos activos
            for job_id, job_info in self.active_renders.items():
                status.append({
                    "id": job_id,
                    "video_path": job_info["params"]["video_path"],
                    "output_path": job_info["params"]["output_path"],
                    "status": "processing",
                    "progress": job_info["progress"],
                    "priority": job_info["params"]["priority"],
                    "start_time": job_info["start_time"]
                })
            
            # Añadir trabajos completados (últimos 10)
            completed = []
            for job_id, result in self.render_results.items():
                if result["status"] in ["completed", "failed"]:
                    completed.append({
                        "id": job_id,
                        "status": result["status"],
                        "output_path": result["output_path"],
                        "error": result["error"]
                    })
            
            # Ordenar por ID (más reciente primero) y limitar a 10
            completed.sort(key=lambda x: x["id"], reverse=True)
            status.extend(completed[:10])
            
            return status
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado de un trabajo específico
        
        Args:
            job_id: ID del trabajo
            
        Returns:
            Información del estado del trabajo
        """
        # Buscar en trabajos pendientes
        with self.queue_lock:
            for job in self.render_queue:
                if job["id"] == job_id:
                    return {
                        "id": job_id,
                        "status": "pending",
                        "progress": 0,
                        "video_path": job["params"]["video_path"],
                        "output_path": job["params"]["output_path"]
                    }
            
            # Buscar en trabajos activos
            if job_id in self.active_renders:
                job_info = self.active_renders[job_id]
                return {
                    "id": job_id,
                    "status": "processing",
                    "progress": job_info["progress"],
                    "video_path": job_info["params"]["video_path"],
                    "output_path": job_info["params"]["output_path"],
                    "start_time": job_info["start_time"]
                }
            
            # Buscar en resultados
            if job_id in self.render_results:
                result = self.render_results[job_id]
                return {
                    "id": job_id,
                    "status": result["status"],
                    "output_path": result["output_path"],
                    "error": result["error"]
                }
            
            # No encontrado
            return {
                "id": job_id,
                "status": "not_found",
                "error": "Trabajo de renderizado no encontrado"
            }
    
    def cancel_job(self, job_id: str) -> bool:
        """
        Cancela un trabajo de renderizado
        
        Args:
            job_id: ID del trabajo a cancelar
            
        Returns:
            True si se canceló correctamente, False en caso contrario
        """
        with self.queue_lock:
            # Buscar en cola de pendientes
            for i, job in enumerate(self.render_queue):
                if job["id"] == job_id:
                    # Eliminar de la cola
                    self.render_queue.pop(i)
                    # Actualizar resultados
                    self.render_results[job_id] = {
                        "status": "cancelled",
                        "output_path": None,
                        "error": "Trabajo cancelado por el usuario"
                    }
                    logger.info(f"Trabajo cancelado: {job_id}")
                    return True
            
            # No se puede cancelar un trabajo activo (por ahora)
            if job_id in self.active_renders:
                logger.warning(f"No se puede cancelar un trabajo activo: {job_id}")
                return False
            
            # No encontrado
            logger.warning(f"Trabajo no encontrado para cancelar: {job_id}")
            return False
    
    def clear_completed_jobs(self) -> int:
        """
        Limpia los trabajos completados de la lista de resultados
        
        Returns:
            Número de trabajos eliminados
        """
        count = 0
        with self.queue_lock:
            to_remove = []
            for job_id, result in self.render_results.items():
                if result["status"] in ["completed", "failed", "cancelled"]:
                    to_remove.append(job_id)
            
            for job_id in to_remove:
                del self.render_results[job_id]
                count += 1
        
        logger.info(f"Se eliminaron {count} trabajos completados de la lista")
        return count
    
    def render_video(self, 
                    video_path: str, 
                    output_path: str = None,
                    format: str = None,
                    resolution: Tuple[int, int] = None,
                    fps: int = None,
                    bitrate: str = None,
                    watermark_path: str = None,
                    optimize_size: bool = True) -> str:
        """
        Renderiza un video directamente (sin usar la cola)
        
        Args:
            video_path: Ruta al video a renderizar
            output_path: Ruta de salida para el video renderizado
            format: Formato de salida (mp4, mov, etc.)
            resolution: Resolución de salida (altura, anchura)
            fps: Frames por segundo
            bitrate: Tasa de bits
            watermark_path: Ruta a la imagen de marca de agua
            optimize_size: Optimizar tamaño del archivo
            
        Returns:
            Ruta al video renderizado o None si hay error
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
                format = format or self.default_format
                output_path = os.path.join(self.output_dir, f"{name}_rendered.{format}")
            
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Configurar parámetros
            format = format or self.default_format
            resolution = resolution or self.default_resolution
            fps = fps or self.default_fps
            bitrate = bitrate or self.default_bitrate
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Redimensionar si es necesario
            if resolution and (video.h != resolution[0] or video.w != resolution[1]):
                video = video.resize(height=resolution[0], width=resolution[1])
            
            # Añadir marca de agua si se proporciona
            if watermark_path and os.path.exists(watermark_path):
                watermark = ImageClip(watermark_path)
                
                # Redimensionar marca de agua (10% del ancho del video)
                watermark_width = int(video.w * 0.1)
                watermark = watermark.resize(width=watermark_width)
                
                # Posicionar en la esquina inferior derecha
                watermark = watermark.set_position(('right', 'bottom'))
                
                # Establecer duración
                watermark = watermark.set_duration(video.duration)
                
                # Componer video con marca de agua
                video = CompositeVideoClip([video, watermark])
            
            # Renderizar video
            if optimize_size:
                # Usar FFmpeg directamente para mejor control de la compresión
                temp_output = os.path.join(self.temp_dir, f"temp_{int(time.time())}.{format}")
                
                # Primero guardar con moviepy
                video.write_videofile(temp_output, fps=fps, codec="libx264", 
                                     audio_codec="aac", threads=4)
                
                # Luego optimizar con FFmpeg
                self._optimize_video(temp_output, output_path, bitrate)
                
                # Eliminar archivo temporal
                if os.path.exists(temp_output):
                    os.remove(temp_output)
            else:
                # Renderizar directamente con moviepy
                video.write_videofile(output_path, fps=fps, codec="libx264", 
                                     audio_codec="aac", bitrate=bitrate, threads=4)
            
            logger.info(f"Video renderizado exitosamente: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error al renderizar video: {str(e)}")
            return None
    
    def start_processing(self) -> bool:
        """
        Inicia el procesamiento de la cola de renderizado
        
        Returns:
            True si se inició correctamente, False en caso contrario
        """
        try:
            # Iniciar procesamiento
            self._process_queue()
            return True
        except Exception as e:
            logger.error(f"Error al iniciar procesamiento: {str(e)}")
            return False
    
    def stop_processing(self) -> bool:
        """
        Detiene el procesamiento de la cola de renderizado
        
        Returns:
            True si se detuvo correctamente, False en caso contrario
        """
        try:
            # Detener executor
            self.executor.shutdown(wait=False)
            
            # Reiniciar executor
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
            # Marcar trabajos activos como fallidos
            with self.queue_lock:
                for job_id, job_info in self.active_renders.items():
                    self.render_results[job_id] = {
                        "status": "failed",
                        "output_path": None,
                        "error": "Procesamiento detenido por el usuario"
                    }
                
                # Limpiar trabajos activos
                self.active_renders = {}
            
            logger.info("Procesamiento de cola detenido")
            return True
        except Exception as e:
            logger.error(f"Error al detener procesamiento: {str(e)}")
            return False
    
    def _process_queue(self):
        """Procesa la cola de renderizado"""
        with self.queue_lock:
            # Verificar si hay capacidad para más trabajos
            available_slots = self.max_workers - len(self.active_renders)
            
            if available_slots <= 0 or not self.render_queue:
                return
            
            # Procesar tantos trabajos como slots disponibles
            for _ in range(min(available_slots, len(self.render_queue))):
                if not self.render_queue:
                    break
                
                # Obtener siguiente trabajo
                job = self.render_queue.pop(0)
                job_id = job["id"]
                params = job["params"]
                
                # Marcar como activo
                params["start_time"] = time.time()
                self.active_renders[job_id] = {
                    "params": params,
                    "progress": 0,
                    "start_time": params["start_time"]
                }
                
                # Enviar a procesar
                self.executor.submit(self._render_job, job_id, params)
                
                logger.info(f"Iniciando renderizado de {job_id}")
    
    def _render_job(self, job_id: str, params: Dict[str, Any]):
        """
        Renderiza un trabajo específico
        
        Args:
            job_id: ID del trabajo
            params: Parámetros de renderizado
        """
        try:
            # Extraer parámetros
            video_path = params["video_path"]
            output_path = params["output_path"]
            format = params["format"]
            resolution = params["resolution"]
            fps = params["fps"]
            bitrate = params["bitrate"]
            watermark_path = params["watermark_path"]
            optimize_size = params["optimize_size"]
            callback = params["callback"]
            
            # Actualizar progreso
            self._update_job_progress(job_id, 10)
            
            # Cargar video
            video = VideoFileClip(video_path)
            
            # Actualizar progreso
            self._update_job_progress(job_id, 20)
            
            # Redimensionar si es necesario
            if resolution and (video.h != resolution[0] or video.w != resolution[1]):
                video = video.resize(height=resolution[0], width=resolution[1])
            
            # Actualizar progreso
            self._update_job_progress(job_id, 30)
            
            # Añadir marca de agua si se proporciona
            if watermark_path and os.path.exists(watermark_path):
                watermark = ImageClip(watermark_path)
                
                # Redimensionar marca de agua (10% del ancho del video)
                watermark_width = int(video.w * 0.1)
                watermark = watermark.resize(width=watermark_width)
                
                # Posicionar en la esquina inferior derecha
                watermark = watermark.set_position(('right', 'bottom'))
                
                # Establecer duración
                watermark = watermark.set_duration(video.duration)
                
                # Componer video con marca de agua
                video = CompositeVideoClip([video, watermark])
            
            # Actualizar progreso
            self._update_job_progress(job_id, 40)
            
            # Renderizar video
            if optimize_size:
                # Usar FFmpeg directamente para mejor control de la compresión
                temp_output = os.path.join(self.temp_dir, f"temp_{job_id}.{format}")
                
                # Primero guardar con moviepy
                video.write_videofile(temp_output, fps=fps, codec="libx264", 
                                     audio_codec="aac", threads=4)
                
                # Actualizar progreso
                self._update_job_progress(job_id, 70)
                
                # Luego optimizar con FFmpeg
                self._optimize_video(temp_output, output_path, bitrate)
                
                # Eliminar archivo temporal
                if os.path.exists(temp_output):
                    os.remove(temp_output)
            else:
                # Renderizar directamente con moviepy
                video.write_videofile(output_path, fps=fps, codec="libx264", 
                                     audio_codec="aac", bitrate=bitrate, threads=4)
            
            # Actualizar progreso
            self._update_job_progress(job_id, 100)
            
            # Marcar como completado
            with self.queue_lock:
                self.render_results[job_id] = {
                    "status": "completed",
                    "output_path": output_path,
                    "error": None
                }
                
                # Eliminar de activos
                if job_id in self.active_renders:
                    self.active_renders[job_id]["progress"] = 100
                    self.active_renders[job_id]["end_time"] = time.time()
                    # No eliminar para mantener historial hasta que se limpie
            
            logger.info(f"Renderizado completado: {job_id} -> {output_path}")
            
            # Llamar callback si existe
            if callback and callable(callback):
                try:
                    callback(job_id, output_path)
                except Exception as e:
                    logger.error(f"Error en callback de {job_id}: {str(e)}")
            
            # Procesar siguiente trabajo en cola
            self._process_queue()
            
        except Exception as e:
            logger.error(f"Error al renderizar {job_id}: {str(e)}")
            
            # Marcar como fallido
            with self.queue_lock:
                self.render_results[job_id] = {
                    "status": "failed",
                    "output_path": None,
                    "error": str(e)
                }
                
                # Actualizar activos
                if job_id in self.active_renders:
                    self.active_renders[job_id]["end_time"] = time.time()
                    # No eliminar para mantener historial hasta que se limpie
            
            # Procesar siguiente trabajo en cola
            self._process_queue()
    
    def _update_job_progress(self, job_id: str, progress: int):
        """
        Actualiza el progreso de un trabajo
        
        Args:
            job_id: ID del trabajo
            progress: Porcentaje de progreso (0-100)
        """
        with self.queue_lock:
            if job_id in self.active_renders:
                self.active_renders[job_id]["progress"] = progress
    
    def _optimize_video(self, input_path: str, output_path: str, bitrate: str):
        """
        Optimiza un video usando FFmpeg
        
        Args:
            input_path: Ruta al video de entrada
            output_path: Ruta al video de salida
            bitrate: Tasa de bits
        """
        try:
            # Comando FFmpeg para optimizar
            cmd = [
                self.ffmpeg_path,
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "slow",  # Mejor compresión, más lento
                "-crf", "23",       # Factor de calidad constante (18-28, menor = mejor)
                "-b:v", bitrate,
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",  # Optimizar para streaming
                "-y",  # Sobrescribir si existe
                output_path
            ]
            
            # Ejecutar comando
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Esperar a que termine
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error al optimizar video: {stderr}")
                # Si falla, copiar el archivo original
                shutil.copy(input_path, output_path)
            else:
                logger.info(f"Video optimizado correctamente: {output_path}")
            
        except Exception as e:
            logger.error(f"Error al optimizar video: {str(e)}")
            # Si falla, copiar el archivo original
            shutil.copy(input_path, output_path)
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        try:
            # Detener executor
            self.executor.shutdown(wait=False)
            
            # Limpiar archivos temporales
            for file in os.listdir(self.temp_dir):
                if file.startswith("temp_"):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except:
                        pass
        except:
            pass