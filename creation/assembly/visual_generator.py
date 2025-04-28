import os
import time
import json
import random
import logging
import requests
from typing import Dict, List, Tuple, Any, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import base64
import hashlib
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VisualGenerator")

class VisualGenerator:
    """
    Generador de contenido visual utilizando múltiples servicios de IA
    
    Soporta:
    - Leonardo.ai (gratuito y premium)
    - Stable Diffusion XL (local)
    - Midjourney (vía API)
    - RunwayML (imágenes y video)
    - Canva Pro (miniaturas)
    
    Con caché inteligente para reutilización de assets
    """
    
    def __init__(self, config_path: str = "config/platforms.json", knowledge_base=None):
        """
        Inicializa el generador de visuales
        
        Args:
            config_path: Ruta al archivo de configuración
            knowledge_base: Instancia de la base de conocimiento
        """
        self.config = self._load_config(config_path)
        self.knowledge_base = knowledge_base
        self.cache_dir = self.config.get("cache_directory", "uploads/cache/visuals")
        self.output_dir = self.config.get("output_directory", "uploads/visuals")
        
        # Crear directorios si no existen
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cargar modelos locales si están configurados
        self.sd_model = None
        if self.config.get("apis", {}).get("stable_diffusion", {}).get("local_enabled", False):
            self._load_stable_diffusion()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Carga la configuración desde un archivo JSON
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Returns:
            Diccionario con la configuración
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # Configuración por defecto para generación visual
            default_config = {
                "apis": {
                    "leonardo": {
                        "enabled": True,
                        "api_key": "",
                        "free_tier": True,
                        "daily_limit": 150
                    },
                    "stable_diffusion": {
                        "local_enabled": True,
                        "api_enabled": False,
                        "api_url": "http://localhost:7860",
                        "api_key": ""
                    },
                    "midjourney": {
                        "enabled": False,
                        "api_key": ""
                    },
                    "runway": {
                        "enabled": False,
                        "api_key": ""
                    },
                    "canva": {
                        "enabled": False,
                        "api_key": ""
                    }
                },
                "cache_directory": "uploads/cache/visuals",
                "output_directory": "uploads/visuals",
                "cache_enabled": True,
                "cache_ttl_days": 30,
                "default_style": "photorealistic",
                "default_aspect_ratio": "16:9",
                "default_resolution": "1024x576",
                "default_quality": 90,
                "watermark_enabled": False,
                "watermark_text": "",
                "watermark_opacity": 0.3
            }
            
            # Combinar configuración por defecto con la cargada
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config.get(key), dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            # Devolver configuración por defecto
            return {
                "apis": {
                    "leonardo": {"enabled": True, "free_tier": True, "daily_limit": 150},
                    "stable_diffusion": {"local_enabled": True}
                },
                "cache_directory": "uploads/cache/visuals",
                "output_directory": "uploads/visuals",
                "cache_enabled": True
            }
    
    def _load_stable_diffusion(self):
        """Carga el modelo local de Stable Diffusion si está habilitado"""
        try:
            # Intentar importar las dependencias necesarias
            import torch
            from diffusers import StableDiffusionXLPipeline
            
            # Verificar si hay GPU disponible
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                logger.warning("No GPU available, Stable Diffusion will be slow")
            
            # Cargar modelo SDXL
            model_id = "stabilityai/stable-diffusion-xl-base-1.0"
            self.sd_model = StableDiffusionXLPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                use_safetensors=True
            )
            self.sd_model = self.sd_model.to(device)
            
            # Optimizar para inferencia
            if device == "cuda":
                self.sd_model.enable_xformers_memory_efficient_attention()
            
            logger.info(f"Loaded Stable Diffusion XL model on {device}")
            
        except ImportError as e:
            logger.error(f"Failed to import Stable Diffusion dependencies: {str(e)}")
            logger.info("Please install with: pip install torch diffusers transformers accelerate xformers")
            self.sd_model = None
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion model: {str(e)}")
            self.sd_model = None
    
    def generate_image(self, 
                     prompt: str, 
                     negative_prompt: str = None,
                     style: str = None,
                     aspect_ratio: str = None,
                     service: str = None,
                     output_path: str = None,
                     use_cache: bool = True) -> str:
        """
        Genera una imagen basada en un prompt
        
        Args:
            prompt: Descripción textual de la imagen
            negative_prompt: Elementos a evitar en la imagen
            style: Estilo visual (photorealistic, anime, digital_art, etc.)
            aspect_ratio: Relación de aspecto (1:1, 16:9, 9:16, etc.)
            service: Servicio a utilizar (leonardo, stable_diffusion, midjourney, runway)
            output_path: Ruta de salida para la imagen
            use_cache: Si se debe buscar en caché antes de generar
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Configurar valores por defecto
            style = style or self.config.get("default_style", "photorealistic")
            aspect_ratio = aspect_ratio or self.config.get("default_aspect_ratio", "16:9")
            
            # Determinar resolución basada en relación de aspecto
            width, height = self._get_resolution_from_aspect_ratio(aspect_ratio)
            
            # Enriquecer prompt con estilo si no está especificado
            if style and style not in prompt.lower():
                prompt = f"{prompt}, {style} style"
            
            # Verificar caché si está habilitado
            if use_cache and self.config.get("cache_enabled", True):
                cache_path = self._check_cache(prompt, negative_prompt, style, aspect_ratio)
                if cache_path:
                    logger.info(f"Using cached image: {cache_path}")
                    
                    # Si se especificó una ruta de salida, copiar la imagen
                    if output_path:
                        import shutil
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        shutil.copy2(cache_path, output_path)
                        return output_path
                    
                    return cache_path
            
            # Determinar servicio a utilizar
            if not service:
                service = self._select_best_service()
            
            # Generar imagen con el servicio seleccionado
            if service == "leonardo":
                image_path = self._generate_with_leonardo(prompt, negative_prompt, width, height, output_path)
            elif service == "stable_diffusion":
                image_path = self._generate_with_stable_diffusion(prompt, negative_prompt, width, height, output_path)
            elif service == "midjourney":
                image_path = self._generate_with_midjourney(prompt, negative_prompt, width, height, output_path)
            elif service == "runway":
                image_path = self._generate_with_runway(prompt, negative_prompt, width, height, output_path)
            else:
                # Fallback a Stable Diffusion local o Leonardo
                if self.sd_model:
                    image_path = self._generate_with_stable_diffusion(prompt, negative_prompt, width, height, output_path)
                else:
                    image_path = self._generate_with_leonardo(prompt, negative_prompt, width, height, output_path)
            
            # Guardar en caché si está habilitado
            if self.config.get("cache_enabled", True) and image_path:
                self._add_to_cache(image_path, prompt, negative_prompt, style, aspect_ratio)
            
            # Añadir marca de agua si está configurado
            if self.config.get("watermark_enabled", False) and image_path:
                watermark_text = self.config.get("watermark_text", "")
                if watermark_text:
                    self._add_watermark(image_path, watermark_text)
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            return None
    
    def generate_image_variation(self,
                               image_path: str,
                               prompt: str = None,
                               strength: float = 0.7,
                               service: str = None,
                               output_path: str = None) -> str:
        """
        Genera una variación de una imagen existente
        
        Args:
            image_path: Ruta a la imagen base
            prompt: Prompt adicional para guiar la variación
            strength: Intensidad de la variación (0.0-1.0)
            service: Servicio a utilizar
            output_path: Ruta de salida para la imagen
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Verificar que la imagen existe
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Determinar servicio a utilizar
            if not service:
                service = self._select_best_service(variation=True)
            
            # Generar variación con el servicio seleccionado
            if service == "leonardo":
                return self._generate_variation_with_leonardo(image_path, prompt, strength, output_path)
            elif service == "stable_diffusion":
                return self._generate_variation_with_stable_diffusion(image_path, prompt, strength, output_path)
            elif service == "runway":
                return self._generate_variation_with_runway(image_path, prompt, strength, output_path)
            else:
                # Fallback a Stable Diffusion local o Leonardo
                if self.sd_model:
                    return self._generate_variation_with_stable_diffusion(image_path, prompt, strength, output_path)
                else:
                    return self._generate_variation_with_leonardo(image_path, prompt, strength, output_path)
                
        except Exception as e:
            logger.error(f"Error generating image variation: {str(e)}")
            return None
    
    def generate_character_image(self,
                               character_name: str,
                               description: str = None,
                               style: str = None,
                               outfit: str = None,
                               emotion: str = "neutral",
                               output_path: str = None) -> str:
        """
        Genera una imagen de un personaje
        
        Args:
            character_name: Nombre del personaje
            description: Descripción adicional
            style: Estilo visual
            outfit: Vestimenta
            emotion: Emoción (happy, sad, angry, etc.)
            output_path: Ruta de salida para la imagen
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Obtener información del personaje desde la base de conocimiento
            character_info = {}
            if self.knowledge_base:
                character_info = self.knowledge_base.get_character(character_name) or {}
            
            # Combinar información del personaje con parámetros
            char_description = description or character_info.get("description", "")
            char_style = style or character_info.get("style", self.config.get("default_style", "photorealistic"))
            char_outfit = outfit or character_info.get("outfit", "")
            
            # Construir prompt completo
            prompt_parts = [
                f"Portrait of {character_name}",
                char_description,
                f"wearing {char_outfit}" if char_outfit else "",
                f"with {emotion} expression" if emotion and emotion != "neutral" else "",
                f"{char_style} style"
            ]
            
            prompt = ", ".join([p for p in prompt_parts if p])
            
            # Generar imagen con relación de aspecto 3:4 (buena para retratos)
            return self.generate_image(
                prompt=prompt,
                style=char_style,
                aspect_ratio="3:4",
                output_path=output_path
            )
            
        except Exception as e:
            logger.error(f"Error generating character image: {str(e)}")
            return None
    
    def generate_scene_image(self,
                           description: str,
                           style: str = None,
                           mood: str = None,
                           time_of_day: str = None,
                           output_path: str = None) -> str:
        """
        Genera una imagen de escena o fondo
        
        Args:
            description: Descripción de la escena
            style: Estilo visual
            mood: Estado de ánimo de la escena
            time_of_day: Momento del día
            output_path: Ruta de salida para la imagen
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Configurar valores por defecto
            style = style or self.config.get("default_style", "photorealistic")
            
            # Construir prompt completo
            prompt_parts = [
                description,
                f"during {time_of_day}" if time_of_day else "",
                f"{mood} mood" if mood else "",
                f"{style} style"
            ]
            
            prompt = ", ".join([p for p in prompt_parts if p])
            
            # Generar imagen con relación de aspecto 16:9 (buena para escenas)
            return self.generate_image(
                prompt=prompt,
                style=style,
                aspect_ratio="16:9",
                output_path=output_path
            )
            
        except Exception as e:
            logger.error(f"Error generating scene image: {str(e)}")
            return None
    
    def generate_storyboard(self,
                          scenes: List[Dict[str, str]],
                          style: str = None,
                          output_dir: str = None) -> List[str]:
        """
        Genera un storyboard a partir de descripciones de escenas
        
        Args:
            scenes: Lista de diccionarios con descripciones de escenas
            style: Estilo visual común para todas las escenas
            output_dir: Directorio de salida
            
        Returns:
            Lista de rutas a las imágenes generadas
        """
        try:
            # Configurar valores por defecto
            style = style or self.config.get("default_style", "photorealistic")
            
            if not output_dir:
                timestamp = int(time.time())
                output_dir = os.path.join(self.output_dir, f"storyboard_{timestamp}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar imágenes para cada escena
            image_paths = []
            
            for i, scene in enumerate(scenes):
                description = scene.get("description", "")
                mood = scene.get("mood")
                time_of_day = scene.get("time_of_day")
                
                # Generar nombre de archivo
                filename = f"scene_{i+1:02d}.png"
                output_path = os.path.join(output_dir, filename)
                
                # Generar imagen de escena
                image_path = self.generate_scene_image(
                    description=description,
                    style=style,
                    mood=mood,
                    time_of_day=time_of_day,
                    output_path=output_path
                )
                
                if image_path:
                    # Añadir texto de descripción a la imagen
                    self._add_caption_to_image(image_path, description)
                    image_paths.append(image_path)
                    
                    # Guardar metadatos
                    metadata_path = os.path.join(output_dir, f"scene_{i+1:02d}_metadata.json")
                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(scene, f, ensure_ascii=False, indent=2)
            
            # Generar archivo de metadatos del storyboard completo
            metadata_path = os.path.join(output_dir, "storyboard_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({
                    "scenes": scenes,
                    "style": style,
                    "created_at": time.time(),
                    "image_paths": image_paths
                }, f, ensure_ascii=False, indent=2)
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Error generating storyboard: {str(e)}")
            return []
    
    def generate_animation(self,
                         prompt: str,
                         duration: float = 3.0,
                         fps: int = 24,
                         style: str = None,
                         aspect_ratio: str = None,
                         output_path: str = None) -> str:
        """
        Genera una animación corta basada en un prompt
        
        Args:
            prompt: Descripción textual de la animación
            duration: Duración en segundos
            fps: Frames por segundo
            style: Estilo visual
            aspect_ratio: Relación de aspecto
            output_path: Ruta de salida para la animación
            
        Returns:
            Ruta a la animación generada
        """
        try:
            # Verificar si RunwayML está habilitado
            runway_config = self.config.get("apis", {}).get("runway", {})
            if not runway_config.get("enabled", False):
                logger.warning("RunwayML API is not enabled for animations")
                return None
            
            # Configurar valores por defecto
            style = style or self.config.get("default_style", "photorealistic")
            aspect_ratio = aspect_ratio or self.config.get("default_aspect_ratio", "16:9")
            
            # Determinar resolución basada en relación de aspecto
            width, height = self._get_resolution_from_aspect_ratio(aspect_ratio)
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"animation_{timestamp}_{random_id}.mp4"
                output_path = os.path.join(self.output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Enriquecer prompt con estilo si no está especificado
            if style and style not in prompt.lower():
                prompt = f"{prompt}, {style} style"
            
            # Llamar a la API de RunwayML para generar la animación
            api_key = runway_config.get("api_key", "")
            api_url = "https://api.runwayml.com/v1/generationVideo"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "negative_prompt": "poor quality, blurry, distorted",
                "width": width,
                "height": height,
                "num_frames": int(duration * fps),
                "fps": fps
            }
            
            # Simulación de llamada a API (en implementación real se usaría requests.post)
            logger.info(f"Generating animation with RunwayML: {prompt}")
            
            # En una implementación real, aquí se llamaría a la API de RunwayML
            # response = requests.post(api_url, headers=headers, json=payload)
            # if response.status_code == 200:
            #     with open(output_path, "wb") as f:
            #         f.write(response.content)
            #     return output_path
            
            # Simulación: generar una imagen estática como fallback
            image_path = self.generate_image(
                prompt=prompt,
                style=style,
                aspect_ratio=aspect_ratio
            )
            
            if image_path:
                logger.warning("Animation generation not implemented, returning static image")
                return image_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating animation: {str(e)}")
            return None
    
    def _select_best_service(self, variation: bool = False) -> str:
        """
        Selecciona el mejor servicio disponible para generación de imágenes
        
        Args:
            variation: Si se trata de una variación de imagen
            
        Returns:
            Nombre del servicio seleccionado
        """
        apis_config = self.config.get("apis", {})
        
        # Verificar servicios habilitados
        available_services = []
        
        # Stable Diffusion local
        if self.sd_model and apis_config.get("stable_diffusion", {}).get("local_enabled", False):
            available_services.append(("stable_diffusion", 3))  # Prioridad alta si está disponible localmente
        
        # Stable Diffusion API
        elif apis_config.get("stable_diffusion", {}).get("api_enabled", False):
            available_services.append(("stable_diffusion", 2))
        
        # Leonardo.ai
        if apis_config.get("leonardo", {}).get("enabled", False):
            # Verificar límite diario en modo gratuito
            if apis_config.get("leonardo", {}).get("free_tier", True):
                daily_limit = apis_config.get("leonardo", {}).get("daily_limit", 150)
                # Prioridad media si quedan generaciones gratuitas
                available_services.append(("leonardo", 1))
            else:
                # Prioridad alta si es cuenta premium
                available_services.append(("leonardo", 3))
        
        # Midjourney (solo para generación, no variaciones)
        if not variation and apis_config.get("midjourney", {}).get("enabled", False):
            available_services.append(("midjourney", 2))
        
        # RunwayML
        if apis_config.get("runway", {}).get("enabled", False):
            available_services.append(("runway", 2))
        
        # Seleccionar servicio con mayor prioridad
        if available_services:
            available_services.sort(key=lambda x: x[1], reverse=True)
            return available_services[0][0]
        
        # Fallback a Leonardo (asumiendo que siempre está disponible en modo gratuito)
        return "leonardo"
    
    def _get_resolution_from_aspect_ratio(self, aspect_ratio: str) -> Tuple[int, int]:
        """
        Determina la resolución basada en la relación de aspecto
        
        Args:
            aspect_ratio: Relación de aspecto (ej: 16:9, 1:1, 9:16)
            
        Returns:
            Tupla (ancho, alto)
        """
        # Resoluciones predefinidas
        resolutions = {
            "1:1": (1024, 1024),
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "4:3": (1024, 768),
            "3:4": (768, 1024),
            "3:2": (1024, 683),
            "2:3": (683, 1024)
        }
        
        # Usar resolución predefinida si está disponible
        if aspect_ratio in resolutions:
            return resolutions[aspect_ratio]
        
        # Parsear relación personalizada
        try:
            w, h = aspect_ratio.split(":")
            w, h = int(w), int(h)
            
            # Mantener área total similar a 1024x1024
            scale = (1024 * 1024 / (w * h)) ** 0.5
            width = int(w * scale)
            height = int(h * scale)
            
            # Asegurar que ambas dimensiones son múltiplos de 8 (requerido por algunos modelos)
            width = (width // 8) * 8
            height = (height // 8) * 8
            
            return (width, height)
            
        except Exception:
            # Fallback a 1:1
            return (1024, 1024)
    
    def _check_cache(self, prompt: str, negative_prompt: str, style: str, aspect_ratio: str) -> Optional[str]:
        """
        Verifica si existe una imagen en caché para los parámetros dados
        
        Args:
            prompt: Prompt de la imagen
            negative_prompt: Prompt negativo
            style: Estilo visual
            aspect_ratio: Relación de aspecto
            
        Returns:
            Ruta a la imagen en caché o None si no existe
        """
        if not self.config.get("cache_enabled", True):
            return None
        
        # Generar hash de los parámetros
        cache_key = self._generate_cache_key(prompt, negative_prompt, style, aspect_ratio)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.png")
        
        # Verificar si existe el archivo
        if os.path.exists(cache_file):
            # Verificar TTL
            cache_ttl_days = self.config.get("cache_ttl_days", 30)
            file_age_days = (time.time() - os.path.getmtime(cache_file)) / (60 * 60 * 24)
            
            if file_age_days <= cache_ttl_days:
                return cache_file
            else:
                # Eliminar archivo expirado
                try:
                    os.remove(cache_file)
                except Exception:
                    pass
        
        return None
    
    def _add_to_cache(self, image_path: str, prompt: str, negative_prompt: str, style: str, aspect_ratio: str) -> None:
        """
        Añade una imagen al caché
        
        Args:
            image_path: Ruta a la imagen
            prompt: Prompt de la imagen
            negative_prompt: Prompt negativo
            style: Estilo visual
            aspect_ratio: Relación de aspecto
        """
        if not self.config.get("cache_enabled", True) or not os.path.exists(image_path):
            return
        
        try:
            # Generar hash de los parámetros
            cache_key = self._generate_cache_key(prompt, negative_prompt, style, aspect_ratio)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.png")
            
            # Copiar archivo a caché
            import shutil
            shutil.copy2(image_path, cache_file)
            
            # Guardar metadatos
            metadata_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "style": style,
                "aspect_ratio": aspect_ratio,
                "created_at": time.time(),
                "original_path": image_path
            }
            
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error adding image to cache: {str(e)}")
    
    def _generate_cache_key(self, prompt: str, negative_prompt: str, style: str, aspect_ratio: str) -> str:
        """
        Genera una clave única para el caché basada en los parámetros
        
        Args:
            prompt: Prompt de la imagen
            negative_prompt: Prompt negativo
            style: Estilo visual
            aspect_ratio: Relación de aspecto
            
        Returns:
            Clave hash para el caché
        """
        # Normalizar parámetros
        prompt = (prompt or "").lower().strip()
        negative_prompt = (negative_prompt or "").lower().strip()
        style = (style or "").lower().strip()
        aspect_ratio = (aspect_ratio or "").lower().strip()
        
        # Concatenar parámetros
        cache_str = f"{prompt}|{negative_prompt}|{style}|{aspect_ratio}"
        
        # Generar hash
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _generate_with_leonardo(self, prompt: str, negative_prompt: str, width: int, height: int, output_path: str) -> str:
        """
        Genera una imagen con Leonardo.ai
        
        Args:
            prompt: Prompt de la imagen
            negative_prompt: Prompt negativo
            width: Ancho de la imagen
            height: Alto de la imagen
            output_path: Ruta de salida
            
        Returns:
            Ruta a la imagen generada
        """
                try:
            # Verificar configuración de Leonardo.ai
            leonardo_config = self.config.get("apis", {}).get("leonardo", {})
            if not leonardo_config.get("enabled", False):
                logger.warning("Leonardo.ai API is not enabled")
                return None
            
            # Obtener API key
            api_key = leonardo_config.get("api_key", "")
            if not api_key:
                logger.warning("Leonardo.ai API key not configured")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"leonardo_{timestamp}_{random_id}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Configurar parámetros para la API
            api_url = "https://cloud.leonardo.ai/api/rest/v1/generations"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Seleccionar modelo
            model_id = leonardo_config.get("model_id", "6bef9f1b-29cb-40c7-b9df-32b51c1f67d3")  # Leonardo Creative por defecto
            
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "modelId": model_id,
                "width": width,
                "height": height,
                "num_images": 1,
                "guidance_scale": 7.0,
                "public": False
            }
            
            # Realizar solicitud a la API
            logger.info(f"Generating image with Leonardo.ai: {prompt}")
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Leonardo.ai API error: {response.status_code} - {response.text}")
                return None
            
            # Procesar respuesta
            response_data = response.json()
            generation_id = response_data.get("sdGenerationJob", {}).get("generationId")
            
            if not generation_id:
                logger.error("Failed to get generation ID from Leonardo.ai")
                return None
            
            # Esperar a que la generación se complete
            max_attempts = 30
            attempt = 0
            
            while attempt < max_attempts:
                # Consultar estado de la generación
                status_url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}"
                status_response = requests.get(status_url, headers=headers)
                
                if status_response.status_code != 200:
                    logger.error(f"Error checking generation status: {status_response.status_code}")
                    time.sleep(2)
                    attempt += 1
                    continue
                
                status_data = status_response.json()
                status = status_data.get("sdGenerationJob", {}).get("status")
                
                if status == "COMPLETE":
                    # Obtener URL de la imagen generada
                    generated_images = status_data.get("sdGenerationJob", {}).get("generatedImages", [])
                    
                    if not generated_images:
                        logger.error("No images generated by Leonardo.ai")
                        return None
                    
                    image_url = generated_images[0].get("url")
                    
                    if not image_url:
                        logger.error("Image URL not found in Leonardo.ai response")
                        return None
                    
                    # Descargar imagen
                    image_response = requests.get(image_url)
                    
                    if image_response.status_code != 200:
                        logger.error(f"Failed to download image: {image_response.status_code}")
                        return None
                    
                    # Guardar imagen
                    with open(output_path, "wb") as f:
                        f.write(image_response.content)
                    
                    logger.info(f"Image generated with Leonardo.ai: {output_path}")
                    return output_path
                
                elif status == "FAILED":
                    logger.error("Leonardo.ai generation failed")
                    return None
                
                # Esperar antes de verificar de nuevo
                time.sleep(2)
                attempt += 1
            
            logger.error("Leonardo.ai generation timed out")
            return None
            
        except Exception as e:
            logger.error(f"Error generating image with Leonardo.ai: {str(e)}")
            return None
    
    def _generate_with_stable_diffusion(self, prompt: str, negative_prompt: str, width: int, height: int, output_path: str) -> str:
        """
        Genera una imagen con Stable Diffusion (local o API)
        
        Args:
            prompt: Prompt de la imagen
            negative_prompt: Prompt negativo
            width: Ancho de la imagen
            height: Alto de la imagen
            output_path: Ruta de salida
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Verificar si se usa modelo local o API
            sd_config = self.config.get("apis", {}).get("stable_diffusion", {})
            use_local = sd_config.get("local_enabled", False) and self.sd_model is not None
            use_api = sd_config.get("api_enabled", False) and not use_local
            
            if not use_local and not use_api:
                logger.warning("Stable Diffusion not available (neither local nor API)")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"sd_{timestamp}_{random_id}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generar con modelo local
            if use_local:
                logger.info(f"Generating image with local Stable Diffusion: {prompt}")
                
                # Configurar parámetros
                num_inference_steps = sd_config.get("num_steps", 30)
                guidance_scale = sd_config.get("guidance_scale", 7.5)
                
                # Generar imagen
                image = self.sd_model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                ).images[0]
                
                # Guardar imagen
                image.save(output_path)
                logger.info(f"Image generated with local Stable Diffusion: {output_path}")
                return output_path
            
            # Generar con API
            else:
                logger.info(f"Generating image with Stable Diffusion API: {prompt}")
                
                # Configurar API
                api_url = sd_config.get("api_url", "http://localhost:7860")
                api_endpoint = f"{api_url}/sdapi/v1/txt2img"
                
                # Configurar parámetros
                payload = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt or "",
                    "width": width,
                    "height": height,
                    "steps": sd_config.get("num_steps", 30),
                    "cfg_scale": sd_config.get("guidance_scale", 7.5),
                    "sampler_name": sd_config.get("sampler", "DPM++ 2M Karras"),
                    "batch_size": 1
                }
                
                # Realizar solicitud a la API
                response = requests.post(api_endpoint, json=payload)
                
                if response.status_code != 200:
                    logger.error(f"Stable Diffusion API error: {response.status_code} - {response.text}")
                    return None
                
                # Procesar respuesta
                response_data = response.json()
                image_data = response_data.get("images", [])[0]
                
                if not image_data:
                    logger.error("No image data in Stable Diffusion API response")
                    return None
                
                # Decodificar imagen base64
                image_bytes = base64.b64decode(image_data)
                
                # Guardar imagen
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                
                logger.info(f"Image generated with Stable Diffusion API: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error generating image with Stable Diffusion: {str(e)}")
            return None
    
    def _generate_with_midjourney(self, prompt: str, negative_prompt: str, width: int, height: int, output_path: str) -> str:
        """
        Genera una imagen con Midjourney API
        
        Args:
            prompt: Prompt de la imagen
            negative_prompt: Prompt negativo
            width: Ancho de la imagen
            height: Alto de la imagen
            output_path: Ruta de salida
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Verificar configuración de Midjourney
            midjourney_config = self.config.get("apis", {}).get("midjourney", {})
            if not midjourney_config.get("enabled", False):
                logger.warning("Midjourney API is not enabled")
                return None
            
            # Obtener API key
            api_key = midjourney_config.get("api_key", "")
            if not api_key:
                logger.warning("Midjourney API key not configured")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"midjourney_{timestamp}_{random_id}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Configurar parámetros para la API
            api_url = "https://api.midjourney.com/v1/imagine"  # URL de ejemplo
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Combinar prompt y negative_prompt
            full_prompt = prompt
            if negative_prompt:
                full_prompt += f" --no {negative_prompt}"
            
            # Añadir parámetros de aspecto
            aspect_param = ""
            if width and height:
                if width == height:
                    aspect_param = "--ar 1:1"
                elif width > height:
                    aspect_param = f"--ar {width}:{height}"
                else:
                    aspect_param = f"--ar {width}:{height}"
            
            if aspect_param:
                full_prompt += f" {aspect_param}"
            
            payload = {
                "prompt": full_prompt,
                "returnUrl": True
            }
            
            # Realizar solicitud a la API
            logger.info(f"Generating image with Midjourney: {full_prompt}")
            
            # Simulación de llamada a API (en implementación real se usaría requests.post)
            # response = requests.post(api_url, headers=headers, json=payload)
            # if response.status_code != 200:
            #     logger.error(f"Midjourney API error: {response.status_code} - {response.text}")
            #     return None
            
            # Simulación: generar una imagen con otro servicio como fallback
            logger.warning("Midjourney API integration not implemented, falling back to Leonardo")
            return self._generate_with_leonardo(prompt, negative_prompt, width, height, output_path)
            
        except Exception as e:
            logger.error(f"Error generating image with Midjourney: {str(e)}")
            return None
    
    def _generate_with_runway(self, prompt: str, negative_prompt: str, width: int, height: int, output_path: str) -> str:
        """
        Genera una imagen con RunwayML
        
        Args:
            prompt: Prompt de la imagen
            negative_prompt: Prompt negativo
            width: Ancho de la imagen
            height: Alto de la imagen
            output_path: Ruta de salida
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Verificar configuración de RunwayML
            runway_config = self.config.get("apis", {}).get("runway", {})
            if not runway_config.get("enabled", False):
                logger.warning("RunwayML API is not enabled")
                return None
            
            # Obtener API key
            api_key = runway_config.get("api_key", "")
            if not api_key:
                logger.warning("RunwayML API key not configured")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"runway_{timestamp}_{random_id}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Configurar parámetros para la API
            api_url = "https://api.runwayml.com/v1/generationImage"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "width": width,
                "height": height,
                "num_outputs": 1,
                "guidance_scale": 7.0
            }
            
            # Realizar solicitud a la API
            logger.info(f"Generating image with RunwayML: {prompt}")
            
            # Simulación de llamada a API (en implementación real se usaría requests.post)
            # response = requests.post(api_url, headers=headers, json=payload)
            # if response.status_code != 200:
            #     logger.error(f"RunwayML API error: {response.status_code} - {response.text}")
            #     return None
            
            # Simulación: generar una imagen con otro servicio como fallback
            logger.warning("RunwayML API integration not implemented, falling back to Leonardo")
            return self._generate_with_leonardo(prompt, negative_prompt, width, height, output_path)
            
        except Exception as e:
            logger.error(f"Error generating image with RunwayML: {str(e)}")
            return None
    
    def _generate_variation_with_leonardo(self, image_path: str, prompt: str, strength: float, output_path: str) -> str:
        """
        Genera una variación de imagen con Leonardo.ai
        
        Args:
            image_path: Ruta a la imagen base
            prompt: Prompt adicional
            strength: Intensidad de la variación
            output_path: Ruta de salida
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Verificar configuración de Leonardo.ai
            leonardo_config = self.config.get("apis", {}).get("leonardo", {})
            if not leonardo_config.get("enabled", False):
                logger.warning("Leonardo.ai API is not enabled")
                return None
            
            # Obtener API key
            api_key = leonardo_config.get("api_key", "")
            if not api_key:
                logger.warning("Leonardo.ai API key not configured")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"leonardo_var_{timestamp}_{random_id}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar y codificar imagen en base64
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            
            # Configurar parámetros para la API
            api_url = "https://cloud.leonardo.ai/api/rest/v1/variations"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Seleccionar modelo
            model_id = leonardo_config.get("model_id", "6bef9f1b-29cb-40c7-b9df-32b51c1f67d3")  # Leonardo Creative por defecto
            
            payload = {
                "image": image_base64,
                "prompt": prompt or "",
                "modelId": model_id,
                "strength": strength,
                "num_images": 1,
                "guidance_scale": 7.0,
                "public": False
            }
            
            # Realizar solicitud a la API
            logger.info(f"Generating image variation with Leonardo.ai")
            
            # Simulación de llamada a API (en implementación real se usaría requests.post)
            # response = requests.post(api_url, headers=headers, json=payload)
            # if response.status_code != 200:
            #     logger.error(f"Leonardo.ai API error: {response.status_code} - {response.text}")
            #     return None
            
            # Simulación: generar una imagen con otro servicio como fallback
            logger.warning("Leonardo.ai variation API not fully implemented, falling back to regular generation")
            return self._generate_with_leonardo(prompt or "Variation of image", None, 1024, 1024, output_path)
            
        except Exception as e:
            logger.error(f"Error generating image variation with Leonardo.ai: {str(e)}")
            return None
    
    def _generate_variation_with_stable_diffusion(self, image_path: str, prompt: str, strength: float, output_path: str) -> str:
        """
        Genera una variación de imagen con Stable Diffusion
        
        Args:
            image_path: Ruta a la imagen base
            prompt: Prompt adicional
            strength: Intensidad de la variación
            output_path: Ruta de salida
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Verificar si se usa modelo local o API
            sd_config = self.config.get("apis", {}).get("stable_diffusion", {})
            use_local = sd_config.get("local_enabled", False) and self.sd_model is not None
            use_api = sd_config.get("api_enabled", False) and not use_local
            
            if not use_local and not use_api:
                logger.warning("Stable Diffusion not available (neither local nor API)")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"sd_var_{timestamp}_{random_id}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generar con modelo local
            if use_local:
                logger.info(f"Generating image variation with local Stable Diffusion")
                
                try:
                    # Importar dependencias necesarias
                    from diffusers import StableDiffusionImg2ImgPipeline
                    
                    # Cargar imagen
                    init_image = Image.open(image_path).convert("RGB")
                    
                    # Redimensionar si es necesario
                    width, height = init_image.size
                    if width > 1024 or height > 1024:
                        ratio = min(1024 / width, 1024 / height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        init_image = init_image.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Configurar pipeline img2img
                    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-1",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    ).to(self.sd_model.device)
                    
                    # Configurar parámetros
                    num_inference_steps = sd_config.get("num_steps", 30)
                    guidance_scale = sd_config.get("guidance_scale", 7.5)
                    
                    # Generar imagen
                    result = img2img_pipe(
                        prompt=prompt or "Variation of the image",
                        image=init_image,
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    ).images[0]
                    
                    # Guardar imagen
                    result.save(output_path)
                    logger.info(f"Image variation generated with local Stable Diffusion: {output_path}")
                    return output_path
                    
                except ImportError:
                    logger.error("Failed to import Stable Diffusion img2img dependencies")
                    return None
                except Exception as e:
                    logger.error(f"Error in local Stable Diffusion img2img: {str(e)}")
                    return None
            
            # Generar con API
            else:
                logger.info(f"Generating image variation with Stable Diffusion API")
                
                # Cargar y codificar imagen en base64
                with open(image_path, "rb") as f:
                    image_data = f.read()
                
                image_base64 = base64.b64encode(image_data).decode("utf-8")
                
                # Configurar API
                api_url = sd_config.get("api_url", "http://localhost:7860")
                api_endpoint = f"{api_url}/sdapi/v1/img2img"
                
                # Configurar parámetros
                payload = {
                    "init_images": [image_base64],
                    "prompt": prompt or "Variation of the image",
                    "denoising_strength": strength,
                    "steps": sd_config.get("num_steps", 30),
                    "cfg_scale": sd_config.get("guidance_scale", 7.5),
                    "sampler_name": sd_config.get("sampler", "DPM++ 2M Karras"),
                    "batch_size": 1
                }
                
                # Realizar solicitud a la API
                response = requests.post(api_endpoint, json=payload)
                
                if response.status_code != 200:
                    logger.error(f"Stable Diffusion API error: {response.status_code} - {response.text}")
                    return None
                
                # Procesar respuesta
                response_data = response.json()
                image_data = response_data.get("images", [])[0]
                
                if not image_data:
                    logger.error("No image data in Stable Diffusion API response")
                    return None
                
                # Decodificar imagen base64
                image_bytes = base64.b64decode(image_data)
                
                # Guardar imagen
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                
                logger.info(f"Image variation generated with Stable Diffusion API: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error generating image variation with Stable Diffusion: {str(e)}")
            return None
    
    def _generate_variation_with_runway(self, image_path: str, prompt: str, strength: float, output_path: str) -> str:
        """
        Genera una variación de imagen con RunwayML
        
        Args:
            image_path: Ruta a la imagen base
            prompt: Prompt adicional
            strength: Intensidad de la variación
            output_path: Ruta de salida
            
        Returns:
            Ruta a la imagen generada
        """
        try:
            # Verificar configuración de RunwayML
            runway_config = self.config.get("apis", {}).get("runway", {})
            if not runway_config.get("enabled", False):
                logger.warning("RunwayML API is not enabled")
                return None
            
            # Obtener API key
            api_key = runway_config.get("api_key", "")
            if not api_key:
                logger.warning("RunwayML API key not configured")
                return None
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"runway_var_{timestamp}_{random_id}.png"
                output_path = os.path.join(self.output_dir, filename)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Cargar y codificar imagen en base64
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            
            # Configurar parámetros para la API
            api_url = "https://api.runwayml.com/v1/variationImage"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "image": image_base64,
                "prompt": prompt or "",
                "strength": strength,
                "num_outputs": 1,
                "guidance_scale": 7.0
            }
            
            # Realizar solicitud a la API
            logger.info(f"Generating image variation with RunwayML")
            
            # Simulación de llamada a API (en implementación real se usaría requests.post)
            # response = requests.post(api_url, headers=headers, json=payload)
            # if response.status_code != 200:
            #     logger.error(f"RunwayML API error: {response.status_code} - {response.text}")
            #     return None
            
            # Simulación: generar una imagen con otro servicio como fallback
            logger.warning("RunwayML variation API not implemented, falling back to Stable Diffusion")
            return self._generate_variation_with_stable_diffusion(image_path, prompt, strength, output_path)
            
        except Exception as e:
            logger.error(f"Error generating image variation with RunwayML: {str(e)}")
            return None
    
    def _add_watermark(self, image_path: str, watermark_text: str) -> None:
        """
        Añade una marca de agua a una imagen
        
        Args:
            image_path: Ruta a la imagen
            watermark_text: Texto de la marca de agua
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Crear capa para marca de agua
            watermark = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark)
            
            # Cargar fuente
            try:
                font_size = int(min(image.width, image.height) / 20)
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                # Fallback a fuente por defecto
                font = ImageFont.load_default()
            
            # Obtener tamaño del texto
            text_width, text_height = draw.textsize(watermark_text, font=font)
            
            # Posición (esquina inferior derecha)
            x = image.width - text_width - 20
            y = image.height - text_height - 20
            
            # Dibujar texto con sombra
            opacity = int(255 * self.config.get("watermark_opacity", 0.3))
            draw.text((x+2, y+2), watermark_text, font=font, fill=(0, 0, 0, opacity))
            draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, opacity))
            
            # Combinar imágenes
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            
            watermarked = Image.alpha_composite(image, watermark)
            watermarked = watermarked.convert("RGB")
            
            # Guardar imagen
            watermarked.save(image_path)
            
        except Exception as e:
            logger.error(f"Error adding watermark: {str(e)}")
    
    def _add_caption_to_image(self, image_path: str, caption: str) -> None:
        """
        Añade un texto de descripción a una imagen
        
        Args:
            image_path: Ruta a la imagen
            caption: Texto de descripción
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Crear nueva imagen con espacio para el texto
            caption_height = int(image.height * 0.15)  # 15% de la altura para el texto
            new_height = image.height + caption_height
            
            new_image = Image.new("RGB", (image.width, new_height), (0, 0, 0))
            new_image.paste(image, (0, 0))
            
            # Preparar para dibujar texto
            draw = ImageDraw.Draw(new_image)
            
            # Cargar fuente
            try:
                font_size = int(image.width / 40)  # Tamaño proporcional al ancho
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                # Fallback a fuente por defecto
                font = ImageFont.load_default()
            
            # Limitar longitud del texto
            max_chars = 100
            if len(caption) > max_chars:
                caption = caption[:max_chars-3] + "..."
            
                        # Posición del texto (centrado en la parte inferior)
            text_width, text_height = draw.textsize(caption, font=font)
            x = (image.width - text_width) // 2
            y = image.height + (caption_height - text_height) // 2
            
            # Dibujar texto con sombra para mejor legibilidad
            draw.text((x+1, y+1), caption, font=font, fill=(0, 0, 0))
            draw.text((x, y), caption, font=font, fill=(255, 255, 255))
            
            # Guardar imagen
            new_image.save(image_path)
            logger.info(f"Caption added to image: {image_path}")
            
        except Exception as e:
            logger.error(f"Error adding caption to image: {str(e)}")
    
    def _enhance_image(self, image_path: str, enhancement_type: str = "auto") -> str:
        """
        Mejora una imagen (ajuste de color, contraste, etc.)
        
        Args:
            image_path: Ruta a la imagen
            enhancement_type: Tipo de mejora (auto, color, contrast, sharpness)
            
        Returns:
            Ruta a la imagen mejorada
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Aplicar mejoras según el tipo
            if enhancement_type == "auto" or enhancement_type == "all":
                # Aplicar todas las mejoras con valores moderados
                image = ImageEnhance.Color(image).enhance(1.2)
                image = ImageEnhance.Contrast(image).enhance(1.15)
                image = ImageEnhance.Brightness(image).enhance(1.05)
                image = ImageEnhance.Sharpness(image).enhance(1.3)
            elif enhancement_type == "color":
                image = ImageEnhance.Color(image).enhance(1.3)
            elif enhancement_type == "contrast":
                image = ImageEnhance.Contrast(image).enhance(1.3)
            elif enhancement_type == "brightness":
                image = ImageEnhance.Brightness(image).enhance(1.2)
            elif enhancement_type == "sharpness":
                image = ImageEnhance.Sharpness(image).enhance(1.5)
            
            # Guardar imagen
            image.save(image_path)
            logger.info(f"Image enhanced ({enhancement_type}): {image_path}")
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image_path
    
    def _resize_image(self, image_path: str, width: int = None, height: int = None, max_size: int = None) -> str:
        """
        Redimensiona una imagen
        
        Args:
            image_path: Ruta a la imagen
            width: Ancho deseado
            height: Alto deseado
            max_size: Tamaño máximo (para el lado más largo)
            
        Returns:
            Ruta a la imagen redimensionada
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            original_width, original_height = image.size
            
            # Determinar nuevas dimensiones
            new_width, new_height = original_width, original_height
            
            if max_size and (original_width > max_size or original_height > max_size):
                # Redimensionar manteniendo proporción
                if original_width > original_height:
                    new_width = max_size
                    new_height = int(original_height * (max_size / original_width))
                else:
                    new_height = max_size
                    new_width = int(original_width * (max_size / original_height))
            elif width and height:
                # Redimensionar a dimensiones específicas
                new_width, new_height = width, height
            elif width:
                # Redimensionar solo ancho, mantener proporción
                new_width = width
                new_height = int(original_height * (width / original_width))
            elif height:
                # Redimensionar solo alto, mantener proporción
                new_height = height
                new_width = int(original_width * (height / original_height))
            
            # Verificar si es necesario redimensionar
            if new_width != original_width or new_height != original_height:
                # Redimensionar imagen
                resized_image = image.resize((new_width, new_height), Image.LANCZOS)
                
                # Guardar imagen
                resized_image.save(image_path)
                logger.info(f"Image resized to {new_width}x{new_height}: {image_path}")
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return image_path
    
    def _crop_image(self, image_path: str, crop_type: str = "center", crop_ratio: float = None) -> str:
        """
        Recorta una imagen
        
        Args:
            image_path: Ruta a la imagen
            crop_type: Tipo de recorte (center, top, bottom, left, right, smart)
            crop_ratio: Relación de aspecto deseada (ancho/alto)
            
        Returns:
            Ruta a la imagen recortada
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            width, height = image.size
            
            # Si no se especifica relación de aspecto, no hacer nada
            if not crop_ratio:
                return image_path
            
            # Calcular dimensiones de recorte
            current_ratio = width / height
            
            if abs(current_ratio - crop_ratio) < 0.01:
                # La imagen ya tiene la relación de aspecto deseada
                return image_path
            
            if current_ratio > crop_ratio:
                # Imagen más ancha que la relación deseada
                new_width = int(height * crop_ratio)
                new_height = height
            else:
                # Imagen más alta que la relación deseada
                new_width = width
                new_height = int(width / crop_ratio)
            
            # Calcular coordenadas de recorte según el tipo
            if crop_type == "center":
                left = (width - new_width) // 2
                top = (height - new_height) // 2
            elif crop_type == "top":
                left = (width - new_width) // 2
                top = 0
            elif crop_type == "bottom":
                left = (width - new_width) // 2
                top = height - new_height
            elif crop_type == "left":
                left = 0
                top = (height - new_height) // 2
            elif crop_type == "right":
                left = width - new_width
                top = (height - new_height) // 2
            elif crop_type == "smart":
                # Implementación básica de recorte inteligente
                # En una implementación real, se usaría un algoritmo de detección de objetos
                try:
                    # Intentar detectar áreas de interés
                    gray = image.convert("L")
                    edges = gray.filter(ImageFilter.FIND_EDGES)
                    
                    # Dividir la imagen en una cuadrícula y encontrar áreas con más bordes
                    grid_size = 8
                    grid_width = width // grid_size
                    grid_height = height // grid_size
                    
                    edge_scores = []
                    for y in range(0, height - grid_height, grid_height):
                        for x in range(0, width - grid_width, grid_width):
                            box = (x, y, x + grid_width, y + grid_height)
                            region = edges.crop(box)
                            score = sum(region.getdata()) / (grid_width * grid_height)
                            edge_scores.append((x, y, score))
                    
                    # Ordenar por puntuación
                    edge_scores.sort(key=lambda x: x[2], reverse=True)
                    
                    # Usar el centro del área con mayor puntuación
                    if edge_scores:
                        center_x = edge_scores[0][0] + grid_width // 2
                        center_y = edge_scores[0][1] + grid_height // 2
                        
                        left = max(0, center_x - new_width // 2)
                        top = max(0, center_y - new_height // 2)
                        
                        # Ajustar si se sale de los límites
                        if left + new_width > width:
                            left = width - new_width
                        if top + new_height > height:
                            top = height - new_height
                    else:
                        # Fallback a recorte central
                        left = (width - new_width) // 2
                        top = (height - new_height) // 2
                except Exception:
                    # Fallback a recorte central
                    left = (width - new_width) // 2
                    top = (height - new_height) // 2
            else:
                # Fallback a recorte central
                left = (width - new_width) // 2
                top = (height - new_height) // 2
            
            # Recortar imagen
            right = left + new_width
            bottom = top + new_height
            
            cropped_image = image.crop((left, top, right, bottom))
            
            # Guardar imagen
            cropped_image.save(image_path)
            logger.info(f"Image cropped to {new_width}x{new_height} ({crop_type}): {image_path}")
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error cropping image: {str(e)}")
            return image_path
    
    def _apply_filter(self, image_path: str, filter_type: str) -> str:
        """
        Aplica un filtro a una imagen
        
        Args:
            image_path: Ruta a la imagen
            filter_type: Tipo de filtro (bw, sepia, blur, etc.)
            
        Returns:
            Ruta a la imagen con filtro
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Aplicar filtro según el tipo
            if filter_type == "bw" or filter_type == "grayscale":
                # Blanco y negro
                image = image.convert("L").convert("RGB")
            elif filter_type == "sepia":
                # Sepia
                sepia_data = []
                for pixel in image.convert("RGB").getdata():
                    r, g, b = pixel
                    new_r = min(255, int(r * 0.393 + g * 0.769 + b * 0.189))
                    new_g = min(255, int(r * 0.349 + g * 0.686 + b * 0.168))
                    new_b = min(255, int(r * 0.272 + g * 0.534 + b * 0.131))
                    sepia_data.append((new_r, new_g, new_b))
                
                image = Image.new("RGB", image.size)
                image.putdata(sepia_data)
            elif filter_type == "blur":
                # Desenfoque
                image = image.filter(ImageFilter.GaussianBlur(radius=2))
            elif filter_type == "contour":
                # Contorno
                image = image.filter(ImageFilter.CONTOUR)
            elif filter_type == "emboss":
                # Relieve
                image = image.filter(ImageFilter.EMBOSS)
            elif filter_type == "edge_enhance":
                # Realce de bordes
                image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            elif filter_type == "smooth":
                # Suavizado
                image = image.filter(ImageFilter.SMOOTH_MORE)
            elif filter_type == "sharpen":
                # Nitidez
                image = image.filter(ImageFilter.SHARPEN)
            
            # Guardar imagen
            image.save(image_path)
            logger.info(f"Filter applied ({filter_type}): {image_path}")
            
            return image_path
            
        except Exception as e:
            logger.error(f"Error applying filter: {str(e)}")
            return image_path