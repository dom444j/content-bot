import os
import time
import random
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

from data.knowledge_base import KnowledgeBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThumbnailMaker:
    """
    Clase para generar miniaturas optimizadas para diferentes plataformas
    con llamadas a la acción (CTAs) integradas.
    """
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.config = self._load_config()
        self.templates = self._load_templates()
        self.fonts = self._load_fonts()
        self.cache = {}
        logger.info("ThumbnailMaker initialized")
        
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración para la generación de miniaturas"""
        try:
            config = self.knowledge_base.get_thumbnail_config()
            if not config:
                # Configuración por defecto
                config = {
                    "output_directory": "uploads/thumbnails",
                    "template_directory": "datasets/thumbnail_templates",
                    "font_directory": "datasets/fonts",
                    "cache_enabled": True,
                    "cache_max_size": 100,
                    "default_quality": 90,
                    "platforms": {
                        "youtube": {
                            "width": 1280,
                            "height": 720,
                            "text_size": 72,
                            "cta_size": 48,
                            "text_color": "#FFFFFF",
                            "cta_color": "#FF0000",
                            "shadow_enabled": True,
                            "shadow_color": "#000000",
                            "shadow_offset": [2, 2],
                            "max_text_length": 40
                        },
                        "tiktok": {
                            "width": 1080,
                            "height": 1920,
                            "text_size": 64,
                            "cta_size": 42,
                            "text_color": "#FFFFFF",
                            "cta_color": "#00F2EA",
                            "shadow_enabled": True,
                            "shadow_color": "#000000",
                            "shadow_offset": [2, 2],
                            "max_text_length": 30
                        },
                        "instagram": {
                            "width": 1080,
                            "height": 1920,
                            "text_size": 64,
                            "cta_size": 42,
                            "text_color": "#FFFFFF",
                            "cta_color": "#C13584",
                            "shadow_enabled": True,
                            "shadow_color": "#000000",
                            "shadow_offset": [2, 2],
                            "max_text_length": 30
                        }
                    },
                    "apis": {
                        "canva": {
                            "enabled": False,
                            "api_key": "",
                            "base_url": "https://api.canva.com/v1"
                        },
                        "leonardo": {
                            "enabled": False,
                            "api_key": "",
                            "base_url": "https://cloud.leonardo.ai/api/rest/v1"
                        }
                    },
                    "cta_templates": [
                        "¡SIGUE AHORA!",
                        "MIRA HASTA EL FINAL",
                        "COMENTA '🔥' PARA MÁS",
                        "GUARDA PARA DESPUÉS",
                        "COMPARTE CON UN AMIGO"
                    ]
                }
                self.knowledge_base.save_thumbnail_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading thumbnail config: {str(e)}")
            # Configuración mínima en caso de error
            return {
                "output_directory": "uploads/thumbnails",
                "platforms": {
                    "youtube": {"width": 1280, "height": 720},
                    "tiktok": {"width": 1080, "height": 1920},
                    "instagram": {"width": 1080, "height": 1920}
                }
            }
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Carga plantillas de miniaturas por plataforma"""
        templates = {}
        template_dir = self.config.get("template_directory", "datasets/thumbnail_templates")
        
        # Asegurar que el directorio existe
        os.makedirs(template_dir, exist_ok=True)
        
        # Cargar plantillas para cada plataforma
        for platform in self.config.get("platforms", {}).keys():
            platform_dir = os.path.join(template_dir, platform)
            os.makedirs(platform_dir, exist_ok=True)
            
            # Buscar archivos de imagen en el directorio de la plataforma
            templates[platform] = []
            if os.path.exists(platform_dir):
                for file in os.listdir(platform_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        templates[platform].append(os.path.join(platform_dir, file))
        
        return templates
    
    def _load_fonts(self) -> Dict[str, str]:
        """Carga fuentes disponibles para texto y CTAs"""
        fonts = {}
        font_dir = self.config.get("font_directory", "datasets/fonts")
        
        # Asegurar que el directorio existe
        os.makedirs(font_dir, exist_ok=True)
        
        # Buscar archivos de fuente en el directorio
        if os.path.exists(font_dir):
            for file in os.listdir(font_dir):
                if file.lower().endswith(('.ttf', '.otf')):
                    font_name = os.path.splitext(file)[0]
                    fonts[font_name] = os.path.join(font_dir, file)
        
        # Añadir fuente por defecto si no hay ninguna
        if not fonts:
            # Usar fuente del sistema
            fonts["default"] = "arial.ttf"
        
        return fonts
    
    def create_thumbnail(self, 
                        title: str, 
                        platform: str = "youtube", 
                        cta_text: str = None,
                        background_image: str = None,
                        character_image: str = None,
                        style: str = "default",
                        output_path: str = None) -> str:
        """
        Crea una miniatura optimizada para la plataforma especificada
        
        Args:
            title: Título principal para la miniatura
            platform: Plataforma (youtube, tiktok, instagram)
            cta_text: Texto de llamada a la acción (opcional)
            background_image: Ruta a imagen de fondo (opcional)
            character_image: Ruta a imagen de personaje (opcional)
            style: Estilo de miniatura (default, minimal, bold)
            output_path: Ruta de salida (opcional)
            
        Returns:
            Ruta a la miniatura generada
        """
        try:
            # Verificar plataforma
            if platform not in self.config.get("platforms", {}):
                logger.warning(f"Platform {platform} not supported, using youtube")
                platform = "youtube"
            
            platform_config = self.config["platforms"][platform]
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                output_dir = self.config.get("output_directory", "uploads/thumbnails")
                os.makedirs(output_dir, exist_ok=True)
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"thumbnail_{platform}_{timestamp}_{random_id}.png"
                output_path = os.path.join(output_dir, filename)
            
            # Seleccionar CTA si no se proporciona
            if not cta_text and self.config.get("cta_templates"):
                cta_text = random.choice(self.config["cta_templates"])
            
            # Crear miniatura según el estilo
            if style == "minimal":
                thumbnail_path = self._create_minimal_thumbnail(
                    title, platform, cta_text, background_image, output_path
                )
            elif style == "bold":
                thumbnail_path = self._create_bold_thumbnail(
                    title, platform, cta_text, background_image, character_image, output_path
                )
            else:  # default
                thumbnail_path = self._create_default_thumbnail(
                    title, platform, cta_text, background_image, character_image, output_path
                )
            
            logger.info(f"Created thumbnail: {thumbnail_path}")
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            raise
    
    def _create_default_thumbnail(self, 
                                title: str, 
                                platform: str,
                                cta_text: str,
                                background_image: str,
                                character_image: str,
                                output_path: str) -> str:
        """Crea una miniatura con estilo por defecto"""
        platform_config = self.config["platforms"][platform]
        width = platform_config.get("width", 1280)
        height = platform_config.get("height", 720)
        
        # Crear imagen base
        if background_image and os.path.exists(background_image):
            # Usar imagen de fondo proporcionada
            try:
                img = Image.open(background_image)
                img = img.resize((width, height), Image.LANCZOS)
            except Exception as e:
                logger.error(f"Error opening background image: {str(e)}")
                # Crear imagen en blanco como fallback
                img = Image.new('RGB', (width, height), color=(33, 33, 33))
        else:
            # Seleccionar plantilla aleatoria si hay disponibles
            templates = self.templates.get(platform, [])
            if templates:
                template_path = random.choice(templates)
                try:
                    img = Image.open(template_path)
                    img = img.resize((width, height), Image.LANCZOS)
                except Exception as e:
                    logger.error(f"Error opening template: {str(e)}")
                    # Crear imagen en blanco como fallback
                    img = Image.new('RGB', (width, height), color=(33, 33, 33))
            else:
                # Crear imagen en blanco
                img = Image.new('RGB', (width, height), color=(33, 33, 33))
        
        # Añadir personaje si se proporciona
        if character_image and os.path.exists(character_image):
            try:
                char_img = Image.open(character_image)
                # Redimensionar manteniendo proporción
                char_width = int(height * 0.8)
                char_height = int(height * 0.8)
                char_img = char_img.resize((char_width, char_height), Image.LANCZOS)
                
                # Si tiene canal alfa, usar como máscara
                if char_img.mode == 'RGBA':
                    # Pegar en la esquina derecha
                    x_offset = width - char_width - int(width * 0.05)
                    y_offset = height - char_height
                    img.paste(char_img, (x_offset, y_offset), char_img)
                else:
                    # Convertir a RGBA para poder usar como máscara
                    char_img = char_img.convert('RGBA')
                    x_offset = width - char_width - int(width * 0.05)
                    y_offset = height - char_height
                    img.paste(char_img, (x_offset, y_offset), char_img)
            except Exception as e:
                logger.error(f"Error adding character image: {str(e)}")
        
        # Preparar para dibujar texto
        draw = ImageDraw.Draw(img)
        
        # Seleccionar fuente para título
        title_font_size = platform_config.get("text_size", 72)
        title_font_path = next(iter(self.fonts.values())) if self.fonts else "arial.ttf"
        try:
            title_font = ImageFont.truetype(title_font_path, title_font_size)
        except Exception as e:
            logger.error(f"Error loading font: {str(e)}")
            title_font = ImageFont.load_default()
        
        # Limitar longitud del título
        max_length = platform_config.get("max_text_length", 40)
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
        
        # Calcular posición del título (centrado horizontalmente, parte superior)
        title_color = platform_config.get("text_color", "#FFFFFF")
        shadow_enabled = platform_config.get("shadow_enabled", True)
        shadow_color = platform_config.get("shadow_color", "#000000")
        shadow_offset = platform_config.get("shadow_offset", [2, 2])
        
        # Dividir título en líneas si es muy largo
        words = title.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            # Estimar ancho del texto
            text_width = draw.textlength(test_line, font=title_font)
            
            if text_width < width * 0.9:  # Dejar margen del 10%
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Si no hay líneas (título vacío), añadir una línea vacía
        if not lines:
            lines = [""]
        
        # Dibujar título con sombra
        y_offset = int(height * 0.1)  # 10% desde arriba
        line_height = title_font_size * 1.2
        
        for line in lines:
            # Calcular ancho del texto para centrarlo
            text_width = draw.textlength(line, font=title_font)
            x_position = (width - text_width) // 2
            
            # Dibujar sombra si está habilitada
            if shadow_enabled:
                shadow_x = x_position + shadow_offset[0]
                shadow_y = y_offset + shadow_offset[1]
                draw.text((shadow_x, shadow_y), line, font=title_font, fill=shadow_color)
            
            # Dibujar texto
            draw.text((x_position, y_offset), line, font=title_font, fill=title_color)
            y_offset += line_height
        
        # Añadir CTA si se proporciona
        if cta_text:
            # Seleccionar fuente para CTA
            cta_font_size = platform_config.get("cta_size", 48)
            cta_font_path = next(iter(self.fonts.values())) if self.fonts else "arial.ttf"
            try:
                cta_font = ImageFont.truetype(cta_font_path, cta_font_size)
            except Exception as e:
                logger.error(f"Error loading CTA font: {str(e)}")
                cta_font = ImageFont.load_default()
            
            # Calcular posición del CTA (centrado, parte inferior)
            cta_color = platform_config.get("cta_color", "#FF0000")
            cta_width = draw.textlength(cta_text, font=cta_font)
            cta_x = (width - cta_width) // 2
            cta_y = height - int(height * 0.15)  # 15% desde abajo
            
            # Dibujar fondo para CTA (rectángulo semitransparente)
            padding = 20
            cta_bg = Image.new('RGBA', (int(cta_width + padding * 2), int(cta_font_size * 1.5)), (0, 0, 0, 180))
            img.paste(cta_bg, (int(cta_x - padding), int(cta_y - padding // 2)), cta_bg)
            
            # Dibujar sombra si está habilitada
            if shadow_enabled:
                shadow_x = cta_x + shadow_offset[0]
                shadow_y = cta_y + shadow_offset[1]
                draw.text((shadow_x, shadow_y), cta_text, font=cta_font, fill=shadow_color)
            
            # Dibujar CTA
            draw.text((cta_x, cta_y), cta_text, font=cta_font, fill=cta_color)
        
        # Guardar imagen
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        quality = self.config.get("default_quality", 90)
        img.save(output_path, quality=quality)
        
        return output_path
    
    def _create_minimal_thumbnail(self, 
                                title: str, 
                                platform: str,
                                cta_text: str,
                                background_image: str,
                                output_path: str) -> str:
        """Crea una miniatura con estilo minimalista"""
        platform_config = self.config["platforms"][platform]
        width = platform_config.get("width", 1280)
        height = platform_config.get("height", 720)
        
        # Crear imagen base (fondo blanco o claro)
        img = Image.new('RGB', (width, height), color=(245, 245, 245))
        
        # Añadir una franja de color en la parte inferior
        draw = ImageDraw.Draw(img)
        stripe_height = int(height * 0.2)
        stripe_color = self._hex_to_rgb(platform_config.get("cta_color", "#FF0000"))
        draw.rectangle([(0, height - stripe_height), (width, height)], fill=stripe_color)
        
        # Seleccionar fuente para título
        title_font_size = platform_config.get("text_size", 72)
        title_font_path = next(iter(self.fonts.values())) if self.fonts else "arial.ttf"
        try:
            title_font = ImageFont.truetype(title_font_path, title_font_size)
        except Exception as e:
            logger.error(f"Error loading font: {str(e)}")
            title_font = ImageFont.load_default()
        
        # Limitar longitud del título
        max_length = platform_config.get("max_text_length", 40)
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
        
        # Dividir título en líneas
        words = title.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            text_width = draw.textlength(test_line, font=title_font)
            
            if text_width < width * 0.8:  # Dejar margen del 20%
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        if not lines:
            lines = [""]
        
        # Dibujar título (texto negro)
        y_offset = int(height * 0.2)  # 20% desde arriba
        line_height = title_font_size * 1.2
        
        for line in lines:
            text_width = draw.textlength(line, font=title_font)
            x_position = (width - text_width) // 2
            draw.text((x_position, y_offset), line, font=title_font, fill="#000000")
            y_offset += line_height
        
        # Añadir CTA si se proporciona
        if cta_text:
            cta_font_size = platform_config.get("cta_size", 48)
            cta_font_path = next(iter(self.fonts.values())) if self.fonts else "arial.ttf"
            try:
                cta_font = ImageFont.truetype(cta_font_path, cta_font_size)
            except Exception as e:
                logger.error(f"Error loading CTA font: {str(e)}")
                cta_font = ImageFont.load_default()
            
            # Calcular posición del CTA (centrado, dentro de la franja)
            cta_width = draw.textlength(cta_text, font=cta_font)
            cta_x = (width - cta_width) // 2
            cta_y = height - stripe_height // 2 - cta_font_size // 2
            
            # Dibujar CTA (texto blanco)
            draw.text((cta_x, cta_y), cta_text, font=cta_font, fill="#FFFFFF")
        
        # Guardar imagen
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        quality = self.config.get("default_quality", 90)
        img.save(output_path, quality=quality)
        
        return output_path
    
    def _create_bold_thumbnail(self, 
                             title: str, 
                             platform: str,
                             cta_text: str,
                             background_image: str,
                             character_image: str,
                             output_path: str) -> str:
        """Crea una miniatura con estilo llamativo"""
        platform_config = self.config["platforms"][platform]
        width = platform_config.get("width", 1280)
        height = platform_config.get("height", 720)
        
        # Crear imagen base
        if background_image and os.path.exists(background_image):
            try:
                img = Image.open(background_image)
                img = img.resize((width, height), Image.LANCZOS)
            except Exception as e:
                logger.error(f"Error opening background image: {str(e)}")
                # Usar color vibrante como fallback
                img = Image.new('RGB', (width, height), color=(255, 50, 50))
        else:
            # Usar color vibrante
            img = Image.new('RGB', (width, height), color=(255, 50, 50))
        
        # Añadir personaje si se proporciona
        if character_image and os.path.exists(character_image):
            try:
                char_img = Image.open(character_image)
                # Redimensionar manteniendo proporción
                char_width = int(height * 0.9)
                char_height = int(height * 0.9)
                char_img = char_img.resize((char_width, char_height), Image.LANCZOS)
                
                # Si tiene canal alfa, usar como máscara
                if char_img.mode == 'RGBA':
                    # Pegar en la esquina derecha
                    x_offset = width - char_width - int(width * 0.05)
                    y_offset = height - char_height
                    img.paste(char_img, (x_offset, y_offset), char_img)
                else:
                    # Convertir a RGBA para poder usar como máscara
                    char_img = char_img.convert('RGBA')
                    x_offset = width - char_width - int(width * 0.05)
                    y_offset = height - char_height
                    img.paste(char_img, (x_offset, y_offset), char_img)
            except Exception as e:
                logger.error(f"Error adding character image: {str(e)}")
        
        # Preparar para dibujar texto
        draw = ImageDraw.Draw(img)
        
        # Seleccionar fuente para título (más grande que el estilo por defecto)
        title_font_size = int(platform_config.get("text_size", 72) * 1.3)
        title_font_path = next(iter(self.fonts.values())) if self.fonts else "arial.ttf"
        try:
            title_font = ImageFont.truetype(title_font_path, title_font_size)
        except Exception as e:
            logger.error(f"Error loading font: {str(e)}")
            title_font = ImageFont.load_default()
        
        # Limitar longitud del título
        max_length = platform_config.get("max_text_length", 40)
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
        
        # Dividir título en líneas
        words = title.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            text_width = draw.textlength(test_line, font=title_font)
            
            if text_width < width * 0.6:  # Dejar espacio para el personaje
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        if not lines:
            lines = [""]
        
        # Dibujar título con contorno
        y_offset = int(height * 0.1)  # 10% desde arriba
        line_height = title_font_size * 1.2
        title_color = "#FFFFFF"  # Blanco
        outline_color = "#000000"  # Negro
        
        for line in lines:
            text_width = draw.textlength(line, font=title_font)
            x_position = int(width * 0.05)  # Alineado a la izquierda con margen
            
            # Dibujar contorno (desplazando el texto en varias direcciones)
            outline_width = 3
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:  # Evitar la posición central
                        draw.text((x_position + dx, y_offset + dy), line, font=title_font, fill=outline_color)
            
            # Dibujar texto principal
            draw.text((x_position, y_offset), line, font=title_font, fill=title_color)
            y_offset += line_height
        
        # Añadir CTA si se proporciona
        if cta_text:
            # Seleccionar fuente para CTA (más grande y llamativa)
            cta_font_size = int(platform_config.get("cta_size", 48) * 1.5)
            cta_font_path = next(iter(self.fonts.values())) if self.fonts else "arial.ttf"
            try:
                cta_font = ImageFont.truetype(cta_font_path, cta_font_size)
            except Exception as e:
                logger.error(f"Error loading CTA font: {str(e)}")
                cta_font = ImageFont.load_default()
            
            # Calcular posición del CTA (esquina inferior izquierda)
            cta_color = "#FFFF00"  # Amarillo brillante
            cta_x = int(width * 0.05)
            cta_y = height - int(height * 0.2)
            
            # Dibujar fondo para CTA (rectángulo con color contrastante)
            cta_width = draw.textlength(cta_text, font=cta_font)
            padding = 20
            cta_bg_color = (0, 0, 0, 200)  # Negro semitransparente
            cta_bg = Image.new('RGBA', (int(cta_width + padding * 2), int(cta_font_size * 1.5)), cta_bg_color)
            img.paste(cta_bg, (int(cta_x - padding), int(cta_y - padding // 2)), cta_bg)
            
            # Dibujar contorno para el CTA
            outline_width = 2
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx != 0 or dy != 0:
                        draw.text((cta_x + dx, cta_y + dy), cta_text, font=cta_font, fill="#000000")
            
            # Dibujar CTA
            draw.text((cta_x, cta_y), cta_text, font=cta_font, fill=cta_color)
        
        # Guardar imagen
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        quality = self.config.get("default_quality", 90)
        img.save(output_path, quality=quality)
        
        return output_path
    
    def create_thumbnail_set(self, 
                           title: str, 
                           platforms: List[str] = None,
                           cta_text: str = None,
                           background_image: str = None,
                           character_image: str = None,
                           style: str = "default") -> Dict[str, str]:
        """
        Crea un conjunto de miniaturas para múltiples plataformas
        
        Args:
            title: Título principal para las miniaturas
            platforms: Lista de plataformas (youtube, tiktok, instagram)
            cta_text: Texto de llamada a la acción (opcional)
            background_image: Ruta a imagen de fondo (opcional)
            character_image: Ruta a imagen de personaje (opcional)
            style: Estilo de miniatura (default, minimal, bold)
            
        Returns:
            Diccionario con rutas a las miniaturas generadas por plataforma
        """
        if not platforms:
            platforms = list(self.config.get("platforms", {}).keys())
        
        results = {}
        for platform in platforms:
            try:
                thumbnail_path = self.create_thumbnail(
                    title=title,
                    platform=platform,
                    cta_text=cta_text,
                    background_image=background_image,
                    character_image=character_image,
                    style=style
                )
                results[platform] = thumbnail_path
            except Exception as e:
                logger.error(f"Error creating thumbnail for {platform}: {str(e)}")
                results[platform] = None
        
        return results
    
        def create_ab_test_thumbnails(self, 
                                title: str, 
                                platform: str = "youtube",
                                variations: int = 3,
                                cta_variations: bool = True,
                                style_variations: bool = True,
                                output_dir: str = None) -> List[str]:
        """
        Crea múltiples variaciones de miniaturas para pruebas A/B
        
        Args:
            title: Título principal para las miniaturas
            platform: Plataforma (youtube, tiktok, instagram)
            variations: Número de variaciones a generar
            cta_variations: Si se deben variar los textos de CTA
            style_variations: Si se deben variar los estilos
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Lista de rutas a las miniaturas generadas
        """
        try:
            # Verificar plataforma
            if platform not in self.config.get("platforms", {}):
                logger.warning(f"Platform {platform} not supported, using youtube")
                platform = "youtube"
            
            # Generar directorio de salida si no se proporciona
            if not output_dir:
                base_dir = self.config.get("output_directory", "uploads/thumbnails")
                timestamp = int(time.time())
                output_dir = os.path.join(base_dir, f"abtest_{platform}_{timestamp}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Estilos disponibles
            styles = ["default", "minimal", "bold"]
            
            # Obtener CTAs disponibles
            ctas = self.config.get("cta_templates", [])
            if not ctas:
                ctas = ["¡MIRA AHORA!", "NO TE LO PIERDAS", "DESCUBRE MÁS"]
            
            # Generar variaciones
            thumbnail_paths = []
            
            for i in range(variations):
                # Seleccionar estilo
                if style_variations and styles:
                    style = random.choice(styles)
                else:
                    style = "default"
                
                # Seleccionar CTA
                if cta_variations and ctas:
                    cta_text = random.choice(ctas)
                else:
                    cta_text = ctas[0] if ctas else None
                
                # Generar nombre de archivo
                filename = f"thumbnail_var{i+1}_{platform}.png"
                output_path = os.path.join(output_dir, filename)
                
                # Crear miniatura
                thumbnail_path = self.create_thumbnail(
                    title=title,
                    platform=platform,
                    cta_text=cta_text,
                    style=style,
                    output_path=output_path
                )
                
                thumbnail_paths.append(thumbnail_path)
                
                # Guardar metadatos de la variación
                metadata_path = os.path.join(output_dir, f"metadata_var{i+1}.json")
                metadata = {
                    "title": title,
                    "platform": platform,
                    "style": style,
                    "cta_text": cta_text,
                    "created_at": time.time(),
                    "thumbnail_path": thumbnail_path
                }
                
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Created {len(thumbnail_paths)} A/B test thumbnails in {output_dir}")
            return thumbnail_paths
            
        except Exception as e:
            logger.error(f"Error creating A/B test thumbnails: {str(e)}")
            return []
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convierte un color hexadecimal a RGB"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def add_overlay(self, 
                   thumbnail_path: str, 
                   overlay_text: str = None,
                   overlay_image: str = None,
                   position: str = "bottom-right",
                   output_path: str = None) -> str:
        """
        Añade una superposición (texto o imagen) a una miniatura existente
        
        Args:
            thumbnail_path: Ruta a la miniatura
            overlay_text: Texto a superponer (opcional)
            overlay_image: Ruta a imagen a superponer (opcional)
            position: Posición (top-left, top-right, bottom-left, bottom-right, center)
            output_path: Ruta de salida (opcional)
            
        Returns:
            Ruta a la miniatura con superposición
        """
        try:
            # Verificar que al menos se proporciona texto o imagen
            if not overlay_text and not overlay_image:
                logger.warning("No overlay text or image provided")
                return thumbnail_path
            
            # Abrir imagen base
            img = Image.open(thumbnail_path)
            width, height = img.size
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                dirname = os.path.dirname(thumbnail_path)
                basename = os.path.basename(thumbnail_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(dirname, f"{name}_overlay{ext}")
            
            # Añadir superposición de imagen
            if overlay_image and os.path.exists(overlay_image):
                try:
                    overlay_img = Image.open(overlay_image)
                    
                    # Redimensionar overlay (máximo 25% del tamaño original)
                    max_width = int(width * 0.25)
                    max_height = int(height * 0.25)
                    
                    overlay_width, overlay_height = overlay_img.size
                    if overlay_width > max_width or overlay_height > max_height:
                        # Mantener proporción
                        ratio = min(max_width / overlay_width, max_height / overlay_height)
                        new_width = int(overlay_width * ratio)
                        new_height = int(overlay_height * ratio)
                        overlay_img = overlay_img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Calcular posición
                    overlay_width, overlay_height = overlay_img.size
                    if position == "top-left":
                        pos = (int(width * 0.05), int(height * 0.05))
                    elif position == "top-right":
                        pos = (width - overlay_width - int(width * 0.05), int(height * 0.05))
                    elif position == "bottom-left":
                        pos = (int(width * 0.05), height - overlay_height - int(height * 0.05))
                    elif position == "bottom-right":
                        pos = (width - overlay_width - int(width * 0.05), height - overlay_height - int(height * 0.05))
                    else:  # center
                        pos = ((width - overlay_width) // 2, (height - overlay_height) // 2)
                    
                    # Pegar imagen con transparencia si tiene canal alfa
                    if overlay_img.mode == 'RGBA':
                        img.paste(overlay_img, pos, overlay_img)
                    else:
                        img.paste(overlay_img, pos)
                    
                except Exception as e:
                    logger.error(f"Error adding overlay image: {str(e)}")
            
            # Añadir superposición de texto
            if overlay_text:
                draw = ImageDraw.Draw(img)
                
                # Seleccionar fuente
                font_size = int(min(width, height) * 0.05)  # 5% del tamaño mínimo
                font_path = next(iter(self.fonts.values())) if self.fonts else "arial.ttf"
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except Exception as e:
                    logger.error(f"Error loading font: {str(e)}")
                    font = ImageFont.load_default()
                
                # Calcular tamaño del texto
                text_width = draw.textlength(overlay_text, font=font)
                text_height = font_size * 1.2
                
                # Calcular posición
                if position == "top-left":
                    text_pos = (int(width * 0.05), int(height * 0.05))
                elif position == "top-right":
                    text_pos = (width - text_width - int(width * 0.05), int(height * 0.05))
                elif position == "bottom-left":
                    text_pos = (int(width * 0.05), height - text_height - int(height * 0.05))
                elif position == "bottom-right":
                    text_pos = (width - text_width - int(width * 0.05), height - text_height - int(height * 0.05))
                else:  # center
                    text_pos = ((width - text_width) // 2, (height - text_height) // 2)
                
                # Dibujar fondo para el texto
                padding = 10
                bg_rect = [
                    (text_pos[0] - padding, text_pos[1] - padding),
                    (text_pos[0] + text_width + padding, text_pos[1] + text_height + padding)
                ]
                draw.rectangle(bg_rect, fill=(0, 0, 0, 180))
                
                # Dibujar texto
                draw.text(text_pos, overlay_text, font=font, fill="#FFFFFF")
            
            # Guardar imagen
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            quality = self.config.get("default_quality", 90)
            img.save(output_path, quality=quality)
            
            logger.info(f"Added overlay to thumbnail: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding overlay to thumbnail: {str(e)}")
            return thumbnail_path
    
    def generate_from_template(self, 
                             template_id: str,
                             title: str,
                             platform: str = "youtube",
                             params: Dict[str, Any] = None,
                             output_path: str = None) -> str:
        """
        Genera una miniatura a partir de una plantilla predefinida
        
        Args:
            template_id: ID de la plantilla
            title: Título para la miniatura
            platform: Plataforma (youtube, tiktok, instagram)
            params: Parámetros adicionales para la plantilla
            output_path: Ruta de salida (opcional)
            
        Returns:
            Ruta a la miniatura generada
        """
        try:
            # Verificar plataforma
            if platform not in self.config.get("platforms", {}):
                logger.warning(f"Platform {platform} not supported, using youtube")
                platform = "youtube"
            
            # Obtener plantilla
            template = self.knowledge_base.get_thumbnail_template(template_id)
            if not template:
                logger.warning(f"Template {template_id} not found, using default style")
                return self.create_thumbnail(title, platform, output_path=output_path)
            
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                output_dir = self.config.get("output_directory", "uploads/thumbnails")
                os.makedirs(output_dir, exist_ok=True)
                timestamp = int(time.time())
                random_id = random.randint(1000, 9999)
                filename = f"thumbnail_{platform}_{timestamp}_{random_id}.png"
                output_path = os.path.join(output_dir, filename)
            
            # Combinar parámetros de la plantilla con los proporcionados
            template_params = template.get("params", {})
            if params:
                template_params.update(params)
            
            # Crear miniatura según el tipo de plantilla
            if template.get("type") == "canva" and self.config.get("apis", {}).get("canva", {}).get("enabled"):
                # Implementar integración con Canva API
                return self._generate_with_canva(template, title, platform, template_params, output_path)
            elif template.get("type") == "leonardo" and self.config.get("apis", {}).get("leonardo", {}).get("enabled"):
                # Implementar integración con Leonardo AI
                return self._generate_with_leonardo(template, title, platform, template_params, output_path)
            else:
                # Usar generación local
                style = template.get("style", "default")
                cta_text = template.get("cta_text")
                background_image = template.get("background_image")
                
                return self.create_thumbnail(
                    title=title,
                    platform=platform,
                    cta_text=cta_text,
                    background_image=background_image,
                    style=style,
                    output_path=output_path
                )
            
        except Exception as e:
            logger.error(f"Error generating thumbnail from template: {str(e)}")
            # Fallback a creación estándar
            return self.create_thumbnail(title, platform, output_path=output_path)
    
    def _generate_with_canva(self, 
                           template: Dict[str, Any],
                           title: str,
                           platform: str,
                           params: Dict[str, Any],
                           output_path: str) -> str:
        """
        Genera una miniatura usando la API de Canva
        
        En una implementación real, aquí se llamaría a la API de Canva.
        """
        logger.info(f"Generating thumbnail with Canva API (template: {template.get('id')})")
        
        # Simulación de llamada a API
        # En una implementación real, aquí se llamaría a la API de Canva
        
        # Crear miniatura local como fallback
        return self.create_thumbnail(
            title=title,
            platform=platform,
            style="default",
            output_path=output_path
        )
    
    def _generate_with_leonardo(self, 
                              template: Dict[str, Any],
                              title: str,
                              platform: str,
                              params: Dict[str, Any],
                              output_path: str) -> str:
        """
        Genera una miniatura usando la API de Leonardo AI
        
        En una implementación real, aquí se llamaría a la API de Leonardo AI.
        """
        logger.info(f"Generating thumbnail with Leonardo AI (template: {template.get('id')})")
        
        # Simulación de llamada a API
        # En una implementación real, aquí se llamaría a la API de Leonardo AI
        
        # Crear miniatura local como fallback
        return self.create_thumbnail(
            title=title,
            platform=platform,
            style="bold",
            output_path=output_path
        )