"""
Visual Identity - Generador de identidades visuales para personajes

Este módulo gestiona la creación de identidades visuales:
- Generación de avatares y elementos visuales
- Integración con servicios de IA generativa
- Gestión de estilos visuales consistentes
- Adaptación visual según nicho y personalidad
"""

import os
import sys
import json
import logging
import random
import base64
import requests
from typing import Dict, List, Any, Optional, Tuple
import time
from PIL import Image
from io import BytesIO

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'visual_identity.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('VisualIdentity')

class VisualIdentity:
    """
    Clase para gestionar la identidad visual de los personajes.
    """

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.config = self._load_config()
        self.cache = {}
        logger.info("VisualIdentity initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración de servicios visuales"""
        try:
            # Intentar cargar desde la base de conocimiento
            config = self.knowledge_base.get_visual_services_config()
            if not config:
                # Configuración por defecto
                config = {
                    "default_service": "stable_diffusion",
                    "services": {
                        "leonardo": {
                            "api_key": "",
                            "base_url": "https://cloud.leonardo.ai/api/rest/v1",
                            "active": False,
                            "free_tokens_per_day": 150,
                            "cost_per_month": 10
                        },
                        "stable_diffusion": {
                            "model_path": "models/stable-diffusion-xl",
                            "active": True,
                            "local": True,
                            "cost_per_month": 0
                        },
                        "midjourney": {
                            "api_key": "",
                            "base_url": "https://api.midjourney.com",
                            "active": False,
                            "cost_per_month": 10
                        },
                        "runwayml": {
                            "api_key": "",
                            "base_url": "https://api.runwayml.com",
                            "active": False,
                            "cost_per_month": 15
                        },
                        "canva": {
                            "api_key": "",
                            "base_url": "https://api.canva.com",
                            "active": False,
                            "cost_per_month": 12
                        }
                    },
                    "cache_enabled": True,
                    "cache_expiry_days": 30,
                    "default_resolution": "1024x1024",
                    "default_format": "png"
                }
                # Guardar en la base de conocimiento
                self.knowledge_base.save_visual_services_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading visual services config: {str(e)}")
            # Retornar configuración mínima
            return {
                "default_service": "stable_diffusion",
                "services": {
                    "stable_diffusion": {
                        "active": True,
                        "local": True
                    }
                },
                "cache_enabled": True
            }

    def generate_identity(self, name: str, niche: str, 
                         visual_style: Dict[str, Any], 
                         personality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera una identidad visual completa para un personaje
        
        Args:
            name: Nombre del personaje
            niche: Nicho del personaje
            visual_style: Diccionario con preferencias de estilo visual
            personality: Diccionario con atributos de personalidad
            
        Returns:
            Diccionario con elementos de identidad visual
        """
        try:
            # Generar ID único para la identidad visual
            identity_id = f"{name.lower().replace(' ', '_')}_{niche}_{int(time.time())}"
            
            # Determinar servicio a utilizar
            service = self._select_service(visual_style.get("preferred_service"))
            
            # Generar elementos visuales
            avatar = self.generate_avatar(name, niche, visual_style, personality, service)
            color_scheme = self.generate_color_scheme(niche, personality)
            typography = self.generate_typography(personality)
            visual_elements = self.generate_visual_elements(niche, visual_style, personality, service)
            
            # Crear identidad visual completa
            visual_identity = {
                "id": identity_id,
                "name": name,
                "niche": niche,
                "avatar": avatar,
                "color_scheme": color_scheme,
                "typography": typography,
                "visual_elements": visual_elements,
                "style_guide": self._generate_style_guide(visual_style, personality),
                "created_at": time.time(),
                "service_used": service
            }
            
            # Guardar en la base de conocimiento
            self.knowledge_base.save_visual_identity(identity_id, visual_identity)
            
            logger.info(f"Generated visual identity for {name} in {niche} niche")
            return visual_identity
            
        except Exception as e:
            logger.error(f"Error generating visual identity: {str(e)}")
            # Retornar identidad visual mínima
            return {
                "id": f"{name.lower().replace(' ', '_')}_{niche}_{int(time.time())}",
                "name": name,
                "niche": niche,
                "avatar": {"url": "", "prompt": ""},
                "color_scheme": self.generate_color_scheme(niche, personality),
                "typography": self.generate_typography(personality),
                "visual_elements": {},
                "style_guide": {},
                "created_at": time.time(),
                "service_used": "none"
            }

    def _select_service(self, preferred_service: Optional[str] = None) -> str:
        """Selecciona el servicio de generación visual a utilizar"""
        services = self.config.get("services", {})
        
        # Si hay un servicio preferido y está activo, usarlo
        if preferred_service and preferred_service in services:
            if services[preferred_service].get("active", False):
                return preferred_service
                
        # Usar el servicio por defecto si está activo
        default_service = self.config.get("default_service", "stable_diffusion")
        if default_service in services and services[default_service].get("active", False):
            return default_service
            
        # Buscar cualquier servicio activo
        for service_name, service_config in services.items():
            if service_config.get("active", False):
                return service_name
                
        # Si no hay servicios activos, usar stable_diffusion (asumiendo local)
        logger.warning("No active visual services found, defaulting to stable_diffusion")
        return "stable_diffusion"

    def generate_avatar(self, name: str, niche: str, 
                       visual_style: Dict[str, Any], 
                       personality: Dict[str, Any],
                       service: str) -> Dict[str, Any]:
        """
        Genera un avatar para el personaje
        
        Args:
            name: Nombre del personaje
            niche: Nicho del personaje
            visual_style: Diccionario con preferencias de estilo visual
            personality: Diccionario con atributos de personalidad
            service: Servicio de generación a utilizar
            
        Returns:
            Diccionario con información del avatar
        """
        # Construir prompt base según nicho
        base_prompts = {
            "finance": "Professional financial advisor, business attire, confident pose",
            "health": "Health and wellness coach, athletic appearance, healthy glow",
            "gaming": "Gamer with headset, casual style, energetic expression",
            "technology": "Tech expert, modern appearance, smart casual style",
            "humor": "Comedian with expressive face, casual and approachable",
            "education": "Educational expert, professional but approachable, intelligent expression"
        }
        
        base_prompt = base_prompts.get(niche, "Professional content creator")
        
        # Ajustar según personalidad
        if personality.get("formality", 0.5) > 0.7:
            base_prompt += ", formal attire"
        elif personality.get("formality", 0.5) < 0.3:
            base_prompt += ", casual attire"
            
        if personality.get("enthusiasm", 0.5) > 0.7:
            base_prompt += ", enthusiastic expression"
        elif personality.get("enthusiasm", 0.5) < 0.3:
            base_prompt += ", calm demeanor"
            
        # Ajustar según estilo visual
        style = visual_style.get("style", "realistic")
        if style == "cartoon":
            base_prompt = f"Cartoon style {base_prompt}, vibrant colors"
        elif style == "anime":
            base_prompt = f"Anime style {base_prompt}, detailed character design"
        elif style == "3d":
            base_prompt = f"3D rendered {base_prompt}, high quality render"
        elif style == "minimalist":
            base_prompt = f"Minimalist style {base_prompt}, clean lines, simple design"
        else:  # realistic
            base_prompt = f"Photorealistic {base_prompt}, high quality portrait"
            
        # Añadir género si está especificado
        gender = visual_style.get("gender", "neutral")
        if gender == "male":
            base_prompt = f"Male {base_prompt}"
        elif gender == "female":
            base_prompt = f"Female {base_prompt}"
            
        # Añadir edad si está especificada
        age = visual_style.get("age")
        if age:
            base_prompt = f"{age} years old {base_prompt}"
            
        # Añadir características específicas
        features = visual_style.get("features", [])
        if features:
            features_str = ", ".join(features)
            base_prompt = f"{base_prompt}, {features_str}"
            
        # Prompt final
        prompt = f"{base_prompt}, portrait, high quality, detailed"
        
        # Verificar caché
        cache_key = f"avatar_{prompt}_{service}"
        if self.config.get("cache_enabled", True) and cache_key in self.cache:
            logger.info(f"Using cached avatar for {name}")
            return self.cache[cache_key]
            
        # Generar avatar según el servicio
        try:
            if service == "stable_diffusion":
                avatar_url = self._generate_with_stable_diffusion(prompt)
            elif service == "leonardo":
                avatar_url = self._generate_with_leonardo(prompt)
            elif service == "midjourney":
                avatar_url = self._generate_with_midjourney(prompt)
            else:
                # Servicio no implementado, usar placeholder
                avatar_url = ""
                logger.warning(f"Service {service} not implemented for avatar generation")
                
            # Crear resultado
            avatar = {
                "url": avatar_url,
                "prompt": prompt,
                "service": service,
                "created_at": time.time()
            }
            
            # Guardar en caché
            if self.config.get("cache_enabled", True):
                self.cache[cache_key] = avatar
                
            return avatar
            
        except Exception as e:
            logger.error(f"Error generating avatar: {str(e)}")
            # Retornar avatar vacío
            return {
                "url": "",
                "prompt": prompt,
                "service": "none",
                "created_at": time.time()
            }

    def generate_color_scheme(self, niche: str, personality: Dict[str, Any]) -> Dict[str, str]:
        """
        Genera un esquema de colores basado en el nicho y personalidad
        
        Args:
            niche: Nicho del personaje
            personality: Diccionario con atributos de personalidad
            
        Returns:
            Diccionario con colores para diferentes elementos
        """
        # Paletas base por nicho
        niche_palettes = {
            "finance": {
                "primary": ["#1F3A93", "#2E86C1", "#26A65B", "#013243"],
                "secondary": ["#F5AB35", "#E67E22", "#F1C40F", "#6C7A89"],
                "accent": ["#E74C3C", "#9B59B6", "#3498DB", "#1ABC9C"]
            },
            "health": {
                "primary": ["#26A65B", "#2ECC71", "#16A085", "#27AE60"],
                "secondary": ["#3498DB", "#1F3A93", "#6BB9F0", "#1E8BC3"],
                "accent": ["#F62459", "#F5AB35", "#E67E22", "#F1C40F"]
            },
            "gaming": {
                                "primary": ["#8E44AD", "#9B59B6", "#3A539B", "#E74C3C"],
                "secondary": ["#2ECC71", "#1ABC9C", "#3498DB", "#F62459"],
                "accent": ["#F1C40F", "#F5AB35", "#E67E22", "#4ECDC4"]
            },
            "technology": {
                "primary": ["#3498DB", "#2980B9", "#1F3A93", "#22313F"],
                "secondary": ["#6BB9F0", "#8E44AD", "#9B59B6", "#6C7A89"],
                "accent": ["#F62459", "#F5AB35", "#E67E22", "#4ECDC4"]
            },
            "humor": {
                "primary": ["#F62459", "#F22613", "#F9690E", "#F39C12"],
                "secondary": ["#9B59B6", "#8E44AD", "#3498DB", "#2ECC71"],
                "accent": ["#F1C40F", "#F5AB35", "#E67E22", "#4ECDC4"]
            },
            "education": {
                "primary": ["#3498DB", "#2980B9", "#1F3A93", "#22313F"],
                "secondary": ["#27AE60", "#2ECC71", "#16A085", "#1ABC9C"],
                "accent": ["#F1C40F", "#F5AB35", "#E67E22", "#F62459"]
            }
        }
        
        # Seleccionar paleta según nicho o usar una genérica
        palette = niche_palettes.get(niche, {
            "primary": ["#3498DB", "#2980B9", "#1F3A93", "#22313F"],
            "secondary": ["#27AE60", "#2ECC71", "#16A085", "#1ABC9C"],
            "accent": ["#F1C40F", "#F5AB35", "#E67E22", "#F62459"]
        })
        
        # Seleccionar colores según personalidad
        enthusiasm = personality.get("enthusiasm", 0.5)
        formality = personality.get("formality", 0.5)
        
        # Más entusiasta = colores más vibrantes
        if enthusiasm > 0.7:
            primary_idx = random.randint(0, 1)  # Colores más vibrantes
            accent_idx = random.randint(0, 1)   # Acentos más vibrantes
        elif enthusiasm < 0.3:
            primary_idx = random.randint(2, 3)  # Colores más sobrios
            accent_idx = random.randint(2, 3)   # Acentos más sobrios
        else:
            primary_idx = random.randint(0, 3)
            accent_idx = random.randint(0, 3)
            
        # Más formal = colores más sobrios para secundarios
        if formality > 0.7:
            secondary_idx = random.randint(2, 3)  # Colores secundarios más sobrios
        elif formality < 0.3:
            secondary_idx = random.randint(0, 1)  # Colores secundarios más vibrantes
        else:
            secondary_idx = random.randint(0, 3)
            
        # Crear esquema de colores
        color_scheme = {
            "primary": palette["primary"][primary_idx],
            "secondary": palette["secondary"][secondary_idx],
            "accent": palette["accent"][accent_idx],
            "background": "#FFFFFF",  # Fondo blanco por defecto
            "text": "#333333",        # Texto oscuro por defecto
            "light_text": "#777777"   # Texto claro por defecto
        }
        
        # Ajustar colores de fondo y texto según formalidad
        if formality > 0.7:
            color_scheme["background"] = "#FFFFFF"  # Fondo blanco para formal
            color_scheme["text"] = "#333333"        # Texto oscuro para formal
        elif formality < 0.3:
            # Para menos formal, posibilidad de fondo oscuro
            if random.random() < 0.3:
                color_scheme["background"] = "#222222"  # Fondo oscuro
                color_scheme["text"] = "#EEEEEE"        # Texto claro
                color_scheme["light_text"] = "#AAAAAA"  # Texto claro secundario
        
        return color_scheme

    def generate_typography(self, personality: Dict[str, Any]) -> Dict[str, str]:
        """
        Genera una tipografía basada en la personalidad
        
        Args:
            personality: Diccionario con atributos de personalidad
            
        Returns:
            Diccionario con fuentes para diferentes elementos
        """
        # Mapeo de personalidad a estilos tipográficos
        formality = personality.get("formality", 0.5)
        enthusiasm = personality.get("enthusiasm", 0.5)
        
        # Fuentes para títulos
        if formality > 0.7:  # Formal
            heading_fonts = ["Georgia", "Playfair Display", "Merriweather", "Times New Roman"]
        elif formality < 0.3:  # Informal
            heading_fonts = ["Montserrat", "Poppins", "Raleway", "Bebas Neue"]
        else:  # Neutral
            heading_fonts = ["Roboto", "Open Sans", "Lato", "Source Sans Pro"]
            
        # Fuentes para cuerpo
        if formality > 0.7:  # Formal
            body_fonts = ["Georgia", "Merriweather", "Lora", "PT Serif"]
        elif formality < 0.3:  # Informal
            body_fonts = ["Roboto", "Open Sans", "Lato", "Nunito"]
        else:  # Neutral
            body_fonts = ["Roboto", "Open Sans", "Source Sans Pro", "Noto Sans"]
            
        # Seleccionar fuentes
        heading_font = random.choice(heading_fonts)
        body_font = random.choice(body_fonts)
        
        # Determinar pesos y tamaños
        if enthusiasm > 0.7:  # Más entusiasta = más bold
            heading_weight = "700"  # Bold
            body_weight = "400"     # Regular
        elif enthusiasm < 0.3:  # Menos entusiasta = más light
            heading_weight = "400"  # Regular
            body_weight = "300"     # Light
        else:  # Neutral
            heading_weight = "600"  # Semi-bold
            body_weight = "400"     # Regular
            
        # Crear tipografía
        typography = {
            "heading_font": heading_font,
            "body_font": body_font,
            "heading_weight": heading_weight,
            "body_weight": body_weight,
            "heading_size": "24px",
            "subheading_size": "18px",
            "body_size": "16px",
            "small_size": "14px"
        }
        
        return typography

    def generate_visual_elements(self, niche: str, visual_style: Dict[str, Any], 
                                personality: Dict[str, Any], service: str) -> Dict[str, Any]:
        """
        Genera elementos visuales adicionales para el personaje
        
        Args:
            niche: Nicho del personaje
            visual_style: Diccionario con preferencias de estilo visual
            personality: Diccionario con atributos de personalidad
            service: Servicio de generación a utilizar
            
        Returns:
            Diccionario con elementos visuales
        """
        # Elementos base
        elements = {
            "logo": {},
            "background": {},
            "icons": [],
            "patterns": [],
            "overlays": []
        }
        
        # Generar logo si se solicita
        if visual_style.get("generate_logo", False):
            elements["logo"] = self._generate_logo(niche, visual_style, personality, service)
            
        # Generar fondo si se solicita
        if visual_style.get("generate_background", False):
            elements["background"] = self._generate_background(niche, visual_style, personality, service)
            
        # Generar iconos según nicho
        elements["icons"] = self._generate_icons(niche)
        
        # Generar patrones según estilo
        elements["patterns"] = self._generate_patterns(visual_style)
        
        # Generar overlays según estilo
        elements["overlays"] = self._generate_overlays(visual_style)
        
        return elements

    def _generate_logo(self, niche: str, visual_style: Dict[str, Any], 
                      personality: Dict[str, Any], service: str) -> Dict[str, Any]:
        """Genera un logo para el personaje"""
        # Construir prompt para el logo
        style = visual_style.get("style", "realistic")
        name = visual_style.get("name", "Content Creator")
        
        if style == "minimalist":
            prompt = f"Minimalist logo for {name}, {niche} content creator, clean lines, simple design"
        elif style == "cartoon":
            prompt = f"Cartoon style logo for {name}, {niche} content creator, vibrant colors"
        elif style == "3d":
            prompt = f"3D rendered logo for {name}, {niche} content creator, high quality render"
        else:
            prompt = f"Professional logo for {name}, {niche} content creator, modern design"
            
        # Verificar caché
        cache_key = f"logo_{prompt}_{service}"
        if self.config.get("cache_enabled", True) and cache_key in self.cache:
            return self.cache[cache_key]
            
        # Generar logo según el servicio
        try:
            if service == "stable_diffusion":
                logo_url = self._generate_with_stable_diffusion(prompt)
            elif service == "leonardo":
                logo_url = self._generate_with_leonardo(prompt)
            else:
                # Servicio no implementado, usar placeholder
                logo_url = ""
                logger.warning(f"Service {service} not implemented for logo generation")
                
            # Crear resultado
            logo = {
                "url": logo_url,
                "prompt": prompt,
                "service": service,
                "created_at": time.time()
            }
            
            # Guardar en caché
            if self.config.get("cache_enabled", True):
                self.cache[cache_key] = logo
                
            return logo
            
        except Exception as e:
            logger.error(f"Error generating logo: {str(e)}")
            # Retornar logo vacío
            return {
                "url": "",
                "prompt": prompt,
                "service": "none",
                "created_at": time.time()
            }

    def _generate_background(self, niche: str, visual_style: Dict[str, Any], 
                            personality: Dict[str, Any], service: str) -> Dict[str, Any]:
        """Genera un fondo para el personaje"""
        # Construir prompt para el fondo
        style = visual_style.get("style", "realistic")
        
        # Fondos según nicho
        niche_backgrounds = {
            "finance": "office environment, financial charts, professional setting",
            "health": "gym, nature, healthy environment, wellness space",
            "gaming": "gaming setup, colorful lights, game-themed environment",
            "technology": "tech workspace, modern office, digital environment",
            "humor": "stage, colorful background, entertainment setting",
            "education": "library, classroom, academic environment"
        }
        
        background_desc = niche_backgrounds.get(niche, "professional content creation space")
        
        if style == "minimalist":
            prompt = f"Minimalist {background_desc}, clean lines, simple design"
        elif style == "cartoon":
            prompt = f"Cartoon style {background_desc}, vibrant colors"
        elif style == "3d":
            prompt = f"3D rendered {background_desc}, high quality render"
        else:
            prompt = f"Professional {background_desc}, modern design"
            
        # Verificar caché
        cache_key = f"background_{prompt}_{service}"
        if self.config.get("cache_enabled", True) and cache_key in self.cache:
            return self.cache[cache_key]
            
        # Generar fondo según el servicio
        try:
            if service == "stable_diffusion":
                background_url = self._generate_with_stable_diffusion(prompt)
            elif service == "leonardo":
                background_url = self._generate_with_leonardo(prompt)
            else:
                # Servicio no implementado, usar placeholder
                background_url = ""
                logger.warning(f"Service {service} not implemented for background generation")
                
            # Crear resultado
            background = {
                "url": background_url,
                "prompt": prompt,
                "service": service,
                "created_at": time.time()
            }
            
            # Guardar en caché
            if self.config.get("cache_enabled", True):
                self.cache[cache_key] = background
                
            return background
            
        except Exception as e:
            logger.error(f"Error generating background: {str(e)}")
            # Retornar fondo vacío
            return {
                "url": "",
                "prompt": prompt,
                "service": "none",
                "created_at": time.time()
            }

    def _generate_icons(self, niche: str) -> List[Dict[str, str]]:
        """Genera iconos relacionados con el nicho"""
        # Iconos por nicho
        niche_icons = {
            "finance": ["💰", "💸", "📈", "🚀", "💎", "🤑", "💹", "📊"],
            "health": ["💪", "🥗", "🧘", "🏃", "❤️", "🧠", "🥦", "🏋️"],
            "gaming": ["🎮", "🕹️", "🏆", "🔥", "⚔️", "🛡️", "🎯", "🎲"],
            "technology": ["💻", "📱", "🤖", "⚙️", "🔋", "🚀", "📡", "💾"],
            "humor": ["😂", "🤣", "😆", "😜", "🤪", "😎", "🎭", "🎪"],
            "education": ["📚", "🧠", "✏️", "🎓", "💡", "🔍", "📝", "🧮"]
        }
        
        # Seleccionar iconos para el nicho
        icons = niche_icons.get(niche, ["✨", "👍", "🙌", "🔥", "💯", "⭐", "🌟", "💫"])
        
        # Convertir a formato de diccionario
        icon_list = []
        for icon in icons:
            icon_list.append({
                "emoji": icon,
                "unicode": f"U+{ord(icon):X}",
                "description": f"{niche} related icon"
            })
            
        return icon_list

    def _generate_patterns(self, visual_style: Dict[str, Any]) -> List[Dict[str, str]]:
        """Genera patrones visuales según el estilo"""
        style = visual_style.get("style", "realistic")
        
        # Patrones por estilo
        style_patterns = {
            "minimalist": [
                {"name": "dots", "description": "Simple dot pattern", "color": "#EEEEEE"},
                {"name": "lines", "description": "Thin line pattern", "color": "#DDDDDD"},
                {"name": "grid", "description": "Minimal grid pattern", "color": "#F5F5F5"}
            ],
            "cartoon": [
                {"name": "bubbles", "description": "Cartoon bubble pattern", "color": "#F5AB35"},
                {"name": "stars", "description": "Star pattern", "color": "#F1C40F"},
                {"name": "waves", "description": "Wavy pattern", "color": "#3498DB"}
            ],
            "3d": [
                {"name": "cubes", "description": "3D cube pattern", "color": "#9B59B6"},
                {"name": "spheres", "description": "3D sphere pattern", "color": "#2ECC71"},
                {"name": "pyramids", "description": "3D pyramid pattern", "color": "#E74C3C"}
            ],
            "realistic": [
                {"name": "subtle_gradient", "description": "Subtle gradient", "color": "#F5F5F5"},
                {"name": "noise", "description": "Subtle noise texture", "color": "#EEEEEE"},
                {"name": "paper", "description": "Paper texture", "color": "#FFFFFF"}
            ]
        }
        
        # Seleccionar patrones para el estilo
        patterns = style_patterns.get(style, style_patterns["realistic"])
        
        # Seleccionar aleatoriamente 2 patrones
        if len(patterns) > 2:
            selected_patterns = random.sample(patterns, 2)
        else:
            selected_patterns = patterns
            
        return selected_patterns

    def _generate_overlays(self, visual_style: Dict[str, Any]) -> List[Dict[str, str]]:
        """Genera overlays visuales según el estilo"""
        style = visual_style.get("style", "realistic")
        
        # Overlays por estilo
        style_overlays = {
            "minimalist": [
                {"name": "corner_accent", "description": "Minimal corner accent", "opacity": 0.2},
                {"name": "subtle_vignette", "description": "Subtle vignette", "opacity": 0.1}
            ],
            "cartoon": [
                {"name": "speech_bubble", "description": "Speech bubble overlay", "opacity": 0.8},
                {"name": "comic_frame", "description": "Comic frame border", "opacity": 0.7},
                {"name": "action_lines", "description": "Action lines", "opacity": 0.6}
            ],
            "3d": [
                {"name": "depth_shadow", "description": "3D depth shadow", "opacity": 0.4},
                {"name": "highlight_glow", "description": "Highlight glow", "opacity": 0.5}
            ],
            "realistic": [
                {"name": "light_leak", "description": "Light leak effect", "opacity": 0.3},
                {"name": "subtle_shadow", "description": "Subtle shadow overlay", "opacity": 0.2},
                {"name": "grain", "description": "Film grain", "opacity": 0.1}
            ]
        }
        
        # Seleccionar overlays para el estilo
        overlays = style_overlays.get(style, style_overlays["realistic"])
        
        # Seleccionar aleatoriamente 1-2 overlays
        num_overlays = random.randint(1, min(2, len(overlays)))
        selected_overlays = random.sample(overlays, num_overlays)
            
        return selected_overlays

    def _generate_style_guide(self, visual_style: Dict[str, Any], 
                             personality: Dict[str, Any]) -> Dict[str, Any]:
        """Genera una guía de estilo para el personaje"""
        # Extraer atributos relevantes
        formality = personality.get("formality", 0.5)
        enthusiasm = personality.get("enthusiasm", 0.5)
        style = visual_style.get("style", "realistic")
        
        # Determinar reglas de composición
        if formality > 0.7:  # Formal
            composition = "Centered, balanced compositions with professional framing"
        elif formality < 0.3:  # Informal
            composition = "Dynamic, asymmetrical compositions with creative framing"
        else:  # Neutral
            composition = "Balanced compositions with natural framing"
            
        # Determinar reglas de iluminación
        if enthusiasm > 0.7:  # Entusiasta
            lighting = "Bright, high-contrast lighting with vibrant colors"
        elif enthusiasm < 0.3:  # Calmado
            lighting = "Soft, low-contrast lighting with subdued colors"
        else:  # Neutral
            lighting = "Natural, balanced lighting with good contrast"
            
        # Determinar reglas de edición
        if style == "cartoon":
            editing = "Vibrant colors, bold outlines, simplified details"
        elif style == "minimalist":
            editing = "Clean, minimal editing with focus on negative space"
        elif style == "3d":
            editing = "Rich textures, depth effects, dramatic lighting"
        else:  # realistic
            editing = "Natural colors, subtle enhancements, realistic details"
            
        # Crear guía de estilo
        style_guide = {
            "composition": composition,
            "lighting": lighting,
            "editing": editing,
            "consistency_rules": [
                "Maintain consistent color palette across all visuals",
                "Use the same typography hierarchy for all content",
                "Keep visual elements aligned with personality traits",
                f"Maintain {style} style consistently"
            ],
            "do_not": [
                "Mix different visual styles",
                "Use colors outside the defined palette",
                "Change typography without reason",
                "Create visuals that contradict the character's personality"
            ]
        }
        
        return style_guide

    def _generate_with_stable_diffusion(self, prompt: str) -> str:
        """
        Genera una imagen con Stable Diffusion
        
        Args:
            prompt: Prompt para la generación
            
        Returns:
            URL o ruta a la imagen generada
        """
        try:
            # Verificar si hay implementación local
            service_config = self.config.get("services", {}).get("stable_diffusion", {})
            model_path = service_config.get("model_path", "models/stable-diffusion-xl")
            
            # Aquí iría la implementación real con la biblioteca de Stable Diffusion
            # Por ahora, simular generación
            logger.info(f"Generating image with Stable Diffusion using prompt: {prompt}")
            
            # Simular generación (en implementación real, aquí se llamaría al modelo)
            # Generar un nombre de archivo único basado en el prompt y timestamp
            filename = f"sd_{hash(prompt)}_{int(time.time())}.png"
            output_path = os.path.join("data", "generated_images", filename)
            
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # En una implementación real, aquí se guardaría la imagen generada
            # Por ahora, crear una imagen de placeholder
            self._create_placeholder_image(output_path, prompt)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating with Stable Diffusion: {str(e)}")
            return ""

    def _generate_with_leonardo(self, prompt: str) -> str:
        """
        Genera una imagen con Leonardo AI
        
        Args:
            prompt: Prompt para la generación
            
        Returns:
            URL a la imagen generada
        """
        try:
            # Obtener configuración de Leonardo
            service_config = self.config.get("services", {}).get("leonardo", {})
            api_key = service_config.get("api_key", "")
            base_url = service_config.get("base_url", "https://cloud.leonardo.ai/api/rest/v1")
            
            if not api_key:
                logger.error("Leonardo API key not configured")
                return ""
                
            # Configurar headers
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            # Crear payload para la generación
            payload = {
                "prompt": prompt,
                "negative_prompt": "low quality, blurry, distorted, deformed",
                "modelId": "6bef9f1b-29cb-40c7-b9df-32b51c1f67d3",  # Leonardo Creative
                "width": 512,
                "height": 512,
                "num_images": 1,
                "guidance_scale": 7
            }
            
            # Realizar solicitud para iniciar generación
            response = requests.post(
                f"{base_url}/generations",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Error from Leonardo API: {response.text}")
                return ""
                
            generation_id = response.json().get("id")
            
            # Esperar a que la generación se complete
            max_attempts = 30
            for attempt in range(max_attempts):
                # Verificar estado
                status_response = requests.get(
                    f"{base_url}/generations/{generation_id}",
                    headers=headers
                )
                
                if status_response.status_code != 200:
                    logger.error(f"Error checking generation status: {status_response.text}")
                    return ""
                    
                status_data = status_response.json()
                
                if status_data.get("status") == "COMPLETE":
                    # Generación completada
                    generated_images = status_data.get("generated_images", [])
                    if generated_images:
                        # Retornar URL de la primera imagen
                        return generated_images[0].get("url", "")
                    break
                    
                elif status_data.get("status") == "FAILED":
                    logger.error("Leonardo generation failed")
                    return ""
                    
                # Esperar antes de verificar de nuevo
                time.sleep(2)
                
            logger.error("Leonardo generation timed out")
            return ""
            
        except Exception as e:
            logger.error(f"Error generating with Leonardo: {str(e)}")
            return ""

    def _generate_with_midjourney(self, prompt: str) -> str:
        """
        Genera una imagen con Midjourney
        
        Args:
            prompt: Prompt para la generación
            
        Returns:
            URL a la imagen generada
        """
        # Nota: Esta es una implementación simulada ya que Midjourney no tiene API oficial
        logger.warning("Midjourney API not implemented, using placeholder")
        
        # Crear un placeholder
        filename = f"mj_{hash(prompt)}_{int(time.time())}.png"
        output_path = os.path.join("data", "generated_images", filename)
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Crear imagen de placeholder
        self._create_placeholder_image(output_path, prompt)
        
        return output_path

    def _create_placeholder_image(self, path: str, text: str = "Generated Image") -> None:
        """
        Crea una imagen de placeholder con texto
        
        Args:
            path: Ruta donde guardar la imagen
            text: Texto a mostrar en la imagen
        """
        try:
            # Crear imagen
            width, height = 512, 512
            image = Image.new("RGB", (width, height), color=(240, 240, 240))
            
            # Guardar imagen
            image.save(path)
            
            logger.info(f"Created placeholder image at {path}")
            
        except Exception as e:
            logger.error(f"Error creating placeholder image: {str(e)}")

    def get_visual_identity(self, identity_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene una identidad visual por su ID
        
        Args:
            identity_id: ID de la identidad visual
            
        Returns:
            Diccionario con la identidad visual o None si no existe
        """
        try:
            return self.knowledge_base.get_visual_identity(identity_id)
        except Exception as e:
            logger.error(f"Error getting visual identity: {str(e)}")
            return None

    def update_visual_identity(self, identity_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualiza una identidad visual existente
        
        Args:
            identity_id: ID de la identidad visual
            updates: Diccionario con actualizaciones
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        try:
            # Obtener identidad actual
            identity = self.knowledge_base.get_visual_identity(identity_id)
            if not identity:
                logger.warning(f"Visual identity not found: {identity_id}")
                return False
                
            # Aplicar actualizaciones
            for key, value in updates.items():
                if key != "id" and key != "created_at":
                    identity[key] = value
                    
            # Guardar actualización
            self.knowledge_base.save_visual_identity(identity_id, identity)
            
            logger.info(f"Updated visual identity: {identity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating visual identity: {str(e)}")
            return False

    def clear_cache(self) -> None:
        """Limpia la caché de identidades visuales"""
        self.cache = {}
        logger.info("Cleared visual identity cache")

    def get_service_status(self) -> Dict[str, bool]:
        """
        Obtiene el estado de los servicios de generación visual
        
        Returns:
            Diccionario con el estado de cada servicio
        """
        services = self.config.get("services", {})
        status = {}
        
        for service_name, service_config in services.items():
            status[service_name] = service_config.get("active", False)
            
        return status

    def update_service_config(self, service_name: str, config_updates: Dict[str, Any]) -> bool:
        """
        Actualiza la configuración de un servicio
        
        Args:
            service_name: Nombre del servicio
            config_updates: Actualizaciones de configuración
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        try:
            services = self.config.get("services", {})
            
            if service_name not in services:
                logger.warning(f"Service not found: {service_name}")
                return False
                
            # Aplicar actualizaciones
            for key, value in config_updates.items():
                services[service_name][key] = value
                
            # Actualizar configuración
            self.config["services"] = services
            
            # Guardar en la base de conocimiento
            self.knowledge_base.save_visual_services_config(self.config)
            
            logger.info(f"Updated service config: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating service config: {str(e)}")
            return False