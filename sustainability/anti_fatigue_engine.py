"""
Anti-Fatigue Engine Module

Este módulo implementa estrategias para prevenir la fatiga de contenido,
manteniendo el interés de la audiencia a través de variaciones controladas
en formatos, estilos, personajes y narrativas.

Características principales:
- Rotación inteligente de formatos de contenido
- Variación de estilos visuales y narrativos
- Gestión de frecuencia de publicación
- Detección de señales de fatiga en la audiencia
- Recomendaciones para renovar contenido
- Planificación de ciclos de contenido
"""

import os
import json
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/anti_fatigue.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AntiFatigue")

class AntiFatigueEngine:
    """
    Motor de prevención de fatiga de contenido para mantener el interés
    de la audiencia a largo plazo.
    """
    
    def __init__(self, config_path: str = "config/anti_fatigue_config.json"):
        """
        Inicializa el motor anti-fatiga.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        
        # Cargar configuración
        self._load_config()
        
        # Historial de contenido
        self.content_history = {}
        
        # Métricas de fatiga
        self.fatigue_metrics = {}
        
        # Ciclos de contenido
        self.content_cycles = {}
        
        # Recomendaciones actuales
        self.current_recommendations = {}
        
        logger.info("Motor anti-fatiga inicializado")
    
    def _load_config(self) -> None:
        """Carga la configuración desde el archivo JSON."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # Configuración por defecto
                self.config = {
                    "fatigue_threshold": 0.15,  # 15% de caída en engagement
                    "rotation_frequency": {
                        "format": 5,        # Cada 5 videos
                        "style": 3,         # Cada 3 videos
                        "character": 7,     # Cada 7 videos
                        "narrative": 10     # Cada 10 videos
                    },
                    "content_formats": [
                        "tutorial", "storytelling", "reaction", "challenge",
                        "review", "interview", "compilation", "behind_the_scenes",
                        "Q&A", "day_in_life", "top_list", "explainer"
                    ],
                    "visual_styles": [
                        "minimal", "vibrant", "retro", "futuristic", 
                        "hand_drawn", "photorealistic", "comic", "cinematic",
                        "neon", "pastel", "monochrome", "gradient"
                    ],
                    "narrative_structures": [
                        "problem_solution", "curiosity_gap", "story_arc", 
                        "how_to", "listicle", "case_study", "comparison",
                        "before_after", "day_in_life", "behind_the_scenes",
                        "myth_busting", "prediction"
                    ],
                    "posting_frequency": {
                        "max_per_day": 2,
                        "min_days_between_similar": 3,
                        "optimal_time_variance": 2  # Horas de variación
                    },
                    "fatigue_signals": {
                        "engagement_drop_threshold": 0.15,  # 15% de caída
                        "retention_drop_threshold": 0.10,   # 10% de caída
                        "comment_sentiment_threshold": 0.20, # 20% de caída
                        "click_through_threshold": 0.25,    # 25% de caída
                        "unsubscribe_increase_threshold": 0.30 # 30% de aumento
                    },
                    "refresh_strategies": {
                        "minor_refresh": {
                            "threshold": 0.10,  # 10% de caída
                            "actions": ["style", "thumbnail", "title"]
                        },
                        "moderate_refresh": {
                            "threshold": 0.20,  # 20% de caída
                            "actions": ["format", "character", "narrative"]
                        },
                        "major_refresh": {
                            "threshold": 0.30,  # 30% de caída
                            "actions": ["topic", "series", "collaboration"]
                        }
                    },
                    "content_cycles": {
                        "micro_cycle": 7,    # 7 días
                        "mid_cycle": 30,     # 30 días
                        "macro_cycle": 90    # 90 días
                    },
                    "seasonal_adjustments": {
                        "summer": ["outdoor", "travel", "adventure"],
                        "fall": ["education", "productivity", "reflection"],
                        "winter": ["cozy", "holidays", "goals"],
                        "spring": ["renewal", "growth", "creativity"]
                    },
                    "audience_segments": {
                        "new_followers": {
                            "max_age": 30,  # Días
                            "content_bias": ["introductory", "foundational"]
                        },
                        "core_audience": {
                            "engagement_rate": 0.10,  # 10% o más
                            "content_bias": ["advanced", "insider"]
                        },
                        "casual_viewers": {
                            "view_frequency": 0.5,  # Menos de 1 de cada 2 videos
                            "content_bias": ["entertaining", "trending"]
                        }
                    },
                    "platform_specific": {
                        "youtube": {
                            "format_bias": ["tutorial", "explainer", "review"],
                            "optimal_duration": [8, 12]  # Minutos
                        },
                        "tiktok": {
                            "format_bias": ["challenge", "storytelling", "reaction"],
                            "optimal_duration": [15, 60]  # Segundos
                        },
                        "instagram": {
                            "format_bias": ["behind_the_scenes", "day_in_life", "aesthetic"],
                            "optimal_duration": [30, 90]  # Segundos
                        }
                    }
                }
                
                # Guardar configuración por defecto
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4)
            
            logger.info(f"Configuración anti-fatiga cargada")
            
        except Exception as e:
            logger.error(f"Error cargando configuración anti-fatiga: {str(e)}")
            # Configuración mínima de respaldo
            self.config = {
                "fatigue_threshold": 0.15,
                "rotation_frequency": {
                    "format": 5, "style": 3, "character": 7, "narrative": 10
                },
                "content_formats": ["tutorial", "storytelling", "reaction"],
                "visual_styles": ["minimal", "vibrant", "retro"],
                "narrative_structures": ["problem_solution", "curiosity_gap", "story_arc"]
            }
    
    def save_config(self) -> None:
        """Guarda la configuración actual en el archivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuración anti-fatiga guardada en {self.config_path}")
        except Exception as e:
            logger.error(f"Error guardando configuración anti-fatiga: {str(e)}")
    
    def update_content_history(self, channel_id: str, content_data: Dict[str, Any]) -> None:
        """
        Actualiza el historial de contenido para un canal específico.
        
        Args:
            channel_id: Identificador del canal
            content_data: Datos del contenido publicado
                {
                    "content_id": "abc123",
                    "title": "Título del video",
                    "format": "tutorial",
                    "style": "minimal",
                    "character": "tech_expert",
                    "narrative": "problem_solution",
                    "topic": "crypto",
                    "publish_date": "2023-06-15",
                    "platform": "youtube",
                    "duration": 480,  # segundos
                    "metrics": {
                        "views": 1000,
                        "engagement_rate": 0.05,
                        "retention_rate": 0.60,
                        "click_through_rate": 0.08,
                        "comment_sentiment": 0.75  # 0-1, 1 siendo positivo
                    }
                }
        """
        try:
            # Inicializar historial del canal si no existe
            if channel_id not in self.content_history:
                self.content_history[channel_id] = []
            
            # Añadir contenido al historial
            self.content_history[channel_id].append(content_data)
            
            # Ordenar por fecha de publicación
            self.content_history[channel_id] = sorted(
                self.content_history[channel_id],
                key=lambda x: x.get("publish_date", "2000-01-01")
            )
            
            # Limitar historial a los últimos 100 contenidos
            if len(self.content_history[channel_id]) > 100:
                self.content_history[channel_id] = self.content_history[channel_id][-100:]
            
            logger.info(f"Historial actualizado para canal {channel_id}, formato: {content_data.get('format')}")
            
        except Exception as e:
            logger.error(f"Error actualizando historial de contenido: {str(e)}")
    
    def detect_fatigue_signals(self, channel_id: str, window_size: int = 10) -> Dict[str, float]:
        """
        Detecta señales de fatiga en el contenido reciente.
        
        Args:
            channel_id: Identificador del canal
            window_size: Tamaño de la ventana de análisis
            
        Returns:
            Diccionario con métricas de fatiga
        """
        try:
            # Verificar si hay suficiente historial
            if channel_id not in self.content_history or len(self.content_history[channel_id]) < window_size:
                logger.warning(f"Historial insuficiente para canal {channel_id}")
                return {
                    "fatigue_detected": False,
                    "engagement_trend": 0.0,
                    "retention_trend": 0.0,
                    "sentiment_trend": 0.0,
                    "click_through_trend": 0.0,
                    "overall_fatigue_score": 0.0
                }
            
            # Obtener contenido reciente
            recent_content = self.content_history[channel_id][-window_size:]
            
            # Calcular tendencias
            engagement_values = [c.get("metrics", {}).get("engagement_rate", 0) for c in recent_content]
            retention_values = [c.get("metrics", {}).get("retention_rate", 0) for c in recent_content]
            sentiment_values = [c.get("metrics", {}).get("comment_sentiment", 0) for c in recent_content]
            ctr_values = [c.get("metrics", {}).get("click_through_rate", 0) for c in recent_content]
            
            # Dividir en dos mitades para comparar tendencias
            half_size = window_size // 2
            
            # Calcular tendencias (cambio porcentual entre primera y segunda mitad)
            engagement_trend = self._calculate_trend(engagement_values, half_size)
            retention_trend = self._calculate_trend(retention_values, half_size)
            sentiment_trend = self._calculate_trend(sentiment_values, half_size)
            ctr_trend = self._calculate_trend(ctr_values, half_size)
            
            # Calcular puntuación general de fatiga (promedio ponderado de tendencias negativas)
            weights = {
                "engagement": 0.4,
                "retention": 0.3,
                "sentiment": 0.2,
                "ctr": 0.1
            }
            
            overall_score = (
                engagement_trend * weights["engagement"] +
                retention_trend * weights["retention"] +
                sentiment_trend * weights["sentiment"] +
                ctr_trend * weights["ctr"]
            )
            
            # Determinar si hay fatiga según umbrales
            thresholds = self.config["fatigue_signals"]
            fatigue_detected = (
                engagement_trend <= -thresholds["engagement_drop_threshold"] or
                retention_trend <= -thresholds["retention_drop_threshold"] or
                sentiment_trend <= -thresholds["comment_sentiment_threshold"] or
                ctr_trend <= -thresholds["click_through_threshold"]
            )
            
            # Guardar métricas de fatiga
            fatigue_metrics = {
                "fatigue_detected": fatigue_detected,
                "engagement_trend": engagement_trend,
                "retention_trend": retention_trend,
                "sentiment_trend": sentiment_trend,
                "click_through_trend": ctr_trend,
                "overall_fatigue_score": overall_score,
                "timestamp": datetime.now().isoformat()
            }
            
            self.fatigue_metrics[channel_id] = fatigue_metrics
            
            logger.info(f"Análisis de fatiga para canal {channel_id}: {fatigue_detected}, score: {overall_score:.2f}")
            return fatigue_metrics
            
        except Exception as e:
            logger.error(f"Error detectando señales de fatiga: {str(e)}")
            return {
                "fatigue_detected": False,
                "engagement_trend": 0.0,
                "retention_trend": 0.0,
                "sentiment_trend": 0.0,
                "click_through_trend": 0.0,
                "overall_fatigue_score": 0.0,
                "error": str(e)
            }
    
    def _calculate_trend(self, values: List[float], half_size: int) -> float:
        """
        Calcula la tendencia entre dos mitades de una lista de valores.
        
        Args:
            values: Lista de valores
            half_size: Tamaño de cada mitad
            
        Returns:
            Cambio porcentual entre las mitades
        """
        if not values or len(values) < 2 or half_size < 1:
            return 0.0
        
        first_half = values[:half_size]
        second_half = values[-half_size:]
        
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        
        if first_avg == 0:
            return 0.0
        
        return (second_avg - first_avg) / first_avg
    
    def recommend_content_variation(self, channel_id: str, niche: str, 
                                   platform: str = "youtube") -> Dict[str, Any]:
        """
        Recomienda variaciones de contenido para prevenir fatiga.
        
        Args:
            channel_id: Identificador del canal
            niche: Nicho del canal (finanzas, salud, gaming, etc.)
            platform: Plataforma objetivo
            
        Returns:
            Recomendaciones de variación de contenido
        """
        try:
            # Detectar fatiga
            fatigue_metrics = self.detect_fatigue_signals(channel_id)
            fatigue_score = fatigue_metrics["overall_fatigue_score"]
            fatigue_detected = fatigue_metrics["fatigue_detected"]
            
            # Obtener historial reciente
            recent_content = self.content_history.get(channel_id, [])[-10:] if channel_id in self.content_history else []
            
            # Extraer formatos, estilos, personajes y narrativas recientes
            recent_formats = [c.get("format") for c in recent_content if "format" in c]
            recent_styles = [c.get("style") for c in recent_content if "style" in c]
            recent_characters = [c.get("character") for c in recent_content if "character" in c]
            recent_narratives = [c.get("narrative") for c in recent_content if "narrative" in c]
            
            # Determinar nivel de refresco necesario
            refresh_level = "minor_refresh"
            if fatigue_score <= -0.30:
                refresh_level = "major_refresh"
            elif fatigue_score <= -0.20:
                refresh_level = "moderate_refresh"
            elif fatigue_score <= -0.10:
                refresh_level = "minor_refresh"
            
            # Obtener estrategias de refresco
            refresh_strategies = self.config["refresh_strategies"].get(refresh_level, {}).get("actions", [])
            
            # Recomendaciones base
            recommendations = {
                "fatigue_detected": fatigue_detected,
                "fatigue_score": fatigue_score,
                "refresh_level": refresh_level,
                "format": self._recommend_format(recent_formats, platform),
                "style": self._recommend_style(recent_styles),
                "narrative": self._recommend_narrative(recent_narratives),
                "character_rotation": self._should_rotate_character(recent_characters),
                "posting_frequency": self._recommend_posting_frequency(channel_id, platform),
                "priority_actions": refresh_strategies,
                "timestamp": datetime.now().isoformat()
            }
            
            # Añadir recomendaciones específicas de plataforma
            if platform in self.config["platform_specific"]:
                platform_config = self.config["platform_specific"][platform]
                recommendations["optimal_duration"] = platform_config.get("optimal_duration", [60, 180])
                
                # Ajustar formato según bias de plataforma si hay fatiga
                if fatigue_detected and "format" in refresh_strategies:
                    platform_formats = platform_config.get("format_bias", [])
                    if platform_formats:
                        # Elegir un formato preferido para la plataforma que no se haya usado recientemente
                        available_formats = [f for f in platform_formats if f not in recent_formats[-3:]]
                        if available_formats:
                            recommendations["format"] = random.choice(available_formats)
            
            # Añadir ajustes estacionales
            current_month = datetime.now().month
            season = ""
            if 3 <= current_month <= 5:
                season = "spring"
            elif 6 <= current_month <= 8:
                season = "summer"
            elif 9 <= current_month <= 11:
                season = "fall"
            else:
                season = "winter"
                
            if season in self.config["seasonal_adjustments"]:
                seasonal_themes = self.config["seasonal_adjustments"][season]
                recommendations["seasonal_themes"] = seasonal_themes
            
            # Guardar recomendaciones actuales
            self.current_recommendations[channel_id] = recommendations
            
            logger.info(f"Recomendaciones generadas para canal {channel_id}, nivel: {refresh_level}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones de variación: {str(e)}")
            return {
                "fatigue_detected": False,
                "fatigue_score": 0.0,
                "refresh_level": "minor_refresh",
                "format": random.choice(self.config["content_formats"]),
                "style": random.choice(self.config["visual_styles"]),
                "narrative": random.choice(self.config["narrative_structures"]),
                "character_rotation": False,
                "error": str(e)
            }
    
    def _recommend_format(self, recent_formats: List[str], platform: str = "youtube") -> str:
        """
        Recomienda un formato de contenido que no se haya usado recientemente.
        
        Args:
            recent_formats: Formatos usados recientemente
            platform: Plataforma objetivo
            
        Returns:
            Formato recomendado
        """
        # Obtener todos los formatos disponibles
        all_formats = self.config["content_formats"]
        
        # Filtrar formatos recientes (últimos 3)
        recent_formats_set = set(recent_formats[-3:]) if recent_formats else set()
        
        # Priorizar formatos específicos de plataforma
        platform_formats = []
        if platform in self.config["platform_specific"]:
            platform_formats = self.config["platform_specific"][platform].get("format_bias", [])
        
        # Candidatos: primero los de plataforma que no sean recientes, luego cualquiera no reciente
        candidates = [f for f in platform_formats if f not in recent_formats_set]
        if not candidates:
            candidates = [f for f in all_formats if f not in recent_formats_set]
        
        # Si no hay candidatos, usar cualquier formato
        if not candidates:
            candidates = all_formats
        
        return random.choice(candidates)
    
    def _recommend_style(self, recent_styles: List[str]) -> str:
        """
        Recomienda un estilo visual que no se haya usado recientemente.
        
        Args:
            recent_styles: Estilos usados recientemente
            
        Returns:
            Estilo recomendado
        """
        all_styles = self.config["visual_styles"]
        recent_styles_set = set(recent_styles[-2:]) if recent_styles else set()
        
        candidates = [s for s in all_styles if s not in recent_styles_set]
        if not candidates:
            candidates = all_styles
        
        return random.choice(candidates)
    
    def _recommend_narrative(self, recent_narratives: List[str]) -> str:
        """
        Recomienda una estructura narrativa que no se haya usado recientemente.
        
        Args:
            recent_narratives: Narrativas usadas recientemente
            
        Returns:
            Narrativa recomendada
        """
        all_narratives = self.config["narrative_structures"]
        recent_narratives_set = set(recent_narratives[-3:]) if recent_narratives else set()
        
        candidates = [n for n in all_narratives if n not in recent_narratives_set]
        if not candidates:
            candidates = all_narratives
        
        return random.choice(candidates)
    
    def _should_rotate_character(self, recent_characters: List[str]) -> bool:
        """
        Determina si se debe rotar el personaje.
        
        Args:
            recent_characters: Personajes usados recientemente
            
        Returns:
            True si se debe rotar el personaje
        """
        # Si no hay personajes recientes, no rotar
        if not recent_characters:
            return False
        
        # Si el mismo personaje se ha usado en los últimos N videos, rotar
        rotation_threshold = self.config["rotation_frequency"]["character"]
        if len(recent_characters) >= rotation_threshold:
            last_character = recent_characters[-1]
            consecutive_count = 0
            
            # Contar cuántos videos consecutivos con el mismo personaje
            for char in reversed(recent_characters):
                if char == last_character:
                    consecutive_count += 1
                else:
                    break
            
            return consecutive_count >= rotation_threshold
        
        return False
    
    def _recommend_posting_frequency(self, channel_id: str, platform: str) -> Dict[str, Any]:
        """
        Recomienda frecuencia de publicación para evitar fatiga.
        
        Args:
            channel_id: Identificador del canal
            platform: Plataforma objetivo
            
        Returns:
            Recomendaciones de frecuencia
        """
        # Configuración base
        base_config = self.config["posting_frequency"]
        max_per_day = base_config["max_per_day"]
        min_days_between_similar = base_config["min_days_between_similar"]
        
        # Ajustar según plataforma
        if platform == "youtube":
            max_per_day = 1  # YouTube generalmente menos frecuente
            min_days_between_similar = 5
        elif platform == "tiktok":
            max_per_day = 3  # TikTok permite mayor frecuencia
            min_days_between_similar = 2
        
        # Ajustar según fatiga detectada
        if channel_id in self.fatigue_metrics and self.fatigue_metrics[channel_id]["fatigue_detected"]:
            # Reducir frecuencia si hay fatiga
            max_per_day = max(1, max_per_day - 1)
            min_days_between_similar += 1
        
        return {
            "max_per_day": max_per_day,
            "min_days_between_similar": min_days_between_similar,
            "optimal_variance": base_config["optimal_time_variance"]
        }
    
    def plan_content_cycle(self, channel_id: str, niche: str, 
                          days: int = 30, platform: str = "youtube") -> Dict[str, Any]:
        """
        Planifica un ciclo de contenido para evitar fatiga.
        
        Args:
            channel_id: Identificador del canal
            niche: Nicho del canal
            days: Número de días a planificar
            platform: Plataforma objetivo
            
        Returns:
            Plan de ciclo de contenido
        """
        try:
            # Determinar tipo de ciclo
            cycle_type = "micro_cycle"
            if days > 60:
                cycle_type = "macro_cycle"
            elif days > 14:
                cycle_type = "mid_cycle"
            
            # Obtener configuración de ciclo
            cycle_length = self.config["content_cycles"][cycle_type]
            
            # Crear plan base
            today = datetime.now().date()
            plan = {
                "channel_id": channel_id,
                "niche": niche,
                "platform": platform,
                "cycle_type": cycle_type,
                "start_date": today.isoformat(),
                "end_date": (today + timedelta(days=days)).isoformat(),
                "content_schedule": [],
                "format_distribution": {},
                "style_distribution": {},
                "narrative_distribution": {}
            }
            
            # Distribución de formatos, estilos y narrativas
            all_formats = self.config["content_formats"]
            all_styles = self.config["visual_styles"]
            all_narratives = self.config["narrative_structures"]
            
            # Priorizar formatos específicos de plataforma
            platform_formats = all_formats
            if platform in self.config["platform_specific"]:
                platform_bias = self.config["platform_specific"][platform].get("format_bias", [])
                if platform_bias:
                    # Mezclar formatos específicos de plataforma con otros
                    platform_formats = platform_bias + [f for f in all_formats if f not in platform_bias]
            
            # Calcular número de publicaciones según frecuencia recomendada
            posting_freq = self._recommend_posting_frequency(channel_id, platform)
            posts_per_day = posting_freq["max_per_day"]
            total_posts = min(days * posts_per_day, 60)  # Limitar a 60 publicaciones
            
            # Distribuir formatos, estilos y narrativas
            format_counts = self._distribute_elements(platform_formats, total_posts)
            style_counts = self._distribute_elements(all_styles, total_posts)
            narrative_counts = self._distribute_elements(all_narratives, total_posts)
            
            plan["format_distribution"] = format_counts
            plan["style_distribution"] = style_counts
            plan["narrative_distribution"] = narrative_counts
            
            # Crear programación de contenido
            formats_list = self._expand_distribution(format_counts)
            styles_list = self._expand_distribution(style_counts)
            narratives_list = self._expand_distribution(narrative_counts)
            
            # Mezclar para evitar patrones predecibles
            random.shuffle(formats_list)
            random.shuffle(styles_list)
            random.shuffle(narratives_list)
            
            # Generar programación
            current_date = today
            for i in range(total_posts):
                # Determinar fecha según frecuencia
                if i > 0 and i % posts_per_day == 0:
                    current_date += timedelta(days=1)
                
                # Seleccionar elementos para este contenido
                format_type = formats_list[i % len(formats_list)]
                style_type = styles_list[i % len(styles_list)]
                narrative_type = narratives_list[i % len(narratives_list)]
                
                # Añadir a programación
                content_item = {
                    "date": current_date.isoformat(),
                    "format": format_type,
                    "style": style_type,
                    "narrative": narrative_type,
                    "platform": platform,
                    "rotation_focus": self._determine_rotation_focus(i, cycle_length)
                }
                
                plan["content_schedule"].append(content_item)
            
            # Guardar plan de ciclo
            self.content_cycles[channel_id] = plan
            
            logger.info(f"Plan de ciclo generado para canal {channel_id}, {total_posts} publicaciones")
            return plan
            
        except Exception as e:
            logger.error(f"Error planificando ciclo de contenido: {str(e)}")
            return {
                "channel_id": channel_id,
                "niche": niche,
                "platform": platform,
                "error": str(e)
            }
    
        def _distribute_elements(self, elements: List[str], total_count: int) -> Dict[str, int]:
        """
        Distribuye elementos según una distribución ponderada.
        
        Args:
            elements: Lista de elementos a distribuir
            total_count: Número total de elementos a asignar
            
        Returns:
            Diccionario con conteo de cada elemento
        """
        if not elements or total_count <= 0:
            return {}
        
        # Crear distribución base (uniforme)
        base_count = total_count // len(elements)
        remainder = total_count % len(elements)
        
        # Asignar conteo base a cada elemento
        distribution = {element: base_count for element in elements}
        
        # Distribuir el resto de forma aleatoria
        for i in range(remainder):
            element = random.choice(elements)
            distribution[element] += 1
        
        return distribution
    
    def _expand_distribution(self, distribution: Dict[str, int]) -> List[str]:
        """
        Expande una distribución de conteo a una lista de elementos.
        
        Args:
            distribution: Diccionario con conteo de cada elemento
            
        Returns:
            Lista expandida de elementos
        """
        expanded = []
        for element, count in distribution.items():
            expanded.extend([element] * count)
        
        return expanded
    
    def _determine_rotation_focus(self, index: int, cycle_length: int) -> str:
        """
        Determina el enfoque de rotación para un índice específico en el ciclo.
        
        Args:
            index: Índice en el ciclo
            cycle_length: Longitud del ciclo
            
        Returns:
            Enfoque de rotación (format, style, narrative, character)
        """
        # Dividir el ciclo en cuatro partes
        quarter = cycle_length // 4
        
        # Determinar enfoque según posición en el ciclo
        if index % cycle_length < quarter:
            return "format"
        elif index % cycle_length < quarter * 2:
            return "style"
        elif index % cycle_length < quarter * 3:
            return "narrative"
        else:
            return "character"
    
    def analyze_content_patterns(self, channel_id: str) -> Dict[str, Any]:
        """
        Analiza patrones en el contenido para identificar tendencias y repeticiones.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Análisis de patrones de contenido
        """
        try:
            if channel_id not in self.content_history or len(self.content_history[channel_id]) < 5:
                logger.warning(f"Historial insuficiente para análisis de patrones: {channel_id}")
                return {"error": "Historial insuficiente para análisis"}
            
            # Obtener historial
            history = self.content_history[channel_id]
            
            # Extraer secuencias
            format_sequence = [c.get("format", "unknown") for c in history if "format" in c]
            style_sequence = [c.get("style", "unknown") for c in history if "style" in c]
            narrative_sequence = [c.get("narrative", "unknown") for c in history if "narrative" in c]
            
            # Análisis de repetición
            format_repetition = self._analyze_repetition(format_sequence)
            style_repetition = self._analyze_repetition(style_sequence)
            narrative_repetition = self._analyze_repetition(narrative_sequence)
            
            # Análisis de frecuencia
            format_frequency = self._analyze_frequency(format_sequence)
            style_frequency = self._analyze_frequency(style_sequence)
            narrative_frequency = self._analyze_frequency(narrative_sequence)
            
            # Análisis de correlación con métricas
            performance_correlation = self._analyze_performance_correlation(history)
            
            # Resultados
            analysis = {
                "repetition_analysis": {
                    "format": format_repetition,
                    "style": style_repetition,
                    "narrative": narrative_repetition
                },
                "frequency_analysis": {
                    "format": format_frequency,
                    "style": style_frequency,
                    "narrative": narrative_frequency
                },
                "performance_correlation": performance_correlation,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Análisis de patrones completado para canal {channel_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando patrones de contenido: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_repetition(self, sequence: List[str]) -> Dict[str, Any]:
        """
        Analiza repeticiones en una secuencia.
        
        Args:
            sequence: Secuencia de elementos
            
        Returns:
            Análisis de repetición
        """
        if not sequence or len(sequence) < 3:
            return {"repetition_index": 0.0, "longest_streak": 0}
        
        # Contar repeticiones consecutivas
        streaks = []
        current_streak = 1
        current_element = sequence[0]
        
        for element in sequence[1:]:
            if element == current_element:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
                current_element = element
        
        # Añadir última racha
        streaks.append(current_streak)
        
        # Calcular índice de repetición (0-1)
        max_streak = max(streaks)
        avg_streak = sum(streaks) / len(streaks)
        repetition_index = avg_streak / len(sequence)
        
        return {
            "repetition_index": round(repetition_index, 2),
            "longest_streak": max_streak,
            "average_streak": round(avg_streak, 2)
        }
    
    def _analyze_frequency(self, sequence: List[str]) -> Dict[str, float]:
        """
        Analiza frecuencia de elementos en una secuencia.
        
        Args:
            sequence: Secuencia de elementos
            
        Returns:
            Frecuencia relativa de cada elemento
        """
        if not sequence:
            return {}
        
        # Contar ocurrencias
        counts = {}
        for element in sequence:
            counts[element] = counts.get(element, 0) + 1
        
        # Calcular frecuencia relativa
        total = len(sequence)
        frequencies = {element: count / total for element, count in counts.items()}
        
        return frequencies
    
    def _analyze_performance_correlation(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza correlación entre elementos y rendimiento.
        
        Args:
            history: Historial de contenido
            
        Returns:
            Correlación entre elementos y métricas
        """
        # Filtrar contenido con métricas
        content_with_metrics = [c for c in history if "metrics" in c and "format" in c]
        
        if len(content_with_metrics) < 5:
            return {"insufficient_data": True}
        
        # Extraer métricas y elementos
        formats = {}
        styles = {}
        narratives = {}
        
        # Agrupar métricas por elemento
        for content in content_with_metrics:
            format_type = content.get("format")
            style_type = content.get("style")
            narrative_type = content.get("narrative")
            
            metrics = content.get("metrics", {})
            engagement = metrics.get("engagement_rate", 0)
            
            # Agrupar por formato
            if format_type:
                if format_type not in formats:
                    formats[format_type] = []
                formats[format_type].append(engagement)
            
            # Agrupar por estilo
            if style_type:
                if style_type not in styles:
                    styles[style_type] = []
                styles[style_type].append(engagement)
            
            # Agrupar por narrativa
            if narrative_type:
                if narrative_type not in narratives:
                    narratives[narrative_type] = []
                narratives[narrative_type].append(engagement)
        
        # Calcular promedios
        format_performance = {f: sum(values) / len(values) for f, values in formats.items() if len(values) >= 2}
        style_performance = {s: sum(values) / len(values) for s, values in styles.items() if len(values) >= 2}
        narrative_performance = {n: sum(values) / len(values) for n, values in narratives.items() if len(values) >= 2}
        
        # Encontrar elementos de mejor y peor rendimiento
        best_format = max(format_performance.items(), key=lambda x: x[1], default=(None, 0))
        worst_format = min(format_performance.items(), key=lambda x: x[1], default=(None, 0))
        
        best_style = max(style_performance.items(), key=lambda x: x[1], default=(None, 0))
        worst_style = min(style_performance.items(), key=lambda x: x[1], default=(None, 0))
        
        best_narrative = max(narrative_performance.items(), key=lambda x: x[1], default=(None, 0))
        worst_narrative = min(narrative_performance.items(), key=lambda x: x[1], default=(None, 0))
        
        return {
            "best_performers": {
                "format": {"name": best_format[0], "engagement": round(best_format[1], 3)},
                "style": {"name": best_style[0], "engagement": round(best_style[1], 3)},
                "narrative": {"name": best_narrative[0], "engagement": round(best_narrative[1], 3)}
            },
            "worst_performers": {
                "format": {"name": worst_format[0], "engagement": round(worst_format[1], 3)},
                "style": {"name": worst_style[0], "engagement": round(worst_style[1], 3)},
                "narrative": {"name": worst_narrative[0], "engagement": round(worst_narrative[1], 3)}
            },
            "format_performance": format_performance,
            "style_performance": style_performance,
            "narrative_performance": narrative_performance
        }
    
    def generate_fatigue_report(self, channel_id: str) -> Dict[str, Any]:
        """
        Genera un informe completo de fatiga y recomendaciones.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Informe de fatiga
        """
        try:
            # Detectar fatiga
            fatigue_metrics = self.detect_fatigue_signals(channel_id)
            
            # Analizar patrones
            pattern_analysis = self.analyze_content_patterns(channel_id)
            
            # Obtener recomendaciones actuales
            recommendations = self.current_recommendations.get(channel_id, {})
            
            # Crear informe
            report = {
                "channel_id": channel_id,
                "timestamp": datetime.now().isoformat(),
                "fatigue_metrics": fatigue_metrics,
                "pattern_analysis": pattern_analysis,
                "recommendations": recommendations,
                "content_history_size": len(self.content_history.get(channel_id, [])),
                "refresh_strategies": self.config["refresh_strategies"]
            }
            
            # Añadir visualizaciones si hay suficientes datos
            if channel_id in self.content_history and len(self.content_history[channel_id]) >= 10:
                report["visualizations"] = self._generate_visualizations(channel_id)
            
            logger.info(f"Informe de fatiga generado para canal {channel_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generando informe de fatiga: {str(e)}")
            return {
                "channel_id": channel_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _generate_visualizations(self, channel_id: str) -> Dict[str, str]:
        """
        Genera visualizaciones para el informe de fatiga.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Rutas a las visualizaciones generadas
        """
        try:
            # Crear directorio para visualizaciones
            vis_dir = f"reports/visualizations/{channel_id}"
            os.makedirs(vis_dir, exist_ok=True)
            
            # Obtener historial
            history = self.content_history.get(channel_id, [])
            
            # Extraer fechas y métricas
            dates = [datetime.fromisoformat(c.get("publish_date", "2000-01-01")) for c in history if "publish_date" in c]
            engagement = [c.get("metrics", {}).get("engagement_rate", 0) for c in history if "metrics" in c]
            retention = [c.get("metrics", {}).get("retention_rate", 0) for c in history if "metrics" in c]
            
            # Verificar datos suficientes
            if len(dates) < 5 or len(engagement) < 5:
                return {"error": "Datos insuficientes para visualizaciones"}
            
            # Gráfico de tendencia de engagement
            engagement_path = f"{vis_dir}/engagement_trend.png"
            plt.figure(figsize=(10, 6))
            plt.plot(dates[-20:], engagement[-20:], marker='o', linestyle='-', color='blue')
            plt.title('Tendencia de Engagement')
            plt.xlabel('Fecha de Publicación')
            plt.ylabel('Tasa de Engagement')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(engagement_path)
            plt.close()
            
            # Gráfico de retención
            if len(retention) >= 5:
                retention_path = f"{vis_dir}/retention_trend.png"
                plt.figure(figsize=(10, 6))
                plt.plot(dates[-20:], retention[-20:], marker='o', linestyle='-', color='green')
                plt.title('Tendencia de Retención')
                plt.xlabel('Fecha de Publicación')
                plt.ylabel('Tasa de Retención')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(retention_path)
                plt.close()
            else:
                retention_path = None
            
            # Gráfico de distribución de formatos
            formats = [c.get("format") for c in history if "format" in c]
            if formats:
                format_counts = {}
                for f in formats:
                    format_counts[f] = format_counts.get(f, 0) + 1
                
                formats_path = f"{vis_dir}/format_distribution.png"
                plt.figure(figsize=(10, 6))
                plt.bar(format_counts.keys(), format_counts.values(), color='purple')
                plt.title('Distribución de Formatos de Contenido')
                plt.xlabel('Formato')
                plt.ylabel('Frecuencia')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(formats_path)
                plt.close()
            else:
                formats_path = None
            
            return {
                "engagement_trend": engagement_path,
                "retention_trend": retention_path,
                "format_distribution": formats_path
            }
            
        except Exception as e:
            logger.error(f"Error generando visualizaciones: {str(e)}")
            return {"error": str(e)}
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Actualiza la configuración del motor anti-fatiga.
        
        Args:
            new_config: Nueva configuración (parcial o completa)
        """
        try:
            # Actualizar configuración
            for key, value in new_config.items():
                if key in self.config:
                    # Si es un diccionario, actualizar recursivamente
                    if isinstance(value, dict) and isinstance(self.config[key], dict):
                        self.config[key].update(value)
                    else:
                        self.config[key] = value
            
            # Guardar configuración
            self.save_config()
            
            logger.info("Configuración anti-fatiga actualizada")
            
        except Exception as e:
            logger.error(f"Error actualizando configuración: {str(e)}")


# Función principal para uso desde línea de comandos
def main():
    """Función principal para uso desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Motor Anti-Fatiga de Contenido")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Comando para detectar fatiga
    detect_parser = subparsers.add_parser("detect", help="Detectar señales de fatiga")
    detect_parser.add_argument("--channel", required=True, help="ID del canal")
    detect_parser.add_argument("--window", type=int, default=10, help="Tamaño de ventana de análisis")
    
    # Comando para recomendar variaciones
    recommend_parser = subparsers.add_parser("recommend", help="Recomendar variaciones de contenido")
    recommend_parser.add_argument("--channel", required=True, help="ID del canal")
    recommend_parser.add_argument("--niche", required=True, help="Nicho del canal")
    recommend_parser.add_argument("--platform", default="youtube", help="Plataforma objetivo")
    
    # Comando para planificar ciclo
    plan_parser = subparsers.add_parser("plan", help="Planificar ciclo de contenido")
    plan_parser.add_argument("--channel", required=True, help="ID del canal")
    plan_parser.add_argument("--niche", required=True, help="Nicho del canal")
    plan_parser.add_argument("--days", type=int, default=30, help="Días a planificar")
    plan_parser.add_argument("--platform", default="youtube", help="Plataforma objetivo")
    plan_parser.add_argument("--output", help="Ruta para guardar el plan (JSON)")
    
    # Comando para actualizar historial
    update_parser = subparsers.add_parser("update", help="Actualizar historial de contenido")
    update_parser.add_argument("--channel", required=True, help="ID del canal")
    update_parser.add_argument("--content", required=True, help="Ruta al archivo JSON con datos de contenido")
    
    # Comando para generar informe
    report_parser = subparsers.add_parser("report", help="Generar informe de fatiga")
    report_parser.add_argument("--channel", required=True, help="ID del canal")
    report_parser.add_argument("--output", help="Ruta para guardar el informe (JSON)")
    
    # Comando para actualizar configuración
    config_parser = subparsers.add_parser("config", help="Actualizar configuración")
    config_parser.add_argument("--update", required=True, help="Ruta al archivo JSON con nueva configuración")
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Crear instancia del motor
    engine = AntiFatigueEngine()
    
    # Ejecutar comando correspondiente
    if args.command == "detect":
        # Detectar fatiga
        fatigue_metrics = engine.detect_fatigue_signals(args.channel, args.window)
        print(json.dumps(fatigue_metrics, indent=4))
    
    elif args.command == "recommend":
        # Recomendar variaciones
        recommendations = engine.recommend_content_variation(args.channel, args.niche, args.platform)
        print(json.dumps(recommendations, indent=4))
    
    elif args.command == "plan":
        # Planificar ciclo
        plan = engine.plan_content_cycle(args.channel, args.niche, args.days, args.platform)
        
        # Guardar o mostrar plan
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=4)
            print(f"Plan guardado en {args.output}")
        else:
            print(json.dumps(plan, indent=4))
    
    elif args.command == "update":
        # Cargar datos de contenido
        try:
            with open(args.content, "r", encoding="utf-8") as f:
                content_data = json.load(f)
            
            # Actualizar historial
            engine.update_content_history(args.channel, content_data)
            print(f"Historial actualizado para canal {args.channel}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    elif args.command == "report":
        # Generar informe
        report = engine.generate_fatigue_report(args.channel)
        
        # Guardar o mostrar informe
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            print(f"Informe guardado en {args.output}")
        else:
            print(json.dumps(report, indent=4))
    
    elif args.command == "config":
        # Cargar nueva configuración
        try:
            with open(args.update, "r", encoding="utf-8") as f:
                new_config = json.load(f)
            
            # Actualizar configuración
            engine.update_config(new_config)
            print("Configuración actualizada")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()