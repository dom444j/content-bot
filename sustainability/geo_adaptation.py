"""
Geographic Adaptation Module

Este módulo permite adaptar el contenido a diferentes regiones geográficas,
optimizando para audiencias específicas con consideraciones culturales,
lingüísticas y de zona horaria.

Características principales:
- Detección de regiones de alto rendimiento
- Adaptación de contenido a normas culturales locales
- Optimización de horarios de publicación por zona horaria
- Traducción y localización de contenido
- Análisis de tendencias regionales
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import pycountry
from googletrans import Translator
from geopy.geocoders import Nominatim
import pytz

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/geo_adaptation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GeoAdaptation")

class GeoAdaptationEngine:
    """
    Motor de adaptación geográfica para optimizar contenido por región.
    """
    
    def __init__(self, config_path: str = "config/geo_config.json"):
        """
        Inicializa el motor de adaptación geográfica.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.translator = Translator()
        self.geolocator = Nominatim(user_agent="content-bot-geo")
        
        # Cargar configuración
        self._load_config()
        
        # Métricas por región
        self.region_metrics = {}
        
        # Historial de rendimiento
        self.performance_history = {}
        
        logger.info("Motor de adaptación geográfica inicializado")
    
    def _load_config(self) -> None:
        """Carga la configuración desde el archivo JSON."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # Configuración por defecto
                self.config = {
                    "target_regions": ["US", "ES", "MX", "AR", "CO", "BR"],
                    "primary_language": "es",
                    "translation_enabled": True,
                    "cultural_adaptation": True,
                    "timezone_optimization": True,
                    "performance_threshold": 0.8,  # 80% del rendimiento promedio
                    "min_audience_size": 1000,
                    "max_regions": 10,
                    "regional_trends_enabled": True,
                    "content_localization": {
                        "US": {"slang": True, "references": True, "holidays": True},
                        "ES": {"slang": True, "references": True, "holidays": True},
                        "MX": {"slang": True, "references": True, "holidays": True},
                        "AR": {"slang": True, "references": True, "holidays": True},
                        "CO": {"slang": True, "references": True, "holidays": True},
                        "BR": {"slang": False, "references": True, "holidays": True}
                    },
                    "optimal_posting_times": {
                        "US": {"weekday": "18:00", "weekend": "12:00"},
                        "ES": {"weekday": "20:00", "weekend": "13:00"},
                        "MX": {"weekday": "19:00", "weekend": "14:00"},
                        "AR": {"weekday": "21:00", "weekend": "15:00"},
                        "CO": {"weekday": "19:00", "weekend": "14:00"},
                        "BR": {"weekday": "20:00", "weekend": "15:00"}
                    },
                    "cultural_references": {
                        "US": ["Super Bowl", "Thanksgiving", "Black Friday"],
                        "ES": ["La Liga", "Semana Santa", "Navidad"],
                        "MX": ["Día de Muertos", "5 de Mayo", "Liga MX"],
                        "AR": ["Mate", "Asado", "Superclásico"],
                        "CO": ["Vallenato", "Carnaval de Barranquilla", "Café"],
                        "BR": ["Carnaval", "Samba", "Futebol"]
                    },
                    "regional_slang": {
                        "US": {"cool": "cool", "friend": "buddy"},
                        "ES": {"cool": "guay", "friend": "colega"},
                        "MX": {"cool": "padre", "friend": "cuate"},
                        "AR": {"cool": "copado", "friend": "che"},
                        "CO": {"cool": "chévere", "friend": "parcero"},
                        "BR": {"cool": "legal", "friend": "amigo"}
                    }
                }
                
                # Guardar configuración por defecto
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4)
            
            logger.info(f"Configuración cargada: {len(self.config['target_regions'])} regiones objetivo")
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {str(e)}")
            # Configuración mínima de respaldo
            self.config = {
                "target_regions": ["US", "ES"],
                "primary_language": "es",
                "translation_enabled": False,
                "cultural_adaptation": False,
                "timezone_optimization": True,
                "performance_threshold": 0.8,
                "min_audience_size": 1000,
                "max_regions": 5
            }
    
    def save_config(self) -> None:
        """Guarda la configuración actual en el archivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuración guardada en {self.config_path}")
        except Exception as e:
            logger.error(f"Error guardando configuración: {str(e)}")
    
    def update_region_metrics(self, metrics_data: Dict[str, Dict]) -> None:
        """
        Actualiza las métricas por región.
        
        Args:
            metrics_data: Diccionario con métricas por región
                {
                    "US": {
                        "views": 1000,
                        "engagement_rate": 0.05,
                        "conversion_rate": 0.02,
                        "audience_size": 5000,
                        "revenue": 50.0
                    },
                    ...
                }
        """
        self.region_metrics.update(metrics_data)
        
        # Actualizar historial de rendimiento
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date not in self.performance_history:
            self.performance_history[current_date] = {}
        
        for region, metrics in metrics_data.items():
            self.performance_history[current_date][region] = metrics
        
        logger.info(f"Métricas actualizadas para {len(metrics_data)} regiones")
    
    def identify_top_regions(self, metric: str = "engagement_rate", top_n: int = 5) -> List[str]:
        """
        Identifica las regiones de mejor rendimiento según una métrica.
        
        Args:
            metric: Métrica a utilizar (views, engagement_rate, conversion_rate, revenue)
            top_n: Número de regiones a devolver
            
        Returns:
            Lista de códigos de país de las mejores regiones
        """
        if not self.region_metrics:
            logger.warning("No hay métricas disponibles para identificar regiones")
            return self.config["target_regions"][:top_n]
        
        # Ordenar regiones por métrica
        sorted_regions = sorted(
            self.region_metrics.items(),
            key=lambda x: x[1].get(metric, 0),
            reverse=True
        )
        
        # Filtrar por tamaño mínimo de audiencia
        filtered_regions = [
            region for region, metrics in sorted_regions
            if metrics.get("audience_size", 0) >= self.config["min_audience_size"]
        ]
        
        # Limitar al número solicitado
        top_regions = [region for region, _ in filtered_regions[:top_n]]
        
        logger.info(f"Top {len(top_regions)} regiones por {metric}: {', '.join(top_regions)}")
        return top_regions
    
    def optimize_posting_schedule(self, region: str, content_type: str = "video") -> Dict[str, str]:
        """
        Determina los horarios óptimos de publicación para una región.
        
        Args:
            region: Código de país (ISO 3166-1 alpha-2)
            content_type: Tipo de contenido (video, post, reel)
            
        Returns:
            Diccionario con horarios óptimos por día de la semana
        """
        # Obtener zona horaria de la región
        try:
            country = pycountry.countries.get(alpha_2=region)
            if not country:
                logger.warning(f"País no encontrado para código {region}")
                country_name = region
            else:
                country_name = country.name
            
            # Obtener zona horaria aproximada (simplificado)
            timezone_map = {
                "US": "America/New_York",
                "ES": "Europe/Madrid",
                "MX": "America/Mexico_City",
                "AR": "America/Argentina/Buenos_Aires",
                "CO": "America/Bogota",
                "BR": "America/Sao_Paulo"
            }
            
            timezone_str = timezone_map.get(region, "UTC")
            timezone = pytz.timezone(timezone_str)
            
            # Horarios óptimos desde configuración
            if region in self.config["optimal_posting_times"]:
                weekday_time = self.config["optimal_posting_times"][region]["weekday"]
                weekend_time = self.config["optimal_posting_times"][region]["weekend"]
            else:
                # Valores por defecto
                weekday_time = "19:00"
                weekend_time = "14:00"
            
            # Crear horario para cada día de la semana
            schedule = {}
            for day_idx in range(7):
                day_name = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][day_idx]
                if day_idx < 5:  # Lunes a viernes
                    schedule[day_name] = weekday_time
                else:  # Fin de semana
                    schedule[day_name] = weekend_time
            
            logger.info(f"Horario optimizado para {country_name} ({timezone_str})")
            return schedule
            
        except Exception as e:
            logger.error(f"Error optimizando horario para {region}: {str(e)}")
            # Horario por defecto
            return {
                "monday": "19:00", "tuesday": "19:00", "wednesday": "19:00",
                "thursday": "19:00", "friday": "19:00",
                "saturday": "14:00", "sunday": "14:00"
            }
    
    def translate_content(self, content: str, target_language: str) -> str:
        """
        Traduce contenido a un idioma específico.
        
        Args:
            content: Texto a traducir
            target_language: Código de idioma (ISO 639-1)
            
        Returns:
            Texto traducido
        """
        if not self.config["translation_enabled"]:
            logger.info("Traducción desactivada en configuración")
            return content
        
        try:
            # Detectar idioma de origen
            detected = self.translator.detect(content)
            source_lang = detected.lang
            
            # Si ya está en el idioma objetivo, devolver sin cambios
            if source_lang == target_language:
                return content
            
            # Traducir
            result = self.translator.translate(content, dest=target_language)
            translated_text = result.text
            
            logger.info(f"Contenido traducido de {source_lang} a {target_language}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Error traduciendo contenido: {str(e)}")
            return content
    
    def adapt_cultural_references(self, content: Dict, target_region: str) -> Dict:
        """
        Adapta referencias culturales para una región específica.
        
        Args:
            content: Diccionario con contenido (título, descripción, guion, etc.)
            target_region: Código de país objetivo
            
        Returns:
            Contenido adaptado
        """
        if not self.config["cultural_adaptation"]:
            logger.info("Adaptación cultural desactivada en configuración")
            return content
        
        try:
            # Verificar si hay referencias culturales para la región
            if target_region not in self.config["cultural_references"]:
                logger.warning(f"No hay referencias culturales para {target_region}")
                return content
            
            # Obtener referencias culturales y slang regional
            cultural_refs = self.config["cultural_references"].get(target_region, [])
            regional_slang = self.config["regional_slang"].get(target_region, {})
            
            # Contenido adaptado
            adapted_content = content.copy()
            
            # Adaptar título y descripción
            if "title" in adapted_content:
                # Reemplazar slang
                for generic, regional in regional_slang.items():
                    adapted_content["title"] = adapted_content["title"].replace(generic, regional)
                
                # Añadir referencia cultural si es apropiado y no existe ya
                for ref in cultural_refs:
                    if ref.lower() not in adapted_content["title"].lower() and len(adapted_content["title"]) < 50:
                        if np.random.random() < 0.3:  # 30% de probabilidad
                            adapted_content["title"] += f" | {ref}"
                            break
            
            # Adaptar descripción
            if "description" in adapted_content:
                # Reemplazar slang
                for generic, regional in regional_slang.items():
                    adapted_content["description"] = adapted_content["description"].replace(generic, regional)
                
                # Añadir referencia cultural si es apropiado
                for ref in cultural_refs:
                    if ref.lower() not in adapted_content["description"].lower():
                        if np.random.random() < 0.4:  # 40% de probabilidad
                            adapted_content["description"] += f"\n\nInspired by {ref}"
                            break
            
            # Adaptar guion
            if "script" in adapted_content:
                # Reemplazar slang
                for generic, regional in regional_slang.items():
                    adapted_content["script"] = adapted_content["script"].replace(generic, regional)
            
            # Adaptar hashtags
            if "hashtags" in adapted_content:
                # Añadir hashtags regionales
                regional_hashtags = [f"#{ref.replace(' ', '')}" for ref in cultural_refs]
                existing_hashtags = set(adapted_content["hashtags"])
                
                # Seleccionar 1-3 hashtags regionales que no existan ya
                new_hashtags = [tag for tag in regional_hashtags if tag not in existing_hashtags]
                selected_hashtags = np.random.choice(
                    new_hashtags, 
                    size=min(3, len(new_hashtags)), 
                    replace=False
                ).tolist() if new_hashtags else []
                
                adapted_content["hashtags"] = adapted_content["hashtags"] + selected_hashtags
            
            logger.info(f"Contenido adaptado culturalmente para {target_region}")
            return adapted_content
            
        except Exception as e:
            logger.error(f"Error adaptando referencias culturales: {str(e)}")
            return content
    
    def analyze_regional_trends(self, region: str, niche: str) -> List[str]:
        """
        Analiza tendencias específicas de una región.
        
        Args:
            region: Código de país
            niche: Nicho de contenido
            
        Returns:
            Lista de tendencias regionales
        """
        if not self.config["regional_trends_enabled"]:
            logger.info("Análisis de tendencias regionales desactivado")
            return []
        
        try:
            # Simulación de tendencias regionales
            # En un sistema real, esto se conectaría a APIs de tendencias
            regional_trends = {
                "US": {
                    "finance": ["crypto regulations", "inflation hedge", "retirement planning"],
                    "health": ["keto diet", "mental wellness", "home workouts"],
                    "gaming": ["Fortnite", "Call of Duty", "Minecraft"],
                    "tech": ["AI assistants", "VR headsets", "smart home"]
                },
                "ES": {
                    "finance": ["inversiones seguras", "ahorro fiscal", "criptomonedas"],
                    "health": ["dieta mediterránea", "yoga", "running"],
                    "gaming": ["FIFA", "League of Legends", "Minecraft"],
                    "tech": ["móviles plegables", "energía solar", "coches eléctricos"]
                },
                "MX": {
                    "finance": ["remesas", "inversión inmobiliaria", "ahorro en dólares"],
                    "health": ["comida saludable", "ejercicio en casa", "suplementos"],
                    "gaming": ["Free Fire", "FIFA", "Fortnite"],
                    "tech": ["smartphones económicos", "internet rural", "apps de delivery"]
                },
                "AR": {
                    "finance": ["dólar blue", "plazo fijo", "inflación"],
                    "health": ["mate beneficios", "rutinas express", "alimentación consciente"],
                    "gaming": ["FIFA", "Counter-Strike", "League of Legends"],
                    "tech": ["billeteras virtuales", "trabajo remoto", "cursos online"]
                },
                "CO": {
                    "finance": ["inversión finca raíz", "ahorro programado", "emprendimiento"],
                    "health": ["alimentación balanceada", "ciclismo", "meditación"],
                    "gaming": ["Free Fire", "FIFA", "Call of Duty Mobile"],
                    "tech": ["apps colombianas", "fintech", "energías renovables"]
                },
                "BR": {
                    "finance": ["pix", "investimentos", "economia digital"],
                    "health": ["treino em casa", "alimentação saudável", "suplementos"],
                    "gaming": ["Free Fire", "League of Legends", "FIFA"],
                    "tech": ["celulares", "internet 5G", "carros elétricos"]
                }
            }
            
            # Obtener tendencias para la región y nicho
            if region in regional_trends and niche in regional_trends[region]:
                trends = regional_trends[region][niche]
                logger.info(f"Tendencias para {region} en {niche}: {trends}")
                return trends
            else:
                logger.warning(f"No hay tendencias disponibles para {region} en {niche}")
                return []
                
        except Exception as e:
            logger.error(f"Error analizando tendencias regionales: {str(e)}")
            return []
    
    def adapt_content_for_region(self, content: Dict, target_region: str, 
                                 translate: bool = True, adapt_culture: bool = True) -> Dict:
        """
        Adapta contenido completo para una región específica.
        
        Args:
            content: Diccionario con contenido
            target_region: Código de país objetivo
            translate: Si se debe traducir el contenido
            adapt_culture: Si se deben adaptar referencias culturales
            
        Returns:
            Contenido adaptado para la región
        """
        try:
            # Determinar idioma objetivo
            language_map = {
                "US": "en", "ES": "es", "MX": "es", 
                "AR": "es", "CO": "es", "BR": "pt"
            }
            target_language = language_map.get(target_region, "en")
            
            # Copia del contenido original
            adapted_content = content.copy()
            
            # Traducir si es necesario
            if translate and self.config["translation_enabled"]:
                if "title" in adapted_content:
                    adapted_content["title"] = self.translate_content(
                        adapted_content["title"], target_language
                    )
                
                if "description" in adapted_content:
                    adapted_content["description"] = self.translate_content(
                        adapted_content["description"], target_language
                    )
                
                if "script" in adapted_content:
                    adapted_content["script"] = self.translate_content(
                        adapted_content["script"], target_language
                    )
            
            # Adaptar culturalmente si es necesario
            if adapt_culture and self.config["cultural_adaptation"]:
                adapted_content = self.adapt_cultural_references(
                    adapted_content, target_region
                )
            
            # Optimizar horario de publicación
            if self.config["timezone_optimization"]:
                content_type = adapted_content.get("content_type", "video")
                adapted_content["posting_schedule"] = self.optimize_posting_schedule(
                    target_region, content_type
                )
            
            # Añadir tendencias regionales si hay nicho
            if "niche" in adapted_content and self.config["regional_trends_enabled"]:
                regional_trends = self.analyze_regional_trends(
                    target_region, adapted_content["niche"]
                )
                
                if regional_trends:
                    adapted_content["regional_trends"] = regional_trends
                    
                    # Incorporar tendencias en hashtags
                    if "hashtags" in adapted_content:
                        trend_hashtags = [f"#{trend.replace(' ', '')}" for trend in regional_trends]
                        adapted_content["hashtags"].extend(trend_hashtags[:2])  # Añadir hasta 2 hashtags
            
            logger.info(f"Contenido adaptado para {target_region}")
            return adapted_content
            
        except Exception as e:
            logger.error(f"Error adaptando contenido para región: {str(e)}")
            return content
    
    def recommend_expansion_regions(self) -> List[Dict[str, Union[str, float]]]:
        """
        Recomienda regiones para expansión basadas en rendimiento y potencial.
        
        Returns:
            Lista de regiones recomendadas con puntuación
        """
        try:
            # Regiones actuales
            current_regions = set(self.config["target_regions"])
            
            # Regiones potenciales (simplificado)
            potential_regions = [
                {"code": "CA", "name": "Canada", "language": "en", "market_size": 0.8},
                {"code": "UK", "name": "United Kingdom", "language": "en", "market_size": 0.9},
                {"code": "AU", "name": "Australia", "language": "en", "market_size": 0.7},
                {"code": "DE", "name": "Germany", "language": "de", "market_size": 0.85},
                {"code": "FR", "name": "France", "language": "fr", "market_size": 0.8},
                {"code": "IT", "name": "Italy", "language": "it", "market_size": 0.75},
                {"code": "CL", "name": "Chile", "language": "es", "market_size": 0.6},
                {"code": "PE", "name": "Peru", "language": "es", "market_size": 0.55},
                {"code": "EC", "name": "Ecuador", "language": "es", "market_size": 0.5},
                {"code": "UY", "name": "Uruguay", "language": "es", "market_size": 0.45},
                {"code": "PT", "name": "Portugal", "language": "pt", "market_size": 0.6}
            ]
            
            # Filtrar regiones que ya están en uso
            filtered_regions = [r for r in potential_regions if r["code"] not in current_regions]
            
            # Calcular puntuación para cada región
            scored_regions = []
            for region in filtered_regions:
                # Factores de puntuación
                language_factor = 1.0 if region["language"] in ["es", "en"] else 0.7
                market_size_factor = region["market_size"]
                
                # Similitud con regiones exitosas
                similarity_factor = 0.5
                if self.region_metrics:
                    # Encontrar región exitosa con mismo idioma
                    for existing_region, metrics in self.region_metrics.items():
                        existing_language = next((r["language"] for r in potential_regions if r["code"] == existing_region), "en")
                        if existing_language == region["language"] and metrics.get("engagement_rate", 0) > 0.05:
                            similarity_factor = 0.9
                            break
                
                # Puntuación final
                score = (language_factor * 0.4) + (market_size_factor * 0.4) + (similarity_factor * 0.2)
                
                # Añadir a lista con puntuación
                scored_regions.append({
                    "code": region["code"],
                    "name": region["name"],
                    "language": region["language"],
                    "score": round(score, 2),
                    "market_size": region["market_size"]
                })
            
            # Ordenar por puntuación
            recommended_regions = sorted(scored_regions, key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Regiones recomendadas para expansión: {[r['code'] for r in recommended_regions[:3]]}")
            return recommended_regions
            
        except Exception as e:
            logger.error(f"Error recomendando regiones para expansión: {str(e)}")
            return []
    
    def generate_geo_report(self) -> Dict:
        """
        Genera un informe de rendimiento geográfico.
        
        Returns:
            Diccionario con informe de rendimiento
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "active_regions": len(self.config["target_regions"]),
                "top_regions": {},
                "underperforming_regions": [],
                "recommended_expansions": [],
                "regional_metrics": {},
                "optimization_status": {
                    "translation_enabled": self.config["translation_enabled"],
                    "cultural_adaptation": self.config["cultural_adaptation"],
                    "timezone_optimization": self.config["timezone_optimization"],
                    "regional_trends_enabled": self.config["regional_trends_enabled"]
                }
            }
            
            # Métricas por región
            if self.region_metrics:
                # Top regiones por métrica
                for metric in ["views", "engagement_rate", "conversion_rate", "revenue"]:
                    top_regions = self.identify_top_regions(metric=metric, top_n=3)
                    report["top_regions"][metric] = top_regions
                
                # Regiones con bajo rendimiento
                avg_engagement = sum(m.get("engagement_rate", 0) for m in self.region_metrics.values()) / len(self.region_metrics)
                threshold = avg_engagement * self.config["performance_threshold"]
                
                underperforming = [
                    region for region, metrics in self.region_metrics.items()
                    if metrics.get("engagement_rate", 0) < threshold
                ]
                report["underperforming_regions"] = underperforming
                
                # Métricas regionales
                report["regional_metrics"] = self.region_metrics
            
            # Regiones recomendadas para expansión
            recommended = self.recommend_expansion_regions()
            report["recommended_expansions"] = recommended[:3]  # Top 3
            
            logger.info(f"Informe geográfico generado: {len(report['regional_metrics'])} regiones analizadas")
            return report
            
        except Exception as e:
            logger.error(f"Error generando informe geográfico: {str(e)}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def update_target_regions(self, add_regions: List[str] = None, remove_regions: List[str] = None) -> None:
        """
        Actualiza las regiones objetivo.
        
        Args:
            add_regions: Lista de regiones a añadir
            remove_regions: Lista de regiones a eliminar
        """
        try:
            current_regions = set(self.config["target_regions"])
            
            # Añadir regiones
            if add_regions:
                for region in add_regions:
                    if region not in current_regions:
                        current_regions.add(region)
                        logger.info(f"Región añadida: {region}")
            
            # Eliminar regiones
            if remove_regions:
                for region in remove_regions:
                    if region in current_regions:
                        current_regions.remove(region)
                        logger.info(f"Región eliminada: {region}")
            
            # Actualizar configuración
            self.config["target_regions"] = list(current_regions)
            
            # Limitar al máximo configurado
            if len(self.config["target_regions"]) > self.config["max_regions"]:
                self.config["target_regions"] = self.config["target_regions"][:self.config["max_regions"]]
                logger.warning(f"Regiones limitadas a {self.config['max_regions']}")
            
            # Guardar configuración
            self.save_config()
            
            logger.info(f"Regiones objetivo actualizadas: {len(self.config['target_regions'])} regiones activas")
            
        except Exception as e:
            logger.error(f"Error actualizando regiones objetivo: {str(e)}")


# Función para uso desde línea de comandos
def main():
    """Función principal para uso desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Motor de adaptación geográfica")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
        # Comando para generar informe
    report_parser = subparsers.add_parser("report", help="Generar informe de rendimiento geográfico")
    report_parser.add_argument("--output", help="Ruta para guardar el informe (formato JSON)")
    
    # Comando para adaptar contenido
    adapt_parser = subparsers.add_parser("adapt", help="Adaptar contenido para una región específica")
    adapt_parser.add_argument("--content", required=True, help="Ruta al archivo JSON con contenido")
    adapt_parser.add_argument("--region", required=True, help="Código de país objetivo (ISO 3166-1 alpha-2)")
    adapt_parser.add_argument("--output", help="Ruta para guardar el contenido adaptado")
    adapt_parser.add_argument("--no-translate", action="store_true", help="Desactivar traducción")
    adapt_parser.add_argument("--no-culture", action="store_true", help="Desactivar adaptación cultural")
    
    # Comando para actualizar regiones objetivo
    regions_parser = subparsers.add_parser("regions", help="Actualizar regiones objetivo")
    regions_parser.add_argument("--add", nargs="+", help="Regiones a añadir")
    regions_parser.add_argument("--remove", nargs="+", help="Regiones a eliminar")
    
    # Comando para actualizar métricas
    metrics_parser = subparsers.add_parser("metrics", help="Actualizar métricas por región")
    metrics_parser.add_argument("--data", required=True, help="Ruta al archivo CSV o JSON con métricas")
    
    # Comando para optimizar horarios
    schedule_parser = subparsers.add_parser("schedule", help="Optimizar horarios de publicación")
    schedule_parser.add_argument("--region", required=True, help="Código de país")
    schedule_parser.add_argument("--content-type", default="video", choices=["video", "post", "reel"], 
                                help="Tipo de contenido")
    
    # Comando para recomendar expansión
    expand_parser = subparsers.add_parser("expand", help="Recomendar regiones para expansión")
    expand_parser.add_argument("--top", type=int, default=3, help="Número de regiones a recomendar")
    expand_parser.add_argument("--output", help="Ruta para guardar recomendaciones")
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Crear instancia del motor
    engine = GeoAdaptationEngine()
    
    # Ejecutar comando correspondiente
    if args.command == "report":
        # Generar informe
        report = engine.generate_geo_report()
        
        # Guardar o mostrar informe
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            print(f"Informe guardado en {args.output}")
        else:
            print(json.dumps(report, indent=4))
    
    elif args.command == "adapt":
        # Cargar contenido
        try:
            with open(args.content, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as e:
            print(f"Error cargando contenido: {str(e)}")
            return
        
        # Adaptar contenido
        adapted_content = engine.adapt_content_for_region(
            content=content,
            target_region=args.region,
            translate=not args.no_translate,
            adapt_culture=not args.no_culture
        )
        
        # Guardar o mostrar contenido adaptado
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(adapted_content, f, indent=4)
            print(f"Contenido adaptado guardado en {args.output}")
        else:
            print(json.dumps(adapted_content, indent=4))
    
    elif args.command == "regions":
        # Actualizar regiones
        engine.update_target_regions(
            add_regions=args.add,
            remove_regions=args.remove
        )
        
        # Mostrar regiones actualizadas
        print(f"Regiones objetivo actualizadas: {engine.config['target_regions']}")
    
    elif args.command == "metrics":
        # Cargar métricas
        try:
            # Determinar formato
            if args.data.endswith(".csv"):
                # Cargar CSV
                df = pd.read_csv(args.data)
                
                # Convertir a diccionario
                metrics_data = {}
                for _, row in df.iterrows():
                    region = row["region"]
                    metrics_data[region] = {
                        col: row[col] for col in df.columns if col != "region"
                    }
            else:
                # Cargar JSON
                with open(args.data, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
            
            # Actualizar métricas
            engine.update_region_metrics(metrics_data)
            print(f"Métricas actualizadas para {len(metrics_data)} regiones")
            
        except Exception as e:
            print(f"Error cargando métricas: {str(e)}")
    
    elif args.command == "schedule":
        # Optimizar horario
        schedule = engine.optimize_posting_schedule(
            region=args.region,
            content_type=args.content_type
        )
        
        # Mostrar horario
        print(f"Horario optimizado para {args.region}:")
        for day, time in schedule.items():
            print(f"  {day.capitalize()}: {time}")
    
    elif args.command == "expand":
        # Recomendar regiones
        recommendations = engine.recommend_expansion_regions()
        
        # Limitar al número solicitado
        top_recommendations = recommendations[:args.top]
        
        # Guardar o mostrar recomendaciones
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(top_recommendations, f, indent=4)
            print(f"Recomendaciones guardadas en {args.output}")
        else:
            print("Regiones recomendadas para expansión:")
            for i, region in enumerate(top_recommendations, 1):
                print(f"{i}. {region['name']} ({region['code']}) - Score: {region['score']}")
                print(f"   Idioma: {region['language']}, Tamaño de mercado: {region['market_size']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()