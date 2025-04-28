import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/competitor_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("competitor_analyzer")

class CompetitorAnalyzer:
    """
    Analiza competidores en diferentes plataformas para identificar estrategias,
    CTAs efectivos, nichos rentables y oportunidades de mercado.
    """
    
    def __init__(self, data_path: str = "data/analysis"):
        """
        Inicializa el analizador de competidores.
        
        Args:
            data_path: Ruta para almacenar datos de análisis
        """
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # Cargar configuración
        self.config_path = "config/platforms.json"
        self.config = self._load_config()
        
        # Inicializar almacenamiento de datos
        self.competitor_data = self._load_data("competitor_data.json", {})
        self.cta_database = self._load_data("cta_database.json", {})
        self.niche_analysis = self._load_data("niche_analysis.json", {})
        self.content_trends = self._load_data("content_trends.json", {})
        
        # Importar adaptadores de plataforma
        self.platform_adapters = {}
        self._load_platform_adapters()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración de plataformas"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def _load_data(self, filename: str, default: Any) -> Any:
        """Carga datos desde un archivo JSON"""
        filepath = os.path.join(self.data_path, filename)
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Error al cargar {filename}: {str(e)}")
            return default
    
    def _save_data(self, filename: str, data: Any) -> bool:
        """Guarda datos en un archivo JSON"""
        filepath = os.path.join(self.data_path, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error al guardar {filename}: {str(e)}")
            return False
    
    def _load_platform_adapters(self):
        """Carga los adaptadores de plataforma disponibles"""
        try:
            # Importar adaptadores dinámicamente
            from platform_adapters.youtube_adapter import YouTubeAdapter
            from platform_adapters.tiktok_adapter import TikTokAdapter
            from platform_adapters.instagram_adapter import InstagramAdapter
            
            # Inicializar adaptadores con configuración
            self.platform_adapters["youtube"] = YouTubeAdapter(self.config.get("youtube", {}))
            self.platform_adapters["tiktok"] = TikTokAdapter(self.config.get("tiktok", {}))
            self.platform_adapters["instagram"] = InstagramAdapter(self.config.get("instagram", {}))
            
            logger.info(f"Adaptadores de plataforma cargados: {list(self.platform_adapters.keys())}")
        except ImportError as e:
            logger.warning(f"No se pudieron cargar todos los adaptadores: {str(e)}")
        except Exception as e:
            logger.error(f"Error al inicializar adaptadores: {str(e)}")
    
    def analyze_competitor(self, platform: str, competitor_id: str, max_content: int = 50) -> Dict[str, Any]:
        """
        Analiza un competidor específico en una plataforma.
        
        Args:
            platform: Plataforma (youtube, tiktok, instagram)
            competitor_id: ID del competidor
            max_content: Cantidad máxima de contenido a analizar
            
        Returns:
            Resultados del análisis
        """
        if platform not in self.platform_adapters:
            logger.error(f"Plataforma no soportada: {platform}")
            return {
                "status": "error",
                "message": f"Plataforma no soportada: {platform}"
            }
        
        try:
            # Obtener datos del competidor
            adapter = self.platform_adapters[platform]
            competitor_info = adapter.get_channel_info(competitor_id)
            
            if not competitor_info:
                return {
                    "status": "error",
                    "message": f"No se pudo obtener información del competidor: {competitor_id}"
                }
            
            # Obtener contenido reciente
            content_list = adapter.get_recent_content(competitor_id, max_content)
            
            # Analizar CTAs, tendencias y patrones
            cta_patterns = self._extract_cta_patterns(content_list, platform)
            content_metrics = self._analyze_content_metrics(content_list)
            posting_schedule = self._analyze_posting_schedule(content_list)
            keywords = self._extract_keywords(content_list)
            
            # Guardar resultados
            analysis_result = {
                "competitor_id": competitor_id,
                "platform": platform,
                "analysis_date": datetime.now().isoformat(),
                "subscriber_count": competitor_info.get("subscriber_count", 0),
                "total_content": competitor_info.get("content_count", 0),
                "engagement_rate": competitor_info.get("engagement_rate", 0),
                "cta_patterns": cta_patterns,
                "content_metrics": content_metrics,
                "posting_schedule": posting_schedule,
                "top_keywords": keywords,
                "niche_category": self._determine_niche(keywords)
            }
            
            # Actualizar base de datos
            competitor_key = f"{platform}_{competitor_id}"
            self.competitor_data[competitor_key] = analysis_result
            
            # Actualizar base de datos de CTAs
            self._update_cta_database(cta_patterns, platform)
            
            # Guardar datos actualizados
            self._save_data("competitor_data.json", self.competitor_data)
            self._save_data("cta_database.json", self.cta_database)
            
            return {
                "status": "success",
                "competitor": competitor_id,
                "platform": platform,
                "analysis": analysis_result
            }
            
        except Exception as e:
            logger.error(f"Error al analizar competidor {competitor_id} en {platform}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar competidor: {str(e)}"
            }
    
    def _extract_cta_patterns(self, content_list: List[Dict[str, Any]], platform: str) -> Dict[str, Any]:
        """
        Extrae patrones de CTAs del contenido analizado.
        
        Args:
            content_list: Lista de contenido
            platform: Plataforma
            
        Returns:
            Patrones de CTAs identificados
        """
        cta_patterns = {
            "text_ctas": [],
            "visual_ctas": [],
            "timing_patterns": [],
            "most_effective": []
        }
        
        # Extraer CTAs de texto (descripciones, títulos)
        cta_phrases = [
            "suscríbete", "subscribe", "follow", "sígueme", "like", "comenta",
            "comparte", "share", "dale like", "hit like", "click", "link in bio",
            "link en bio", "únete", "join", "más información", "more info"
        ]
        
        all_descriptions = []
        all_titles = []
        cta_timings = []
        
        for content in content_list:
            # Recopilar texto
            if "description" in content:
                all_descriptions.append(content["description"].lower())
            if "title" in content:
                all_titles.append(content["title"].lower())
            
            # Recopilar timings de CTAs (si están disponibles)
            if "cta_timing" in content:
                cta_timings.append(content["cta_timing"])
        
        # Analizar CTAs en texto
        text_ctas = []
        for phrase in cta_phrases:
            for desc in all_descriptions:
                if phrase in desc:
                    # Extraer contexto alrededor del CTA
                    start = max(0, desc.find(phrase) - 20)
                    end = min(len(desc), desc.find(phrase) + len(phrase) + 20)
                    context = desc[start:end].strip()
                    text_ctas.append({"phrase": phrase, "context": context})
        
        # Analizar timings de CTAs
        if cta_timings:
            avg_timing = sum(cta_timings) / len(cta_timings)
            timing_distribution = {}
            for timing in cta_timings:
                bucket = f"{int(timing // 5 * 5)}-{int(timing // 5 * 5 + 5)}s"
                timing_distribution[bucket] = timing_distribution.get(bucket, 0) + 1
            
            cta_patterns["timing_patterns"] = {
                "average_timing": round(avg_timing, 1),
                "distribution": timing_distribution
            }
        
        # Identificar CTAs más efectivos (basado en engagement)
        effective_ctas = []
        for content in content_list:
            if "engagement_rate" in content and "description" in content:
                for phrase in cta_phrases:
                    if phrase in content["description"].lower():
                        effective_ctas.append({
                            "phrase": phrase,
                            "engagement_rate": content["engagement_rate"],
                            "content_id": content.get("content_id", "")
                        })
        
        # Ordenar por tasa de engagement
        effective_ctas.sort(key=lambda x: x["engagement_rate"], reverse=True)
        
        cta_patterns["text_ctas"] = text_ctas
        cta_patterns["most_effective"] = effective_ctas[:5]  # Top 5
        
        return cta_patterns
    
    def _analyze_content_metrics(self, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza métricas de contenido para identificar patrones.
        
        Args:
            content_list: Lista de contenido
            
        Returns:
            Métricas y patrones identificados
        """
        if not content_list:
            return {}
        
        # Extraer métricas
        views = [content.get("view_count", 0) for content in content_list]
        likes = [content.get("like_count", 0) for content in content_list]
        comments = [content.get("comment_count", 0) for content in content_list]
        durations = [content.get("duration", 0) for content in content_list]
        
        # Calcular promedios
        avg_views = sum(views) / len(views) if views else 0
        avg_likes = sum(likes) / len(likes) if likes else 0
        avg_comments = sum(comments) / len(comments) if comments else 0
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calcular engagement
        engagement_rates = []
        for content in content_list:
            view_count = content.get("view_count", 0)
            if view_count > 0:
                engagement = (content.get("like_count", 0) + content.get("comment_count", 0)) / view_count
                engagement_rates.append(engagement)
        
        avg_engagement = sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0
        
        # Identificar contenido más exitoso
        content_list_with_metrics = []
        for content in content_list:
            if "view_count" in content:
                content_list_with_metrics.append({
                    "content_id": content.get("content_id", ""),
                    "title": content.get("title", ""),
                    "view_count": content.get("view_count", 0),
                    "like_count": content.get("like_count", 0),
                    "comment_count": content.get("comment_count", 0),
                    "engagement_rate": (content.get("like_count", 0) + content.get("comment_count", 0)) / content.get("view_count", 1)
                })
        
        # Ordenar por vistas
        top_by_views = sorted(content_list_with_metrics, key=lambda x: x["view_count"], reverse=True)[:5]
        
        # Ordenar por engagement
        top_by_engagement = sorted(content_list_with_metrics, key=lambda x: x["engagement_rate"], reverse=True)[:5]
        
        # Analizar duración óptima
        duration_engagement = {}
        for content in content_list:
            if "duration" in content and "view_count" in content and content["view_count"] > 0:
                duration_bucket = f"{int(content['duration'] // 15 * 15)}-{int(content['duration'] // 15 * 15 + 15)}s"
                engagement = (content.get("like_count", 0) + content.get("comment_count", 0)) / content["view_count"]
                
                if duration_bucket not in duration_engagement:
                    duration_engagement[duration_bucket] = []
                
                duration_engagement[duration_bucket].append(engagement)
        
        # Calcular engagement promedio por duración
        avg_duration_engagement = {}
        for bucket, engagements in duration_engagement.items():
            avg_duration_engagement[bucket] = sum(engagements) / len(engagements)
        
        # Encontrar duración óptima
        optimal_duration = max(avg_duration_engagement.items(), key=lambda x: x[1])[0] if avg_duration_engagement else "N/A"
        
        return {
            "average_metrics": {
                "views": round(avg_views, 1),
                "likes": round(avg_likes, 1),
                "comments": round(avg_comments, 1),
                "duration": round(avg_duration, 1),
                "engagement_rate": round(avg_engagement * 100, 2)
            },
            "top_content": {
                "by_views": top_by_views,
                "by_engagement": top_by_engagement
            },
            "optimal_duration": optimal_duration,
            "duration_engagement": avg_duration_engagement
        }
    
    def _analyze_posting_schedule(self, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza el calendario de publicación del competidor.
        
        Args:
            content_list: Lista de contenido
            
        Returns:
            Patrones de publicación identificados
        """
        if not content_list:
            return {}
        
        # Extraer fechas de publicación
        posting_dates = []
        for content in content_list:
            if "publish_date" in content:
                try:
                    date = datetime.fromisoformat(content["publish_date"])
                    posting_dates.append(date)
                except (ValueError, TypeError):
                    continue
        
        if not posting_dates:
            return {}
        
        # Ordenar fechas
        posting_dates.sort()
        
        # Analizar días de la semana
        weekdays = [date.weekday() for date in posting_dates]
        weekday_counts = Counter(weekdays)
        weekday_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        weekday_distribution = {weekday_names[day]: count for day, count in weekday_counts.items()}
        
        # Analizar horas del día
        hours = [date.hour for date in posting_dates]
        hour_counts = Counter(hours)
        hour_distribution = {f"{hour}:00": count for hour, count in hour_counts.items()}
        
        # Calcular frecuencia de publicación
        if len(posting_dates) >= 2:
            time_diffs = [(posting_dates[i] - posting_dates[i-1]).days for i in range(1, len(posting_dates))]
            avg_frequency = sum(time_diffs) / len(time_diffs)
        else:
            avg_frequency = 0
        
        # Identificar día y hora más comunes
        most_common_day = max(weekday_distribution.items(), key=lambda x: x[1])[0] if weekday_distribution else "N/A"
        most_common_hour = max(hour_distribution.items(), key=lambda x: x[1])[0] if hour_distribution else "N/A"
        
        return {
            "weekday_distribution": weekday_distribution,
            "hour_distribution": hour_distribution,
            "average_frequency_days": round(avg_frequency, 1),
            "most_common_day": most_common_day,
            "most_common_hour": most_common_hour,
            "first_analyzed_post": posting_dates[0].isoformat() if posting_dates else None,
            "last_analyzed_post": posting_dates[-1].isoformat() if posting_dates else None
        }
    
    def _extract_keywords(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extrae palabras clave del contenido analizado.
        
        Args:
            content_list: Lista de contenido
            
        Returns:
            Palabras clave identificadas
        """
        # Combinar todos los títulos y descripciones
        all_text = ""
        for content in content_list:
            if "title" in content:
                all_text += " " + content["title"]
            if "description" in content:
                all_text += " " + content["description"]
            if "tags" in content and isinstance(content["tags"], list):
                all_text += " " + " ".join(content["tags"])
        
        # Convertir a minúsculas
        all_text = all_text.lower()
        
        # Palabras a ignorar
        stopwords = [
            "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "si",
            "de", "del", "a", "ante", "con", "en", "para", "por", "según", "sin", "sobre",
            "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "for", "with", "on",
            "at", "from", "by", "about", "as", "into", "like", "through", "after", "over",
            "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "this", "that",
            "these", "those", "mi", "tu", "su", "nuestro", "vuestro", "my", "your", "our",
            "their", "his", "her", "its", "que", "what", "who", "whom", "whose", "which",
            "cuando", "como", "where", "when", "why", "how"
        ]
        
        # Dividir en palabras
        words = all_text.split()
        
        # Filtrar palabras vacías y contar frecuencias
        word_counts = Counter([word for word in words if word not in stopwords and len(word) > 3])
        
        # Obtener las palabras más comunes
        top_keywords = word_counts.most_common(20)
        
        # Formatear resultados
        return [{"keyword": keyword, "count": count} for keyword, count in top_keywords]
    
    def _determine_niche(self, keywords: List[Dict[str, Any]]) -> str:
        """
        Determina el nicho del competidor basado en palabras clave.
        
        Args:
            keywords: Lista de palabras clave
            
        Returns:
            Nicho identificado
        """
        # Mapeo de palabras clave a nichos
        niche_keywords = {
            "finanzas": ["dinero", "inversión", "ahorro", "finanzas", "crypto", "bitcoin", "trading", "bolsa", "economía"],
            "salud": ["salud", "fitness", "ejercicio", "dieta", "nutrición", "bienestar", "yoga", "meditación", "mindfulness"],
            "tecnología": ["tech", "tecnología", "gadget", "smartphone", "computadora", "inteligencia", "artificial", "ia", "ai", "programación"],
            "gaming": ["juego", "gaming", "videojuego", "playstation", "xbox", "nintendo", "steam", "fortnite", "minecraft"],
            "humor": ["humor", "comedia", "risa", "gracioso", "meme", "divertido", "broma", "chiste", "parodia"],
            "educación": ["educación", "aprender", "curso", "tutorial", "guía", "enseñar", "escuela", "universidad", "estudiante"],
            "viajes": ["viaje", "travel", "turismo", "vacaciones", "hotel", "aventura", "destino", "playa", "montaña"],
            "moda": ["moda", "fashion", "ropa", "estilo", "belleza", "makeup", "cosmética", "tendencia", "outfit"],
            "cocina": ["cocina", "receta", "comida", "chef", "gastronomía", "restaurante", "delicioso", "sabroso", "gourmet"]
        }
        
        # Contar coincidencias por nicho
        niche_scores = {niche: 0 for niche in niche_keywords}
        
        for kw_entry in keywords:
            keyword = kw_entry["keyword"]
            count = kw_entry["count"]
            
            for niche, niche_kws in niche_keywords.items():
                if any(niche_kw in keyword for niche_kw in niche_kws):
                    niche_scores[niche] += count
        
        # Determinar el nicho con mayor puntuación
        if not niche_scores:
            return "desconocido"
        
        top_niche = max(niche_scores.items(), key=lambda x: x[1])
        
        # Si la puntuación es 0, no se pudo determinar
        if top_niche[1] == 0:
            return "desconocido"
        
        return top_niche[0]
    
    def _update_cta_database(self, cta_patterns: Dict[str, Any], platform: str):
        """
        Actualiza la base de datos de CTAs con nuevos patrones.
        
        Args:
            cta_patterns: Patrones de CTAs identificados
            platform: Plataforma
        """
        if platform not in self.cta_database:
            self.cta_database[platform] = {
                "text_ctas": [],
                "timing_patterns": {},
                "effective_ctas": []
            }
        
        # Actualizar CTAs de texto
        existing_ctas = set(cta["phrase"] for cta in self.cta_database[platform]["text_ctas"])
        for cta in cta_patterns.get("text_ctas", []):
            if cta["phrase"] not in existing_ctas:
                self.cta_database[platform]["text_ctas"].append(cta)
                existing_ctas.add(cta["phrase"])
        
        # Actualizar patrones de timing
        if "timing_patterns" in cta_patterns and cta_patterns["timing_patterns"]:
            if "timing_patterns" not in self.cta_database[platform]:
                self.cta_database[platform]["timing_patterns"] = {}
            
            # Actualizar distribución
            for bucket, count in cta_patterns["timing_patterns"].get("distribution", {}).items():
                if bucket in self.cta_database[platform]["timing_patterns"]:
                    self.cta_database[platform]["timing_patterns"][bucket] += count
                else:
                    self.cta_database[platform]["timing_patterns"][bucket] = count
        
        # Actualizar CTAs efectivos
        for cta in cta_patterns.get("most_effective", []):
            self.cta_database[platform]["effective_ctas"].append(cta)
        
        # Ordenar CTAs efectivos por engagement y limitar a los 20 mejores
        self.cta_database[platform]["effective_ctas"] = sorted(
            self.cta_database[platform]["effective_ctas"],
            key=lambda x: x["engagement_rate"],
            reverse=True
        )[:20]
    
    def analyze_niche(self, niche: str, platform: str, max_competitors: int = 10) -> Dict[str, Any]:
        """
        Analiza un nicho específico en una plataforma.
        
        Args:
            niche: Nicho a analizar
            platform: Plataforma
            max_competitors: Número máximo de competidores a analizar
            
        Returns:
            Análisis del nicho
        """
        if platform not in self.platform_adapters:
            logger.error(f"Plataforma no soportada: {platform}")
            return {
                "status": "error",
                "message": f"Plataforma no soportada: {platform}"
            }
        
        try:
            # Buscar competidores en el nicho
            adapter = self.platform_adapters[platform]
            competitors = adapter.search_channels(niche, max_results=max_competitors)
            
            if not competitors:
                return {
                    "status": "warning",
                    "message": f"No se encontraron competidores para el nicho: {niche}"
                }
            
            # Analizar cada competidor
            niche_data = {
                "niche": niche,
                "platform": platform,
                "analysis_date": datetime.now().isoformat(),
                "competitors": [],
                "aggregate_metrics": {
                    "total_subscribers": 0,
                    "average_engagement": 0,
                    "content_frequency": 0,
                    "optimal_duration": "",
                    "best_posting_times": {},
                    "top_ctas": []
                },
                "saturation_index": 0
            }
            
            for competitor in competitors:
                competitor_id = competitor.get("channel_id", "")
                if not competitor_id:
                    continue
                
                # Analizar competidor
                analysis = self.analyze_competitor(platform, competitor_id)
                
                if analysis.get("status") == "success":
                    niche_data["competitors"].append({
                        "competitor_id": competitor_id,
                        "name": competitor.get("name", ""),
                        "subscriber_count": competitor.get("subscriber_count", 0),
                        "engagement_rate": analysis["analysis"].get("engagement_rate", 0),
                        "content_count": competitor.get("content_count", 0)
                    })
                    
                    # Actualizar métricas agregadas
                    niche_data["aggregate_metrics"]["total_subscribers"] += competitor.get("subscriber_count", 0)
            
            # Calcular métricas agregadas
            if niche_data["competitors"]:
                # Promedio de engagement
                engagement_rates = [comp.get("engagement_rate", 0) for comp in niche_data["competitors"]]
                niche_data["aggregate_metrics"]["average_engagement"] = sum(engagement_rates) / len(engagement_rates)
                
                # Mejores CTAs
                all_ctas = []
                for competitor_key in [f"{platform}_{comp['competitor_id']}" for comp in niche_data["competitors"]]:
                    if competitor_key in self.competitor_data:
                        comp_data = self.competitor_data[competitor_key]
                        for cta in comp_data.get("cta_patterns", {}).get("most_effective", []):
                            all_ctas.append(cta)
                
                # Ordenar por engagement
                all_ctas.sort(key=lambda x: x["engagement_rate"], reverse=True)
                niche_data["aggregate_metrics"]["top_ctas"] = all_ctas[:10]  # Top 10
                
                # Calcular índice de saturación
                total_subs = niche_data["aggregate_metrics"]["total_subscribers"]
                avg_engagement = niche_data["aggregate_metrics"]["average_engagement"]
                competitor_count = len(niche_data["competitors"])
                
                # Fórmula simple para saturación: (competidores * log(suscriptores)) / engagement
                import math
                if avg_engagement > 0:
                    saturation = (competitor_count * math.log10(total_subs + 1)) / (avg_engagement * 100)
                    niche_data["saturation_index"] = min(100, max(0, round(saturation * 10, 1)))
                else:
                    niche_data["saturation_index"] = 50  # Valor por defecto
            
            # Guardar análisis de nicho
            self.niche_analysis[f"{platform}_{niche}"] = niche_data
            self._save_data("niche_analysis.json", self.niche_analysis)
            
            return {
                "status": "success",
                "niche": niche,
                "platform": platform,
                "analysis": niche_data
            }
            
        except Exception as e:
            logger.error(f"Error al analizar nicho {niche} en {platform}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al analizar nicho: {str(e)}"
            }
    
        def get_cta_recommendations(self, niche: str, platform: str) -> Dict[str, Any]:
        """
        Obtiene recomendaciones de CTAs para un nicho específico.
        
        Args:
            niche: Nicho
            platform: Plataforma
            
        Returns:
            Recomendaciones de CTAs
        """
        # Verificar si tenemos datos para este nicho
        niche_key = f"{platform}_{niche}"
        if niche_key not in self.niche_analysis:
            logger.warning(f"No hay análisis disponible para el nicho: {niche} en {platform}")
            return {
                "status": "warning",
                "message": f"No hay análisis disponible para el nicho: {niche} en {platform}",
                "recommendations": []
            }
        
        # Obtener datos del nicho
        niche_data = self.niche_analysis[niche_key]
        
        # Verificar si hay CTAs efectivos en el nicho
        if not niche_data.get("aggregate_metrics", {}).get("top_ctas", []):
            logger.warning(f"No hay CTAs efectivos identificados para el nicho: {niche}")
            return {
                "status": "warning",
                "message": f"No hay CTAs efectivos identificados para el nicho: {niche}",
                "recommendations": []
            }
        
        # Obtener CTAs efectivos del nicho
        top_ctas = niche_data["aggregate_metrics"]["top_ctas"]
        
        # Obtener CTAs generales de la plataforma
        platform_ctas = self.cta_database.get(platform, {}).get("effective_ctas", [])
        
        # Combinar y ordenar por engagement
        all_ctas = top_ctas + [cta for cta in platform_ctas if cta not in top_ctas]
        all_ctas.sort(key=lambda x: x["engagement_rate"], reverse=True)
        
        # Generar recomendaciones
        recommendations = []
        for cta in all_ctas[:10]:  # Top 10
            # Buscar contexto para este CTA
            context = ""
            for comp_key, comp_data in self.competitor_data.items():
                if comp_data["platform"] == platform:
                    for text_cta in comp_data.get("cta_patterns", {}).get("text_ctas", []):
                        if text_cta["phrase"] == cta["phrase"]:
                            context = text_cta.get("context", "")
                            break
                    if context:
                        break
            
            recommendations.append({
                "phrase": cta["phrase"],
                "engagement_rate": round(cta["engagement_rate"] * 100, 2),
                "context": context,
                "source": "niche" if cta in top_ctas else "platform"
            })
        
        # Obtener patrones de timing
        timing_patterns = {}
        for comp_key, comp_data in self.competitor_data.items():
            if comp_data["platform"] == platform and comp_key.split("_")[1] in [comp["competitor_id"] for comp in niche_data.get("competitors", [])]:
                timing = comp_data.get("cta_patterns", {}).get("timing_patterns", {})
                if timing:
                    timing_patterns[comp_key] = timing
        
        # Calcular timing óptimo
        optimal_timing = "N/A"
        if timing_patterns:
            all_timings = []
            for comp_key, timing in timing_patterns.items():
                if "average_timing" in timing:
                    all_timings.append(timing["average_timing"])
            
            if all_timings:
                optimal_timing = f"{round(sum(all_timings) / len(all_timings), 1)}s"
        
        return {
            "status": "success",
            "niche": niche,
            "platform": platform,
            "recommendations": recommendations,
            "optimal_timing": optimal_timing,
            "saturation_index": niche_data.get("saturation_index", 0)
        }
    
    def track_content_trends(self, platform: str, keywords: List[str], days: int = 30) -> Dict[str, Any]:
        """
        Rastrea tendencias de contenido en una plataforma.
        
        Args:
            platform: Plataforma
            keywords: Palabras clave a rastrear
            days: Número de días a analizar
            
        Returns:
            Tendencias de contenido
        """
        if platform not in self.platform_adapters:
            logger.error(f"Plataforma no soportada: {platform}")
            return {
                "status": "error",
                "message": f"Plataforma no soportada: {platform}"
            }
        
        try:
            # Obtener adaptador
            adapter = self.platform_adapters[platform]
            
            # Calcular fecha de inicio
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Inicializar resultados
            trend_results = {
                "platform": platform,
                "analysis_date": datetime.now().isoformat(),
                "period_days": days,
                "keywords": {},
                "trending_content": []
            }
            
            # Analizar cada palabra clave
            for keyword in keywords:
                # Buscar contenido relacionado
                content_list = adapter.search_content(keyword, max_results=50, start_date=start_date)
                
                if not content_list:
                    continue
                
                # Analizar métricas
                views = [content.get("view_count", 0) for content in content_list]
                likes = [content.get("like_count", 0) for content in content_list]
                comments = [content.get("comment_count", 0) for content in content_list]
                
                # Calcular promedios
                avg_views = sum(views) / len(views) if views else 0
                avg_likes = sum(likes) / len(likes) if likes else 0
                avg_comments = sum(comments) / len(comments) if comments else 0
                
                # Calcular engagement
                engagement_rates = []
                for content in content_list:
                    view_count = content.get("view_count", 0)
                    if view_count > 0:
                        engagement = (content.get("like_count", 0) + content.get("comment_count", 0)) / view_count
                        engagement_rates.append(engagement)
                
                avg_engagement = sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0
                
                # Guardar métricas
                trend_results["keywords"][keyword] = {
                    "content_count": len(content_list),
                    "average_views": round(avg_views, 1),
                    "average_likes": round(avg_likes, 1),
                    "average_comments": round(avg_comments, 1),
                    "average_engagement": round(avg_engagement * 100, 2),
                    "trending_score": round((avg_views * 0.4 + avg_engagement * 100 * 0.6), 2)
                }
                
                # Añadir contenido a la lista de tendencias
                for content in content_list:
                    if "view_count" in content and content["view_count"] > 0:
                        trend_results["trending_content"].append({
                            "content_id": content.get("content_id", ""),
                            "title": content.get("title", ""),
                            "channel_id": content.get("channel_id", ""),
                            "channel_name": content.get("channel_name", ""),
                            "view_count": content.get("view_count", 0),
                            "like_count": content.get("like_count", 0),
                            "comment_count": content.get("comment_count", 0),
                            "publish_date": content.get("publish_date", ""),
                            "keywords": [keyword],
                            "engagement_rate": (content.get("like_count", 0) + content.get("comment_count", 0)) / content.get("view_count", 1) * 100,
                            "trending_score": (content.get("view_count", 0) * 0.4 + 
                                              ((content.get("like_count", 0) + content.get("comment_count", 0)) / 
                                               content.get("view_count", 1) * 100) * 0.6)
                        })
            
            # Eliminar duplicados y ordenar por puntuación de tendencia
            unique_content = {}
            for content in trend_results["trending_content"]:
                content_id = content["content_id"]
                if content_id in unique_content:
                    # Combinar palabras clave
                    unique_content[content_id]["keywords"].extend(content["keywords"])
                    unique_content[content_id]["keywords"] = list(set(unique_content[content_id]["keywords"]))
                else:
                    unique_content[content_id] = content
            
            # Convertir de nuevo a lista y ordenar
            trend_results["trending_content"] = sorted(
                list(unique_content.values()),
                key=lambda x: x["trending_score"],
                reverse=True
            )[:20]  # Top 20
            
            # Ordenar palabras clave por puntuación de tendencia
            sorted_keywords = sorted(
                [(k, v["trending_score"]) for k, v in trend_results["keywords"].items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            trend_results["top_keywords"] = [{"keyword": k, "trending_score": s} for k, s in sorted_keywords]
            
            # Guardar resultados
            trend_key = f"{platform}_{datetime.now().strftime('%Y%m%d')}"
            self.content_trends[trend_key] = trend_results
            self._save_data("content_trends.json", self.content_trends)
            
            return {
                "status": "success",
                "platform": platform,
                "trends": trend_results
            }
            
        except Exception as e:
            logger.error(f"Error al rastrear tendencias en {platform}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al rastrear tendencias: {str(e)}"
            }
    
    def find_market_gaps(self, niche: str, platform: str) -> Dict[str, Any]:
        """
        Identifica oportunidades y nichos de mercado subexplotados.
        
        Args:
            niche: Nicho principal
            platform: Plataforma
            
        Returns:
            Oportunidades de mercado identificadas
        """
        # Verificar si tenemos datos para este nicho
        niche_key = f"{platform}_{niche}"
        if niche_key not in self.niche_analysis:
            logger.warning(f"No hay análisis disponible para el nicho: {niche} en {platform}")
            return {
                "status": "warning",
                "message": f"No hay análisis disponible para el nicho: {niche} en {platform}",
                "opportunities": []
            }
        
        try:
            # Obtener datos del nicho
            niche_data = self.niche_analysis[niche_key]
            
            # Obtener palabras clave relacionadas
            related_keywords = set()
            for competitor_id in [comp["competitor_id"] for comp in niche_data.get("competitors", [])]:
                comp_key = f"{platform}_{competitor_id}"
                if comp_key in self.competitor_data:
                    for keyword in self.competitor_data[comp_key].get("top_keywords", []):
                        related_keywords.add(keyword["keyword"])
            
            # Analizar tendencias para estas palabras clave
            if related_keywords:
                trends = self.track_content_trends(platform, list(related_keywords)[:20])
                
                if trends.get("status") != "success":
                    return {
                        "status": "error",
                        "message": "No se pudieron obtener tendencias para el análisis de oportunidades"
                    }
                
                trend_data = trends["trends"]
            else:
                return {
                    "status": "warning",
                    "message": "No hay suficientes palabras clave para analizar oportunidades",
                    "opportunities": []
                }
            
            # Identificar subnichos con alta demanda y baja competencia
            opportunities = []
            
            for keyword_data in trend_data["top_keywords"]:
                keyword = keyword_data["keyword"]
                
                # Buscar competidores específicos para este subnicho
                subniche_competitors = self.platform_adapters[platform].search_channels(keyword, max_results=5)
                
                if not subniche_competitors:
                    continue
                
                # Calcular métricas de competencia
                total_subs = sum(comp.get("subscriber_count", 0) for comp in subniche_competitors)
                avg_subs = total_subs / len(subniche_competitors) if subniche_competitors else 0
                
                # Obtener métricas de demanda
                demand_score = trend_data["keywords"].get(keyword, {}).get("trending_score", 0)
                
                # Calcular puntuación de oportunidad (demanda alta + competencia baja = buena oportunidad)
                opportunity_score = 0
                if avg_subs > 0:
                    opportunity_score = demand_score / (math.log10(avg_subs + 1) * 0.5)
                
                # Añadir a la lista de oportunidades
                opportunities.append({
                    "keyword": keyword,
                    "demand_score": demand_score,
                    "competition_level": {
                        "competitor_count": len(subniche_competitors),
                        "average_subscribers": round(avg_subs, 0),
                        "total_subscribers": total_subs
                    },
                    "opportunity_score": round(opportunity_score, 2)
                })
            
            # Ordenar por puntuación de oportunidad
            opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
            
            return {
                "status": "success",
                "niche": niche,
                "platform": platform,
                "opportunities": opportunities[:10],  # Top 10
                "related_keywords": list(related_keywords)[:20]
            }
            
        except Exception as e:
            logger.error(f"Error al identificar oportunidades de mercado: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al identificar oportunidades de mercado: {str(e)}"
            }
    
    def generate_competitor_report(self, competitor_id: str, platform: str) -> Dict[str, Any]:
        """
        Genera un informe detallado sobre un competidor específico.
        
        Args:
            competitor_id: ID del competidor
            platform: Plataforma
            
        Returns:
            Informe detallado del competidor
        """
        competitor_key = f"{platform}_{competitor_id}"
        if competitor_key not in self.competitor_data:
            logger.warning(f"No hay datos disponibles para el competidor: {competitor_id} en {platform}")
            return {
                "status": "warning",
                "message": f"No hay datos disponibles para el competidor: {competitor_id} en {platform}",
                "report": {}
            }
        
        try:
            # Obtener datos del competidor
            competitor_data = self.competitor_data[competitor_key]
            
            # Obtener información adicional del adaptador
            adapter = self.platform_adapters.get(platform)
            if not adapter:
                return {
                    "status": "error",
                    "message": f"Plataforma no soportada: {platform}"
                }
            
            competitor_info = adapter.get_channel_info(competitor_id)
            
            # Generar informe
            report = {
                "competitor_id": competitor_id,
                "platform": platform,
                "report_date": datetime.now().isoformat(),
                "basic_info": {
                    "name": competitor_info.get("name", ""),
                    "description": competitor_info.get("description", ""),
                    "subscriber_count": competitor_info.get("subscriber_count", 0),
                    "content_count": competitor_info.get("content_count", 0),
                    "creation_date": competitor_info.get("creation_date", ""),
                    "profile_url": competitor_info.get("profile_url", "")
                },
                "engagement_metrics": {
                    "engagement_rate": competitor_data.get("engagement_rate", 0),
                    "average_views": competitor_data.get("content_metrics", {}).get("average_metrics", {}).get("views", 0),
                    "average_likes": competitor_data.get("content_metrics", {}).get("average_metrics", {}).get("likes", 0),
                    "average_comments": competitor_data.get("content_metrics", {}).get("average_metrics", {}).get("comments", 0)
                },
                "content_strategy": {
                    "posting_schedule": competitor_data.get("posting_schedule", {}),
                    "optimal_duration": competitor_data.get("content_metrics", {}).get("optimal_duration", "N/A"),
                    "top_content": competitor_data.get("content_metrics", {}).get("top_content", {})
                },
                "audience_insights": {
                    "niche_category": competitor_data.get("niche_category", "desconocido"),
                    "top_keywords": competitor_data.get("top_keywords", [])
                },
                "cta_strategy": {
                    "text_ctas": competitor_data.get("cta_patterns", {}).get("text_ctas", []),
                    "timing_patterns": competitor_data.get("cta_patterns", {}).get("timing_patterns", {}),
                    "most_effective": competitor_data.get("cta_patterns", {}).get("most_effective", [])
                }
            }
            
            # Añadir recomendaciones
            report["recommendations"] = {
                "content_strategy": [
                    f"Publicar contenido los {report['content_strategy']['posting_schedule'].get('most_common_day', 'N/A')} a las {report['content_strategy']['posting_schedule'].get('most_common_hour', 'N/A')}",
                    f"Crear videos de duración {report['content_strategy']['optimal_duration']} para maximizar engagement",
                    "Enfocarse en los temas de mayor engagement del competidor"
                ],
                "cta_strategy": [
                    f"Utilizar CTAs como '{cta['phrase']}'" for cta in report["cta_strategy"]["most_effective"][:3]
                ] if report["cta_strategy"]["most_effective"] else ["No hay datos suficientes para recomendar CTAs"],
                "keywords": [
                    f"Incorporar palabras clave como '{kw['keyword']}'" for kw in report["audience_insights"]["top_keywords"][:5]
                ] if report["audience_insights"]["top_keywords"] else ["No hay datos suficientes para recomendar palabras clave"]
            }
            
            return {
                "status": "success",
                "competitor_id": competitor_id,
                "platform": platform,
                "report": report
            }
            
        except Exception as e:
            logger.error(f"Error al generar informe del competidor: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al generar informe del competidor: {str(e)}"
            }
    
    def export_analysis_data(self, format_type: str = "json") -> Dict[str, Any]:
        """
        Exporta todos los datos de análisis para uso externo.
        
        Args:
            format_type: Formato de exportación (json, csv)
            
        Returns:
            Resultado de la exportación
        """
        valid_formats = ["json", "csv"]
        if format_type not in valid_formats:
            return {
                "status": "error",
                "message": f"Formato no válido. Debe ser uno de: {', '.join(valid_formats)}"
            }
        
        try:
            # Crear directorio de exportación
            export_dir = os.path.join(self.data_path, "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            # Generar nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"competitor_analysis_export_{timestamp}.{format_type}"
            filepath = os.path.join(export_dir, filename)
            
            # Preparar datos para exportación
            export_data = {
                "metadata": {
                    "export_date": datetime.now().isoformat(),
                    "competitor_count": len(self.competitor_data),
                    "niche_count": len(self.niche_analysis),
                    "platform_count": len(set(comp_data["platform"] for comp_data in self.competitor_data.values()))
                },
                "competitors": self.competitor_data,
                "niches": self.niche_analysis,
                "cta_database": self.cta_database,
                "content_trends": self.content_trends
            }
            
            # Exportar datos
            if format_type == "json":
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=4)
            
            elif format_type == "csv":
                # Para CSV, exportar tablas separadas
                import csv
                
                # Exportar competidores
                comp_filepath = os.path.join(export_dir, f"competitors_{timestamp}.csv")
                with open(comp_filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["competitor_id", "platform", "subscriber_count", "engagement_rate", "niche_category"])
                    
                    for comp_key, comp_data in self.competitor_data.items():
                        writer.writerow([
                            comp_data.get("competitor_id", ""),
                            comp_data.get("platform", ""),
                            comp_data.get("subscriber_count", 0),
                            comp_data.get("engagement_rate", 0),
                            comp_data.get("niche_category", "")
                        ])
                
                # Exportar nichos
                niche_filepath = os.path.join(export_dir, f"niches_{timestamp}.csv")
                with open(niche_filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["niche", "platform", "total_subscribers", "average_engagement", "saturation_index"])
                    
                    for niche_key, niche_data in self.niche_analysis.items():
                        writer.writerow([
                            niche_data.get("niche", ""),
                            niche_data.get("platform", ""),
                            niche_data.get("aggregate_metrics", {}).get("total_subscribers", 0),
                            niche_data.get("aggregate_metrics", {}).get("average_engagement", 0),
                            niche_data.get("saturation_index", 0)
                        ])
                
                # Actualizar filepath para incluir todos los archivos
                filepath = export_dir
            
            logger.info(f"Datos de análisis exportados: {filepath}")
            
            return {
                "status": "success",
                "message": "Datos exportados correctamente",
                "filepath": filepath,
                "format": format_type
            }
            
        except Exception as e:
            logger.error(f"Error al exportar datos: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al exportar datos: {str(e)}"
            }