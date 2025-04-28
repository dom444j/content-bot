import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import requests
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class NicheSaturationDetector:
    """
    Detector de saturación de nichos que analiza tendencias, competencia y rendimiento
    para identificar cuando un nicho está saturándose y requiere pivote o adaptación.
    """
    
    def __init__(self, config_path: str = "../config/niches.json"):
        """
        Inicializa el detector de saturación de nichos
        
        Args:
            config_path: Ruta al archivo de configuración de nichos
        """
        self.logger = logging.getLogger("niche_saturation")
        handler = logging.FileHandler("../logs/niche_saturation.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Cargar configuración
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.logger.info(f"Configuración cargada desde {config_path}")
        except Exception as e:
            self.logger.error(f"Error al cargar configuración: {str(e)}")
            self.config = {
                "niches": {},
                "saturation_thresholds": {
                    "competition_growth_rate": 0.3,  # 30% de crecimiento mensual en competencia
                    "engagement_decline_rate": 0.2,  # 20% de caída en engagement
                    "cpm_decline_rate": 0.15,        # 15% de caída en CPM
                    "content_similarity": 0.7,       # 70% de similitud en contenido
                    "trend_decline_rate": 0.25       # 25% de caída en búsquedas
                },
                "analysis_period_days": 30,
                "warning_threshold": 0.6,            # 60% del umbral para advertencia
                "critical_threshold": 0.8,           # 80% del umbral para alerta crítica
                "api_keys": {
                    "google_trends": "",
                    "youtube": "",
                    "tiktok": ""
                }
            }
        
        # Inicializar datos de nichos
        self.niche_data = {}
        self.load_niche_data()
        
        # Métricas de saturación por nicho
        self.saturation_metrics = {}
    
    def load_niche_data(self) -> None:
        """Carga datos históricos de nichos desde archivos"""
        try:
            niche_data_path = "../data/niche_data.json"
            if os.path.exists(niche_data_path):
                with open(niche_data_path, 'r', encoding='utf-8') as f:
                    self.niche_data = json.load(f)
                self.logger.info(f"Datos de nichos cargados desde {niche_data_path}")
            else:
                self.logger.warning(f"No se encontró archivo de datos de nichos en {niche_data_path}")
                # Inicializar estructura de datos
                for niche_id, niche_info in self.config.get("niches", {}).items():
                    self.niche_data[niche_id] = {
                        "name": niche_info.get("name", ""),
                        "keywords": niche_info.get("keywords", []),
                        "platforms": niche_info.get("platforms", []),
                        "historical_data": [],
                        "competition_data": [],
                        "trend_data": [],
                        "last_update": datetime.now().isoformat()
                    }
        except Exception as e:
            self.logger.error(f"Error al cargar datos de nichos: {str(e)}")
    
    def save_niche_data(self) -> None:
        """Guarda datos de nichos en archivo"""
        try:
            niche_data_path = "../data/niche_data.json"
            with open(niche_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.niche_data, f, indent=2)
            self.logger.info(f"Datos de nichos guardados en {niche_data_path}")
        except Exception as e:
            self.logger.error(f"Error al guardar datos de nichos: {str(e)}")
    
    def update_niche_data(self, niche_id: str, platform: str, metrics: Dict[str, Any]) -> None:
        """
        Actualiza datos históricos de un nicho con nuevas métricas
        
        Args:
            niche_id: ID del nicho
            platform: Plataforma (youtube, tiktok, instagram, etc.)
            metrics: Métricas actualizadas (engagement, cpm, views, etc.)
        """
        try:
            if niche_id not in self.niche_data:
                self.logger.warning(f"Nicho {niche_id} no encontrado, creando nuevo registro")
                self.niche_data[niche_id] = {
                    "name": niche_id,
                    "keywords": [],
                    "platforms": [platform],
                    "historical_data": [],
                    "competition_data": [],
                    "trend_data": [],
                    "last_update": datetime.now().isoformat()
                }
            
            # Añadir nuevos datos históricos
            entry = {
                "date": datetime.now().isoformat(),
                "platform": platform,
                "metrics": metrics
            }
            
            self.niche_data[niche_id]["historical_data"].append(entry)
            self.niche_data[niche_id]["last_update"] = datetime.now().isoformat()
            
            # Guardar cambios
            self.save_niche_data()
            
            self.logger.info(f"Datos actualizados para nicho {niche_id} en plataforma {platform}")
        except Exception as e:
            self.logger.error(f"Error al actualizar datos de nicho {niche_id}: {str(e)}")
    
    def fetch_competition_data(self, niche_id: str) -> None:
        """
        Obtiene datos de competencia para un nicho específico
        
        Args:
            niche_id: ID del nicho
        """
        try:
            if niche_id not in self.niche_data:
                self.logger.error(f"Nicho {niche_id} no encontrado")
                return
            
            niche_info = self.niche_data[niche_id]
            keywords = niche_info.get("keywords", [])
            platforms = niche_info.get("platforms", [])
            
            competition_data = {
                "date": datetime.now().isoformat(),
                "metrics": {}
            }
            
            # Simular obtención de datos de competencia
            # En producción, esto se conectaría a APIs reales
            for platform in platforms:
                if platform == "youtube":
                    # Simulación de datos de YouTube
                    competition_data["metrics"][platform] = {
                        "channel_count": np.random.randint(100, 10000),
                        "avg_views": np.random.randint(1000, 100000),
                        "avg_engagement_rate": np.random.uniform(0.01, 0.2),
                        "content_growth_rate": np.random.uniform(0.05, 0.5)
                    }
                elif platform == "tiktok":
                    # Simulación de datos de TikTok
                    competition_data["metrics"][platform] = {
                        "creator_count": np.random.randint(500, 20000),
                        "avg_views": np.random.randint(5000, 500000),
                        "avg_engagement_rate": np.random.uniform(0.05, 0.3),
                        "content_growth_rate": np.random.uniform(0.1, 0.7)
                    }
                elif platform == "instagram":
                    # Simulación de datos de Instagram
                    competition_data["metrics"][platform] = {
                        "creator_count": np.random.randint(300, 15000),
                        "avg_views": np.random.randint(2000, 200000),
                        "avg_engagement_rate": np.random.uniform(0.03, 0.25),
                        "content_growth_rate": np.random.uniform(0.08, 0.6)
                    }
            
            # Guardar datos de competencia
            self.niche_data[niche_id]["competition_data"].append(competition_data)
            self.niche_data[niche_id]["last_update"] = datetime.now().isoformat()
            
            # Guardar cambios
            self.save_niche_data()
            
            self.logger.info(f"Datos de competencia actualizados para nicho {niche_id}")
        except Exception as e:
            self.logger.error(f"Error al obtener datos de competencia para nicho {niche_id}: {str(e)}")
    
    def fetch_trend_data(self, niche_id: str) -> None:
        """
        Obtiene datos de tendencias para un nicho específico
        
        Args:
            niche_id: ID del nicho
        """
        try:
            if niche_id not in self.niche_data:
                self.logger.error(f"Nicho {niche_id} no encontrado")
                return
            
            niche_info = self.niche_data[niche_id]
            keywords = niche_info.get("keywords", [])
            
            trend_data = {
                "date": datetime.now().isoformat(),
                "keywords": {}
            }
            
            # Simular obtención de datos de tendencias
            # En producción, esto se conectaría a Google Trends API
            for keyword in keywords:
                # Simulación de datos de tendencias
                trend_data["keywords"][keyword] = {
                    "search_volume": np.random.randint(1000, 1000000),
                    "growth_rate": np.random.uniform(-0.3, 0.5),
                    "competition_level": np.random.uniform(0.1, 0.9)
                }
            
            # Guardar datos de tendencias
            self.niche_data[niche_id]["trend_data"].append(trend_data)
            self.niche_data[niche_id]["last_update"] = datetime.now().isoformat()
            
            # Guardar cambios
            self.save_niche_data()
            
            self.logger.info(f"Datos de tendencias actualizados para nicho {niche_id}")
        except Exception as e:
            self.logger.error(f"Error al obtener datos de tendencias para nicho {niche_id}: {str(e)}")
    
    def analyze_niche_saturation(self, niche_id: str) -> Dict[str, Any]:
        """
        Analiza la saturación de un nicho específico
        
        Args:
            niche_id: ID del nicho
            
        Returns:
            Diccionario con métricas de saturación
        """
        try:
            if niche_id not in self.niche_data:
                self.logger.error(f"Nicho {niche_id} no encontrado")
                return {"error": "Nicho no encontrado"}
            
            niche_info = self.niche_data[niche_id]
            
            # Obtener datos recientes si es necesario
            cutoff_date = (datetime.now() - timedelta(days=self.config.get("analysis_period_days", 30)))
            cutoff_date_str = cutoff_date.isoformat()
            
            if not niche_info["competition_data"] or datetime.fromisoformat(niche_info["competition_data"][-1]["date"]) < cutoff_date:
                self.fetch_competition_data(niche_id)
            
            if not niche_info["trend_data"] or datetime.fromisoformat(niche_info["trend_data"][-1]["date"]) < cutoff_date:
                self.fetch_trend_data(niche_id)
            
            # Filtrar datos históricos recientes
            recent_historical = [
                entry for entry in niche_info["historical_data"] 
                if datetime.fromisoformat(entry["date"]) >= cutoff_date
            ]
            
            recent_competition = [
                entry for entry in niche_info["competition_data"] 
                if datetime.fromisoformat(entry["date"]) >= cutoff_date
            ]
            
            recent_trends = [
                entry for entry in niche_info["trend_data"] 
                if datetime.fromisoformat(entry["date"]) >= cutoff_date
            ]
            
            # Calcular métricas de saturación
            saturation_metrics = {
                "competition_growth": self._calculate_competition_growth(recent_competition),
                "engagement_decline": self._calculate_engagement_decline(recent_historical),
                "cpm_decline": self._calculate_cpm_decline(recent_historical),
                "content_similarity": self._calculate_content_similarity(niche_id),
                "trend_decline": self._calculate_trend_decline(recent_trends)
            }
            
            # Calcular puntuación de saturación (0-1)
            thresholds = self.config.get("saturation_thresholds", {})
            
            saturation_score = 0
            factor_count = 0
            
            if "competition_growth" in saturation_metrics and "competition_growth_rate" in thresholds:
                factor = min(1.0, saturation_metrics["competition_growth"] / thresholds["competition_growth_rate"])
                saturation_score += factor
                factor_count += 1
            
            if "engagement_decline" in saturation_metrics and "engagement_decline_rate" in thresholds:
                factor = min(1.0, saturation_metrics["engagement_decline"] / thresholds["engagement_decline_rate"])
                saturation_score += factor
                factor_count += 1
            
            if "cpm_decline" in saturation_metrics and "cpm_decline_rate" in thresholds:
                factor = min(1.0, saturation_metrics["cpm_decline"] / thresholds["cpm_decline_rate"])
                saturation_score += factor
                factor_count += 1
            
            if "content_similarity" in saturation_metrics and "content_similarity" in thresholds:
                factor = min(1.0, saturation_metrics["content_similarity"] / thresholds["content_similarity"])
                saturation_score += factor
                factor_count += 1
            
            if "trend_decline" in saturation_metrics and "trend_decline_rate" in thresholds:
                factor = min(1.0, saturation_metrics["trend_decline"] / thresholds["trend_decline_rate"])
                saturation_score += factor
                factor_count += 1
            
            if factor_count > 0:
                saturation_score /= factor_count
            
            # Determinar nivel de alerta
            warning_threshold = self.config.get("warning_threshold", 0.6)
            critical_threshold = self.config.get("critical_threshold", 0.8)
            
            if saturation_score >= critical_threshold:
                alert_level = "CRÍTICO"
            elif saturation_score >= warning_threshold:
                alert_level = "ADVERTENCIA"
            else:
                alert_level = "NORMAL"
            
            # Generar recomendaciones
            recommendations = self._generate_recommendations(niche_id, saturation_metrics, saturation_score)
            
            # Guardar métricas de saturación
            self.saturation_metrics[niche_id] = {
                "date": datetime.now().isoformat(),
                "metrics": saturation_metrics,
                "saturation_score": saturation_score,
                "alert_level": alert_level,
                "recommendations": recommendations
            }
            
            result = {
                "niche_id": niche_id,
                "niche_name": niche_info.get("name", niche_id),
                "saturation_metrics": saturation_metrics,
                "saturation_score": saturation_score,
                "alert_level": alert_level,
                "recommendations": recommendations,
                "analysis_date": datetime.now().isoformat()
            }
            
            self.logger.info(f"Análisis de saturación completado para nicho {niche_id}: {alert_level}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al analizar saturación de nicho {niche_id}: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_competition_growth(self, competition_data: List[Dict[str, Any]]) -> float:
        """
        Calcula la tasa de crecimiento de la competencia
        
        Args:
            competition_data: Datos de competencia
            
        Returns:
            Tasa de crecimiento de la competencia
        """
        if len(competition_data) < 2:
            return 0.0
        
        try:
            # Ordenar por fecha
            sorted_data = sorted(competition_data, key=lambda x: x["date"])
            
            # Calcular crecimiento para cada plataforma
            growth_rates = []
            
            for platform in sorted_data[-1]["metrics"].keys():
                if platform in sorted_data[0]["metrics"]:
                    # Usar métricas relevantes para cada plataforma
                    if platform == "youtube":
                        metric_key = "channel_count"
                    elif platform in ["tiktok", "instagram"]:
                        metric_key = "creator_count"
                    else:
                        continue
                    
                    initial_value = sorted_data[0]["metrics"][platform].get(metric_key, 0)
                    final_value = sorted_data[-1]["metrics"][platform].get(metric_key, 0)
                    
                    if initial_value > 0:
                        growth_rate = (final_value - initial_value) / initial_value
                        growth_rates.append(growth_rate)
            
            # Promedio de tasas de crecimiento
            if growth_rates:
                return sum(growth_rates) / len(growth_rates)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error al calcular crecimiento de competencia: {str(e)}")
            return 0.0
    
    def _calculate_engagement_decline(self, historical_data: List[Dict[str, Any]]) -> float:
        """
        Calcula la tasa de disminución del engagement
        
        Args:
            historical_data: Datos históricos
            
        Returns:
            Tasa de disminución del engagement
        """
        if len(historical_data) < 2:
            return 0.0
        
        try:
            # Agrupar por plataforma
            platform_data = {}
            
            for entry in historical_data:
                platform = entry["platform"]
                if platform not in platform_data:
                    platform_data[platform] = []
                
                if "engagement_rate" in entry["metrics"]:
                    platform_data[platform].append({
                        "date": entry["date"],
                        "engagement_rate": entry["metrics"]["engagement_rate"]
                    })
            
            # Calcular tendencia para cada plataforma
            decline_rates = []
            
            for platform, data in platform_data.items():
                if len(data) < 2:
                    continue
                
                # Ordenar por fecha
                sorted_data = sorted(data, key=lambda x: x["date"])
                
                # Calcular tendencia lineal
                dates = [datetime.fromisoformat(entry["date"]).timestamp() for entry in sorted_data]
                rates = [entry["engagement_rate"] for entry in sorted_data]
                
                if len(dates) >= 3:
                    # Suavizar datos con filtro Savitzky-Golay si hay suficientes puntos
                    window_length = min(len(dates), 5)
                    if window_length % 2 == 0:
                        window_length -= 1
                    
                    if window_length >= 3:
                        rates = savgol_filter(rates, window_length, 1).tolist()
                
                # Calcular pendiente
                if len(dates) > 1 and max(rates) > 0:
                    # Normalizar fechas
                    min_date = min(dates)
                    norm_dates = [(d - min_date) / (max(dates) - min_date) if max(dates) > min_date else 0 for d in dates]
                    
                    # Calcular pendiente con regresión lineal simple
                    n = len(norm_dates)
                    sum_x = sum(norm_dates)
                    sum_y = sum(rates)
                    sum_xy = sum(x * y for x, y in zip(norm_dates, rates))
                    sum_xx = sum(x * x for x in norm_dates)
                    
                    if n * sum_xx - sum_x * sum_x != 0:
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                        
                        # Convertir pendiente a tasa de cambio relativa
                        avg_rate = sum_y / n
                        if avg_rate > 0:
                            relative_change = -slope / avg_rate  # Negativo porque buscamos disminución
                            decline_rates.append(max(0, relative_change))  # Solo considerar disminuciones
            
            # Promedio de tasas de disminución
            if decline_rates:
                return sum(decline_rates) / len(decline_rates)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error al calcular disminución de engagement: {str(e)}")
            return 0.0
    
    def _calculate_cpm_decline(self, historical_data: List[Dict[str, Any]]) -> float:
        """
        Calcula la tasa de disminución del CPM
        
        Args:
            historical_data: Datos históricos
            
        Returns:
            Tasa de disminución del CPM
        """
        if len(historical_data) < 2:
            return 0.0
        
        try:
            # Agrupar por plataforma
            platform_data = {}
            
            for entry in historical_data:
                platform = entry["platform"]
                if platform not in platform_data:
                    platform_data[platform] = []
                
                if "cpm" in entry["metrics"]:
                    platform_data[platform].append({
                        "date": entry["date"],
                        "cpm": entry["metrics"]["cpm"]
                    })
            
            # Calcular tendencia para cada plataforma
            decline_rates = []
            
            for platform, data in platform_data.items():
                if len(data) < 2:
                    continue
                
                # Ordenar por fecha
                sorted_data = sorted(data, key=lambda x: x["date"])
                
                # Calcular tendencia lineal
                dates = [datetime.fromisoformat(entry["date"]).timestamp() for entry in sorted_data]
                cpms = [entry["cpm"] for entry in sorted_data]
                
                if len(dates) >= 3:
                    # Suavizar datos con filtro Savitzky-Golay si hay suficientes puntos
                    window_length = min(len(dates), 5)
                    if window_length % 2 == 0:
                        window_length -= 1
                    
                    if window_length >= 3:
                        cpms = savgol_filter(cpms, window_length, 1).tolist()
                
                # Calcular pendiente
                if len(dates) > 1 and max(cpms) > 0:
                    # Normalizar fechas
                    min_date = min(dates)
                    norm_dates = [(d - min_date) / (max(dates) - min_date) if max(dates) > min_date else 0 for d in dates]
                    
                    # Calcular pendiente con regresión lineal simple
                    n = len(norm_dates)
                    sum_x = sum(norm_dates)
                    sum_y = sum(cpms)
                    sum_xy = sum(x * y for x, y in zip(norm_dates, cpms))
                    sum_xx = sum(x * x for x in norm_dates)
                    
                    if n * sum_xx - sum_x * sum_x != 0:
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
                        
                        # Convertir pendiente a tasa de cambio relativa
                        avg_cpm = sum_y / n
                        if avg_cpm > 0:
                            relative_change = -slope / avg_cpm  # Negativo porque buscamos disminución
                            decline_rates.append(max(0, relative_change))  # Solo considerar disminuciones
            
            # Promedio de tasas de disminución
            if decline_rates:
                return sum(decline_rates) / len(decline_rates)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error al calcular disminución de CPM: {str(e)}")
            return 0.0
    
    def _calculate_content_similarity(self, niche_id: str) -> float:
        """
        Calcula la similitud de contenido en el nicho
        
        Args:
            niche_id: ID del nicho
            
        Returns:
            Índice de similitud de contenido (0-1)
        """
        try:
            # En una implementación real, esto analizaría contenido real
            # Aquí simulamos un valor basado en datos existentes
            
            if niche_id not in self.niche_data:
                return 0.5  # Valor predeterminado
            
            niche_info = self.niche_data[niche_id]
            
            # Usar datos de competencia si están disponibles
            if niche_info["competition_data"]:
                latest_data = niche_info["competition_data"][-1]
                
                # Calcular promedio ponderado de similitud por plataforma
                similarity = 0.0
                weights = 0.0
                
                for platform, metrics in latest_data["metrics"].items():
                    if "content_growth_rate" in metrics:
                        # Mayor crecimiento de contenido puede indicar mayor similitud
                        platform_similarity = min(0.9, metrics["content_growth_rate"] * 1.5)
                        
                        # Ponderación por plataforma
                        if platform == "youtube":
                            weight = 1.0
                        elif platform == "tiktok":
                            weight = 1.2  # TikTok tiende a tener más contenido similar
                        elif platform == "instagram":
                            weight = 0.8
                        else:
                            weight = 1.0
                        
                        similarity += platform_similarity * weight
                        weights += weight
                
                if weights > 0:
                    return similarity / weights
            
            # Valor predeterminado basado en tendencias
            if niche_info["trend_data"]:
                latest_trends = niche_info["trend_data"][-1]
                
                # Calcular promedio de nivel de competencia
                competition_levels = [
                    data["competition_level"] 
                    for keyword, data in latest_trends["keywords"].items() 
                    if "competition_level" in data
                ]
                
                if competition_levels:
                    return sum(competition_levels) / len(competition_levels)
            
            return 0.5  # Valor predeterminado
                
        except Exception as e:
            self.logger.error(f"Error al calcular similitud de contenido: {str(e)}")
            return 0.5
    
    def _calculate_trend_decline(self, trend_data: List[Dict[str, Any]]) -> float:
        """
        Calcula la tasa de disminución de tendencias
        
        Args:
            trend_data: Datos de tendencias
            
        Returns:
            Tasa de disminución de tendencias
        """
        if len(trend_data) < 2:
            return 0.0
        
        try:
            # Ordenar por fecha
            sorted_data = sorted(trend_data, key=lambda x: x["date"])
            
            # Extraer tasas de crecimiento para cada keyword
            keyword_growth_rates = {}
            
            for entry in sorted_data:
                for keyword, data in entry["keywords"].items():
                    if keyword not in keyword_growth_rates:
                        keyword_growth_rates[keyword] = []
                    
                    if "growth_rate" in data:
                        keyword_growth_rates[keyword].append(data["growth_rate"])
            
            # Calcular tendencia para cada keyword
            decline_rates = []
            
            for keyword, rates in keyword_growth_rates.items():
                if len(rates) < 2:
                    continue
                
                # Calcular cambio en la tasa de crecimiento
                initial_rate = rates[0]
                final_rate = rates[-1]
                
                # Si la tasa pasó de positiva a negativa o disminuyó
                if (initial_rate > 0 and final_rate < 0) or final_rate < initial_rate:
                    decline = initial_rate - final_rate
                    decline_rates.append(decline)
            
            # Promedio de tasas de disminución
            if decline_rates:
                return sum(decline_rates) / len(decline_rates)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error al calcular disminución de tendencias: {str(e)}")
            return 0.0
    
    def _generate_recommendations(self, niche_id: str, metrics: Dict[str, float], saturation_score: float) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones basadas en métricas de saturación
        
        Args:
            niche_id: ID del nicho
            metrics: Métricas de saturación
            saturation_score: Puntuación de saturación
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        try:
                        # Recomendaciones basadas en nivel de saturación
            if saturation_score >= 0.8:
                # Nivel crítico - Recomendar pivote o cambio significativo
                recommendations.append({
                    "type": "critical",
                    "title": "Pivote de nicho recomendado",
                    "description": "El nicho está altamente saturado. Se recomienda pivotar hacia sub-nichos menos competitivos o nichos relacionados emergentes.",
                    "actions": [
                        "Identificar sub-nichos específicos con menor competencia",
                        "Explorar nichos adyacentes con audiencia similar",
                        "Reducir inversión en contenido genérico del nicho",
                        "Desarrollar propuesta de valor única y diferenciada"
                    ]
                })
                
                # Recomendar diferenciación de contenido
                recommendations.append({
                    "type": "critical",
                    "title": "Diferenciación urgente de contenido",
                    "description": "La similitud de contenido es alta. Se requiere diferenciación inmediata para destacar.",
                    "actions": [
                        "Adoptar formatos innovadores no utilizados por competidores",
                        "Desarrollar ángulos únicos sobre temas populares",
                        "Invertir en calidad de producción superior",
                        "Crear contenido de nicho profundo para audiencias específicas"
                    ]
                })
                
                # Recomendar diversificación de plataformas
                recommendations.append({
                    "type": "critical",
                    "title": "Diversificación de plataformas",
                    "description": "Diversificar hacia plataformas menos saturadas para el nicho.",
                    "actions": [
                        "Identificar plataformas emergentes con menor competencia",
                        "Adaptar contenido a formatos específicos de plataformas alternativas",
                        "Mantener presencia reducida en plataformas principales",
                        "Experimentar con plataformas de nicho especializado"
                    ]
                })
            
            elif saturation_score >= 0.6:
                # Nivel de advertencia - Recomendar adaptación
                recommendations.append({
                    "type": "warning",
                    "title": "Adaptación de estrategia recomendada",
                    "description": "El nicho muestra signos de saturación. Se recomienda adaptar la estrategia para mantener competitividad.",
                    "actions": [
                        "Enfocarse en sub-nichos específicos menos competitivos",
                        "Desarrollar contenido más especializado y profundo",
                        "Optimizar para palabras clave de cola larga",
                        "Mejorar la calidad y diferenciación del contenido"
                    ]
                })
                
                # Recomendar optimización de engagement
                if metrics.get("engagement_decline", 0) > 0.1:
                    recommendations.append({
                        "type": "warning",
                        "title": "Optimización de engagement",
                        "description": "Se detecta disminución en tasas de engagement. Se recomienda optimizar interacción.",
                        "actions": [
                            "Aumentar elementos interactivos en el contenido",
                            "Implementar estrategias de gamificación",
                            "Mejorar llamados a la acción",
                            "Experimentar con nuevos formatos de mayor engagement"
                        ]
                    })
                
                # Recomendar optimización de monetización
                if metrics.get("cpm_decline", 0) > 0.1:
                    recommendations.append({
                        "type": "warning",
                        "title": "Diversificación de monetización",
                        "description": "Se detecta disminución en CPM. Se recomienda diversificar fuentes de ingresos.",
                        "actions": [
                            "Explorar modelos de afiliación relevantes",
                            "Desarrollar productos digitales propios",
                            "Implementar estrategias de membresía o suscripción",
                            "Optimizar contenido para nichos publicitarios premium"
                        ]
                    })
            
            else:
                # Nivel normal - Recomendar optimización
                recommendations.append({
                    "type": "info",
                    "title": "Optimización continua",
                    "description": "El nicho muestra niveles saludables de competencia. Se recomienda optimización continua.",
                    "actions": [
                        "Monitorear tendencias emergentes dentro del nicho",
                        "Optimizar SEO y descubribilidad del contenido",
                        "Mejorar calidad de producción incrementalmente",
                        "Fortalecer comunidad y engagement"
                    ]
                })
                
                # Si hay crecimiento de competencia, recomendar diferenciación preventiva
                if metrics.get("competition_growth", 0) > 0.15:
                    recommendations.append({
                        "type": "info",
                        "title": "Diferenciación preventiva",
                        "description": "Se detecta crecimiento en competencia. Se recomienda diferenciación preventiva.",
                        "actions": [
                            "Desarrollar características distintivas de marca",
                            "Invertir en formatos o tecnologías innovadoras",
                            "Crear series de contenido único y reconocible",
                            "Establecer autoridad mediante contenido educativo profundo"
                        ]
                    })
            
            # Recomendaciones específicas basadas en métricas individuales
            if metrics.get("trend_decline", 0) > 0.2:
                recommendations.append({
                    "type": "warning" if metrics["trend_decline"] > 0.3 else "info",
                    "title": "Adaptación a cambios de tendencias",
                    "description": "Se detecta disminución en tendencias de búsqueda. Se recomienda adaptación a nuevos intereses.",
                    "actions": [
                        "Investigar tendencias emergentes relacionadas",
                        "Adaptar contenido a nuevos términos de búsqueda populares",
                        "Desarrollar contenido evergreen menos dependiente de tendencias",
                        "Realizar encuestas a la audiencia para identificar nuevos intereses"
                    ]
                })
            
            if metrics.get("content_similarity", 0) > 0.6:
                recommendations.append({
                    "type": "warning" if metrics["content_similarity"] > 0.7 else "info",
                    "title": "Diferenciación de contenido",
                    "description": "Se detecta alta similitud de contenido en el nicho. Se recomienda mayor diferenciación.",
                    "actions": [
                        "Desarrollar ángulos únicos sobre temas comunes",
                        "Crear formatos de contenido innovadores",
                        "Combinar temas de nichos adyacentes",
                        "Invertir en investigación original y datos exclusivos"
                    ]
                })
            
            # Añadir recomendación de análisis de competencia
            recommendations.append({
                "type": "info",
                "title": "Análisis de competencia",
                "description": "Realizar análisis detallado de competidores principales para identificar oportunidades.",
                "actions": [
                    "Identificar brechas de contenido no cubiertas por competidores",
                    "Analizar estrategias de los competidores más exitosos",
                    "Evaluar puntos débiles de competidores principales",
                    "Monitorear nuevos entrantes al nicho"
                ]
            })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones para nicho {niche_id}: {str(e)}")
            return [{
                "type": "error",
                "title": "Error en generación de recomendaciones",
                "description": f"No se pudieron generar recomendaciones: {str(e)}",
                "actions": ["Verificar datos de entrada", "Intentar nuevamente más tarde"]
            }]
    
    def visualize_saturation_trends(self, niche_id: str, save_path: Optional[str] = None) -> str:
        """
        Genera visualización de tendencias de saturación para un nicho
        
        Args:
            niche_id: ID del nicho
            save_path: Ruta para guardar la visualización (opcional)
            
        Returns:
            Ruta al archivo de visualización guardado o mensaje de error
        """
        try:
            if niche_id not in self.niche_data:
                return f"Error: Nicho {niche_id} no encontrado"
            
            niche_info = self.niche_data[niche_id]
            
            # Crear figura con múltiples subplots
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Análisis de Saturación: {niche_info.get("name", niche_id)}', fontsize=16)
            
            # 1. Gráfico de competencia
            ax1 = axs[0, 0]
            self._plot_competition_trends(niche_id, ax1)
            ax1.set_title('Tendencias de Competencia')
            ax1.set_ylabel('Cantidad de Competidores')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # 2. Gráfico de engagement
            ax2 = axs[0, 1]
            self._plot_engagement_trends(niche_id, ax2)
            ax2.set_title('Tendencias de Engagement')
            ax2.set_ylabel('Tasa de Engagement')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # 3. Gráfico de CPM
            ax3 = axs[1, 0]
            self._plot_cpm_trends(niche_id, ax3)
            ax3.set_title('Tendencias de CPM')
            ax3.set_ylabel('CPM ($)')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # 4. Gráfico de tendencias de búsqueda
            ax4 = axs[1, 1]
            self._plot_search_trends(niche_id, ax4)
            ax4.set_title('Tendencias de Búsqueda')
            ax4.set_ylabel('Volumen de Búsqueda (normalizado)')
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Ajustar layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Guardar o mostrar
            if save_path:
                # Asegurar que el directorio existe
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                self.logger.info(f"Visualización guardada en {save_path}")
                return save_path
            else:
                # Crear un directorio temporal para guardar
                temp_dir = "../reports/visualizations"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = f"{temp_dir}/saturation_{niche_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(temp_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                self.logger.info(f"Visualización guardada en {temp_path}")
                return temp_path
                
        except Exception as e:
            self.logger.error(f"Error al visualizar tendencias de saturación para nicho {niche_id}: {str(e)}")
            return f"Error: {str(e)}"