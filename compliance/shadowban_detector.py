"""
Shadowban Detector - Detector de Shadowbans

Este módulo se encarga de detectar posibles shadowbans en las plataformas,
monitoreando métricas de rendimiento y visibilidad para identificar
patrones anómalos que puedan indicar restricciones no declaradas.
"""

import os
import json
import logging
import time
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/shadowban_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ShadowbanDetector")

class ShadowbanDetector:
    """
    Detecta posibles shadowbans analizando métricas de rendimiento
    y visibilidad en diferentes plataformas.
    """
    
    def __init__(self, config_path: str = "config/platforms.json", 
                 history_path: str = "data/shadowban_history.json"):
        """
        Inicializa el detector de shadowbans.
        
        Args:
            config_path: Ruta al archivo de configuración de plataformas
            history_path: Ruta al archivo de historial de shadowbans
        """
        self.config_path = config_path
        self.history_path = history_path
        
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        # Cargar configuración
        self.platforms = self._load_config()
        
        # Cargar historial
        self.shadowban_history = self._load_history()
        
        # Umbrales de detección
        self.thresholds = {
            "youtube": {
                "views_drop_percent": 50,      # Caída de vistas en porcentaje
                "engagement_drop_percent": 40,  # Caída de engagement
                "min_samples": 5,               # Mínimo de muestras para comparar
                "baseline_days": 30,            # Días para establecer línea base
                "anomaly_z_score": 2.0          # Z-score para considerar anomalía
            },
            "tiktok": {
                "views_drop_percent": 70,       # TikTok suele tener más volatilidad
                "engagement_drop_percent": 60,
                "min_samples": 5,
                "baseline_days": 14,            # Línea base más corta por la naturaleza de TikTok
                "anomaly_z_score": 2.5
            },
            "instagram": {
                "views_drop_percent": 60,
                "engagement_drop_percent": 50,
                "min_samples": 5,
                "baseline_days": 21,
                "anomaly_z_score": 2.0
            }
        }
        
        logger.info(f"ShadowbanDetector inicializado para {len(self.platforms)} plataformas")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración de plataformas"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def _load_history(self) -> Dict[str, Any]:
        """Carga el historial de shadowbans"""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.info(f"Creando nuevo historial de shadowbans")
                return {
                    "detections": [],
                    "metrics": {}
                }
        except Exception as e:
            logger.error(f"Error al cargar historial: {str(e)}")
            return {
                "detections": [],
                "metrics": {}
            }
    
    def _save_history(self) -> bool:
        """Guarda el historial de shadowbans"""
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.shadowban_history, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error al guardar historial: {str(e)}")
            return False
    
    def register_metrics(self, platform: str, content_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Registra métricas de un contenido para análisis
        
        Args:
            platform: Plataforma del contenido
            content_id: ID del contenido
            metrics: Diccionario con métricas
                - views: Número de vistas
                - likes: Número de likes
                - comments: Número de comentarios
                - shares: Número de compartidos
                - timestamp: Timestamp de las métricas (opcional)
                - hashtags: Lista de hashtags (opcional)
                - duration: Duración en segundos (opcional)
                - title: Título del contenido (opcional)
            
        Returns:
            True si se registró correctamente, False en caso contrario
        """
        try:
            # Verificar plataforma
            if platform not in self.thresholds:
                logger.warning(f"Plataforma no soportada: {platform}")
                return False
            
            # Añadir timestamp si no existe
            if "timestamp" not in metrics:
                metrics["timestamp"] = datetime.now().isoformat()
            
            # Inicializar estructura si no existe
            if "metrics" not in self.shadowban_history:
                self.shadowban_history["metrics"] = {}
            
            if platform not in self.shadowban_history["metrics"]:
                self.shadowban_history["metrics"][platform] = {}
            
            if content_id not in self.shadowban_history["metrics"][platform]:
                self.shadowban_history["metrics"][platform][content_id] = []
            
            # Añadir métricas
            self.shadowban_history["metrics"][platform][content_id].append(metrics)
            
            # Guardar historial
            self._save_history()
            
            logger.info(f"Métricas registradas para {platform}/{content_id}")
            return True
        except Exception as e:
            logger.error(f"Error al registrar métricas: {str(e)}")
            return False
    
    def detect_shadowban(self, platform: str, content_id: str = None) -> Dict[str, Any]:
        """
        Detecta posibles shadowbans en una plataforma
        
        Args:
            platform: Plataforma a analizar
            content_id: ID del contenido específico (opcional)
            
        Returns:
            Resultado del análisis con detecciones
        """
        try:
                        # Verificar plataforma
            if platform not in self.thresholds:
                logger.warning(f"Plataforma no soportada: {platform}")
                return {"status": "error", "message": "Plataforma no soportada"}
            
            # Inicializar resultado
            result = {
                "platform": platform,
                "timestamp": datetime.now().isoformat(),
                "detections": [],
                "status": "normal"
            }
            
            # Obtener métricas
            if "metrics" not in self.shadowban_history or platform not in self.shadowban_history["metrics"]:
                logger.warning(f"No hay métricas registradas para {platform}")
                return {"status": "insufficient_data", "message": "No hay métricas registradas"}
            
            # Si se especifica un contenido, analizar solo ese
            if content_id:
                if content_id not in self.shadowban_history["metrics"][platform]:
                    logger.warning(f"No hay métricas para {platform}/{content_id}")
                    return {"status": "insufficient_data", "message": "No hay métricas para este contenido"}
                
                content_metrics = self.shadowban_history["metrics"][platform][content_id]
                detection = self._analyze_content_metrics(platform, content_id, content_metrics)
                
                if detection:
                    result["detections"].append(detection)
                    result["status"] = "shadowban_detected"
            else:
                # Analizar todos los contenidos de la plataforma
                for content_id, content_metrics in self.shadowban_history["metrics"][platform].items():
                    detection = self._analyze_content_metrics(platform, content_id, content_metrics)
                    
                    if detection:
                        result["detections"].append(detection)
                        result["status"] = "shadowban_detected"
            
            # Si se detectaron shadowbans, registrar en historial
            if result["status"] == "shadowban_detected":
                if "detections" not in self.shadowban_history:
                    self.shadowban_history["detections"] = []
                
                for detection in result["detections"]:
                    self.shadowban_history["detections"].append(detection)
                
                self._save_history()
            
            return result
        except Exception as e:
            logger.error(f"Error al detectar shadowban: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_content_metrics(self, platform: str, content_id: str, metrics: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Analiza las métricas de un contenido para detectar shadowbans
        
        Args:
            platform: Plataforma del contenido
            content_id: ID del contenido
            metrics: Lista de métricas del contenido
            
        Returns:
            Detección de shadowban o None si no se detecta
        """
        # Verificar que hay suficientes métricas
        if len(metrics) < self.thresholds[platform]["min_samples"]:
            logger.info(f"Insuficientes muestras para {platform}/{content_id}: {len(metrics)}/{self.thresholds[platform]['min_samples']}")
            return None
        
        # Ordenar métricas por timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.get("timestamp", ""))
        
        # Convertir a DataFrame para análisis
        df = pd.DataFrame(sorted_metrics)
        
        # Convertir timestamps a datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        
        # Calcular métricas de engagement
        if "likes" in df.columns and "views" in df.columns and df["views"].max() > 0:
            df["like_rate"] = df["likes"] / df["views"].clip(lower=1)
        else:
            df["like_rate"] = 0
        
        if "comments" in df.columns and "views" in df.columns and df["views"].max() > 0:
            df["comment_rate"] = df["comments"] / df["views"].clip(lower=1)
        else:
            df["comment_rate"] = 0
        
        if "shares" in df.columns and "views" in df.columns and df["views"].max() > 0:
            df["share_rate"] = df["shares"] / df["views"].clip(lower=1)
        else:
            df["share_rate"] = 0
        
        # Calcular tasa de engagement general
        df["engagement_rate"] = df["like_rate"] + df["comment_rate"] + df["share_rate"]
        
        # Dividir en línea base y período reciente
        baseline_days = self.thresholds[platform]["baseline_days"]
        
        if "timestamp" in df.columns and len(df) >= 2:
            latest_date = df["timestamp"].max()
            baseline_cutoff = latest_date - timedelta(days=baseline_days)
            
            baseline_df = df[df["timestamp"] <= baseline_cutoff]
            recent_df = df[df["timestamp"] > baseline_cutoff]
            
            # Si no hay suficientes datos para línea base o recientes, usar división por mitad
            if len(baseline_df) < 2 or len(recent_df) < 2:
                midpoint = len(df) // 2
                baseline_df = df.iloc[:midpoint]
                recent_df = df.iloc[midpoint:]
        else:
            # Si no hay timestamps, dividir por la mitad
            midpoint = len(df) // 2
            baseline_df = df.iloc[:midpoint]
            recent_df = df.iloc[midpoint:]
        
        # Verificar que hay datos en ambos períodos
        if len(baseline_df) == 0 or len(recent_df) == 0:
            logger.info(f"Insuficientes datos para comparar en {platform}/{content_id}")
            return None
        
        # Calcular promedios
        baseline_views_avg = baseline_df["views"].mean() if "views" in baseline_df else 0
        recent_views_avg = recent_df["views"].mean() if "views" in recent_df else 0
        
        baseline_engagement_avg = baseline_df["engagement_rate"].mean() if "engagement_rate" in baseline_df else 0
        recent_engagement_avg = recent_df["engagement_rate"].mean() if "engagement_rate" in recent_df else 0
        
        # Calcular cambios porcentuales
        if baseline_views_avg > 0:
            views_change_percent = ((recent_views_avg - baseline_views_avg) / baseline_views_avg) * 100
        else:
            views_change_percent = 0
        
        if baseline_engagement_avg > 0:
            engagement_change_percent = ((recent_engagement_avg - baseline_engagement_avg) / baseline_engagement_avg) * 100
        else:
            engagement_change_percent = 0
        
        # Calcular Z-scores para detectar anomalías
        if "views" in df and len(df["views"]) > 2:
            views_mean = df["views"].mean()
            views_std = df["views"].std() if df["views"].std() > 0 else 1
            recent_views_zscore = (recent_views_avg - views_mean) / views_std
        else:
            recent_views_zscore = 0
        
        if "engagement_rate" in df and len(df["engagement_rate"]) > 2:
            engagement_mean = df["engagement_rate"].mean()
            engagement_std = df["engagement_rate"].std() if df["engagement_rate"].std() > 0 else 1
            recent_engagement_zscore = (recent_engagement_avg - engagement_mean) / engagement_std
        else:
            recent_engagement_zscore = 0
        
        # Verificar umbrales para shadowban
        views_drop_threshold = -self.thresholds[platform]["views_drop_percent"]
        engagement_drop_threshold = -self.thresholds[platform]["engagement_drop_percent"]
        anomaly_zscore_threshold = -self.thresholds[platform]["anomaly_z_score"]
        
        # Determinar si hay shadowban
        is_shadowban = False
        reasons = []
        
        if views_change_percent <= views_drop_threshold:
            is_shadowban = True
            reasons.append(f"Caída de vistas: {views_change_percent:.1f}% (umbral: {views_drop_threshold}%)")
        
        if engagement_change_percent <= engagement_drop_threshold:
            is_shadowban = True
            reasons.append(f"Caída de engagement: {engagement_change_percent:.1f}% (umbral: {engagement_drop_threshold}%)")
        
        if recent_views_zscore <= anomaly_zscore_threshold:
            is_shadowban = True
            reasons.append(f"Anomalía en vistas: Z-score {recent_views_zscore:.2f} (umbral: {anomaly_zscore_threshold})")
        
        if recent_engagement_zscore <= anomaly_zscore_threshold:
            is_shadowban = True
            reasons.append(f"Anomalía en engagement: Z-score {recent_engagement_zscore:.2f} (umbral: {anomaly_zscore_threshold})")
        
        # Si se detecta shadowban, crear detección
        if is_shadowban:
            # Obtener datos adicionales del contenido
            content_info = {}
            if len(metrics) > 0:
                latest_metric = metrics[-1]
                if "title" in latest_metric:
                    content_info["title"] = latest_metric["title"]
                if "hashtags" in latest_metric and isinstance(latest_metric["hashtags"], list):
                    content_info["hashtags"] = latest_metric["hashtags"]
                if "duration" in latest_metric:
                    content_info["duration"] = latest_metric["duration"]
            
            detection = {
                "platform": platform,
                "content_id": content_id,
                "timestamp": datetime.now().isoformat(),
                "reasons": reasons,
                "metrics": {
                    "baseline_views_avg": float(baseline_views_avg),
                    "recent_views_avg": float(recent_views_avg),
                    "views_change_percent": float(views_change_percent),
                    "baseline_engagement_avg": float(baseline_engagement_avg),
                    "recent_engagement_avg": float(recent_engagement_avg),
                    "engagement_change_percent": float(engagement_change_percent),
                    "views_zscore": float(recent_views_zscore),
                    "engagement_zscore": float(recent_engagement_zscore)
                },
                "content_info": content_info
            }
            
            logger.warning(f"Shadowban detectado en {platform}/{content_id}: {reasons}")
            return detection
        
        return None
    
    def get_shadowban_history(self, platform: str = None, days: int = 90) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de shadowbans detectados
        
        Args:
            platform: Filtrar por plataforma (opcional)
            days: Número de días hacia atrás para filtrar
            
        Returns:
            Lista de detecciones de shadowban
        """
        if "detections" not in self.shadowban_history:
            return []
        
        # Filtrar por fecha
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        filtered_detections = []
        for detection in self.shadowban_history["detections"]:
            # Filtrar por plataforma si se especifica
            if platform and detection.get("platform") != platform:
                continue
            
            # Filtrar por fecha
            detection_date = detection.get("timestamp", "")
            if detection_date >= cutoff_str:
                filtered_detections.append(detection)
        
        # Ordenar por fecha (más reciente primero)
        filtered_detections.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return filtered_detections
    
    def analyze_shadowban_patterns(self) -> Dict[str, Any]:
        """
        Analiza patrones en los shadowbans detectados
        
        Returns:
            Análisis de patrones de shadowban
        """
        if "detections" not in self.shadowban_history or not self.shadowban_history["detections"]:
            return {"status": "no_data", "message": "No hay detecciones para analizar"}
        
        # Inicializar resultado
        result = {
            "total_detections": len(self.shadowban_history["detections"]),
            "by_platform": {},
            "common_hashtags": {},
            "common_words": {},
            "time_patterns": {},
            "recommendations": []
        }
        
        # Contar por plataforma
        platform_counts = {}
        for detection in self.shadowban_history["detections"]:
            platform = detection.get("platform", "unknown")
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        result["by_platform"] = platform_counts
        
        # Analizar hashtags comunes
        all_hashtags = []
        for detection in self.shadowban_history["detections"]:
            content_info = detection.get("content_info", {})
            if "hashtags" in content_info and isinstance(content_info["hashtags"], list):
                all_hashtags.extend(content_info["hashtags"])
        
        if all_hashtags:
            hashtag_counter = Counter(all_hashtags)
            result["common_hashtags"] = dict(hashtag_counter.most_common(10))
            
            # Recomendar evitar hashtags problemáticos
            if len(hashtag_counter) > 0:
                top_hashtags = hashtag_counter.most_common(3)
                if top_hashtags[0][1] >= 2:  # Si aparece al menos 2 veces
                    result["recommendations"].append({
                        "type": "hashtag",
                        "message": f"Considera evitar estos hashtags: {', '.join(['#' + h[0] for h in top_hashtags])}"
                    })
        
        # Analizar palabras comunes en títulos
        all_words = []
        for detection in self.shadowban_history["detections"]:
            content_info = detection.get("content_info", {})
            if "title" in content_info and isinstance(content_info["title"], str):
                # Tokenizar título
                words = content_info["title"].lower().split()
                # Filtrar palabras comunes
                filtered_words = [w for w in words if len(w) > 3]
                all_words.extend(filtered_words)
        
        if all_words:
            word_counter = Counter(all_words)
            result["common_words"] = dict(word_counter.most_common(10))
            
            # Recomendar evitar palabras problemáticas
            if len(word_counter) > 0:
                top_words = word_counter.most_common(3)
                if top_words[0][1] >= 2:  # Si aparece al menos 2 veces
                    result["recommendations"].append({
                        "type": "words",
                        "message": f"Considera evitar estas palabras en títulos: {', '.join([w[0] for w in top_words])}"
                    })
        
        # Analizar patrones temporales
        timestamps = []
        for detection in self.shadowban_history["detections"]:
            if "timestamp" in detection:
                try:
                    dt = datetime.fromisoformat(detection["timestamp"])
                    timestamps.append(dt)
                except (ValueError, TypeError):
                    pass
        
        if timestamps:
            # Contar por día de la semana
            weekday_counts = Counter([dt.weekday() for dt in timestamps])
            weekday_names = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
            result["time_patterns"]["weekday"] = {weekday_names[day]: count for day, count in weekday_counts.items()}
            
            # Contar por hora del día
            hour_counts = Counter([dt.hour for dt in timestamps])
            result["time_patterns"]["hour"] = dict(hour_counts)
            
            # Recomendar mejores momentos para publicar
            if len(weekday_counts) > 0 and len(hour_counts) > 0:
                worst_day = max(weekday_counts.items(), key=lambda x: x[1])[0]
                worst_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
                
                result["recommendations"].append({
                    "type": "timing",
                    "message": f"Evita publicar los {weekday_names[worst_day]} a las {worst_hour}:00, cuando se detectan más shadowbans"
                })
        
        # Añadir recomendaciones generales
        if result["total_detections"] > 5:
            result["recommendations"].append({
                "type": "general",
                "message": "Considera revisar tus políticas de contenido y CTAs para reducir el riesgo de shadowban"
            })
        
        return result
    
    def generate_shadowban_report(self, platform: str = None, days: int = 30, output_path: str = None) -> Dict[str, Any]:
        """
        Genera un informe detallado de shadowbans
        
        Args:
            platform: Filtrar por plataforma (opcional)
            days: Número de días hacia atrás para filtrar
            output_path: Ruta para guardar gráficos (opcional)
            
        Returns:
            Informe detallado de shadowbans
        """
        # Obtener historial filtrado
        detections = self.get_shadowban_history(platform, days)
        
        # Inicializar informe
        report = {
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "total_detections": len(detections),
            "platforms": {},
            "patterns": {},
            "affected_content": [],
            "recommendations": []
        }
        
        # Si no hay detecciones, devolver informe vacío
        if not detections:
            report["status"] = "no_detections"
            return report
        
        # Analizar por plataforma
        platform_detections = {}
        for detection in detections:
            platform_name = detection.get("platform", "unknown")
            if platform_name not in platform_detections:
                platform_detections[platform_name] = []
            platform_detections[platform_name].append(detection)
        
        # Procesar cada plataforma
        for platform_name, platform_data in platform_detections.items():
            platform_report = {
                "total": len(platform_data),
                "recent": sum(1 for d in platform_data if datetime.fromisoformat(d.get("timestamp", "")) > datetime.now() - timedelta(days=7)),
                "common_reasons": self._get_common_reasons(platform_data),
                "affected_metrics": self._get_affected_metrics(platform_data)
            }
            
            report["platforms"][platform_name] = platform_report
        
        # Analizar patrones
        patterns = self.analyze_shadowban_patterns()
        report["patterns"] = patterns.get("time_patterns", {})
        
        # Añadir contenido afectado (los 10 más recientes)
        for detection in detections[:10]:
            content_info = detection.get("content_info", {})
            affected_content = {
                "platform": detection.get("platform", "unknown"),
                "content_id": detection.get("content_id", ""),
                "title": content_info.get("title", ""),
                "timestamp": detection.get("timestamp", ""),
                "reasons": detection.get("reasons", [])
            }
            report["affected_content"].append(affected_content)
        
        # Añadir recomendaciones
        report["recommendations"] = patterns.get("recommendations", [])
        
        # Generar gráficos si se especifica ruta
        if output_path:
            self._generate_report_charts(detections, output_path)
            report["charts_path"] = output_path
        
        return report
    
    def _get_common_reasons(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """Obtiene las razones más comunes de shadowban"""
        all_reasons = []
        for detection in detections:
            if "reasons" in detection and isinstance(detection["reasons"], list):
                all_reasons.extend([r.split(":")[0].strip() for r in detection["reasons"]])
        
        return dict(Counter(all_reasons).most_common(5))
    
    def _get_affected_metrics(self, detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """Obtiene estadísticas sobre métricas afectadas"""
        views_changes = []
        engagement_changes = []
        
        for detection in detections:
            metrics = detection.get("metrics", {})
            if "views_change_percent" in metrics:
                views_changes.append(metrics["views_change_percent"])
            if "engagement_change_percent" in metrics:
                engagement_changes.append(metrics["engagement_change_percent"])
        
        result = {}
        if views_changes:
            result["avg_views_drop"] = sum(views_changes) / len(views_changes)
        if engagement_changes:
            result["avg_engagement_drop"] = sum(engagement_changes) / len(engagement_changes)
        
        return result
    
    def _generate_report_charts(self, detections: List[Dict[str, Any]], output_path: str) -> None:
        """Genera gráficos para el informe"""
        try:
            # Crear directorio si no existe
            os.makedirs(output_path, exist_ok=True)
            
            # Preparar datos
            platforms = [d.get("platform", "unknown") for d in detections]
            platform_counts = Counter(platforms)
            
            # Gráfico de plataformas
            plt.figure(figsize=(10, 6))
            plt.bar(platform_counts.keys(), platform_counts.values())
            plt.title("Shadowbans por Plataforma")
            plt.xlabel("Plataforma")
            plt.ylabel("Número de Detecciones")
            plt.savefig(os.path.join(output_path, "shadowbans_by_platform.png"))
            plt.close()
            
            # Gráfico de tendencia temporal
            timestamps = []
            for detection in detections:
                if "timestamp" in detection:
                    try:
                        dt = datetime.fromisoformat(detection["timestamp"])
                        timestamps.append(dt)
                    except (ValueError, TypeError):
                        pass
            
            if timestamps:
                # Ordenar por fecha
                timestamps.sort()
                
                # Contar por día
                date_counts = Counter([dt.date() for dt in timestamps])
                dates = list(date_counts.keys())
                counts = list(date_counts.values())
                
                plt.figure(figsize=(12, 6))
                plt.plot(dates, counts, marker='o')
                plt.title("Tendencia de Shadowbans")
                plt.xlabel("Fecha")
                plt.ylabel("Número de Detecciones")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "shadowbans_trend.png"))
                plt.close()
            
            # Gráfico de razones comunes
            all_reasons = []
            for detection in detections:
                if "reasons" in detection and isinstance(detection["reasons"], list):
                    all_reasons.extend([r.split(":")[0].strip() for r in detection["reasons"]])
            
            reason_counts = Counter(all_reasons)
            
            if reason_counts:
                reasons = list(reason_counts.keys())
                counts = list(reason_counts.values())
                
                plt.figure(figsize=(10, 6))
                plt.bar(reasons, counts)
                plt.title("Razones Comunes de Shadowban")
                plt.xlabel("Razón")
                plt.ylabel("Frecuencia")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "shadowban_reasons.png"))
                plt.close()
            
            logger.info(f"Gráficos generados en: {output_path}")
        except Exception as e:
            logger.error(f"Error al generar gráficos: {str(e)}")
    
    def reset_detection_history(self) -> bool:
        """
        Reinicia el historial de detecciones (mantiene las métricas)
        
        Returns:
            True si se reinició correctamente, False en caso contrario
        """
        try:
            if "detections" in self.shadowban_history:
                self.shadowban_history["detections"] = []
                self._save_history()
                logger.info("Historial de detecciones reiniciado")
                return True
            return False
        except Exception as e:
            logger.error(f"Error al reiniciar historial: {str(e)}")
            return False
    
    def export_metrics(self, output_path: str = None) -> str:
        """
        Exporta las métricas a un archivo JSON
        
        Args:
            output_path: Ruta de salida (opcional)
            
        Returns:
            Ruta al archivo exportado
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/exports/shadowban_metrics_{timestamp}.json"
        
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Exportar métricas
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.shadowban_history.get("metrics", {}), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Métricas exportadas a: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error al exportar métricas: {str(e)}")
            return None

# Ejemplo de uso
if __name__ == "__main__":
    detector = ShadowbanDetector()
    
    # Registrar métricas de ejemplo
    detector.register_metrics(
        platform="youtube",
        content_id="video123",
        metrics={
            "views": 1000,
            "likes": 100,
            "comments": 50,
            "shares": 20,
            "title": "Mi video de ejemplo",
            "hashtags": ["tutorial", "ejemplo", "python"]
        }
    )
    
    # Detectar shadowbans
    results = detector.detect_shadowban("youtube")
    print(json.dumps(results, indent=2))
    
    # Generar informe
    report = detector.generate_shadowban_report(days=90, output_path="reports/shadowban")
    print(json.dumps(report, indent=2))