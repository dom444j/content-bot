"""
Módulo de Contingencia para Algoritmos

Este módulo implementa estrategias para detectar y responder a cambios
en los algoritmos de las plataformas, asegurando la sostenibilidad del
contenido frente a actualizaciones de las plataformas.

Características principales:
- Detección de cambios algorítmicos
- Estrategias de adaptación rápida
- Monitoreo de métricas clave
- Planes de contingencia por plataforma
- Análisis de tendencias de distribución
- Alertas tempranas de cambios
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/algo_contingency.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("AlgoContingency")

class AlgoContingencyEngine:
    """
    Motor de contingencia para cambios algorítmicos en plataformas.
    """
    
    def __init__(self, config_path: str = "config/algo_contingency_config.json"):
        """
        Inicializa el motor de contingencia.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        
        # Cargar configuración
        self._load_config()
        
        # Historial de métricas
        self.metrics_history = {}
        
        # Detecciones de cambios
        self.detected_changes = {}
        
        # Planes de contingencia activos
        self.active_contingencies = {}
        
        # Alertas recientes
        self.recent_alerts = []
        
        logger.info("Motor de contingencia algorítmica inicializado")
    
    def _load_config(self) -> None:
        """Carga la configuración desde el archivo JSON."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # Configuración por defecto
                self.config = {
                    "detection_thresholds": {
                        "reach_drop": 0.25,        # 25% de caída en alcance
                        "engagement_drop": 0.30,   # 30% de caída en engagement
                        "views_drop": 0.35,        # 35% de caída en visualizaciones
                        "distribution_change": 0.40, # 40% de cambio en distribución
                        "recommendation_drop": 0.30  # 30% de caída en recomendaciones
                    },
                    "detection_window": {
                        "short_term": 7,   # 7 días
                        "medium_term": 14, # 14 días
                        "long_term": 30    # 30 días
                    },
                    "anomaly_detection": {
                        "sensitivity": 0.05,  # Sensibilidad (0.01-0.1)
                        "min_samples": 10,    # Mínimo de muestras necesarias
                        "contamination": 0.05 # Contaminación esperada (outliers)
                    },
                    "alert_levels": {
                        "warning": {
                            "threshold": 0.15,  # 15% de cambio
                            "confidence": 0.70  # 70% de confianza
                        },
                        "alert": {
                            "threshold": 0.25,  # 25% de cambio
                            "confidence": 0.80  # 80% de confianza
                        },
                        "critical": {
                            "threshold": 0.40,  # 40% de cambio
                            "confidence": 0.90  # 90% de confianza
                        }
                    },
                    "platform_specific": {
                        "youtube": {
                            "key_metrics": ["views", "watch_time", "ctr", "avg_view_duration", "impressions"],
                            "response_strategies": ["metadata_optimization", "thumbnail_refresh", "keyword_update"]
                        },
                        "tiktok": {
                            "key_metrics": ["views", "completion_rate", "shares", "comments", "follower_growth"],
                            "response_strategies": ["trend_alignment", "sound_optimization", "hashtag_refresh"]
                        },
                        "instagram": {
                            "key_metrics": ["reach", "saves", "shares", "profile_visits", "follower_growth"],
                            "response_strategies": ["carousel_focus", "story_integration", "hashtag_refresh"]
                        }
                    },
                    "contingency_plans": {
                        "reach_focused": {
                            "actions": [
                                "Aumentar frecuencia de publicación en 30%",
                                "Priorizar formatos con mayor alcance histórico",
                                "Incrementar colaboraciones con creadores afines",
                                "Optimizar títulos y miniaturas para CTR"
                            ],
                            "metrics_to_monitor": ["reach", "impressions", "unique_viewers"]
                        },
                        "engagement_focused": {
                            "actions": [
                                "Incrementar llamados a la acción",
                                "Formular preguntas directas a la audiencia",
                                "Crear contenido que genere debate",
                                "Responder a comentarios en primeras 2 horas"
                            ],
                            "metrics_to_monitor": ["comments", "likes", "shares", "saves"]
                        },
                        "retention_focused": {
                            "actions": [
                                "Acortar introducción en 30%",
                                "Añadir gancho en primeros 3 segundos",
                                "Incrementar cambios de ritmo narrativo",
                                "Optimizar estructura para retención"
                            ],
                            "metrics_to_monitor": ["avg_view_duration", "completion_rate", "retention_graph"]
                        },
                        "distribution_focused": {
                            "actions": [
                                "Diversificar formatos de contenido",
                                "Experimentar con nuevos nichos relacionados",
                                "Crear series temáticas con continuidad",
                                "Optimizar para búsqueda y descubrimiento"
                            ],
                            "metrics_to_monitor": ["impressions_source", "traffic_source", "search_views"]
                        }
                    },
                    "recovery_thresholds": {
                        "min_recovery_time": 14,  # 14 días mínimo
                        "metrics_recovery": 0.80, # 80% de recuperación
                        "confidence_level": 0.85  # 85% de confianza
                    }
                }
                
                # Guardar configuración por defecto
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4)
            
            logger.info(f"Configuración de contingencia cargada")
            
        except Exception as e:
            logger.error(f"Error cargando configuración de contingencia: {str(e)}")
            # Configuración mínima de respaldo
            self.config = {
                "detection_thresholds": {
                    "reach_drop": 0.25,
                    "engagement_drop": 0.30
                },
                "detection_window": {
                    "short_term": 7,
                    "medium_term": 14
                },
                "alert_levels": {
                    "warning": {"threshold": 0.15},
                    "alert": {"threshold": 0.25},
                    "critical": {"threshold": 0.40}
                }
            }
    
    def save_config(self) -> None:
        """Guarda la configuración actual en el archivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuración de contingencia guardada en {self.config_path}")
        except Exception as e:
                        logger.error(f"Error guardando configuración de contingencia: {str(e)}")
    
    def update_metrics_history(self, channel_id: str, platform: str, metrics_data: Dict[str, Any]) -> None:
        """
        Actualiza el historial de métricas para un canal específico.
        
        Args:
            channel_id: Identificador del canal
            platform: Plataforma (youtube, tiktok, instagram, etc.)
            metrics_data: Datos de métricas
                {
                    "date": "2023-06-15",
                    "content_id": "abc123",
                    "metrics": {
                        "views": 1000,
                        "watch_time": 5000,
                        "ctr": 0.05,
                        "avg_view_duration": 120,
                        "impressions": 20000,
                        "likes": 100,
                        "comments": 20,
                        "shares": 10,
                        "saves": 5,
                        "reach": 15000,
                        "follower_growth": 25,
                        "completion_rate": 0.60
                    }
                }
        """
        try:
            # Inicializar historial del canal si no existe
            if channel_id not in self.metrics_history:
                self.metrics_history[channel_id] = {}
            
            # Inicializar historial de plataforma si no existe
            if platform not in self.metrics_history[channel_id]:
                self.metrics_history[channel_id][platform] = []
            
            # Añadir métricas al historial
            self.metrics_history[channel_id][platform].append(metrics_data)
            
            # Ordenar por fecha
            self.metrics_history[channel_id][platform] = sorted(
                self.metrics_history[channel_id][platform],
                key=lambda x: x.get("date", "2000-01-01")
            )
            
            # Limitar historial a los últimos 90 días
            if len(self.metrics_history[channel_id][platform]) > 90:
                self.metrics_history[channel_id][platform] = self.metrics_history[channel_id][platform][-90:]
            
            logger.info(f"Historial de métricas actualizado para canal {channel_id} en {platform}")
            
        except Exception as e:
            logger.error(f"Error actualizando historial de métricas: {str(e)}")
    
    def detect_algorithm_changes(self, channel_id: str, platform: str, 
                                window_size: str = "medium_term") -> Dict[str, Any]:
        """
        Detecta cambios en el algoritmo basado en el historial de métricas.
        
        Args:
            channel_id: Identificador del canal
            platform: Plataforma a analizar
            window_size: Tamaño de ventana de análisis (short_term, medium_term, long_term)
            
        Returns:
            Resultados de detección de cambios
        """
        try:
            # Verificar si hay suficiente historial
            if (channel_id not in self.metrics_history or 
                platform not in self.metrics_history.get(channel_id, {}) or
                len(self.metrics_history[channel_id][platform]) < self.config["anomaly_detection"]["min_samples"]):
                
                logger.warning(f"Historial insuficiente para canal {channel_id} en {platform}")
                return {
                    "change_detected": False,
                    "confidence": 0.0,
                    "alert_level": "none",
                    "affected_metrics": [],
                    "timestamp": datetime.now().isoformat(),
                    "error": "Historial insuficiente"
                }
            
            # Obtener tamaño de ventana en días
            days = self.config["detection_window"].get(window_size, 14)
            
            # Obtener métricas recientes
            recent_metrics = self.metrics_history[channel_id][platform]
            
            # Filtrar por fecha (últimos N días)
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            recent_metrics = [m for m in recent_metrics if m.get("date", "2000-01-01") >= cutoff_date]
            
            if len(recent_metrics) < self.config["anomaly_detection"]["min_samples"]:
                logger.warning(f"Métricas recientes insuficientes para ventana {window_size}")
                return {
                    "change_detected": False,
                    "confidence": 0.0,
                    "alert_level": "none",
                    "affected_metrics": [],
                    "timestamp": datetime.now().isoformat(),
                    "error": f"Métricas insuficientes para ventana {window_size}"
                }
            
            # Obtener métricas clave para la plataforma
            key_metrics = self.config["platform_specific"].get(platform, {}).get("key_metrics", [])
            if not key_metrics:
                key_metrics = ["views", "engagement", "reach"]  # Métricas por defecto
            
            # Extraer series temporales de métricas clave
            metric_series = {}
            for metric in key_metrics:
                values = [m.get("metrics", {}).get(metric, 0) for m in recent_metrics if metric in m.get("metrics", {})]
                if len(values) >= self.config["anomaly_detection"]["min_samples"]:
                    metric_series[metric] = values
            
            # Verificar si hay suficientes series de métricas
            if not metric_series:
                logger.warning(f"No hay suficientes series de métricas para análisis")
                return {
                    "change_detected": False,
                    "confidence": 0.0,
                    "alert_level": "none",
                    "affected_metrics": [],
                    "timestamp": datetime.now().isoformat(),
                    "error": "No hay suficientes series de métricas"
                }
            
            # Detectar anomalías en cada métrica
            anomalies = {}
            for metric, values in metric_series.items():
                # Dividir en dos mitades para comparar
                half_size = len(values) // 2
                if half_size < 3:  # Necesitamos al menos 3 puntos en cada mitad
                    continue
                
                first_half = values[:half_size]
                second_half = values[-half_size:]
                
                # Calcular cambio porcentual
                first_avg = sum(first_half) / len(first_half) if first_half else 0
                second_avg = sum(second_half) / len(second_half) if second_half else 0
                
                if first_avg == 0:
                    continue
                
                percent_change = (second_avg - first_avg) / first_avg
                
                # Realizar prueba estadística (t-test)
                t_stat, p_value = stats.ttest_ind(first_half, second_half, equal_var=False)
                confidence = 1.0 - p_value if not np.isnan(p_value) else 0.0
                
                # Determinar si hay anomalía según umbrales
                threshold_key = f"{metric}_drop" if percent_change < 0 else f"{metric}_increase"
                threshold = self.config["detection_thresholds"].get(
                    threshold_key, 
                    self.config["detection_thresholds"].get(f"{metric}_change", 0.25)
                )
                
                is_anomaly = abs(percent_change) >= threshold and confidence >= 0.70
                
                anomalies[metric] = {
                    "percent_change": percent_change,
                    "confidence": confidence,
                    "is_anomaly": is_anomaly,
                    "p_value": p_value,
                    "first_avg": first_avg,
                    "second_avg": second_avg
                }
            
            # Determinar si hay cambio algorítmico
            affected_metrics = [m for m, data in anomalies.items() if data["is_anomaly"]]
            
            # Calcular nivel de alerta
            alert_level = "none"
            max_change = max([abs(data["percent_change"]) for data in anomalies.values()], default=0)
            max_confidence = max([data["confidence"] for data in anomalies.values()], default=0)
            
            for level in ["warning", "alert", "critical"]:
                level_config = self.config["alert_levels"].get(level, {})
                threshold = level_config.get("threshold", 0.15)
                conf_threshold = level_config.get("confidence", 0.70)
                
                if max_change >= threshold and max_confidence >= conf_threshold:
                    alert_level = level
            
            # Crear resultado de detección
            detection_result = {
                "change_detected": len(affected_metrics) > 0,
                "confidence": max_confidence,
                "alert_level": alert_level,
                "affected_metrics": affected_metrics,
                "metric_details": anomalies,
                "window_size": window_size,
                "days_analyzed": days,
                "timestamp": datetime.now().isoformat()
            }
            
            # Guardar detección si hay cambio
            if detection_result["change_detected"]:
                if channel_id not in self.detected_changes:
                    self.detected_changes[channel_id] = {}
                
                if platform not in self.detected_changes[channel_id]:
                    self.detected_changes[channel_id][platform] = []
                
                self.detected_changes[channel_id][platform].append(detection_result)
                
                # Limitar a las últimas 10 detecciones
                if len(self.detected_changes[channel_id][platform]) > 10:
                    self.detected_changes[channel_id][platform] = self.detected_changes[channel_id][platform][-10:]
                
                # Generar alerta
                self._generate_alert(channel_id, platform, detection_result)
            
            logger.info(f"Análisis de cambio algorítmico para {channel_id} en {platform}: {alert_level}")
            return detection_result
            
        except Exception as e:
            logger.error(f"Error detectando cambios algorítmicos: {str(e)}")
            return {
                "change_detected": False,
                "confidence": 0.0,
                "alert_level": "none",
                "affected_metrics": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _generate_alert(self, channel_id: str, platform: str, detection_result: Dict[str, Any]) -> None:
        """
        Genera una alerta basada en la detección de cambios.
        
        Args:
            channel_id: Identificador del canal
            platform: Plataforma afectada
            detection_result: Resultado de la detección
        """
        try:
            alert = {
                "channel_id": channel_id,
                "platform": platform,
                "alert_level": detection_result["alert_level"],
                "affected_metrics": detection_result["affected_metrics"],
                "confidence": detection_result["confidence"],
                "timestamp": datetime.now().isoformat(),
                "message": f"Posible cambio algorítmico detectado en {platform} con nivel {detection_result['alert_level']}",
                "recommended_actions": self._get_recommended_actions(platform, detection_result)
            }
            
            # Añadir a alertas recientes
            self.recent_alerts.append(alert)
            
            # Limitar a las últimas 20 alertas
            if len(self.recent_alerts) > 20:
                self.recent_alerts = self.recent_alerts[-20:]
            
            logger.warning(f"ALERTA: {alert['message']} para canal {channel_id}")
            
        except Exception as e:
            logger.error(f"Error generando alerta: {str(e)}")
    
    def _get_recommended_actions(self, platform: str, detection_result: Dict[str, Any]) -> List[str]:
        """
        Obtiene acciones recomendadas según la plataforma y métricas afectadas.
        
        Args:
            platform: Plataforma afectada
            detection_result: Resultado de la detección
            
        Returns:
            Lista de acciones recomendadas
        """
        try:
            affected_metrics = detection_result["affected_metrics"]
            
            # Determinar plan de contingencia principal
            primary_plan = "distribution_focused"  # Plan por defecto
            
            if "reach" in affected_metrics or "impressions" in affected_metrics:
                primary_plan = "reach_focused"
            elif "likes" in affected_metrics or "comments" in affected_metrics or "shares" in affected_metrics:
                primary_plan = "engagement_focused"
            elif "avg_view_duration" in affected_metrics or "completion_rate" in affected_metrics:
                primary_plan = "retention_focused"
            
            # Obtener acciones del plan
            actions = self.config["contingency_plans"].get(primary_plan, {}).get("actions", [])
            
            # Añadir estrategias específicas de plataforma
            platform_strategies = self.config["platform_specific"].get(platform, {}).get("response_strategies", [])
            
            platform_actions = []
            for strategy in platform_strategies:
                if strategy == "metadata_optimization":
                    platform_actions.append("Optimizar metadatos (títulos, descripciones, tags)")
                elif strategy == "thumbnail_refresh":
                    platform_actions.append("Actualizar miniaturas con mayor contraste y claridad")
                elif strategy == "keyword_update":
                    platform_actions.append("Actualizar palabras clave según tendencias actuales")
                elif strategy == "trend_alignment":
                    platform_actions.append("Alinear contenido con tendencias actuales de la plataforma")
                elif strategy == "sound_optimization":
                    platform_actions.append("Utilizar sonidos y música tendencia")
                elif strategy == "hashtag_refresh":
                    platform_actions.append("Actualizar hashtags según análisis de tendencias")
                elif strategy == "carousel_focus":
                    platform_actions.append("Priorizar formato carrusel para mayor engagement")
                elif strategy == "story_integration":
                    platform_actions.append("Integrar stories para promocionar contenido principal")
            
            # Combinar acciones
            all_actions = actions + platform_actions
            
            # Limitar a 5 acciones principales
            return all_actions[:5]
            
        except Exception as e:
            logger.error(f"Error obteniendo acciones recomendadas: {str(e)}")
            return ["Revisar métricas afectadas", "Ajustar estrategia de contenido"]
    
    def activate_contingency_plan(self, channel_id: str, platform: str, 
                                 plan_type: str = None) -> Dict[str, Any]:
        """
        Activa un plan de contingencia específico.
        
        Args:
            channel_id: Identificador del canal
            platform: Plataforma afectada
            plan_type: Tipo de plan (reach_focused, engagement_focused, etc.)
            
        Returns:
            Plan de contingencia activado
        """
        try:
            # Detectar cambios si no se especifica plan
            if plan_type is None:
                detection = self.detect_algorithm_changes(channel_id, platform)
                
                if detection["change_detected"]:
                    affected_metrics = detection["affected_metrics"]
                    
                    # Determinar plan según métricas afectadas
                    if "reach" in affected_metrics or "impressions" in affected_metrics:
                        plan_type = "reach_focused"
                    elif "likes" in affected_metrics or "comments" in affected_metrics:
                        plan_type = "engagement_focused"
                    elif "avg_view_duration" in affected_metrics or "completion_rate" in affected_metrics:
                        plan_type = "retention_focused"
                    else:
                        plan_type = "distribution_focused"
                else:
                    # No se detectaron cambios
                    return {
                        "activated": False,
                        "plan_type": None,
                        "reason": "No se detectaron cambios algorítmicos",
                        "timestamp": datetime.now().isoformat()
                    }
            
            # Verificar si el plan existe
            if plan_type not in self.config["contingency_plans"]:
                logger.warning(f"Plan de contingencia no encontrado: {plan_type}")
                return {
                    "activated": False,
                    "plan_type": plan_type,
                    "reason": f"Plan de contingencia no encontrado: {plan_type}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Obtener plan
            plan = self.config["contingency_plans"][plan_type]
            
            # Crear plan activado
            activated_plan = {
                "channel_id": channel_id,
                "platform": platform,
                "plan_type": plan_type,
                "actions": plan["actions"],
                "metrics_to_monitor": plan["metrics_to_monitor"],
                "platform_specific_actions": self._get_recommended_actions(platform, {"affected_metrics": []}),
                "activation_date": datetime.now().isoformat(),
                "status": "active",
                "results": {},
                "recovery_threshold": self.config["recovery_thresholds"]["metrics_recovery"]
            }
            
            # Guardar plan activo
            if channel_id not in self.active_contingencies:
                self.active_contingencies[channel_id] = {}
            
            self.active_contingencies[channel_id][platform] = activated_plan
            
            logger.info(f"Plan de contingencia {plan_type} activado para {channel_id} en {platform}")
            return activated_plan
            
        except Exception as e:
            logger.error(f"Error activando plan de contingencia: {str(e)}")
            return {
                "activated": False,
                "plan_type": plan_type,
                "reason": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def monitor_contingency_plan(self, channel_id: str, platform: str) -> Dict[str, Any]:
        """
        Monitorea el progreso de un plan de contingencia activo.
        
        Args:
            channel_id: Identificador del canal
            platform: Plataforma
            
        Returns:
            Estado actual del plan de contingencia
        """
        try:
            # Verificar si hay plan activo
            if (channel_id not in self.active_contingencies or 
                platform not in self.active_contingencies.get(channel_id, {})):
                
                logger.warning(f"No hay plan de contingencia activo para {channel_id} en {platform}")
                return {
                    "status": "no_plan",
                    "message": "No hay plan de contingencia activo",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Obtener plan activo
            plan = self.active_contingencies[channel_id][platform]
            
            # Verificar si hay suficiente historial para monitoreo
            if (channel_id not in self.metrics_history or 
                platform not in self.metrics_history.get(channel_id, {})):
                
                logger.warning(f"No hay historial de métricas para monitoreo de {channel_id} en {platform}")
                return {
                    "status": "insufficient_data",
                    "plan_type": plan["plan_type"],
                    "message": "No hay suficientes datos para monitoreo",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Obtener fecha de activación
            activation_date = datetime.fromisoformat(plan["activation_date"])
            days_active = (datetime.now() - activation_date).days
            
            # Verificar tiempo mínimo de recuperación
            min_recovery_time = self.config["recovery_thresholds"]["min_recovery_time"]
            if days_active < min_recovery_time:
                return {
                    "status": "monitoring",
                    "plan_type": plan["plan_type"],
                    "days_active": days_active,
                    "min_recovery_time": min_recovery_time,
                    "message": f"Plan activo por {days_active} días, monitoreo en progreso",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Obtener métricas a monitorear
            metrics_to_monitor = plan["metrics_to_monitor"]
            
            # Obtener métricas antes y después de la activación
            metrics_history = self.metrics_history[channel_id][platform]
            
            # Filtrar por fecha
            pre_activation = [
                m for m in metrics_history 
                if datetime.fromisoformat(m.get("date", "2000-01-01")) < activation_date
            ]
            
            post_activation = [
                m for m in metrics_history 
                if datetime.fromisoformat(m.get("date", "2000-01-01")) >= activation_date
            ]
            
            # Verificar datos suficientes
            if len(pre_activation) < 5 or len(post_activation) < 5:
                return {
                    "status": "insufficient_data",
                    "plan_type": plan["plan_type"],
                    "days_active": days_active,
                    "message": "Datos insuficientes para evaluar recuperación",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Calcular promedios de métricas antes y después
            recovery_metrics = {}
            for metric in metrics_to_monitor:
                pre_values = [m.get("metrics", {}).get(metric, 0) for m in pre_activation if metric in m.get("metrics", {})]
                post_values = [m.get("metrics", {}).get(metric, 0) for m in post_activation if metric in m.get("metrics", {})]
                
                if not pre_values or not post_values:
                    continue
                
                pre_avg = sum(pre_values) / len(pre_values)
                post_avg = sum(post_values) / len(post_values)
                
                if pre_avg == 0:
                    recovery_ratio = 1.0
                else:
                    recovery_ratio = post_avg / pre_avg
                
                recovery_metrics[metric] = {
                    "pre_activation_avg": pre_avg,
                    "post_activation_avg": post_avg,
                    "recovery_ratio": recovery_ratio,
                    "recovered": recovery_ratio >= plan["recovery_threshold"]
                }
            
            # Determinar si hay recuperación general
            recovered_metrics = [m for m, data in recovery_metrics.items() if data["recovered"]]
            recovery_percentage = len(recovered_metrics) / len(recovery_metrics) if recovery_metrics else 0
            
            # Actualizar resultados del plan
            plan["results"] = {
                "days_active": days_active,
                "recovery_metrics": recovery_metrics,
                "recovered_metrics": recovered_metrics,
                "recovery_percentage": recovery_percentage,
                "last_updated": datetime.now().isoformat()
            }
            
            # Determinar estado del plan
            confidence_level = self.config["recovery_thresholds"]["confidence_level"]
            if recovery_percentage >= confidence_level:
                plan["status"] = "recovered"
                message = f"Plan completado con éxito, {len(recovered_metrics)}/{len(recovery_metrics)} métricas recuperadas"
            else:
                plan["status"] = "active"
                message = f"Plan en progreso, {len(recovered_metrics)}/{len(recovery_metrics)} métricas recuperadas"
            
            # Actualizar plan activo
            self.active_contingencies[channel_id][platform] = plan
            
            # Crear informe de monitoreo
            monitoring_report = {
                "status": plan["status"],
                "plan_type": plan["plan_type"],
                "days_active": days_active,
                "recovery_percentage": recovery_percentage,
                "recovered_metrics": recovered_metrics,
                "metrics_details": recovery_metrics,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Monitoreo de plan para {channel_id} en {platform}: {plan['status']}")
            return monitoring_report
            
        except Exception as e:
            logger.error(f"Error monitoreando plan de contingencia: {str(e)}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_platform_trends(self, platform: str, days: int = 30) -> Dict[str, Any]:
        """
        Analiza tendencias generales en una plataforma entre múltiples canales.
        
        Args:
            platform: Plataforma a analizar
            days: Número de días a analizar
            
        Returns:
            Análisis de tendencias de plataforma
        """
        try:
            # Verificar si hay suficientes canales con datos
            channels_with_data = [
                channel_id for channel_id, platforms in self.metrics_history.items()
                if platform in platforms and len(platforms[platform]) > 0
            ]
            
            if len(channels_with_data) < 2:
                logger.warning(f"Datos insuficientes para análisis de tendencias en {platform}")
                return {
                    "platform": platform,
                    "channels_analyzed": len(channels_with_data),
                    "error": "Datos insuficientes para análisis de tendencias",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Fecha de corte
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Recopilar métricas de todos los canales
            all_metrics = {}
            for channel_id in channels_with_data:
                channel_metrics = self.metrics_history[channel_id][platform]
                
                # Filtrar por fecha
                recent_metrics = [
                    m for m in channel_metrics 
                    if m.get("date", "2000-01-01") >= cutoff_date
                ]
                
                # Extraer métricas clave
                for metric_data in recent_metrics:
                    date = metric_data.get("date", "2000-01-01")
                    metrics = metric_data.get("metrics", {})
                    
                    for metric_name, value in metrics.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = {}
                        
                        if date not in all_metrics[metric_name]:
                            all_metrics[metric_name][date] = []
                        
                        all_metrics[metric_name][date].append(value)
            
            # Calcular promedios diarios por métrica
            daily_averages = {}
            for metric_name, dates in all_metrics.items():
                daily_averages[metric_name] = {}
                for date, values in dates.items():
                    daily_averages[metric_name][date] = sum(values) / len(values)
            
            # Calcular tendencias por métrica
            metric_trends = {}
            for metric_name, date_values in daily_averages.items():
                # Ordenar por fecha
                sorted_dates = sorted(date_values.keys())
                
                if len(sorted_dates) < 5:
                    continue
                
                # Dividir en dos mitades
                half_size = len(sorted_dates) // 2
                first_half_dates = sorted_dates[:half_size]
                second_half_dates = sorted_dates[-half_size:]
                
                first_half_values = [date_values[d] for d in first_half_dates]
                second_half_values = [date_values[d] for d in second_half_dates]
                
                # Calcular promedios
                first_avg = sum(first_half_values) / len(first_half_values)
                second_avg = sum(second_half_values) / len(second_half_values)
                
                # Calcular cambio porcentual
                if first_avg == 0:
                    percent_change = 0
                else:
                    percent_change = (second_avg - first_avg) / first_avg
                
                # Realizar prueba estadística
                t_stat, p_value = stats.ttest_ind(first_half_values, second_half_values, equal_var=False)
                confidence = 1.0 - p_value if not np.isnan(p_value) else 0.0
                
                # Guardar tendencia
                metric_trends[metric_name] = {
                    "percent_change": percent_change,
                    "confidence": confidence,
                    "p_value": p_value,
                    "first_half_avg": first_avg,
                    "second_half_avg": second_avg,
                    "is_significant": confidence >= 0.80
                }
            
            # Identificar métricas con cambios significativos
            significant_changes = {
                m: data for m, data in metric_trends.items() 
                if data["is_significant"] and abs(data["percent_change"]) >= 0.10
            }
            
            # Crear informe de tendencias
            trend_report = {
                "platform": platform,
                "days_analyzed": days,
                "channels_analyzed": len(channels_with_data),
                "metric_trends": metric_trends,
                "significant_changes": significant_changes,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Análisis de tendencias completado para {platform}, {len(significant_changes)} cambios significativos")
            return trend_report
            
        except Exception as e:
            logger.error(f"Error analizando tendencias de plataforma: {str(e)}")
            return {
                "platform": platform,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_contingency_report(self, channel_id: str = None) -> Dict[str, Any]:
        """
        Genera un informe completo de contingencia.
        
        Args:
            channel_id: Identificador del canal (opcional, si es None se incluyen todos)
            
        Returns:
            Informe de contingencia
        """
        try:
            # Determinar canales a incluir
            channels = [channel_id] if channel_id else list(self.metrics_history.keys())
            
            # Crear informe base
            report = {
                "timestamp": datetime.now().isoformat(),
                "channels_analyzed": len(channels),
                "active_contingencies": 0,
                "recent_detections": [],
                "recent_alerts": self.recent_alerts[-5:] if self.recent_alerts else [],
                "channel_reports": {}
            }
            
            # Analizar cada canal
            for ch_id in channels:
                if ch_id not in self.metrics_history:
                    continue
                
                                # Plataformas del canal
                platforms = list(self.metrics_history[ch_id].keys())
                
                # Inicializar reporte del canal
                channel_report = {
                    "channel_id": ch_id,
                    "platforms": platforms,
                    "active_plans": {},
                    "recent_detections": {},
                    "metrics_summary": {}
                }
                
                # Analizar cada plataforma
                for platform in platforms:
                    # Verificar planes activos
                    active_plan = None
                    if (ch_id in self.active_contingencies and 
                        platform in self.active_contingencies.get(ch_id, {})):
                        active_plan = self.active_contingencies[ch_id][platform]
                        report["active_contingencies"] += 1
                        channel_report["active_plans"][platform] = active_plan
                    
                    # Obtener detecciones recientes
                    recent_detections = []
                    if (ch_id in self.detected_changes and 
                        platform in self.detected_changes.get(ch_id, {})):
                        recent_detections = self.detected_changes[ch_id][platform][-3:]  # Últimas 3
                        channel_report["recent_detections"][platform] = recent_detections
                        
                        # Añadir a detecciones globales si son críticas
                        critical_detections = [
                            d for d in recent_detections 
                            if d.get("alert_level") == "critical"
                        ]
                        if critical_detections:
                            for detection in critical_detections:
                                detection_summary = {
                                    "channel_id": ch_id,
                                    "platform": platform,
                                    "alert_level": detection.get("alert_level"),
                                    "affected_metrics": detection.get("affected_metrics"),
                                    "timestamp": detection.get("timestamp")
                                }
                                report["recent_detections"].append(detection_summary)
                    
                    # Calcular resumen de métricas
                    metrics_summary = self._calculate_metrics_summary(ch_id, platform)
                    channel_report["metrics_summary"][platform] = metrics_summary
                
                # Añadir reporte del canal al informe general
                report["channel_reports"][ch_id] = channel_report
            
            # Ordenar detecciones recientes por fecha
            report["recent_detections"] = sorted(
                report["recent_detections"],
                key=lambda x: x.get("timestamp", "2000-01-01T00:00:00"),
                reverse=True
            )
            
            # Limitar a las 10 detecciones más recientes
            report["recent_detections"] = report["recent_detections"][:10]
            
            logger.info(f"Informe de contingencia generado para {len(channels)} canales")
            return report
            
        except Exception as e:
            logger.error(f"Error generando informe de contingencia: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_metrics_summary(self, channel_id: str, platform: str) -> Dict[str, Any]:
        """
        Calcula un resumen de métricas para un canal y plataforma.
        
        Args:
            channel_id: Identificador del canal
            platform: Plataforma
            
        Returns:
            Resumen de métricas
        """
        try:
            # Verificar si hay métricas
            if (channel_id not in self.metrics_history or
                platform not in self.metrics_history.get(channel_id, {})):
                return {
                    "error": "No hay métricas disponibles"
                }
            
            # Obtener métricas
            metrics_history = self.metrics_history[channel_id][platform]
            
            if not metrics_history:
                return {
                    "error": "Historial de métricas vacío"
                }
            
            # Obtener fechas para diferentes períodos
            now = datetime.now()
            last_7_days = (now - timedelta(days=7)).strftime("%Y-%m-%d")
            last_30_days = (now - timedelta(days=30)).strftime("%Y-%m-%d")
            
            # Filtrar por períodos
            metrics_7d = [m for m in metrics_history if m.get("date", "2000-01-01") >= last_7_days]
            metrics_30d = [m for m in metrics_history if m.get("date", "2000-01-01") >= last_30_days]
            
            # Obtener métricas clave para la plataforma
            key_metrics = self.config["platform_specific"].get(platform, {}).get("key_metrics", [])
            if not key_metrics:
                key_metrics = ["views", "engagement", "reach"]  # Métricas por defecto
            
            # Calcular promedios por período
            averages_7d = {}
            averages_30d = {}
            
            for metric in key_metrics:
                # Últimos 7 días
                values_7d = [m.get("metrics", {}).get(metric, 0) for m in metrics_7d if metric in m.get("metrics", {})]
                if values_7d:
                    averages_7d[metric] = sum(values_7d) / len(values_7d)
                
                # Últimos 30 días
                values_30d = [m.get("metrics", {}).get(metric, 0) for m in metrics_30d if metric in m.get("metrics", {})]
                if values_30d:
                    averages_30d[metric] = sum(values_30d) / len(values_30d)
            
            # Calcular tendencias (comparación 7d vs 30d)
            trends = {}
            for metric in key_metrics:
                if metric in averages_7d and metric in averages_30d and averages_30d[metric] > 0:
                    percent_change = (averages_7d[metric] - averages_30d[metric]) / averages_30d[metric]
                    trends[metric] = {
                        "percent_change": percent_change,
                        "direction": "up" if percent_change > 0 else "down",
                        "significant": abs(percent_change) >= 0.10
                    }
            
            # Crear resumen
            summary = {
                "total_content": len(metrics_history),
                "recent_content_7d": len(metrics_7d),
                "recent_content_30d": len(metrics_30d),
                "averages_7d": averages_7d,
                "averages_30d": averages_30d,
                "trends": trends,
                "last_updated": datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculando resumen de métricas: {str(e)}")
            return {
                "error": str(e)
            }
    
    def visualize_metrics_trend(self, channel_id: str, platform: str, 
                               metric: str, days: int = 30) -> str:
        """
        Genera una visualización de tendencia para una métrica específica.
        
        Args:
            channel_id: Identificador del canal
            platform: Plataforma
            metric: Métrica a visualizar
            days: Número de días a visualizar
            
        Returns:
            Ruta al archivo de imagen generado
        """
        try:
            # Verificar si hay métricas
            if (channel_id not in self.metrics_history or
                platform not in self.metrics_history.get(channel_id, {})):
                logger.warning(f"No hay métricas disponibles para {channel_id} en {platform}")
                return ""
            
            # Obtener métricas
            metrics_history = self.metrics_history[channel_id][platform]
            
            # Fecha de corte
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Filtrar por fecha y métrica
            filtered_metrics = [
                m for m in metrics_history 
                if m.get("date", "2000-01-01") >= cutoff_date and metric in m.get("metrics", {})
            ]
            
            if not filtered_metrics:
                logger.warning(f"No hay datos suficientes para visualizar {metric}")
                return ""
            
            # Extraer fechas y valores
            dates = [m.get("date") for m in filtered_metrics]
            values = [m.get("metrics", {}).get(metric, 0) for m in filtered_metrics]
            
            # Convertir fechas a objetos datetime para mejor visualización
            x_dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
            
            # Crear figura
            plt.figure(figsize=(10, 6))
            plt.plot(x_dates, values, marker='o', linestyle='-', color='#3366cc')
            
            # Añadir línea de tendencia
            z = np.polyfit(range(len(x_dates)), values, 1)
            p = np.poly1d(z)
            plt.plot(x_dates, p(range(len(x_dates))), "r--", alpha=0.8)
            
            # Añadir etiquetas y título
            plt.title(f'Tendencia de {metric} en {platform} - Canal {channel_id}')
            plt.xlabel('Fecha')
            plt.ylabel(metric.capitalize())
            plt.grid(True, alpha=0.3)
            
            # Formatear eje x para mostrar fechas
            plt.gcf().autofmt_xdate()
            
            # Crear directorio para gráficos si no existe
            output_dir = "reports/metrics_visualizations"
            os.makedirs(output_dir, exist_ok=True)
            
            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_dir}/{channel_id}_{platform}_{metric}_{timestamp}.png"
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Visualización generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando visualización: {str(e)}")
            return ""
    
    def export_metrics_data(self, channel_id: str = None, platform: str = None, 
                           format: str = "csv") -> str:
        """
        Exporta datos de métricas a un archivo.
        
        Args:
            channel_id: Identificador del canal (opcional)
            platform: Plataforma (opcional)
            format: Formato de exportación (csv o json)
            
        Returns:
            Ruta al archivo exportado
        """
        try:
            # Determinar qué datos exportar
            data_to_export = {}
            
            if channel_id and platform:
                # Exportar datos específicos de canal y plataforma
                if (channel_id in self.metrics_history and 
                    platform in self.metrics_history.get(channel_id, {})):
                    data_to_export = {channel_id: {platform: self.metrics_history[channel_id][platform]}}
                else:
                    logger.warning(f"No hay datos para {channel_id} en {platform}")
                    return ""
            elif channel_id:
                # Exportar todas las plataformas de un canal
                if channel_id in self.metrics_history:
                    data_to_export = {channel_id: self.metrics_history[channel_id]}
                else:
                    logger.warning(f"No hay datos para el canal {channel_id}")
                    return ""
            elif platform:
                # Exportar todos los canales de una plataforma
                platform_data = {}
                for ch_id, platforms in self.metrics_history.items():
                    if platform in platforms:
                        platform_data[ch_id] = {platform: platforms[platform]}
                
                if platform_data:
                    data_to_export = platform_data
                else:
                    logger.warning(f"No hay datos para la plataforma {platform}")
                    return ""
            else:
                # Exportar todos los datos
                data_to_export = self.metrics_history
            
            # Crear directorio para exportaciones si no existe
            output_dir = "exports/metrics_data"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ch_part = f"{channel_id}_" if channel_id else ""
            platform_part = f"{platform}_" if platform else ""
            filename = f"{ch_part}{platform_part}metrics_{timestamp}"
            
            # Exportar según formato
            if format.lower() == "json":
                output_path = f"{output_dir}/{filename}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_export, f, indent=4)
            else:  # CSV por defecto
                output_path = f"{output_dir}/{filename}.csv"
                
                # Convertir datos jerárquicos a formato plano para CSV
                rows = []
                for ch_id, platforms in data_to_export.items():
                    for plat, metrics_list in platforms.items():
                        for metric_data in metrics_list:
                            row = {
                                "channel_id": ch_id,
                                "platform": plat,
                                "date": metric_data.get("date", ""),
                                "content_id": metric_data.get("content_id", "")
                            }
                            
                            # Añadir métricas como columnas
                            for metric_name, value in metric_data.get("metrics", {}).items():
                                row[metric_name] = value
                            
                            rows.append(row)
                
                # Crear DataFrame y exportar a CSV
                if rows:
                    df = pd.DataFrame(rows)
                    df.to_csv(output_path, index=False)
                else:
                    logger.warning("No hay datos para exportar")
                    return ""
            
            logger.info(f"Datos exportados a {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exportando datos: {str(e)}")
            return ""
    
    def reset_contingency_plans(self, channel_id: str = None, platform: str = None) -> Dict[str, Any]:
        """
        Reinicia los planes de contingencia activos.
        
        Args:
            channel_id: Identificador del canal (opcional)
            platform: Plataforma (opcional)
            
        Returns:
            Resultado del reinicio
        """
        try:
            reset_count = 0
            
            if channel_id and platform:
                # Reiniciar plan específico
                if (channel_id in self.active_contingencies and 
                    platform in self.active_contingencies.get(channel_id, {})):
                    del self.active_contingencies[channel_id][platform]
                    reset_count = 1
                    
                    # Eliminar diccionario de canal si está vacío
                    if not self.active_contingencies[channel_id]:
                        del self.active_contingencies[channel_id]
            
            elif channel_id:
                # Reiniciar todos los planes de un canal
                if channel_id in self.active_contingencies:
                    reset_count = len(self.active_contingencies[channel_id])
                    del self.active_contingencies[channel_id]
            
            elif platform:
                # Reiniciar todos los planes de una plataforma
                channels_to_check = list(self.active_contingencies.keys())
                for ch_id in channels_to_check:
                    if platform in self.active_contingencies.get(ch_id, {}):
                        del self.active_contingencies[ch_id][platform]
                        reset_count += 1
                        
                        # Eliminar diccionario de canal si está vacío
                        if not self.active_contingencies[ch_id]:
                            del self.active_contingencies[ch_id]
            
            else:
                # Reiniciar todos los planes
                total_plans = 0
                for ch_plans in self.active_contingencies.values():
                    total_plans += len(ch_plans)
                
                reset_count = total_plans
                self.active_contingencies = {}
            
            logger.info(f"Reiniciados {reset_count} planes de contingencia")
            return {
                "reset_count": reset_count,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error reiniciando planes de contingencia: {str(e)}")
            return {
                "reset_count": 0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualiza la configuración del motor de contingencia.
        
        Args:
            config_updates: Diccionario con actualizaciones de configuración
            
        Returns:
            Resultado de la actualización
        """
        try:
            # Función recursiva para actualizar diccionario anidado
            def update_nested_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                        d[k] = update_nested_dict(d[k], v)
                    else:
                        d[k] = v
                return d
            
            # Actualizar configuración
            self.config = update_nested_dict(self.config, config_updates)
            
            # Guardar configuración actualizada
            self.save_config()
            
            logger.info("Configuración de contingencia actualizada")
            return {
                "success": True,
                "message": "Configuración actualizada correctamente",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error actualizando configuración: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }