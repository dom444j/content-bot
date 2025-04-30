"""
ShadowbanManager - Gestor de detección y recuperación de shadowbans

Este módulo se encarga de:
1. Detectar shadowbans en diferentes plataformas
2. Implementar estrategias de recuperación
3. Monitorear el estado de recuperación
4. Proporcionar análisis de riesgo de shadowban
5. Generar recomendaciones para evitar shadowbans
"""

import logging
import time
import datetime
import random
import json
import re
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum, auto
import requests

# Configuración del logger
logger = logging.getLogger("orchestrator.shadowban")

class ShadowbanStatus(Enum):
    """Enumeración de estados de shadowban"""
    NORMAL = auto()             # Funcionamiento normal
    SUSPECTED = auto()          # Posible shadowban, requiere verificación
    CONFIRMED = auto()          # Shadowban confirmado
    RECOVERING = auto()         # En proceso de recuperación
    RECOVERED = auto()          # Recuperado de un shadowban previo


class ShadowbanType(Enum):
    """Tipos de shadowban que pueden ocurrir"""
    HASHTAG_BAN = auto()        # Shadowban en hashtags específicos
    EXPLORE_BAN = auto()        # No aparece en explorar/descubrir
    SEARCH_BAN = auto()         # No aparece en búsquedas
    REACH_LIMITATION = auto()   # Limitación de alcance general
    COMMENT_BAN = auto()        # Comentarios no visibles para otros
    FULL_BAN = auto()           # Shadowban completo (todas las anteriores)
    TEMPORARY_BAN = auto()      # Shadowban temporal (24-72h)
    ALGORITHM_PENALTY = auto()  # Penalización algorítmica


class RecoveryPhase(Enum):
    """Fases de recuperación de shadowban"""
    ANALYSIS = auto()           # Análisis de la causa
    PAUSE = auto()              # Pausa en publicaciones
    CLEANUP = auto()            # Limpieza de contenido problemático
    GRADUAL_RETURN = auto()     # Retorno gradual a la actividad
    MONITORING = auto()         # Monitoreo post-recuperación
    COMPLETED = auto()          # Recuperación completada


class ShadowbanManager:
    """
    Gestor para la detección y recuperación de shadowbans en diferentes plataformas.
    
    Este gestor implementa:
    - Algoritmos de detección basados en métricas de engagement
    - Estrategias de recuperación personalizadas por plataforma
    - Monitoreo continuo del estado de shadowban
    - Análisis predictivo de riesgo de shadowban
    - Recomendaciones para evitar shadowbans
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el gestor de shadowbans.
        
        Args:
            config: Configuración del gestor
        """
        self.config = config or {}
        
        # Umbrales de detección por plataforma
        self.detection_thresholds = self.config.get("detection_thresholds", {
            "youtube": {
                "engagement_drop_percent": 50,  # Caída de engagement >50%
                "view_drop_percent": 60,        # Caída de vistas >60%
                "comment_visibility": 70,        # <70% de comentarios visibles
                "min_samples": 5                 # Mínimo de muestras para comparar
            },
            "tiktok": {
                "engagement_drop_percent": 70,   # Caída de engagement >70%
                "fyp_visibility": 50,            # <50% de apariciones en FYP
                "hashtag_visibility": 60,        # <60% de visibilidad en hashtags
                "min_samples": 3                 # Mínimo de muestras para comparar
            },
            "instagram": {
                "engagement_drop_percent": 60,   # Caída de engagement >60%
                "explore_visibility": 40,        # <40% de apariciones en explorar
                "hashtag_visibility": 50,        # <50% de visibilidad en hashtags
                "min_samples": 4                 # Mínimo de muestras para comparar
            }
        })
        
        # Estrategias de recuperación por plataforma
        self.recovery_strategies = self.config.get("recovery_strategies", {
            "youtube": [
                {"phase": RecoveryPhase.ANALYSIS.name, "duration_days": 1, "actions": ["audit_content", "check_community_guidelines"]},
                {"phase": RecoveryPhase.PAUSE.name, "duration_days": 3, "actions": ["pause_publishing"]},
                {"phase": RecoveryPhase.CLEANUP.name, "duration_days": 2, "actions": ["remove_flagged_content", "update_metadata"]},
                {"phase": RecoveryPhase.GRADUAL_RETURN.name, "duration_days": 7, "actions": ["publish_safe_content", "limited_frequency"]},
                {"phase": RecoveryPhase.MONITORING.name, "duration_days": 14, "actions": ["monitor_metrics", "engagement_analysis"]}
            ],
            "tiktok": [
                {"phase": RecoveryPhase.ANALYSIS.name, "duration_days": 1, "actions": ["audit_content", "check_community_guidelines"]},
                {"phase": RecoveryPhase.PAUSE.name, "duration_days": 2, "actions": ["pause_publishing", "engage_with_others"]},
                {"phase": RecoveryPhase.CLEANUP.name, "duration_days": 1, "actions": ["remove_flagged_content", "update_profile"]},
                {"phase": RecoveryPhase.GRADUAL_RETURN.name, "duration_days": 5, "actions": ["publish_safe_content", "limited_frequency"]},
                {"phase": RecoveryPhase.MONITORING.name, "duration_days": 10, "actions": ["monitor_metrics", "engagement_analysis"]}
            ],
            "instagram": [
                {"phase": RecoveryPhase.ANALYSIS.name, "duration_days": 1, "actions": ["audit_content", "check_community_guidelines"]},
                {"phase": RecoveryPhase.PAUSE.name, "duration_days": 2, "actions": ["pause_publishing", "engage_with_others"]},
                {"phase": RecoveryPhase.CLEANUP.name, "duration_days": 1, "actions": ["remove_flagged_content", "update_profile"]},
                {"phase": RecoveryPhase.GRADUAL_RETURN.name, "duration_days": 5, "actions": ["publish_safe_content", "limited_frequency"]},
                {"phase": RecoveryPhase.MONITORING.name, "duration_days": 10, "actions": ["monitor_metrics", "engagement_analysis"]}
            ]
        })
        
        # Palabras clave de alto riesgo por plataforma
        self.high_risk_keywords = self.config.get("high_risk_keywords", {
            "youtube": ["hack", "crack", "pirate", "free download", "leaked", "copyright", "nsfw", "gambling"],
            "tiktok": ["banned", "shadowban", "nsfw", "adult", "gambling", "weapons", "drugs", "covid"],
            "instagram": ["banned", "shadowban", "nsfw", "adult", "gambling", "weapons", "drugs", "covid"]
        })
        
        # Estado actual de shadowban por canal
        self.channel_status = {}
        
        # Historial de shadowbans
        self.shadowban_history = {}
        
        # Planes de recuperación activos
        self.active_recovery_plans = {}
        
        # Caché de métricas para detección
        self.metrics_cache = {}
        
        logger.info("ShadowbanManager inicializado")
    
    def detect_shadowban(self, channel_id: str, platform: str, 
                         recent_metrics: List[Dict[str, Any]], 
                         baseline_metrics: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detecta si un canal está bajo shadowban basado en métricas recientes vs históricas.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma (youtube, tiktok, instagram, etc.)
            recent_metrics: Métricas recientes (últimos 3-5 posts)
            baseline_metrics: Métricas históricas para comparación (opcional)
            
        Returns:
            Dict[str, Any]: Resultado de la detección con estado y detalles
        """
        logger.info(f"Detectando shadowban para canal {channel_id} en {platform}")
        
        # Si no hay métricas recientes, no podemos detectar
        if not recent_metrics:
            return {
                "status": ShadowbanStatus.NORMAL.name,
                "confidence": 0,
                "details": "No hay suficientes métricas recientes para análisis",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Si no se proporcionan métricas de línea base, usar caché o retornar normal
        if not baseline_metrics:
            if channel_id in self.metrics_cache and platform in self.metrics_cache[channel_id]:
                baseline_metrics = self.metrics_cache[channel_id][platform]
            else:
                return {
                    "status": ShadowbanStatus.NORMAL.name,
                    "confidence": 0,
                    "details": "No hay métricas históricas para comparación",
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
        # Obtener umbrales para la plataforma
        thresholds = self.detection_thresholds.get(platform, {
            "engagement_drop_percent": 60,
            "view_drop_percent": 70,
            "min_samples": 5
        })
        
        # Verificar si tenemos suficientes muestras
        if len(recent_metrics) < thresholds.get("min_samples", 3):
            return {
                "status": ShadowbanStatus.NORMAL.name,
                "confidence": 0,
                "details": f"Insuficientes muestras recientes ({len(recent_metrics)})",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # Calcular métricas promedio
        recent_avg = self._calculate_average_metrics(recent_metrics)
        baseline_avg = self._calculate_average_metrics(baseline_metrics)
        
        # Calcular porcentajes de caída
        drops = self._calculate_metric_drops(recent_avg, baseline_avg)
        
        # Determinar si hay shadowban basado en los umbrales
        shadowban_indicators = []
        confidence_scores = []
        
        # Verificar caída de engagement
        if drops.get("engagement_rate", 0) > thresholds.get("engagement_drop_percent", 60):
            shadowban_indicators.append(f"Caída de engagement: {drops['engagement_rate']:.1f}%")
            confidence_scores.append(min(100, drops["engagement_rate"]))
        
        # Verificar caída de vistas
        if drops.get("views", 0) > thresholds.get("view_drop_percent", 70):
            shadowban_indicators.append(f"Caída de vistas: {drops['views']:.1f}%")
            confidence_scores.append(min(100, drops["views"]))
        
        # Verificar caída de alcance
        if drops.get("reach", 0) > thresholds.get("reach_drop_percent", 65):
            shadowban_indicators.append(f"Caída de alcance: {drops['reach']:.1f}%")
            confidence_scores.append(min(100, drops["reach"]))
        
        # Verificar caída de comentarios
        if drops.get("comments", 0) > thresholds.get("comment_drop_percent", 75):
            shadowban_indicators.append(f"Caída de comentarios: {drops['comments']:.1f}%")
            confidence_scores.append(min(100, drops["comments"]))
        
        # Determinar estado y confianza
        if not shadowban_indicators:
            status = ShadowbanStatus.NORMAL.name
            confidence = 0
            details = "No se detectaron indicadores de shadowban"
        else:
            # Calcular confianza promedio
            confidence = statistics.mean(confidence_scores) if confidence_scores else 0
            
            # Determinar estado basado en confianza
            if confidence >= 80:
                status = ShadowbanStatus.CONFIRMED.name
                details = f"Shadowban confirmado con {len(shadowban_indicators)} indicadores"
            elif confidence >= 50:
                status = ShadowbanStatus.SUSPECTED.name
                details = f"Posible shadowban con {len(shadowban_indicators)} indicadores"
            else:
                status = ShadowbanStatus.NORMAL.name
                details = f"Indicadores débiles, no suficientes para confirmar shadowban"
        
        # Determinar tipo de shadowban
        shadowban_type = self._determine_shadowban_type(drops, platform, recent_metrics)
        
        # Actualizar estado del canal
        self.channel_status[channel_id] = {
            "platform": platform,
            "status": status,
            "confidence": confidence,
            "shadowban_type": shadowban_type,
            "last_checked": datetime.datetime.now().isoformat()
        }
        
        # Guardar métricas en caché para futuras comparaciones
        if channel_id not in self.metrics_cache:
            self.metrics_cache[channel_id] = {}
        self.metrics_cache[channel_id][platform] = recent_metrics
        
        # Registrar en historial si es un shadowban confirmado o sospechado
        if status in [ShadowbanStatus.CONFIRMED.name, ShadowbanStatus.SUSPECTED.name]:
            if channel_id not in self.shadowban_history:
                self.shadowban_history[channel_id] = []
            
            self.shadowban_history[channel_id].append({
                "platform": platform,
                "status": status,
                "confidence": confidence,
                "shadowban_type": shadowban_type,
                "indicators": shadowban_indicators,
                "timestamp": datetime.datetime.now().isoformat(),
                "metrics_snapshot": {
                    "recent": recent_avg,
                    "baseline": baseline_avg,
                    "drops": drops
                }
            })
        
        # Preparar resultado
        result = {
            "channel_id": channel_id,
            "platform": platform,
            "status": status,
            "confidence": confidence,
            "shadowban_type": shadowban_type,
            "details": details,
            "indicators": shadowban_indicators,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics_comparison": {
                "recent": recent_avg,
                "baseline": baseline_avg,
                "drops": drops
            }
        }
        
        logger.info(f"Detección de shadowban para {channel_id} en {platform}: {status} (confianza: {confidence:.1f}%)")
        return result
    
    def _calculate_average_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calcula métricas promedio de una lista de métricas.
        
        Args:
            metrics_list: Lista de diccionarios con métricas
            
        Returns:
            Dict[str, float]: Métricas promedio
        """
        if not metrics_list:
            return {}
        
        # Inicializar acumuladores
        totals = defaultdict(float)
        counts = defaultdict(int)
        
        # Sumar todas las métricas
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    totals[key] += value
                    counts[key] += 1
        
        # Calcular promedios
        averages = {key: totals[key] / counts[key] if counts[key] > 0 else 0 
                   for key in totals}
        
        return averages
    
    def _calculate_metric_drops(self, recent: Dict[str, float], 
                               baseline: Dict[str, float]) -> Dict[str, float]:
        """
        Calcula el porcentaje de caída entre métricas recientes y línea base.
        
        Args:
            recent: Métricas recientes promedio
            baseline: Métricas de línea base promedio
            
        Returns:
            Dict[str, float]: Porcentajes de caída por métrica
        """
        drops = {}
        
        for key in baseline:
            if key in recent and baseline[key] > 0:
                # Calcular porcentaje de caída
                drop_percent = ((baseline[key] - recent[key]) / baseline[key]) * 100
                drops[key] = max(0, drop_percent)  # Solo considerar caídas, no aumentos
        
        # Calcular engagement rate si tenemos los componentes necesarios
        if all(k in recent for k in ["likes", "comments", "shares", "views"]) and \
           all(k in baseline for k in ["likes", "comments", "shares", "views"]):
            
            # Engagement reciente = (likes + comments + shares) / views
            recent_engagement = (recent["likes"] + recent["comments"] + recent["shares"]) / max(1, recent["views"])
            baseline_engagement = (baseline["likes"] + baseline["comments"] + baseline["shares"]) / max(1, baseline["views"])
            
            if baseline_engagement > 0:
                engagement_drop = ((baseline_engagement - recent_engagement) / baseline_engagement) * 100
                drops["engagement_rate"] = max(0, engagement_drop)
        
        return drops
    
    def _determine_shadowban_type(self, drops: Dict[str, float], platform: str,
                                 recent_metrics: List[Dict[str, Any]]) -> str:
        """
        Determina el tipo de shadowban basado en los patrones de caída de métricas.
        
        Args:
            drops: Porcentajes de caída por métrica
            platform: Plataforma
            recent_metrics: Métricas recientes para análisis adicional
            
        Returns:
            str: Tipo de shadowban (de la enumeración ShadowbanType)
        """
        # Verificar patrones específicos para cada tipo de shadowban
        
        # Si hay caída extrema en todas las métricas, es un shadowban completo
        if all(drop > 80 for key, drop in drops.items() if key in ["views", "reach", "likes", "comments"]):
            return ShadowbanType.FULL_BAN.name
        
        # Si hay caída principalmente en hashtags/explorar pero engagement normal con seguidores
        if drops.get("reach", 0) > 70 and drops.get("new_followers", 0) > 70 and drops.get("likes", 0) < 40:
            if platform == "instagram":
                return ShadowbanType.EXPLORE_BAN.name
            elif platform == "tiktok":
                return ShadowbanType.HASHTAG_BAN.name
        
        # Si hay caída principalmente en comentarios visibles
        if drops.get("comments", 0) > 80 and drops.get("views", 0) < 50:
            return ShadowbanType.COMMENT_BAN.name
        
        # Si hay caída general moderada en todas las métricas
        if all(30 < drop < 70 for key, drop in drops.items() if key in ["views", "reach", "likes"]):
            return ShadowbanType.ALGORITHM_PENALTY.name
        
        # Si hay caída severa pero reciente (verificar timestamps en recent_metrics)
        if recent_metrics and len(recent_metrics) >= 3:
            # Ordenar por timestamp (asumiendo que tienen campo 'timestamp')
            if all('timestamp' in m for m in recent_metrics):
                sorted_metrics = sorted(recent_metrics, key=lambda m: m.get('timestamp', ''))
                # Si solo los últimos 1-2 posts tienen caída severa
                if len(sorted_metrics) >= 3:
                    recent_two = sorted_metrics[-2:]
                    older = sorted_metrics[:-2]
                    
                    recent_avg = self._calculate_average_metrics(recent_two)
                    older_avg = self._calculate_average_metrics(older)
                    
                    recent_drops = self._calculate_metric_drops(recent_avg, older_avg)
                    
                    if all(drop > 70 for key, drop in recent_drops.items() if key in ["views", "reach"]):
                        return ShadowbanType.TEMPORARY_BAN.name
        
        # Por defecto, limitación de alcance general
        return ShadowbanType.REACH_LIMITATION.name
    
    def create_recovery_plan(self, channel_id: str, platform: str, 
                            shadowban_type: str = None) -> Dict[str, Any]:
        """
        Crea un plan de recuperación para un canal con shadowban.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            shadowban_type: Tipo de shadowban (opcional)
            
        Returns:
            Dict[str, Any]: Plan de recuperación
        """
        logger.info(f"Creando plan de recuperación para canal {channel_id} en {platform}")
        
        # Verificar si el canal tiene estado de shadowban
        if channel_id not in self.channel_status:
            return {
                "success": False,
                "error": "Canal no tiene estado de shadowban registrado"
            }
        
        channel_status = self.channel_status[channel_id]
        
        # Si no se proporciona tipo de shadowban, usar el del estado del canal
        if not shadowban_type:
            shadowban_type = channel_status.get("shadowban_type", ShadowbanType.REACH_LIMITATION.name)
        
        # Obtener estrategia de recuperación para la plataforma
        strategy = self.recovery_strategies.get(platform, [])
        if not strategy:
            return {
                "success": False,
                "error": f"No hay estrategia de recuperación definida para {platform}"
            }
        
        # Crear plan de recuperación
        start_date = datetime.datetime.now()
        current_date = start_date
        
        plan = {
            "channel_id": channel_id,
            "platform": platform,
            "shadowban_type": shadowban_type,
            "start_date": start_date.isoformat(),
            "current_phase": RecoveryPhase.ANALYSIS.name,
            "phases": [],
            "status": "active",
            "progress": 0,
            "estimated_completion_date": None,
            "notes": []
        }
        
        # Generar fases del plan
        for phase_data in strategy:
            phase_name = phase_data["phase"]
            duration_days = phase_data["duration_days"]
            actions = phase_data["actions"]
            
            # Ajustar duración según tipo de shadowban
            if shadowban_type == ShadowbanType.FULL_BAN.name:
                duration_days = int(duration_days * 1.5)  # 50% más tiempo para bans completos
            elif shadowban_type == ShadowbanType.TEMPORARY_BAN.name:
                duration_days = max(1, int(duration_days * 0.7))  # 30% menos tiempo para bans temporales
            
            # Calcular fechas
            phase_start = current_date
            phase_end = current_date + datetime.timedelta(days=duration_days)
            current_date = phase_end
            
            # Añadir fase al plan
            plan["phases"].append({
                "name": phase_name,
                "start_date": phase_start.isoformat(),
                "end_date": phase_end.isoformat(),
                "duration_days": duration_days,
                "actions": actions,
                "completed_actions": [],
                "status": "pending" if phase_name != RecoveryPhase.ANALYSIS.name else "active",
                "notes": []
            })
        
        # Establecer fecha estimada de finalización
        plan["estimated_completion_date"] = current_date.isoformat()
        
        # Guardar plan en planes activos
        self.active_recovery_plans[channel_id] = plan
        
        # Actualizar estado del canal
        self.channel_status[channel_id]["status"] = ShadowbanStatus.RECOVERING.name
        self.channel_status[channel_id]["recovery_start_date"] = start_date.isoformat()
        self.channel_status[channel_id]["estimated_recovery_date"] = current_date.isoformat()
        
        logger.info(f"Plan de recuperación creado para {channel_id} en {platform}. Duración estimada: {(current_date - start_date).days} días")
        return plan
    
    def update_recovery_progress(self, channel_id: str, completed_action: str = None, 
                                phase_complete: bool = False, notes: str = None) -> Dict[str, Any]:
        """
        Actualiza el progreso de un plan de recuperación.
        
        Args:
            channel_id: ID del canal
            completed_action: Acción completada (opcional)
            phase_complete: Indica si la fase actual está completa (opcional)
            notes: Notas adicionales (opcional)
            
        Returns:
            Dict[str, Any]: Estado actualizado del plan
        """
        # Verificar si existe un plan activo
        if channel_id not in self.active_recovery_plans:
            return {
                "success": False,
                "error": "No hay plan de recuperación activo para este canal"
            }
        
        plan = self.active_recovery_plans[channel_id]
        
        # Encontrar la fase actual
        current_phase_index = None
        for i, phase in enumerate(plan["phases"]):
            if phase["status"] == "active":
                current_phase_index = i
                break
        
        if current_phase_index is None:
            return {
                "success": False,
                "error": "No se encontró fase activa en el plan"
            }
        
        current_phase = plan["phases"][current_phase_index]
        
        # Añadir acción completada
        if completed_action:
            if completed_action in current_phase["actions"] and completed_action not in current_phase["completed_actions"]:
                current_phase["completed_actions"].append(completed_action)
                
                # Añadir nota sobre la acción
                if notes:
                    current_phase["notes"].append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "action": completed_action,
                        "note": notes
                    })
        
        # Marcar fase como completa si se indica
        if phase_complete or set(current_phase["actions"]) == set(current_phase["completed_actions"]):
            current_phase["status"] = "completed"
            
            # Activar siguiente fase si existe
            if current_phase_index < len(plan["phases"]) - 1:
                next_phase = plan["phases"][current_phase_index + 1]
                next_phase["status"] = "active"
                plan["current_phase"] = next_phase["name"]
            else:
                # Plan completado
                plan["status"] = "completed"
                plan["progress"] = 100
                
                # Actualizar estado del canal
                self.channel_status[channel_id]["status"] = ShadowbanStatus.RECOVERED.name
                self.channel_status[channel_id]["recovery_end_date"] = datetime.datetime.now().isoformat()
                
                logger.info(f"Plan de recuperación completado para canal {channel_id}")
                
                # Mover de planes activos a historial
                if channel_id not in self.shadowban_history:
                    self.shadowban_history[channel_id] = []
                
                self.shadowban_history[channel_id].append({
                    "type": "recovery_plan",
                    "plan": plan,
                    "completion_date": datetime.datetime.now().isoformat()
                })
                
                # Eliminar de planes activos
                del self.active_recovery_plans[channel_id]
                
                return {
                    "success": True,
                    "status": "completed",
                    "message": "Plan de recuperación completado exitosamente"
                }
        
        # Calcular progreso general
        completed_phases = sum(1 for phase in plan["phases"] if phase["status"] == "completed")
        current_phase_progress = len(current_phase["completed_actions"]) / max(1, len(current_phase["actions"]))
        plan["progress"] = (completed_phases + current_phase_progress) / len(plan["phases"]) * 100
        
        # Añadir notas generales
        if notes and not completed_action:
            plan["notes"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "note": notes
            })
        
        logger.info(f"Progreso de recuperación actualizado para canal {channel_id}: {plan['progress']:.1f}%")
        return {
            "success": True,
            "status": plan["status"],
            "progress": plan["progress"],
            "current_phase": plan["current_phase"],
            "message": "Progreso actualizado correctamente"
        }
    
    def get_recovery_plan(self, channel_id: str) -> Dict[str, Any]:
        """
        Obtiene el plan de recuperación activo para un canal.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            Dict[str, Any]: Plan de recuperación o error
        """
        if channel_id in self.active_recovery_plans:
            return {
                "success": True,
                "plan": self.active_recovery_plans[channel_id]
            }
        else:
            return {
                "success": False,
                "error": "No hay plan de recuperación activo para este canal"
            }
    
    def get_channel_status(self, channel_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado actual de shadowban de un canal.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            Dict[str, Any]: Estado del canal o error
        """
        if channel_id in self.channel_status:
            return {
                "success": True,
                "status": self.channel_status[channel_id]
            }
        else:
            return {
                "success": False,
                "error": "No hay información de estado para este canal"
            }
    
    def get_shadowban_history(self, channel_id: str) -> Dict[str, Any]:
        """
        Obtiene el historial de shadowbans de un canal.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            Dict[str, Any]: Historial de shadowbans o error
        """
        Returns:
            Dict[str, Any]: Historial de shadowbans o error
        """
        if channel_id in self.shadowban_history:
            return {
                "success": True,
                "history": self.shadowban_history[channel_id]
            }
        else:
            return {
                "success": False,
                "error": "No hay historial de shadowbans para este canal"
            }
    
    def analyze_shadowban_risk(self, channel_id: str, platform: str, 
                              content_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza el riesgo de shadowban para un contenido específico antes de publicarlo.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            content_data: Datos del contenido a analizar (texto, hashtags, etc.)
            
        Returns:
            Dict[str, Any]: Análisis de riesgo con recomendaciones
        """
        logger.info(f"Analizando riesgo de shadowban para canal {channel_id} en {platform}")
        
        # Inicializar resultado
        result = {
            "channel_id": channel_id,
            "platform": platform,
            "risk_level": "low",
            "risk_score": 0,
            "risk_factors": [],
            "recommendations": [],
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Obtener palabras clave de alto riesgo para la plataforma
        high_risk_keywords = self.high_risk_keywords.get(platform, [])
        
        # Extraer texto y hashtags del contenido
        text = content_data.get("caption", "") + " " + content_data.get("description", "")
        hashtags = content_data.get("hashtags", [])
        
        # Verificar palabras clave de alto riesgo en el texto
        risk_keywords_found = []
        for keyword in high_risk_keywords:
            if keyword.lower() in text.lower():
                risk_keywords_found.append(keyword)
        
        # Calcular puntuación de riesgo basada en palabras clave encontradas
        keyword_risk_score = len(risk_keywords_found) * 20  # 20 puntos por cada palabra clave de riesgo
        
        # Verificar cantidad de hashtags (demasiados pueden ser riesgosos)
        hashtag_count = len(hashtags)
        hashtag_risk_score = 0
        
        if platform == "instagram" and hashtag_count > 25:
            hashtag_risk_score = 30
            result["risk_factors"].append(f"Demasiados hashtags ({hashtag_count}/30)")
            result["recommendations"].append("Reducir hashtags a un máximo de 20-25")
        elif platform == "tiktok" and hashtag_count > 10:
            hashtag_risk_score = 20
            result["risk_factors"].append(f"Demasiados hashtags ({hashtag_count}/10)")
            result["recommendations"].append("Reducir hashtags a un máximo de 5-7")
        
        # Verificar hashtags de alto riesgo
        risky_hashtags = []
        for hashtag in hashtags:
            if any(keyword.lower() in hashtag.lower() for keyword in high_risk_keywords):
                risky_hashtags.append(hashtag)
        
        hashtag_keyword_risk_score = len(risky_hashtags) * 15  # 15 puntos por cada hashtag riesgoso
        
        # Verificar frecuencia de publicación reciente
        frequency_risk_score = 0
        if channel_id in self.metrics_cache and platform in self.metrics_cache[channel_id]:
            recent_metrics = self.metrics_cache[channel_id][platform]
            if len(recent_metrics) >= 2:
                # Ordenar por timestamp (asumiendo que tienen campo 'timestamp')
                if all('timestamp' in m for m in recent_metrics):
                    sorted_metrics = sorted(recent_metrics, key=lambda m: m.get('timestamp', ''))
                    
                    # Verificar si las últimas publicaciones fueron muy cercanas
                    if len(sorted_metrics) >= 2:
                        latest = datetime.datetime.fromisoformat(sorted_metrics[-1].get('timestamp'))
                        previous = datetime.datetime.fromisoformat(sorted_metrics[-2].get('timestamp'))
                        time_diff = (latest - previous).total_seconds() / 3600  # horas
                        
                        if time_diff < 1:  # menos de 1 hora
                            frequency_risk_score = 40
                            result["risk_factors"].append(f"Publicaciones muy frecuentes ({time_diff:.1f} horas entre posts)")
                            result["recommendations"].append("Esperar al menos 3-4 horas entre publicaciones")
                        elif time_diff < 3:  # menos de 3 horas
                            frequency_risk_score = 20
                            result["risk_factors"].append(f"Publicaciones frecuentes ({time_diff:.1f} horas entre posts)")
                            result["recommendations"].append("Considerar espaciar más las publicaciones")
        
        # Verificar historial de shadowbans
        history_risk_score = 0
        if channel_id in self.shadowban_history:
            history = self.shadowban_history[channel_id]
            recent_shadowbans = [sb for sb in history if 
                                isinstance(sb, dict) and 
                                sb.get('platform') == platform and
                                'timestamp' in sb and
                                (datetime.datetime.now() - datetime.datetime.fromisoformat(sb['timestamp'])).days < 30]
            
            if recent_shadowbans:
                history_risk_score = min(50, len(recent_shadowbans) * 25)  # Máximo 50 puntos
                result["risk_factors"].append(f"Historial reciente de {len(recent_shadowbans)} shadowbans")
                result["recommendations"].append("Extremar precauciones y ser más conservador en contenido")
        
        # Calcular puntuación total de riesgo
        total_risk_score = keyword_risk_score + hashtag_risk_score + hashtag_keyword_risk_score + frequency_risk_score + history_risk_score
        
        # Normalizar a 100
        total_risk_score = min(100, total_risk_score)
        
        # Determinar nivel de riesgo
        if total_risk_score >= 70:
            risk_level = "high"
        elif total_risk_score >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Añadir factores de riesgo específicos
        if risk_keywords_found:
            result["risk_factors"].append(f"Palabras clave de riesgo: {', '.join(risk_keywords_found)}")
            result["recommendations"].append(f"Evitar palabras como: {', '.join(risk_keywords_found)}")
        
        if risky_hashtags:
            result["risk_factors"].append(f"Hashtags de riesgo: {', '.join(risky_hashtags)}")
            result["recommendations"].append(f"Evitar hashtags como: {', '.join(risky_hashtags)}")
        
        # Actualizar resultado
        result["risk_score"] = total_risk_score
        result["risk_level"] = risk_level
        
        # Si no hay factores de riesgo, añadir mensaje positivo
        if not result["risk_factors"]:
            result["risk_factors"].append("No se detectaron factores de riesgo específicos")
            result["recommendations"].append("Contenido seguro para publicar")
        
        logger.info(f"Análisis de riesgo para {channel_id} en {platform}: {risk_level} (score: {total_risk_score})")
        return result
    
    def generate_shadowban_recommendations(self, channel_id: str, platform: str) -> Dict[str, Any]:
        """
        Genera recomendaciones personalizadas para evitar shadowbans basadas en el historial y estado del canal.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            
        Returns:
            Dict[str, Any]: Recomendaciones personalizadas
        """
        logger.info(f"Generando recomendaciones para evitar shadowbans para canal {channel_id} en {platform}")
        
        recommendations = []
        
        # Verificar estado actual del canal
        channel_status = self.channel_status.get(channel_id, {})
        current_status = channel_status.get("status", ShadowbanStatus.NORMAL.name)
        
        # Recomendaciones generales para todas las plataformas
        general_recommendations = [
            "Mantener una frecuencia de publicación constante pero no excesiva",
            "Evitar usar demasiados hashtags en una sola publicación",
            "No usar hashtags prohibidos o restringidos",
            "Evitar mencionar temas sensibles como COVID, política, drogas, etc.",
            "Interactuar genuinamente con otros usuarios (comentarios, likes)",
            "Evitar usar bots o herramientas automatizadas para interacciones",
            "No publicar contenido que pueda violar las normas de la comunidad",
            "Variar el contenido para evitar repeticiones excesivas"
        ]
        
        # Recomendaciones específicas por plataforma
        platform_specific = {
            "youtube": [
                "Evitar títulos clickbait o engañosos",
                "No solicitar likes/suscripciones de manera agresiva",
                "Mantener una duración de video adecuada (>3 minutos para monetización)",
                "Evitar reutilizar miniaturas o títulos idénticos",
                "Asegurar que el contenido sea adecuado para anunciantes"
            ],
            "tiktok": [
                "Limitar hashtags a 5-7 por publicación",
                "Evitar publicar más de 2-3 veces al día",
                "No republicar contenido de otras plataformas con marcas de agua",
                "Evitar mencionar directamente a la plataforma o shadowbans",
                "Usar música de la biblioteca de TikTok en lugar de audio externo"
            ],
            "instagram": [
                "Limitar hashtags a 20-25 por publicación",
                "Evitar editar publicaciones inmediatamente después de publicarlas",
                "No usar los mismos hashtags en todas las publicaciones",
                "Evitar seguir/dejar de seguir a muchas cuentas rápidamente",
                "Publicar en horarios regulares pero no exactamente iguales"
            ]
        }
        
        # Añadir recomendaciones generales
        recommendations.extend(random.sample(general_recommendations, min(5, len(general_recommendations))))
        
        # Añadir recomendaciones específicas de la plataforma
        if platform in platform_specific:
            recommendations.extend(random.sample(platform_specific[platform], min(3, len(platform_specific[platform]))))
        
        # Recomendaciones basadas en el estado actual
        if current_status == ShadowbanStatus.SUSPECTED.name:
            status_recommendations = [
                "Reducir temporalmente la frecuencia de publicación",
                "Evitar usar hashtags populares por unos días",
                "Interactuar más con otros creadores de contenido",
                "Revisar y posiblemente eliminar publicaciones recientes que puedan violar normas",
                "Monitorear métricas de engagement cuidadosamente"
            ]
            recommendations.extend(status_recommendations)
        elif current_status == ShadowbanStatus.CONFIRMED.name:
            status_recommendations = [
                "Pausar publicaciones por 24-48 horas",
                "Eliminar cualquier contenido que pueda violar las normas",
                "Considerar contactar al soporte de la plataforma",
                "Seguir el plan de recuperación recomendado",
                "No crear nuevas cuentas para eludir restricciones"
            ]
            recommendations.extend(status_recommendations)
        elif current_status == ShadowbanStatus.RECOVERING.name:
            status_recommendations = [
                "Continuar con el plan de recuperación actual",
                "Publicar contenido seguro y no controversial",
                "Mantener interacciones genuinas con la comunidad",
                "Evitar cambios drásticos en patrones de publicación",
                "Monitorear métricas para confirmar mejora"
            ]
            recommendations.extend(status_recommendations)
        
        # Recomendaciones basadas en historial
        if channel_id in self.shadowban_history:
            history = self.shadowban_history[channel_id]
            shadowban_types = [sb.get("shadowban_type") for sb in history if isinstance(sb, dict) and "shadowban_type" in sb]
            
            # Contar tipos de shadowban
            type_counts = defaultdict(int)
            for sb_type in shadowban_types:
                type_counts[sb_type] += 1
            
            # Recomendaciones basadas en tipos más frecuentes
            if type_counts:
                most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
                
                if most_common_type == ShadowbanType.HASHTAG_BAN.name:
                    recommendations.append("Crear una lista de hashtags seguros y rotar entre ellos")
                    recommendations.append("Evitar hashtags muy populares o trending")
                elif most_common_type == ShadowbanType.EXPLORE_BAN.name:
                    recommendations.append("Enfocarse en construir audiencia fiel en lugar de alcance en explorar")
                    recommendations.append("Mejorar la calidad del contenido para aumentar retención")
                elif most_common_type == ShadowbanType.COMMENT_BAN.name:
                    recommendations.append("Evitar palabras clave sensibles en comentarios")
                    recommendations.append("Moderar comentarios para eliminar spam o contenido inapropiado")
                elif most_common_type == ShadowbanType.FULL_BAN.name:
                    recommendations.append("Considerar una pausa más larga (1-2 semanas)")
                    recommendations.append("Revisar a fondo todas las normas de la comunidad")
        
        # Eliminar duplicados y limitar a 10 recomendaciones
        unique_recommendations = list(dict.fromkeys(recommendations))
        final_recommendations = unique_recommendations[:10]
        
        result = {
            "channel_id": channel_id,
            "platform": platform,
            "current_status": current_status,
            "recommendations": final_recommendations,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Generadas {len(final_recommendations)} recomendaciones para {channel_id} en {platform}")
        return result
    
    def simulate_recovery(self, channel_id: str, platform: str, 
                         days: int = 30) -> Dict[str, Any]:
        """
        Simula el proceso de recuperación de shadowban para estimar tiempos y probabilidades.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma
            days: Número de días a simular
            
        Returns:
            Dict[str, Any]: Resultados de la simulación
        """
        logger.info(f"Simulando recuperación de shadowban para canal {channel_id} en {platform} ({days} días)")
        
        # Verificar si el canal tiene estado de shadowban
        if channel_id not in self.channel_status:
            return {
                "success": False,
                "error": "Canal no tiene estado de shadowban registrado"
            }
        
        channel_status = self.channel_status[channel_id]
        current_status = channel_status.get("status", ShadowbanStatus.NORMAL.name)
        shadowban_type = channel_status.get("shadowban_type", ShadowbanType.REACH_LIMITATION.name)
        
        # Si no está en shadowban, no hay nada que simular
        if current_status not in [ShadowbanStatus.SUSPECTED.name, ShadowbanStatus.CONFIRMED.name]:
            return {
                "success": False,
                "error": f"Canal no está en estado de shadowban (estado actual: {current_status})"
            }
        
        # Parámetros de simulación según tipo de shadowban
        recovery_params = {
            ShadowbanType.HASHTAG_BAN.name: {
                "base_days": 7,
                "success_probability": 0.85,
                "daily_improvement": 0.12
            },
            ShadowbanType.EXPLORE_BAN.name: {
                "base_days": 10,
                "success_probability": 0.80,
                "daily_improvement": 0.10
            },
            ShadowbanType.COMMENT_BAN.name: {
                "base_days": 5,
                "success_probability": 0.90,
                "daily_improvement": 0.15
            },
            ShadowbanType.REACH_LIMITATION.name: {
                "base_days": 14,
                "success_probability": 0.75,
                "daily_improvement": 0.08
            },
            ShadowbanType.ALGORITHM_PENALTY.name: {
                "base_days": 21,
                "success_probability": 0.70,
                "daily_improvement": 0.05
            },
            ShadowbanType.TEMPORARY_BAN.name: {
                "base_days": 3,
                "success_probability": 0.95,
                "daily_improvement": 0.25
            },
            ShadowbanType.FULL_BAN.name: {
                "base_days": 30,
                "success_probability": 0.60,
                "daily_improvement": 0.03
            }
        }
        
        # Obtener parámetros para el tipo de shadowban
        params = recovery_params.get(shadowban_type, {
            "base_days": 14,
            "success_probability": 0.75,
            "daily_improvement": 0.08
        })
        
        # Ajustar parámetros según plataforma
        if platform == "youtube":
            params["base_days"] = int(params["base_days"] * 1.2)  # YouTube suele tardar más
            params["success_probability"] *= 0.9  # Menor probabilidad de éxito
        elif platform == "tiktok":
            params["base_days"] = int(params["base_days"] * 0.8)  # TikTok suele ser más rápido
            params["daily_improvement"] *= 1.2  # Mejora más rápida
        
        # Inicializar simulación
        simulation_days = []
        current_day = 0
        current_probability = 0
        recovery_day = None
        
        # Simular día a día
        while current_day < days:
            current_day += 1
            
            # Calcular probabilidad de recuperación para este día
            if current_day <= params["base_days"]:
                # Durante el período base, probabilidad baja
                day_probability = params["success_probability"] * (current_day / params["base_days"])
            else:
                # Después del período base, mejora diaria
                day_probability = min(0.99, params["success_probability"] + 
                                    (current_day - params["base_days"]) * params["daily_improvement"])
            
            # Guardar estado del día
            simulation_days.append({
                "day": current_day,
                "recovery_probability": day_probability,
                "status": "recovering"
            })
            
            # Verificar si se recupera este día
            if random.random() < day_probability and recovery_day is None:
                recovery_day = current_day
                simulation_days[-1]["status"] = "recovered"
            
            # Si ya se recuperó, marcar días restantes como recuperados
            if recovery_day is not None:
                simulation_days[-1]["status"] = "recovered"
        
        # Calcular estadísticas
        recovery_probability = simulation_days[-1]["recovery_probability"] if simulation_days else 0
        estimated_days = recovery_day if recovery_day else params["base_days"] + int((0.95 - params["success_probability"]) / params["daily_improvement"])
        
        # Preparar resultado
        result = {
            "channel_id": channel_id,
            "platform": platform,
            "shadowban_type": shadowban_type,
            "simulation_days": days,
            "recovery_day": recovery_day,
            "estimated_recovery_days": min(days, estimated_days),
            "final_recovery_probability": recovery_probability,
            "simulation_data": simulation_days,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        logger.info(f"Simulación completada para {channel_id}. Recuperación estimada en {estimated_days} días (prob: {recovery_probability:.2f})")
        return result
    
    def reset_channel_status(self, channel_id: str) -> Dict[str, Any]:
        """
        Restablece el estado de shadowban de un canal a normal.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            Dict[str, Any]: Resultado de la operación
        """
        if channel_id in self.channel_status:
            # Guardar estado anterior en historial
            if channel_id not in self.shadowban_history:
                self.shadowban_history[channel_id] = []
            
            previous_status = self.channel_status[channel_id]
            self.shadowban_history[channel_id].append({
                "type": "status_reset",
                "previous_status": previous_status,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # Restablecer estado
            self.channel_status[channel_id]["status"] = ShadowbanStatus.NORMAL.name
            self.channel_status[channel_id]["last_reset"] = datetime.datetime.now().isoformat()
            
            # Eliminar plan de recuperación si existe
            if channel_id in self.active_recovery_plans:
                del self.active_recovery_plans[channel_id]
            
            logger.info(f"Estado de shadowban restablecido para canal {channel_id}")
            return {
                "success": True,
                "message": f"Estado restablecido a {ShadowbanStatus.NORMAL.name}",
                "previous_status": previous_status.get("status")
            }
        else:
            return {
                "success": False,
                "error": "No hay información de estado para este canal"
            }
    
    def export_shadowban_data(self, channel_id: str = None) -> Dict[str, Any]:
        """
        Exporta datos de shadowban para análisis o respaldo.
        
        Args:
            channel_id: ID del canal (opcional, si no se proporciona exporta todos)
            
        Returns:
            Dict[str, Any]: Datos exportados
        """
        if channel_id:
            # Exportar datos de un canal específico
            data = {
                "channel_id": channel_id,
                "status": self.channel_status.get(channel_id, {}),
                "history": self.shadowban_history.get(channel_id, []),
                "active_recovery_plan": self.active_recovery_plans.get(channel_id, None),
                "export_timestamp": datetime.datetime.now().isoformat()
            }
        else:
            # Exportar todos los datos
            data = {
                "channel_status": self.channel_status,
                "shadowban_history": self.shadowban_history,
                "active_recovery_plans": self.active_recovery_plans,
                "export_timestamp": datetime.datetime.now().isoformat()
            }
        
        logger.info(f"Datos de shadowban exportados: {len(json.dumps(data))} bytes")
        return data
    
    def import_shadowban_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Importa datos de shadowban desde un respaldo.
        
        Args:
            data: Datos a importar
            
        Returns:
            Dict[str, Any]: Resultado de la importación
        """
        try:
            if "channel_id" in data:
                # Importar datos de un canal específico
                channel_id = data["channel_id"]
                
                if "status" in data:
                    self.channel_status[channel_id] = data["status"]
                
                if "history" in data:
                    self.shadowban_history[channel_id] = data["history"]
                
                if "active_recovery_plan" in data and data["active_recovery_plan"]:
                    self.active_recovery_plans[channel_id] = data["active_recovery_plan"]
                
                logger.info(f"Datos de shadowban importados para canal {channel_id}")
                return {
                    "success": True,
                    "message": f"Datos importados para canal {channel_id}",
                    "imported_items": {
                        "status": "status" in data,
                        "history": "history" in data,
                        "recovery_plan": "active_recovery_plan" in data and data["active_recovery_plan"] is not None
                    }
                }
            else:
                # Importar todos los datos
                if "channel_status" in data:
                    self.channel_status = data["channel_status"]
                
                if "shadowban_history" in data:
                    self.shadowban_history = data["shadowban_history"]
                
                if "active_recovery_plans" in data:
                    self.active_recovery_plans = data["active_recovery_plans"]
                
                logger.info(f"Datos globales de shadowban importados")
                return {
                    "success": True,
                    "message": "Datos globales importados correctamente",
                    "imported_items": {
                        "channel_status": "channel_status" in data,
                        "shadowban_history": "shadowban_history" in data,
                        "active_recovery_plans": "active_recovery_plans" in data
                    }
                }
        except Exception as e:
            logger.error(f"Error al importar datos de shadowban: {str(e)}")
            return {
                "success": False,
                "error": f"Error al importar datos: {str(e)}"
            }