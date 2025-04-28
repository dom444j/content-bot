import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
import numpy as np
from collections import defaultdict

class TrafficRedistributor:
    """
    Sistema de redistribución inteligente de tráfico y recursos entre canales
    
    Este módulo analiza el rendimiento de diferentes canales y plataformas,
    identificando aquellos con mejor ROI para redistribuir recursos de inversión,
    promoción y esfuerzo de creación de contenido.
    """
    
    def __init__(self, config_path: str = "../config/strategy.json", 
                 analytics_path: str = "../data/analytics_data.json",
                 min_data_points: int = 10,
                 redistribution_threshold: float = 0.5,  # 50% ROI mínimo para redistribución
                 low_performance_threshold: float = 0.05,  # 5% CTR mínimo
                 reallocation_percentage: float = 0.3):  # 30% de recursos a redistribuir
        """
        Inicializa el redistribuidor de tráfico
        
        Args:
            config_path: Ruta al archivo de configuración de estrategia
            analytics_path: Ruta al archivo de datos analíticos
            min_data_points: Número mínimo de puntos de datos para análisis
            redistribution_threshold: ROI mínimo para considerar un canal como destino de redistribución
            low_performance_threshold: Umbral de CTR bajo para considerar redistribución
            reallocation_percentage: Porcentaje de recursos a redistribuir
        """
        self.logger = logging.getLogger("TrafficRedistributor")
        self.config_path = config_path
        self.analytics_path = analytics_path
        self.min_data_points = min_data_points
        self.redistribution_threshold = redistribution_threshold
        self.low_performance_threshold = low_performance_threshold
        self.reallocation_percentage = reallocation_percentage
        
        # Cargar configuración y datos
        self._load_config()
        self._load_analytics()
        
        # Historial de redistribuciones
        self.redistribution_history = []
        
        # Métricas de rendimiento
        self.performance_metrics = {}
        
    def _load_config(self) -> None:
        """Carga la configuración de estrategia"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"Configuración cargada desde {self.config_path}")
            else:
                self.logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                self.config = {
                    "channels": {},
                    "redistribution_settings": {
                        "enabled": True,
                        "min_roi_threshold": self.redistribution_threshold,
                        "low_ctr_threshold": self.low_performance_threshold,
                        "reallocation_percentage": self.reallocation_percentage
                    }
                }
        except Exception as e:
            self.logger.error(f"Error al cargar configuración: {str(e)}")
            self.config = {"channels": {}}
            
    def _load_analytics(self) -> None:
        """Carga los datos analíticos"""
        try:
            if os.path.exists(self.analytics_path):
                with open(self.analytics_path, 'r', encoding='utf-8') as f:
                    self.analytics = json.load(f)
                self.logger.info(f"Datos analíticos cargados desde {self.analytics_path}")
            else:
                self.logger.warning(f"Archivo de datos analíticos no encontrado: {self.analytics_path}")
                self.analytics = {"channels": {}}
        except Exception as e:
            self.logger.error(f"Error al cargar datos analíticos: {str(e)}")
            self.analytics = {"channels": {}}
    
    def _save_config(self) -> None:
        """Guarda la configuración actualizada"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"Configuración guardada en {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error al guardar configuración: {str(e)}")
    
    def analyze_channel_performance(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Analiza el rendimiento de todos los canales en el período especificado
        
        Args:
            days: Número de días para el análisis
            
        Returns:
            Diccionario con métricas de rendimiento por canal
        """
        self.logger.info(f"Analizando rendimiento de canales en los últimos {days} días")
        
        # Fecha límite para el análisis
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Métricas por canal
        channel_metrics = {}
        
        try:
            # Analizar cada canal
            for channel_id, channel_data in self.analytics.get("channels", {}).items():
                # Filtrar datos por fecha
                recent_data = [
                    entry for entry in channel_data.get("performance", [])
                    if entry.get("date", "") >= cutoff_date
                ]
                
                # Verificar datos suficientes
                if len(recent_data) < self.min_data_points:
                    self.logger.info(f"Canal {channel_id} tiene insuficientes datos ({len(recent_data)})")
                    continue
                
                # Calcular métricas clave
                views = sum(entry.get("views", 0) for entry in recent_data)
                revenue = sum(entry.get("revenue", 0) for entry in recent_data)
                cost = sum(entry.get("cost", 0) for entry in recent_data)
                clicks = sum(entry.get("clicks", 0) for entry in recent_data)
                subscribers = sum(entry.get("new_subscribers", 0) for entry in recent_data)
                
                # Evitar división por cero
                if views == 0 or cost == 0:
                    continue
                
                # Calcular métricas derivadas
                ctr = clicks / views if views > 0 else 0
                cpm = (revenue / views) * 1000 if views > 0 else 0
                roi = (revenue - cost) / cost if cost > 0 else 0
                subscriber_cost = cost / subscribers if subscribers > 0 else float('inf')
                
                # Calcular tendencia de ROI
                if len(recent_data) >= 2:
                    # Ordenar por fecha
                    sorted_data = sorted(recent_data, key=lambda x: x.get("date", ""))
                    
                    # Dividir en dos mitades para comparar tendencia
                    mid_point = len(sorted_data) // 2
                    first_half = sorted_data[:mid_point]
                    second_half = sorted_data[mid_point:]
                    
                    # Calcular ROI para cada mitad
                    first_revenue = sum(entry.get("revenue", 0) for entry in first_half)
                    first_cost = sum(entry.get("cost", 0) for entry in first_half)
                    second_revenue = sum(entry.get("revenue", 0) for entry in second_half)
                    second_cost = sum(entry.get("cost", 0) for entry in second_half)
                    
                    first_roi = (first_revenue - first_cost) / first_cost if first_cost > 0 else 0
                    second_roi = (second_revenue - second_cost) / second_cost if second_cost > 0 else 0
                    
                    roi_trend = second_roi - first_roi
                else:
                    roi_trend = 0
                
                # Guardar métricas
                channel_metrics[channel_id] = {
                    "views": views,
                    "revenue": revenue,
                    "cost": cost,
                    "clicks": clicks,
                    "subscribers": subscribers,
                    "ctr": ctr,
                    "cpm": cpm,
                    "roi": roi,
                    "roi_trend": roi_trend,
                    "subscriber_cost": subscriber_cost,
                    "performance_score": self._calculate_performance_score(ctr, roi, roi_trend)
                }
                
                self.logger.info(f"Canal {channel_id}: ROI={roi:.2f}, CTR={ctr:.2f}, Score={channel_metrics[channel_id]['performance_score']:.2f}")
            
            # Actualizar métricas de rendimiento
            self.performance_metrics = channel_metrics
            return channel_metrics
            
        except Exception as e:
            self.logger.error(f"Error al analizar rendimiento de canales: {str(e)}")
            return {}
    
    def _calculate_performance_score(self, ctr: float, roi: float, roi_trend: float) -> float:
        """
        Calcula una puntuación de rendimiento compuesta
        
        Args:
            ctr: Click-through rate
            roi: Return on investment
            roi_trend: Tendencia del ROI
            
        Returns:
            Puntuación de rendimiento (0-100)
        """
        # Pesos para cada métrica
        ctr_weight = 0.3
        roi_weight = 0.5
        trend_weight = 0.2
        
        # Normalizar métricas
        norm_ctr = min(1.0, ctr / 0.2)  # 20% CTR es el máximo normalizado
        norm_roi = min(1.0, roi / 2.0)  # 200% ROI es el máximo normalizado
        norm_trend = min(1.0, max(0.0, (roi_trend + 0.5) / 1.0))  # Normalizar tendencia entre 0 y 1
        
        # Calcular puntuación ponderada
        score = (
            ctr_weight * norm_ctr +
            roi_weight * norm_roi +
            trend_weight * norm_trend
        ) * 100
        
        return score
    
    def identify_redistribution_opportunities(self) -> Tuple[List[str], List[str]]:
        """
        Identifica canales de bajo rendimiento y alto rendimiento para redistribución
        
        Returns:
            Tupla de (canales_bajo_rendimiento, canales_alto_rendimiento)
        """
        if not self.performance_metrics:
            self.analyze_channel_performance()
        
        low_performers = []
        high_performers = []
        
        try:
            for channel_id, metrics in self.performance_metrics.items():
                # Identificar canales de bajo rendimiento (CTR bajo)
                if metrics["ctr"] < self.low_performance_threshold:
                    low_performers.append(channel_id)
                
                # Identificar canales de alto rendimiento (ROI alto)
                if metrics["roi"] >= self.redistribution_threshold:
                    high_performers.append(channel_id)
            
            self.logger.info(f"Identificados {len(low_performers)} canales de bajo rendimiento y {len(high_performers)} de alto rendimiento")
            return low_performers, high_performers
            
        except Exception as e:
            self.logger.error(f"Error al identificar oportunidades de redistribución: {str(e)}")
            return [], []
    
    def calculate_redistribution_plan(self) -> Dict[str, Dict[str, Any]]:
        """
        Calcula un plan de redistribución de recursos entre canales
        
        Returns:
            Plan de redistribución con asignaciones por canal
        """
        low_performers, high_performers = self.identify_redistribution_opportunities()
        
        redistribution_plan = {}
        
        try:
            # Si no hay canales de alto rendimiento o bajo rendimiento, no hay redistribución
            if not high_performers or not low_performers:
                self.logger.info("No hay suficientes canales para redistribución")
                return {}
            
            # Calcular recursos totales a redistribuir
            total_resources = 0
            for channel_id in low_performers:
                if channel_id in self.performance_metrics:
                    # Usar costo como proxy para recursos
                    channel_resources = self.performance_metrics[channel_id]["cost"]
                    # Aplicar porcentaje de reasignación
                    resources_to_redistribute = channel_resources * self.reallocation_percentage
                    total_resources += resources_to_redistribute
                    
                    # Registrar en el plan
                    redistribution_plan[channel_id] = {
                        "action": "reduce",
                        "current_resources": channel_resources,
                        "reduction_amount": resources_to_redistribute,
                        "new_resources": channel_resources - resources_to_redistribute,
                        "reason": f"CTR bajo ({self.performance_metrics[channel_id]['ctr']:.2%})"
                    }
            
            # Si no hay recursos para redistribuir, terminar
            if total_resources <= 0:
                self.logger.info("No hay recursos suficientes para redistribuir")
                return redistribution_plan
            
            # Calcular ponderación para canales de alto rendimiento
            total_score = sum(self.performance_metrics[channel_id]["performance_score"] 
                             for channel_id in high_performers)
            
            # Distribuir recursos a canales de alto rendimiento
            for channel_id in high_performers:
                if channel_id in self.performance_metrics and total_score > 0:
                    # Calcular proporción basada en puntuación
                    channel_score = self.performance_metrics[channel_id]["performance_score"]
                    proportion = channel_score / total_score
                    
                    # Calcular recursos a asignar
                    additional_resources = total_resources * proportion
                    current_resources = self.performance_metrics[channel_id]["cost"]
                    
                    # Registrar en el plan
                    redistribution_plan[channel_id] = {
                        "action": "increase",
                        "current_resources": current_resources,
                        "increase_amount": additional_resources,
                        "new_resources": current_resources + additional_resources,
                        "reason": f"ROI alto ({self.performance_metrics[channel_id]['roi']:.2%})"
                    }
            
            self.logger.info(f"Plan de redistribución calculado para {len(redistribution_plan)} canales")
            return redistribution_plan
            
        except Exception as e:
            self.logger.error(f"Error al calcular plan de redistribución: {str(e)}")
            return {}
    
    def apply_redistribution_plan(self, plan: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
        """
        Aplica el plan de redistribución actualizando la configuración
        
        Args:
            plan: Plan de redistribución (si es None, se calcula)
            
        Returns:
            True si se aplicó correctamente, False en caso contrario
        """
        if plan is None:
            plan = self.calculate_redistribution_plan()
        
        if not plan:
            self.logger.info("No hay plan de redistribución para aplicar")
            return False
        
        try:
            # Actualizar configuración de canales
            for channel_id, action_plan in plan.items():
                if channel_id not in self.config.get("channels", {}):
                    self.config.setdefault("channels", {})[channel_id] = {}
                
                # Actualizar presupuesto/recursos
                self.config["channels"][channel_id]["budget"] = action_plan["new_resources"]
                
                # Actualizar prioridad si es un canal de alto rendimiento
                if action_plan["action"] == "increase":
                    self.config["channels"][channel_id]["priority"] = "high"
                elif action_plan["action"] == "reduce":
                    self.config["channels"][channel_id]["priority"] = "low"
            
            # Guardar configuración actualizada
            self._save_config()
            
            # Registrar redistribución en historial
            self.redistribution_history.append({
                "date": datetime.now().isoformat(),
                "plan": plan,
                "total_redistributed": sum(
                    action["increase_amount"] for action in plan.values() 
                    if action["action"] == "increase"
                )
            })
            
            self.logger.info(f"Plan de redistribución aplicado exitosamente para {len(plan)} canales")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al aplicar plan de redistribución: {str(e)}")
            return False
    
    def get_redistribution_recommendations(self) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones de redistribución para revisión humana
        
        Returns:
            Lista de recomendaciones
        """
        plan = self.calculate_redistribution_plan()
        recommendations = []
        
        try:
            for channel_id, action_plan in plan.items():
                channel_name = self.config.get("channels", {}).get(channel_id, {}).get("name", channel_id)
                
                recommendation = {
                    "channel_id": channel_id,
                    "channel_name": channel_name,
                    "action": action_plan["action"],
                    "current_budget": action_plan["current_resources"],
                    "recommended_budget": action_plan["new_resources"],
                    "change_amount": action_plan["increase_amount"] if action_plan["action"] == "increase" else -action_plan["reduction_amount"],
                    "change_percentage": (action_plan["new_resources"] / action_plan["current_resources"] - 1) * 100 if action_plan["current_resources"] > 0 else 0,
                    "reason": action_plan["reason"],
                    "metrics": {
                        "roi": self.performance_metrics.get(channel_id, {}).get("roi", 0),
                        "ctr": self.performance_metrics.get(channel_id, {}).get("ctr", 0),
                        "cpm": self.performance_metrics.get(channel_id, {}).get("cpm", 0),
                        "performance_score": self.performance_metrics.get(channel_id, {}).get("performance_score", 0)
                    }
                }
                
                recommendations.append(recommendation)
            
            # Ordenar por cambio porcentual (descendente)
            recommendations.sort(key=lambda x: abs(x["change_percentage"]), reverse=True)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones de redistribución: {str(e)}")
            return []
    
    def get_redistribution_history(self, days: int = 90) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de redistribuciones
        
        Args:
            days: Número de días para el historial
            
        Returns:
            Lista de redistribuciones históricas
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        try:
            # Filtrar por fecha
            recent_history = [
                entry for entry in self.redistribution_history
                if entry.get("date", "") >= cutoff_date
            ]
            
            return recent_history
            
        except Exception as e:
            self.logger.error(f"Error al obtener historial de redistribución: {str(e)}")
            return []
    
    def analyze_redistribution_impact(self, days_before: int = 30, days_after: int = 30) -> Dict[str, Any]:
        """
        Analiza el impacto de las redistribuciones anteriores
        
        Args:
            days_before: Días antes de la redistribución para comparar
            days_after: Días después de la redistribución para comparar
            
        Returns:
            Análisis de impacto
        """
        if not self.redistribution_history:
            self.logger.info("No hay historial de redistribución para analizar")
            return {"success": False, "reason": "No hay historial de redistribución"}
        
        try:
            # Obtener la redistribución más reciente con suficientes datos posteriores
            cutoff_date = (datetime.now() - timedelta(days=days_after)).isoformat()
            
            valid_redistributions = [
                entry for entry in self.redistribution_history
                if entry.get("date", "") <= cutoff_date
            ]
            
            if not valid_redistributions:
                return {"success": False, "reason": "No hay redistribuciones con suficientes datos posteriores"}
            
            # Usar la redistribución más reciente con datos suficientes
            latest_redistribution = sorted(valid_redistributions, key=lambda x: x.get("date", ""))[-1]
            redistribution_date = datetime.fromisoformat(latest_redistribution["date"])
            
            # Definir períodos antes y después
            before_start = (redistribution_date - timedelta(days=days_before)).isoformat()
            before_end = redistribution_date.isoformat()
            after_start = redistribution_date.isoformat()
            after_end = (redistribution_date + timedelta(days=days_after)).isoformat()
            
            # Analizar rendimiento antes y después por canal
            impact_analysis = {
                "redistribution_date": redistribution_date.isoformat(),
                "channels": {},
                "overall": {
                    "before": {"roi": 0, "ctr": 0, "revenue": 0, "cost": 0},
                    "after": {"roi": 0, "ctr": 0, "revenue": 0, "cost": 0},
                    "change": {"roi": 0, "ctr": 0, "revenue": 0, "cost": 0}
                }
            }
            
            # Analizar cada canal afectado
            for channel_id, action_plan in latest_redistribution["plan"].items():
                # Obtener datos del canal
                channel_data = self.analytics.get("channels", {}).get(channel_id, {}).get("performance", [])
                
                # Filtrar datos para períodos antes y después
                before_data = [
                    entry for entry in channel_data
                    if before_start <= entry.get("date", "") <= before_end
                ]
                
                after_data = [
                    entry for entry in channel_data
                    if after_start <= entry.get("date", "") <= after_end
                ]
                
                # Verificar datos suficientes
                if len(before_data) < 5 or len(after_data) < 5:
                    continue
                
                # Calcular métricas para antes
                before_views = sum(entry.get("views", 0) for entry in before_data)
                before_revenue = sum(entry.get("revenue", 0) for entry in before_data)
                before_cost = sum(entry.get("cost", 0) for entry in before_data)
                before_clicks = sum(entry.get("clicks", 0) for entry in before_data)
                
                before_ctr = before_clicks / before_views if before_views > 0 else 0
                before_roi = (before_revenue - before_cost) / before_cost if before_cost > 0 else 0
                
                # Calcular métricas para después
                after_views = sum(entry.get("views", 0) for entry in after_data)
                after_revenue = sum(entry.get("revenue", 0) for entry in after_data)
                after_cost = sum(entry.get("cost", 0) for entry in after_data)
                after_clicks = sum(entry.get("clicks", 0) for entry in after_data)
                
                after_ctr = after_clicks / after_views if after_views > 0 else 0
                after_roi = (after_revenue - after_cost) / after_cost if after_cost > 0 else 0
                
                # Calcular cambios
                roi_change = after_roi - before_roi
                roi_change_percent = (after_roi / before_roi - 1) * 100 if before_roi > 0 else float('inf')
                
                ctr_change = after_ctr - before_ctr
                ctr_change_percent = (after_ctr / before_ctr - 1) * 100 if before_ctr > 0 else float('inf')
                
                revenue_change = after_revenue - before_revenue
                revenue_change_percent = (after_revenue / before_revenue - 1) * 100 if before_revenue > 0 else float('inf')
                
                # Guardar análisis del canal
                impact_analysis["channels"][channel_id] = {
                    "action": action_plan["action"],
                    "before": {
                        "roi": before_roi,
                        "ctr": before_ctr,
                        "revenue": before_revenue,
                        "cost": before_cost
                    },
                    "after": {
                        "roi": after_roi,
                        "ctr": after_ctr,
                        "revenue": after_revenue,
                        "cost": after_cost
                    },
                    "change": {
                        "roi": roi_change,
                        "roi_percent": roi_change_percent,
                        "ctr": ctr_change,
                        "ctr_percent": ctr_change_percent,
                        "revenue": revenue_change,
                        "revenue_percent": revenue_change_percent
                    },
                    "success": (action_plan["action"] == "increase" and roi_change > 0) or 
                              (action_plan["action"] == "reduce" and roi_change >= 0)
                }
                
                # Acumular para análisis general
                impact_analysis["overall"]["before"]["roi"] += before_roi
                impact_analysis["overall"]["before"]["ctr"] += before_ctr
                impact_analysis["overall"]["before"]["revenue"] += before_revenue
                impact_analysis["overall"]["before"]["cost"] += before_cost
                
                impact_analysis["overall"]["after"]["roi"] += after_roi
                impact_analysis["overall"]["after"]["ctr"] += after_ctr
                impact_analysis["overall"]["after"]["revenue"] += after_revenue
                impact_analysis["overall"]["after"]["cost"] += after_cost
            
            # Calcular cambios generales
            channel_count = len(impact_analysis["channels"])
            if channel_count > 0:
                # Promediar ROI y CTR
                impact_analysis["overall"]["before"]["roi"] /= channel_count
                impact_analysis["overall"]["before"]["ctr"] /= channel_count
                impact_analysis["overall"]["after"]["roi"] /= channel_count
                impact_analysis["overall"]["after"]["ctr"] /= channel_count
                
                # Calcular cambios
                impact_analysis["overall"]["change"]["roi"] = (
                    impact_analysis["overall"]["after"]["roi"] - 
                    impact_analysis["overall"]["before"]["roi"]
                )
                
                impact_analysis["overall"]["change"]["ctr"] = (
                    impact_analysis["overall"]["after"]["ctr"] - 
                    impact_analysis["overall"]["before"]["ctr"]
                )
                
                impact_analysis["overall"]["change"]["revenue"] = (
                    impact_analysis["overall"]["after"]["revenue"] - 
                    impact_analysis["overall"]["before"]["revenue"]
                )
                
                impact_analysis["overall"]["change"]["cost"] = (
                    impact_analysis["overall"]["after"]["cost"] - 
                    impact_analysis["overall"]["before"]["cost"]
                )
                
                # Calcular éxito general
                successful_channels = sum(
                    1 for channel_data in impact_analysis["channels"].values()
                    if channel_data.get("success", False)
                )
                
                impact_analysis["overall"]["success_rate"] = successful_channels / channel_count
                impact_analysis["overall"]["roi_improvement"] = impact_analysis["overall"]["change"]["roi"] > 0
                impact_analysis["overall"]["revenue_improvement"] = impact_analysis["overall"]["change"]["revenue"] > 0
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error al analizar impacto de redistribución: {str(e)}")
            return {"success": False, "reason": str(e)}
    
    def run_redistribution_cycle(self, apply_automatically: bool = False) -> Dict[str, Any]:
        """
        Ejecuta un ciclo completo de redistribución
        
        Args:
            apply_automatically: Si es True, aplica el plan automáticamente
            
        Returns:
            Resultado del ciclo de redistribución
        """
        try:
            # 1. Analizar rendimiento
            self.analyze_channel_performance()
            
            # 2. Calcular plan de redistribución
            redistribution_plan = self.calculate_redistribution_plan()
            
            # 3. Generar recomendaciones
            recommendations = self.get_redistribution_recommendations()
            
            # 4. Aplicar plan si es automático
            applied = False
            if apply_automatically and redistribution_plan:
                applied = self.apply_redistribution_plan(redistribution_plan)
            
            # 5. Analizar impacto de redistribuciones anteriores
            impact_analysis = self.analyze_redistribution_impact()
            
            # 6. Preparar resultado
            result = {
                "date": datetime.now().isoformat(),
                "channels_analyzed": len(self.performance_metrics),
                "redistribution_plan": redistribution_plan,
                "recommendations": recommendations,
                "applied": applied,
                "previous_impact": impact_analysis
            }
            
            self.logger.info(f"Ciclo de redistribución completado: {len(redistribution_plan)} canales afectados")
            return result
            
        except Exception as e:
            self.logger.error(f"Error en ciclo de redistribución: {str(e)}")
            return {"success": False, "reason": str(e)}