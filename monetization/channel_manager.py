"""
Módulo de gestión de canales de monetización.

Este módulo gestiona los diferentes canales de monetización y sus configuraciones,
permitiendo optimizar la distribución de contenido entre plataformas.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'monetization.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('channel_manager')

class ChannelManager:
    """
    Gestor de canales para optimizar la distribución y monetización de contenido.
    """
    
    def __init__(self, config_path: str = 'config/channels.json'):
        """
        Inicializa el gestor de canales.
        
        Args:
            config_path: Ruta al archivo de configuración de canales
        """
        self.config_path = config_path
        self.load_config()
        
        # Canales registrados
        self.channels = {}
        
        # Métricas de canales
        self.channel_metrics = defaultdict(lambda: defaultdict(float))
        
        # Historial de rendimiento
        self.performance_history = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Cargar datos si existen
        self.load_channel_data()
    
    def load_config(self) -> None:
        """Carga la configuración de canales desde el archivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info("Configuración de canales cargada correctamente")
            else:
                # Configuración por defecto
                self.config = {
                    "platforms": {
                        "youtube": {
                            "monetization_options": ["adsense", "memberships", "sponsorships", "affiliate"],
                            "content_types": ["video", "shorts", "live"],
                            "audience_reach": 0.8,
                            "monetization_potential": 0.9,
                            "growth_rate": 0.05
                        },
                        "tiktok": {
                            "monetization_options": ["creator_fund", "affiliate", "sponsorships"],
                            "content_types": ["short_video", "live"],
                            "audience_reach": 0.9,
                            "monetization_potential": 0.6,
                            "growth_rate": 0.08
                        },
                        "instagram": {
                            "monetization_options": ["bonuses", "affiliate", "sponsorships"],
                            "content_types": ["reels", "posts", "stories", "live"],
                            "audience_reach": 0.85,
                            "monetization_potential": 0.7,
                            "growth_rate": 0.06
                        },
                        "twitter": {
                            "monetization_options": ["super_follows", "affiliate", "sponsorships"],
                            "content_types": ["tweets", "spaces", "fleets"],
                            "audience_reach": 0.6,
                            "monetization_potential": 0.5,
                            "growth_rate": 0.03
                        },
                        "twitch": {
                            "monetization_options": ["subscriptions", "bits", "affiliate", "sponsorships"],
                            "content_types": ["live", "clips"],
                            "audience_reach": 0.4,
                            "monetization_potential": 0.8,
                            "growth_rate": 0.04
                        },
                        "website": {
                            "monetization_options": ["adsense", "affiliate", "products", "memberships"],
                            "content_types": ["blog", "courses", "downloads"],
                            "audience_reach": 0.3,
                            "monetization_potential": 1.0,
                            "growth_rate": 0.02
                        }
                    },
                    "content_types": {
                        "video": {
                            "optimal_platforms": ["youtube", "website"],
                            "monetization_potential": 0.9,
                            "production_cost": 0.8,
                            "audience_engagement": 0.8
                        },
                        "short_video": {
                            "optimal_platforms": ["tiktok", "instagram", "youtube"],
                            "monetization_potential": 0.6,
                            "production_cost": 0.4,
                            "audience_engagement": 0.9
                        },
                        "live": {
                            "optimal_platforms": ["twitch", "youtube", "instagram"],
                            "monetization_potential": 0.8,
                            "production_cost": 0.7,
                            "audience_engagement": 0.9
                        },
                        "blog": {
                            "optimal_platforms": ["website", "twitter"],
                            "monetization_potential": 0.7,
                            "production_cost": 0.3,
                            "audience_engagement": 0.6
                        },
                        "podcast": {
                            "optimal_platforms": ["youtube", "website", "twitter"],
                            "monetization_potential": 0.7,
                            "production_cost": 0.5,
                            "audience_engagement": 0.7
                        }
                    },
                    "audience_demographics": {
                        "gen_z": {
                            "platforms": ["tiktok", "instagram", "youtube"],
                            "content_preferences": ["short_video", "live", "podcast"]
                        },
                        "millennials": {
                            "platforms": ["instagram", "youtube", "twitter"],
                            "content_preferences": ["video", "blog", "podcast"]
                        },
                        "gen_x": {
                            "platforms": ["youtube", "website", "twitter"],
                            "content_preferences": ["video", "blog", "podcast"]
                        },
                        "boomers": {
                            "platforms": ["youtube", "website", "facebook"],
                            "content_preferences": ["video", "blog"]
                        }
                    },
                    "cross_promotion": {
                        "effectiveness": 0.7,
                        "optimal_frequency": 3,
                        "platform_pairs": [
                            ["youtube", "instagram"],
                            ["tiktok", "instagram"],
                            ["youtube", "twitter"],
                            ["website", "youtube"]
                        ]
                    }
                }
                
                # Guardar configuración por defecto
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
                logger.info("Creada configuración de canales por defecto")
        except Exception as e:
            logger.error(f"Error al cargar la configuración de canales: {e}")
            self.config = {"platforms": {}, "content_types": {}, "audience_demographics": {}, "cross_promotion": {}}
    
    def load_channel_data(self) -> None:
        """Carga los datos de canales si existen."""
        channel_data_path = 'data/channels.json'
        try:
            if os.path.exists(channel_data_path):
                with open(channel_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Cargar canales
                    if 'channels' in data:
                        self.channels = data['channels']
                    
                    # Cargar métricas
                    if 'metrics' in data:
                        for channel_id, metrics in data['metrics'].items():
                            for metric, value in metrics.items():
                                self.channel_metrics[channel_id][metric] = value
                    
                    # Cargar historial
                    if 'history' in data:
                        for channel_id, dates in data['history'].items():
                            for date_str, metrics in dates.items():
                                for metric, value in metrics.items():
                                    self.performance_history[channel_id][date_str][metric] = value
                    
                logger.info("Datos de canales cargados correctamente")
        except Exception as e:
            logger.error(f"Error al cargar datos de canales: {e}")
    
    def save_channel_data(self) -> None:
        """Guarda los datos de canales."""
        channel_data_path = 'data/channels.json'
        try:
            os.makedirs(os.path.dirname(channel_data_path), exist_ok=True)
            
            # Preparar datos para guardar
            data = {
                'channels': self.channels,
                'metrics': {k: dict(v) for k, v in self.channel_metrics.items()},
                'history': {
                    k: {
                        date: dict(metrics) for date, metrics in dates.items()
                    } for k, dates in self.performance_history.items()
                },
                'last_update': datetime.now().isoformat()
            }
            
            with open(channel_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Datos de canales guardados correctamente")
        except Exception as e:
            logger.error(f"Error al guardar datos de canales: {e}")
    
    def register_channel(self, channel_id: str, platform: str, name: str, 
                        url: str, niche: str, target_audience: str) -> Dict[str, Any]:
        """
        Registra un nuevo canal.
        
        Args:
            channel_id: Identificador único del canal
            platform: Plataforma del canal (youtube, tiktok, etc.)
            name: Nombre del canal
            url: URL del canal
            niche: Nicho del canal
            target_audience: Audiencia objetivo
            
        Returns:
            Datos del canal registrado
        """
        # Verificar si la plataforma está soportada
        if platform not in self.config.get('platforms', {}):
            logger.warning(f"Plataforma no soportada: {platform}")
            platform = "other"
        
        # Crear canal
        channel = {
            'channel_id': channel_id,
            'platform': platform,
            'name': name,
            'url': url,
            'niche': niche,
            'target_audience': target_audience,
            'monetization_options': self.config.get('platforms', {}).get(platform, {}).get('monetization_options', []),
            'content_types': self.config.get('platforms', {}).get(platform, {}).get('content_types', []),
                        'created_at': datetime.now().isoformat(),
            'status': 'active',
            'metrics': {
                'subscribers': 0,
                'views': 0,
                'engagement_rate': 0.0,
                'revenue': 0.0
            }
        }
        
        # Guardar canal
        self.channels[channel_id] = channel
        self.save_channel_data()
        
        logger.info(f"Canal registrado: {name} ({platform})")
        
        return channel
    
    def update_channel_metrics(self, channel_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualiza las métricas de un canal.
        
        Args:
            channel_id: Identificador del canal
            metrics: Diccionario con métricas a actualizar
            
        Returns:
            Métricas actualizadas
        """
        if channel_id not in self.channels:
            logger.warning(f"Canal no encontrado: {channel_id}")
            return {}
        
        # Actualizar métricas
        for metric, value in metrics.items():
            self.channel_metrics[channel_id][metric] = value
            
            # Actualizar métricas en el canal
            if 'metrics' in self.channels[channel_id] and metric in self.channels[channel_id]['metrics']:
                self.channels[channel_id]['metrics'][metric] = value
        
        # Registrar en historial
        today = datetime.now().strftime('%Y-%m-%d')
        for metric, value in metrics.items():
            self.performance_history[channel_id][today][metric] = value
        
        # Guardar datos
        self.save_channel_data()
        
        logger.info(f"Métricas actualizadas para canal: {channel_id}")
        
        return dict(self.channel_metrics[channel_id])
    
    def get_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un canal.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Datos del canal o None si no existe
        """
        return self.channels.get(channel_id)
    
    def get_all_channels(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene todos los canales registrados.
        
        Returns:
            Diccionario con todos los canales
        """
        return self.channels
    
    def get_channel_metrics(self, channel_id: str) -> Dict[str, float]:
        """
        Obtiene las métricas de un canal.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Métricas del canal
        """
        return dict(self.channel_metrics.get(channel_id, {}))
    
    def get_channel_performance_history(self, channel_id: str, days: int = 30) -> Dict[str, Dict[str, float]]:
        """
        Obtiene el historial de rendimiento de un canal.
        
        Args:
            channel_id: Identificador del canal
            days: Número de días para obtener el historial
            
        Returns:
            Historial de rendimiento
        """
        if channel_id not in self.performance_history:
            return {}
        
        # Filtrar por fecha
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        filtered_history = {}
        for date_str, metrics in self.performance_history[channel_id].items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            if start_date <= date <= end_date:
                filtered_history[date_str] = dict(metrics)
        
        return filtered_history
    
    def optimize_channel_distribution(self, content_type: str, target_audience: str) -> Dict[str, Any]:
        """
        Optimiza la distribución de contenido entre canales.
        
        Args:
            content_type: Tipo de contenido a distribuir
            target_audience: Audiencia objetivo
            
        Returns:
            Recomendaciones de distribución
        """
        # Obtener plataformas óptimas para el tipo de contenido
        content_config = self.config.get('content_types', {}).get(content_type, {})
        optimal_platforms = content_config.get('optimal_platforms', [])
        
        # Obtener plataformas preferidas por la audiencia
        audience_config = self.config.get('audience_demographics', {}).get(target_audience, {})
        audience_platforms = audience_config.get('platforms', [])
        
        # Intersección de plataformas óptimas y preferidas
        recommended_platforms = [p for p in optimal_platforms if p in audience_platforms]
        
        # Si no hay intersección, usar plataformas óptimas
        if not recommended_platforms:
            recommended_platforms = optimal_platforms
        
        # Calcular puntuación para cada plataforma
        platform_scores = {}
        for platform in recommended_platforms:
            platform_config = self.config.get('platforms', {}).get(platform, {})
            
            # Factores de puntuación
            audience_reach = platform_config.get('audience_reach', 0.5)
            monetization_potential = platform_config.get('monetization_potential', 0.5)
            growth_rate = platform_config.get('growth_rate', 0.03)
            
            # Calcular puntuación
            score = (audience_reach * 0.4) + (monetization_potential * 0.4) + (growth_rate * 0.2)
            platform_scores[platform] = score
        
        # Ordenar plataformas por puntuación
        sorted_platforms = sorted(platform_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Crear recomendaciones
        recommendations = {
            'content_type': content_type,
            'target_audience': target_audience,
            'primary_platform': sorted_platforms[0][0] if sorted_platforms else None,
            'secondary_platforms': [p[0] for p in sorted_platforms[1:3]] if len(sorted_platforms) > 1 else [],
            'platform_scores': {p: round(s, 2) for p, s in sorted_platforms},
            'distribution_strategy': self._generate_distribution_strategy(sorted_platforms),
            'cross_promotion': self._generate_cross_promotion(sorted_platforms)
        }
        
        return recommendations
    
    def _generate_distribution_strategy(self, sorted_platforms: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Genera una estrategia de distribución de esfuerzo entre plataformas.
        
        Args:
            sorted_platforms: Lista de plataformas ordenadas por puntuación
            
        Returns:
            Estrategia de distribución
        """
        # Si no hay plataformas, retornar vacío
        if not sorted_platforms:
            return {}
        
        # Normalizar puntuaciones
        total_score = sum(score for _, score in sorted_platforms)
        
        if total_score == 0:
            # Distribución equitativa si todas las puntuaciones son 0
            return {platform: round(1.0 / len(sorted_platforms), 2) for platform, _ in sorted_platforms}
        
        # Calcular distribución basada en puntuaciones
        distribution = {}
        for platform, score in sorted_platforms:
            distribution[platform] = round((score / total_score), 2)
        
        return distribution
    
    def _generate_cross_promotion(self, sorted_platforms: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones de promoción cruzada entre plataformas.
        
        Args:
            sorted_platforms: Lista de plataformas ordenadas por puntuación
            
        Returns:
            Lista de recomendaciones de promoción cruzada
        """
        cross_promotion = []
        
        # Obtener pares de plataformas para promoción cruzada
        platform_pairs = self.config.get('cross_promotion', {}).get('platform_pairs', [])
        
        # Plataformas disponibles
        available_platforms = [p for p, _ in sorted_platforms]
        
        # Verificar pares válidos
        for pair in platform_pairs:
            if len(pair) >= 2 and pair[0] in available_platforms and pair[1] in available_platforms:
                # Crear recomendación
                recommendation = {
                    'from_platform': pair[0],
                    'to_platform': pair[1],
                    'frequency': self.config.get('cross_promotion', {}).get('optimal_frequency', 3),
                    'effectiveness': self.config.get('cross_promotion', {}).get('effectiveness', 0.7)
                }
                
                cross_promotion.append(recommendation)
        
        return cross_promotion
    
    def analyze_channel_performance(self, channel_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Analiza el rendimiento de un canal.
        
        Args:
            channel_id: Identificador del canal
            days: Número de días para el análisis
            
        Returns:
            Análisis de rendimiento
        """
        if channel_id not in self.channels:
            logger.warning(f"Canal no encontrado: {channel_id}")
            return {
                'channel_id': channel_id,
                'status': 'not_found',
                'message': 'Canal no encontrado'
            }
        
        # Obtener datos del canal
        channel = self.channels[channel_id]
        
        # Obtener historial de rendimiento
        history = self.get_channel_performance_history(channel_id, days)
        
        # Métricas actuales
        current_metrics = self.get_channel_metrics(channel_id)
        
        # Calcular tendencias
        trends = self._calculate_trends(history)
        
        # Calcular KPIs
        kpis = self._calculate_kpis(channel, current_metrics, history)
        
        # Crear análisis
        analysis = {
            'channel_id': channel_id,
            'platform': channel.get('platform'),
            'name': channel.get('name'),
            'current_metrics': current_metrics,
            'trends': trends,
            'kpis': kpis,
            'recommendations': self._generate_performance_recommendations(channel, trends, kpis),
            'period': {
                'days': days,
                'start_date': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d')
            }
        }
        
        return analysis
    
    def _calculate_trends(self, history: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """
        Calcula tendencias a partir del historial de rendimiento.
        
        Args:
            history: Historial de rendimiento
            
        Returns:
            Tendencias calculadas
        """
        trends = {}
        
        # Si no hay suficientes datos, retornar vacío
        if len(history) < 2:
            return trends
        
        # Ordenar fechas
        dates = sorted(history.keys())
        
        # Obtener métricas disponibles
        metrics = set()
        for date_metrics in history.values():
            metrics.update(date_metrics.keys())
        
        # Calcular tendencia para cada métrica
        for metric in metrics:
            # Recopilar valores
            values = []
            for date in dates:
                if metric in history[date]:
                    values.append(history[date][metric])
            
            if len(values) < 2:
                continue
            
            # Calcular cambio porcentual
            first_value = values[0]
            last_value = values[-1]
            
            if first_value > 0:
                percent_change = ((last_value - first_value) / first_value) * 100
            else:
                percent_change = 0.0
            
            # Calcular tendencia (pendiente)
            x = np.arange(len(values))
            if len(x) > 1:
                slope, _ = np.polyfit(x, values, 1)
            else:
                slope = 0.0
            
            trend_direction = "up" if slope > 0 else "down" if slope < 0 else "stable"
            
            # Guardar tendencia
            trends[metric] = {
                'first_value': first_value,
                'last_value': last_value,
                'percent_change': round(percent_change, 2),
                'trend_direction': trend_direction,
                'slope': slope
            }
        
        return trends
    
    def _calculate_kpis(self, channel: Dict[str, Any], current_metrics: Dict[str, float], 
                       history: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Calcula KPIs para un canal.
        
        Args:
            channel: Datos del canal
            current_metrics: Métricas actuales
            history: Historial de rendimiento
            
        Returns:
            KPIs calculados
        """
        kpis = {}
        
        # Engagement rate
        if 'engagement' in current_metrics and 'views' in current_metrics and current_metrics['views'] > 0:
            kpis['engagement_rate'] = round((current_metrics['engagement'] / current_metrics['views']) * 100, 2)
        else:
            kpis['engagement_rate'] = 0.0
        
        # Revenue per view
        if 'revenue' in current_metrics and 'views' in current_metrics and current_metrics['views'] > 0:
            kpis['revenue_per_view'] = round(current_metrics['revenue'] / current_metrics['views'], 4)
        else:
            kpis['revenue_per_view'] = 0.0
        
        # Revenue per subscriber
        if 'revenue' in current_metrics and 'subscribers' in current_metrics and current_metrics['subscribers'] > 0:
            kpis['revenue_per_subscriber'] = round(current_metrics['revenue'] / current_metrics['subscribers'], 4)
        else:
            kpis['revenue_per_subscriber'] = 0.0
        
        # Growth rate (subscribers)
        if 'subscribers' in current_metrics and len(history) > 1:
            dates = sorted(history.keys())
            first_date = dates[0]
            last_date = dates[-1]
            
            if 'subscribers' in history.get(first_date, {}) and 'subscribers' in history.get(last_date, {}):
                first_subs = history[first_date]['subscribers']
                last_subs = history[last_date]['subscribers']
                
                if first_subs > 0:
                    days_diff = (datetime.strptime(last_date, '%Y-%m-%d') - datetime.strptime(first_date, '%Y-%m-%d')).days
                    if days_diff > 0:
                        growth_rate = ((last_subs / first_subs) ** (30 / days_diff)) - 1
                        kpis['monthly_growth_rate'] = round(growth_rate * 100, 2)
                    else:
                        kpis['monthly_growth_rate'] = 0.0
                else:
                    kpis['monthly_growth_rate'] = 0.0
            else:
                kpis['monthly_growth_rate'] = 0.0
        else:
            kpis['monthly_growth_rate'] = 0.0
        
        # Monetization efficiency
        platform = channel.get('platform')
        platform_config = self.config.get('platforms', {}).get(platform, {})
        monetization_potential = platform_config.get('monetization_potential', 0.5)
        
        if 'revenue' in current_metrics and monetization_potential > 0:
            # Normalizar ingresos (asumiendo que 1000 es un buen ingreso mensual)
            normalized_revenue = min(current_metrics.get('revenue', 0) / 1000, 1.0)
            kpis['monetization_efficiency'] = round((normalized_revenue / monetization_potential) * 100, 2)
        else:
            kpis['monetization_efficiency'] = 0.0
        
        return kpis
    
    def _generate_performance_recommendations(self, channel: Dict[str, Any], 
                                            trends: Dict[str, Dict[str, Any]], 
                                            kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones basadas en el rendimiento del canal.
        
        Args:
            channel: Datos del canal
            trends: Tendencias calculadas
            kpis: KPIs calculados
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Recomendación basada en engagement
        engagement_rate = kpis.get('engagement_rate', 0.0)
        if engagement_rate < 2.0:
            recommendations.append({
                'type': 'engagement',
                'priority': 'high',
                'message': 'El engagement rate es bajo. Considera mejorar la calidad del contenido y fomentar la interacción.',
                'actions': [
                    'Incluir llamadas a la acción en el contenido',
                    'Responder a comentarios de forma regular',
                    'Crear contenido más interactivo'
                ]
            })
        
        # Recomendación basada en crecimiento
        growth_rate = kpis.get('monthly_growth_rate', 0.0)
        if growth_rate < 5.0:
            recommendations.append({
                'type': 'growth',
                'priority': 'medium',
                'message': 'La tasa de crecimiento es baja. Considera estrategias para aumentar la visibilidad.',
                'actions': [
                    'Optimizar SEO y palabras clave',
                    'Aumentar frecuencia de publicación',
                    'Colaborar con otros creadores'
                ]
            })
        
        # Recomendación basada en monetización
        monetization_efficiency = kpis.get('monetization_efficiency', 0.0)
        if monetization_efficiency < 30.0:
            recommendations.append({
                'type': 'monetization',
                'priority': 'high',
                'message': 'La eficiencia de monetización es baja. Considera diversificar fuentes de ingresos.',
                'actions': [
                    'Explorar opciones de afiliados',
                    'Considerar membresías o productos propios',
                    'Optimizar CTAs para conversiones'
                ]
            })
        
        # Recomendación basada en tendencias de vistas
        if 'views' in trends and trends['views'].get('trend_direction') == 'down':
            recommendations.append({
                'type': 'content',
                'priority': 'high',
                'message': 'Las vistas están disminuyendo. Considera revisar la estrategia de contenido.',
                'actions': [
                    'Analizar qué contenido ha funcionado mejor',
                    'Experimentar con nuevos formatos',
                    'Revisar horarios de publicación'
                ]
            })
        
        return recommendations
    
    def get_monetization_opportunities(self, channel_id: str) -> Dict[str, Any]:
        """
        Identifica oportunidades de monetización para un canal.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Oportunidades de monetización
        """
        if channel_id not in self.channels:
            logger.warning(f"Canal no encontrado: {channel_id}")
            return {
                'channel_id': channel_id,
                'status': 'not_found',
                'message': 'Canal no encontrado'
            }
        
        # Obtener datos del canal
        channel = self.channels[channel_id]
        platform = channel.get('platform')
        
        # Obtener métricas
        metrics = self.get_channel_metrics(channel_id)
        subscribers = metrics.get('subscribers', 0)
        
        # Obtener opciones de monetización disponibles para la plataforma
        platform_config = self.config.get('platforms', {}).get(platform, {})
        available_options = platform_config.get('monetization_options', [])
        
        # Opciones de monetización actuales
        current_options = channel.get('monetization_options', [])
        
        # Identificar oportunidades
        opportunities = []
        
        # Verificar cada opción disponible
        for option in available_options:
            if option in current_options:
                # Ya está implementada
                status = 'active'
                message = 'Ya implementado'
            else:
                # Nueva oportunidad
                status = 'opportunity'
                message = 'Nueva oportunidad de monetización'
            
            # Determinar requisitos y potencial
            requirements = {}
            potential = 0.0
            
            if option == 'adsense':
                requirements = {
                    'subscribers': 1000,
                    'watch_hours': 4000
                }
                potential = min(subscribers / 10000, 1.0) * 100
            
            elif option == 'memberships':
                requirements = {
                    'subscribers': 30000,
                    'age': 18
                }
                potential = min(subscribers / 50000, 1.0) * 100
            
            elif option == 'sponsorships':
                requirements = {
                    'subscribers': 5000,
                    'engagement_rate': 3.0
                }
                potential = min(subscribers / 20000, 1.0) * 100
            
            elif option == 'affiliate':
                requirements = {
                    'subscribers': 1000,
                    'engagement_rate': 2.0
                }
                potential = min(subscribers / 5000, 1.0) * 100
            
            elif option == 'products':
                requirements = {
                    'subscribers': 5000,
                    'engagement_rate': 3.0
                }
                potential = min(subscribers / 15000, 1.0) * 100
            
            elif option == 'creator_fund':
                requirements = {
                    'followers': 10000,
                    'views': 100000,
                    'age': 18
                }
                potential = min(subscribers / 20000, 1.0) * 100
            
            elif option == 'bonuses':
                requirements = {
                    'followers': 5000,
                    'reels': 5
                }
                potential = min(subscribers / 10000, 1.0) * 100
            
            # Verificar elegibilidad
            eligible = True
            for req_key, req_value in requirements.items():
                if req_key in metrics and metrics[req_key] < req_value:
                    eligible = False
                    break
            
            # Crear oportunidad
            opportunity = {
                'option': option,
                'status': status,
                'message': message,
                'requirements': requirements,
                'eligible': eligible,
                'potential': round(potential, 2),
                'priority': 'high' if potential > 70 else 'medium' if potential > 40 else 'low'
            }
            
            opportunities.append(opportunity)
        
        # Ordenar oportunidades por potencial
        opportunities.sort(key=lambda x: x['potential'], reverse=True)
        
        # Crear resultado
        result = {
            'channel_id': channel_id,
            'platform': platform,
            'name': channel.get('name'),
            'current_options': current_options,
            'opportunities': opportunities,
            'recommended_next': next((o['option'] for o in opportunities if o['status'] == 'opportunity' and o['eligible']), None),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def compare_channels(self, channel_ids: List[str]) -> Dict[str, Any]:
        """
        Compara el rendimiento de múltiples canales.
        
        Args:
            channel_ids: Lista de identificadores de canales
            
        Returns:
            Comparación de canales
        """
        # Verificar canales válidos
        valid_channels = [cid for cid in channel_ids if cid in self.channels]
        
        if not valid_channels:
            logger.warning("No se encontraron canales válidos para comparar")
            return {
                'status': 'error',
                'message': 'No se encontraron canales válidos para comparar'
            }
        
        # Recopilar datos de canales
        channels_data = []
        
        for channel_id in valid_channels:
            channel = self.channels[channel_id]
            metrics = self.get_channel_metrics(channel_id)
            
            # Calcular KPIs
            kpis = {}
            
            # Engagement rate
            if 'engagement' in metrics and 'views' in metrics and metrics['views'] > 0:
                kpis['engagement_rate'] = round((metrics['engagement'] / metrics['views']) * 100, 2)
            else:
                kpis['engagement_rate'] = 0.0
            
            # Revenue per view
            if 'revenue' in metrics and 'views' in metrics and metrics['views'] > 0:
                kpis['revenue_per_view'] = round(metrics['revenue'] / metrics['views'], 4)
            else:
                kpis['revenue_per_view'] = 0.0
            
            # Revenue per subscriber
            if 'revenue' in metrics and 'subscribers' in metrics and metrics['subscribers'] > 0:
                kpis['revenue_per_subscriber'] = round(metrics['revenue'] / metrics['subscribers'], 4)
            else:
                kpis['revenue_per_subscriber'] = 0.0
            
            # Crear datos del canal
            channel_data = {
                'channel_id': channel_id,
                'name': channel.get('name'),
                'platform': channel.get('platform'),
                'metrics': metrics,
                'kpis': kpis
            }
            
            channels_data.append(channel_data)
        
        # Comparar métricas
        comparison = {}
        
        # Métricas a comparar
        metrics_to_compare = ['subscribers', 'views', 'engagement', 'revenue']
        kpis_to_compare = ['engagement_rate', 'revenue_per_view', 'revenue_per_subscriber']
        
        # Comparar cada métrica
        for metric in metrics_to_compare:
            values = [(c['channel_id'], c['metrics'].get(metric, 0)) for c in channels_data]
            values.sort(key=lambda x: x[1], reverse=True)
            
            comparison[metric] = {
                'best': {
                    'channel_id': values[0][0] if values else None,
                    'value': values[0][1] if values else 0
                },
                'worst': {
                    'channel_id': values[-1][0] if values else None,
                    'value': values[-1][1] if values else 0
                },
                'average': sum(v[1] for v in values) / len(values) if values else 0,
                'values': {v[0]: v[1] for v in values}
            }
        
        # Comparar cada KPI
        for kpi in kpis_to_compare:
            values = [(c['channel_id'], c['kpis'].get(kpi, 0)) for c in channels_data]
            values.sort(key=lambda x: x[1], reverse=True)
            
            comparison[kpi] = {
                'best': {
                    'channel_id': values[0][0] if values else None,
                    'value': values[0][1] if values else 0
                },
                'worst': {
                    'channel_id': values[-1][0] if values else None,
                    'value': values[-1][1] if values else 0
                },
                'average': sum(v[1] for v in values) / len(values) if values else 0,
                'values': {v[0]: v[1] for v in values}
            }
        
        # Crear resultado
        result = {
            'channels': channels_data,
            'comparison': comparison,
            'best_overall': self._determine_best_overall(comparison),
            'recommendations': self._generate_comparison_recommendations(channels_data, comparison),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _determine_best_overall(self, comparison: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Determina el mejor canal en general basado en la comparación.
        
        Args:
            comparison: Comparación de métricas y KPIs
            
        Returns:
            Identificador del mejor canal o None
        """
        # Contar cuántas veces cada canal es el mejor
        best_counts = defaultdict(int)
        
        for metric, data in comparison.items():
            best_channel = data.get('best', {}).get('channel_id')
            if best_channel:
                best_counts[best_channel] += 1
        
        # Encontrar el canal con más "mejores"
        if best_counts:
            return max(best_counts.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _generate_comparison_recommendations(self, channels_data: List[Dict[str, Any]], 
                                           comparison: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones basadas en la comparación de canales.
        
        Args:
            channels_data: Datos de los canales
            comparison: Comparación de métricas y KPIs
            
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Recomendación para el canal con peor engagement
        worst_engagement = comparison.get('engagement_rate', {}).get('worst', {}).get('channel_id')
        if worst_engagement:
            worst_platform = next((c['platform'] for c in channels_data if c['channel_id'] == worst_engagement), None)
            
            recommendations.append({
                'channel_id': worst_engagement,
                'type': 'engagement',
                'priority': 'high',
                'message': f'Este canal tiene el peor engagement rate. Considera mejorar la interacción con la audiencia.',
                'actions': [
                    'Incluir más llamadas a la acción',
                    'Responder a comentarios de forma regular',
                    f'Estudiar el formato de contenido más exitoso en {worst_platform}'
                ]
            })
        
        # Recomendación para el canal con peor monetización
        worst_revenue = comparison.get('revenue_per_subscriber', {}).get('worst', {}).get('channel_id')
        if worst_revenue:
            recommendations.append({
                'channel_id': worst_revenue,
                'type': 'monetization',
                'priority': 'high',
                                'message': 'Este canal tiene el peor ingreso por suscriptor. Considera optimizar la estrategia de monetización.',
                'actions': [
                    'Revisar y activar fuentes de ingresos adicionales',
                    'Mejorar CTAs para conversiones',
                    'Crear contenido orientado a productos o servicios'
                ]
            })
        
        # Recomendación para el canal con menos suscriptores
        worst_subscribers = comparison.get('subscribers', {}).get('worst', {}).get('channel_id')
        if worst_subscribers:
            recommendations.append({
                'channel_id': worst_subscribers,
                'type': 'growth',
                'priority': 'medium',
                'message': 'Este canal tiene el menor número de suscriptores. Considera estrategias de crecimiento.',
                'actions': [
                    'Aumentar frecuencia de publicación',
                    'Mejorar SEO y palabras clave',
                    'Promocionar el canal en otras plataformas'
                ]
            })
        
        # Recomendación para aprender del mejor canal
        best_overall = self._determine_best_overall(comparison)
        if best_overall:
            best_platform = next((c['platform'] for c in channels_data if c['channel_id'] == best_overall), None)
            
            # Excluir el mejor canal de las recomendaciones
            for channel_data in channels_data:
                if channel_data['channel_id'] != best_overall:
                    recommendations.append({
                        'channel_id': channel_data['channel_id'],
                        'type': 'learning',
                        'priority': 'medium',
                        'message': f'Aprende de las estrategias del canal con mejor rendimiento ({best_platform}).',
                        'actions': [
                            'Analizar tipo de contenido y formato',
                            'Estudiar frecuencia y horarios de publicación',
                            'Revisar estrategias de interacción con la audiencia'
                        ]
                    })
        
        return recommendations
    
    def update_channel_status(self, channel_id: str, status: str) -> Dict[str, Any]:
        """
        Actualiza el estado de un canal.
        
        Args:
            channel_id: Identificador del canal
            status: Nuevo estado (active, inactive, archived)
            
        Returns:
            Datos del canal actualizado
        """
        if channel_id not in self.channels:
            logger.warning(f"Canal no encontrado: {channel_id}")
            return {
                'status': 'error',
                'message': 'Canal no encontrado'
            }
        
        # Validar estado
        valid_statuses = ['active', 'inactive', 'archived']
        if status not in valid_statuses:
            logger.warning(f"Estado no válido: {status}")
            return {
                'status': 'error',
                'message': f'Estado no válido. Debe ser uno de: {", ".join(valid_statuses)}'
            }
        
        # Actualizar estado
        self.channels[channel_id]['status'] = status
        self.channels[channel_id]['updated_at'] = datetime.now().isoformat()
        
        # Guardar datos
        self.save_channel_data()
        
        logger.info(f"Estado del canal {channel_id} actualizado a: {status}")
        
        return self.channels[channel_id]
    
    def delete_channel(self, channel_id: str) -> Dict[str, Any]:
        """
        Elimina un canal.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Resultado de la operación
        """
        if channel_id not in self.channels:
            logger.warning(f"Canal no encontrado: {channel_id}")
            return {
                'status': 'error',
                'message': 'Canal no encontrado'
            }
        
        # Eliminar canal
        channel = self.channels.pop(channel_id)
        
        # Eliminar métricas
        if channel_id in self.channel_metrics:
            del self.channel_metrics[channel_id]
        
        # Eliminar historial
        if channel_id in self.performance_history:
            del self.performance_history[channel_id]
        
        # Guardar datos
        self.save_channel_data()
        
        logger.info(f"Canal eliminado: {channel_id}")
        
        return {
            'status': 'success',
            'message': f'Canal {channel["name"]} eliminado correctamente',
            'channel_id': channel_id
        }
    
    def get_channel_content_recommendations(self, channel_id: str) -> Dict[str, Any]:
        """
        Genera recomendaciones de contenido para un canal.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Recomendaciones de contenido
        """
        if channel_id not in self.channels:
            logger.warning(f"Canal no encontrado: {channel_id}")
            return {
                'status': 'error',
                'message': 'Canal no encontrado'
            }
        
        # Obtener datos del canal
        channel = self.channels[channel_id]
        platform = channel.get('platform')
        niche = channel.get('niche')
        target_audience = channel.get('target_audience')
        
        # Obtener tipos de contenido disponibles para la plataforma
        platform_config = self.config.get('platforms', {}).get(platform, {})
        available_content_types = platform_config.get('content_types', [])
        
        # Obtener preferencias de contenido para la audiencia objetivo
        audience_config = self.config.get('audience_demographics', {}).get(target_audience, {})
        audience_preferences = audience_config.get('content_preferences', [])
        
        # Intersección de tipos disponibles y preferencias
        recommended_types = [ct for ct in available_content_types if ct in audience_preferences]
        
        # Si no hay intersección, usar tipos disponibles
        if not recommended_types:
            recommended_types = available_content_types
        
        # Calcular puntuación para cada tipo de contenido
        content_scores = {}
        for content_type in recommended_types:
            content_config = self.config.get('content_types', {}).get(content_type, {})
            
            # Factores de puntuación
            monetization_potential = content_config.get('monetization_potential', 0.5)
            audience_engagement = content_config.get('audience_engagement', 0.5)
            production_cost = content_config.get('production_cost', 0.5)
            
            # Calcular puntuación (mayor monetización y engagement, menor costo)
            score = (monetization_potential * 0.4) + (audience_engagement * 0.4) - (production_cost * 0.2)
            content_scores[content_type] = score
        
        # Ordenar tipos de contenido por puntuación
        sorted_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Crear recomendaciones
        recommendations = {
            'channel_id': channel_id,
            'platform': platform,
            'name': channel.get('name'),
            'niche': niche,
            'target_audience': target_audience,
            'recommended_content_types': [
                {
                    'type': content_type,
                    'score': round(score, 2),
                    'monetization_potential': self.config.get('content_types', {}).get(content_type, {}).get('monetization_potential', 0.5),
                    'audience_engagement': self.config.get('content_types', {}).get(content_type, {}).get('audience_engagement', 0.5),
                    'production_cost': self.config.get('content_types', {}).get(content_type, {}).get('production_cost', 0.5)
                }
                for content_type, score in sorted_content[:3]  # Top 3 recomendaciones
            ],
            'content_ideas': self._generate_content_ideas(niche, target_audience, sorted_content[:2]),
            'timestamp': datetime.now().isoformat()
        }
        
        return recommendations
    
    def _generate_content_ideas(self, niche: str, target_audience: str, 
                              content_types: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """
        Genera ideas de contenido basadas en nicho, audiencia y tipos de contenido.
        
        Args:
            niche: Nicho del canal
            target_audience: Audiencia objetivo
            content_types: Lista de tipos de contenido con puntuación
            
        Returns:
            Lista de ideas de contenido
        """
        ideas = []
        
        # Ideas genéricas por tipo de contenido
        content_ideas = {
            'video': [
                'Tutorial paso a paso sobre {niche}',
                'Revisión de productos relacionados con {niche}',
                'Entrevista con expertos en {niche}',
                'Comparativa de las mejores opciones en {niche}'
            ],
            'short_video': [
                'Tip rápido sobre {niche}',
                'Dato curioso sobre {niche}',
                'Antes y después en {niche}',
                'Tendencia viral aplicada a {niche}'
            ],
            'live': [
                'Sesión de preguntas y respuestas sobre {niche}',
                'Demostración en vivo de {niche}',
                'Evento especial relacionado con {niche}',
                'Colaboración en vivo con otro creador de {niche}'
            ],
            'blog': [
                'Guía completa sobre {niche}',
                'Los 10 errores comunes en {niche}',
                'Cómo empezar en {niche} para principiantes',
                'Tendencias futuras en {niche}'
            ],
            'podcast': [
                'Entrevista con experto en {niche}',
                'Debate sobre tendencias en {niche}',
                'Historia y evolución de {niche}',
                'Consejos semanales sobre {niche}'
            ]
        }
        
        # Generar ideas para cada tipo de contenido recomendado
        for content_type, _ in content_types:
            if content_type in content_ideas:
                for idea_template in content_ideas[content_type]:
                    # Reemplazar placeholders
                    idea = idea_template.replace('{niche}', niche)
                    
                    # Crear idea
                    idea_obj = {
                        'title': idea,
                        'content_type': content_type,
                        'target_audience': target_audience,
                        'estimated_engagement': random.uniform(0.6, 0.9),  # Simulación
                        'estimated_production_time': self._estimate_production_time(content_type)
                    }
                    
                    ideas.append(idea_obj)
        
        # Limitar a 10 ideas
        return ideas[:10]
    
    def _estimate_production_time(self, content_type: str) -> Dict[str, Any]:
        """
        Estima el tiempo de producción para un tipo de contenido.
        
        Args:
            content_type: Tipo de contenido
            
        Returns:
            Estimación de tiempo
        """
        # Estimaciones por tipo de contenido (en horas)
        estimates = {
            'video': {'min': 4, 'max': 16, 'unit': 'hours'},
            'short_video': {'min': 1, 'max': 4, 'unit': 'hours'},
            'live': {'min': 1, 'max': 3, 'unit': 'hours'},
            'blog': {'min': 2, 'max': 8, 'unit': 'hours'},
            'podcast': {'min': 2, 'max': 6, 'unit': 'hours'},
            'reels': {'min': 1, 'max': 3, 'unit': 'hours'},
            'posts': {'min': 0.5, 'max': 2, 'unit': 'hours'},
            'stories': {'min': 0.2, 'max': 1, 'unit': 'hours'},
            'tweets': {'min': 0.1, 'max': 0.5, 'unit': 'hours'},
            'spaces': {'min': 1, 'max': 3, 'unit': 'hours'}
        }
        
        # Usar estimación por defecto si no existe
        if content_type not in estimates:
            return {'min': 1, 'max': 4, 'unit': 'hours'}
        
        return estimates[content_type]
    
    def export_channel_data(self, channel_id: str, format: str = 'json') -> Dict[str, Any]:
        """
        Exporta los datos de un canal en el formato especificado.
        
        Args:
            channel_id: Identificador del canal
            format: Formato de exportación (json, csv)
            
        Returns:
            Resultado de la exportación
        """
        if channel_id not in self.channels:
            logger.warning(f"Canal no encontrado: {channel_id}")
            return {
                'status': 'error',
                'message': 'Canal no encontrado'
            }
        
        # Obtener datos del canal
        channel = self.channels[channel_id]
        metrics = self.get_channel_metrics(channel_id)
        history = self.get_channel_performance_history(channel_id)
        
        # Crear directorio de exportación
        export_dir = 'exports'
        os.makedirs(export_dir, exist_ok=True)
        
        # Nombre de archivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{export_dir}/channel_{channel_id}_{timestamp}"
        
        # Exportar según formato
        if format.lower() == 'json':
            # Preparar datos
            export_data = {
                'channel': channel,
                'metrics': metrics,
                'history': history,
                'exported_at': datetime.now().isoformat()
            }
            
            # Guardar archivo
            with open(f"{filename}.json", 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)
            
            return {
                'status': 'success',
                'message': f'Datos exportados en formato JSON',
                'file': f"{filename}.json"
            }
        
        elif format.lower() == 'csv':
            # Exportar datos del canal
            with open(f"{filename}_info.csv", 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Field', 'Value'])
                for key, value in channel.items():
                    if isinstance(value, dict):
                        writer.writerow([key, json.dumps(value)])
                    else:
                        writer.writerow([key, value])
            
            # Exportar métricas
            with open(f"{filename}_metrics.csv", 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for metric, value in metrics.items():
                    writer.writerow([metric, value])
            
            # Exportar historial
            with open(f"{filename}_history.csv", 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # Obtener todas las métricas disponibles
                all_metrics = set()
                for date_metrics in history.values():
                    all_metrics.update(date_metrics.keys())
                
                # Escribir encabezado
                header = ['Date'] + list(all_metrics)
                writer.writerow(header)
                
                # Escribir datos
                for date, date_metrics in sorted(history.items()):
                    row = [date]
                    for metric in all_metrics:
                        row.append(date_metrics.get(metric, ''))
                    writer.writerow(row)
            
            return {
                'status': 'success',
                'message': f'Datos exportados en formato CSV',
                'files': [
                    f"{filename}_info.csv",
                    f"{filename}_metrics.csv",
                    f"{filename}_history.csv"
                ]
            }
        
        else:
            return {
                'status': 'error',
                'message': f'Formato no soportado: {format}. Formatos disponibles: json, csv'
            }
    
    def import_channel_data(self, file_path: str) -> Dict[str, Any]:
        """
        Importa datos de un canal desde un archivo JSON.
        
        Args:
            file_path: Ruta al archivo de importación
            
        Returns:
            Resultado de la importación
        """
        try:
            if not os.path.exists(file_path):
                return {
                    'status': 'error',
                    'message': f'Archivo no encontrado: {file_path}'
                }
            
            # Verificar extensión
            if not file_path.endswith('.json'):
                return {
                    'status': 'error',
                    'message': 'Solo se admiten archivos JSON para importación'
                }
            
            # Cargar datos
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Verificar estructura
            if 'channel' not in import_data:
                return {
                    'status': 'error',
                    'message': 'Formato de archivo inválido: falta información del canal'
                }
            
            # Obtener datos del canal
            channel = import_data['channel']
            
            if 'channel_id' not in channel:
                return {
                    'status': 'error',
                    'message': 'Formato de archivo inválido: falta identificador del canal'
                }
            
            channel_id = channel['channel_id']
            
            # Importar canal
            self.channels[channel_id] = channel
            
            # Importar métricas si existen
            if 'metrics' in import_data:
                for metric, value in import_data['metrics'].items():
                    self.channel_metrics[channel_id][metric] = value
            
            # Importar historial si existe
            if 'history' in import_data:
                for date_str, metrics in import_data['history'].items():
                    for metric, value in metrics.items():
                        self.performance_history[channel_id][date_str][metric] = value
            
            # Guardar datos
            self.save_channel_data()
            
            logger.info(f"Datos importados para canal: {channel_id}")
            
            return {
                'status': 'success',
                'message': f'Datos importados correctamente para canal: {channel.get("name", channel_id)}',
                'channel_id': channel_id
            }
            
        except Exception as e:
            logger.error(f"Error al importar datos: {e}")
            return {
                'status': 'error',
                'message': f'Error al importar datos: {str(e)}'
            }