"""
MonetizationManager - Gestor de estrategias de monetización

Este módulo se encarga de implementar, gestionar y optimizar las diferentes
estrategias de monetización para los canales de contenido, incluyendo:
- Anuncios (AdSense, TikTok Creator Fund, etc.)
- Afiliados y marketing de afiliación
- Productos digitales y físicos
- NFTs y tokens
- Suscripciones y membresías
- Patrocinios y colaboraciones B2B
- Marketplace de contenido
"""

import logging
import datetime
import uuid
import threading
import json
import math
import statistics
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum, auto
from collections import defaultdict

# Configuración de logging
logger = logging.getLogger('orchestrator.monetization')

class MonetizationStrategy(Enum):
    """Enumeración de estrategias de monetización disponibles."""
    ADS = auto()               # Anuncios en plataformas
    AFFILIATE = auto()         # Marketing de afiliación
    PRODUCTS = auto()          # Productos propios
    SUBSCRIPTIONS = auto()     # Membresías y suscripciones
    SPONSORSHIPS = auto()      # Patrocinios de marcas
    CREATOR_FUND = auto()      # Fondos de creadores (TikTok, etc.)
    NFT = auto()               # Tokens no fungibles
    TOKENIZATION = auto()      # Tokenización de contenido
    B2B_MARKETPLACE = auto()   # Marketplace B2B
    DONATIONS = auto()         # Donaciones y propinas

class MonetizationStatus(Enum):
    """Enumeración de estados de monetización."""
    INACTIVE = auto()          # No activada
    PENDING = auto()           # En proceso de activación
    ACTIVE = auto()            # Activa y funcionando
    OPTIMIZING = auto()        # En proceso de optimización
    PAUSED = auto()            # Pausada temporalmente
    BLOCKED = auto()           # Bloqueada por la plataforma
    FAILED = auto()            # Falló la activación

class MonetizationManager:
    """
    Gestor de estrategias de monetización para canales de contenido.
    
    Esta clase implementa la lógica para activar, gestionar, analizar y optimizar
    diferentes estrategias de monetización en múltiples plataformas.
    """
    
    def __init__(self, config: Dict[str, Any] = None, 
                 platform_adapters: Dict[str, Any] = None,
                 analysis_manager: Any = None,
                 compliance_manager: Any = None):
        """
        Inicializa el gestor de monetización.
        
        Args:
            config: Configuración del gestor
            platform_adapters: Adaptadores de plataforma para interactuar con APIs
            analysis_manager: Gestor de análisis para obtener métricas
            compliance_manager: Gestor de cumplimiento normativo
        """
        self.config = config or {}
        self.platform_adapters = platform_adapters or {}
        self.analysis_manager = analysis_manager
        self.compliance_manager = compliance_manager
        
        # Estado de monetización por canal y estrategia
        self.monetization_status = defaultdict(dict)
        
        # Historial de ingresos por canal y estrategia
        self.revenue_history = defaultdict(lambda: defaultdict(list))
        
        # Configuración de estrategias por canal
        self.channel_strategies = defaultdict(dict)
        
        # Métricas de rendimiento por estrategia
        self.performance_metrics = defaultdict(lambda: defaultdict(dict))
        
        # Bloqueo para operaciones concurrentes
        self.lock = threading.RLock()
        
        # Cargar configuración inicial
        self._load_initial_config()
        
        logger.info("MonetizationManager inicializado correctamente")
    
    def _load_initial_config(self):
        """Carga la configuración inicial de monetización."""
        try:
            # Cargar configuración de estrategias predeterminadas
            default_strategies = self.config.get('default_strategies', {})
            
            # Cargar umbrales de activación
            self.activation_thresholds = self.config.get('activation_thresholds', {
                'ADS': {'followers': 1000, 'views': 4000},
                'AFFILIATE': {'followers': 500, 'engagement_rate': 0.02},
                'PRODUCTS': {'followers': 5000, 'engagement_rate': 0.03},
                'SUBSCRIPTIONS': {'followers': 10000, 'retention_rate': 0.4},
                'SPONSORSHIPS': {'followers': 10000, 'engagement_rate': 0.05},
                'CREATOR_FUND': {'followers': 10000, 'views': 100000},
                'NFT': {'followers': 5000, 'engagement_rate': 0.04},
                'TOKENIZATION': {'followers': 20000, 'engagement_rate': 0.05},
                'B2B_MARKETPLACE': {'followers': 5000, 'content_quality': 0.7},
                'DONATIONS': {'followers': 1000, 'engagement_rate': 0.03}
            })
            
            # Cargar configuraciones específicas por plataforma
            self.platform_configs = self.config.get('platform_configs', {
                'youtube': {
                    'ADS': {'min_views': 4000, 'min_subscribers': 1000},
                    'SUBSCRIPTIONS': {'feature_name': 'Channel Memberships'}
                },
                'tiktok': {
                    'CREATOR_FUND': {'min_followers': 10000, 'min_views': 100000},
                    'ADS': {'feature_name': 'Creator Marketplace'}
                },
                'instagram': {
                    'ADS': {'feature_name': 'Branded Content'},
                    'AFFILIATE': {'feature_name': 'Shopping Tags'}
                }
            })
            
            logger.debug("Configuración inicial de monetización cargada correctamente")
        except Exception as e:
            logger.error(f"Error al cargar configuración inicial de monetización: {str(e)}")
            # Establecer valores predeterminados básicos
            self.activation_thresholds = {}
            self.platform_configs = {}
    
    def get_available_strategies(self, channel_id: str, platform: str) -> List[MonetizationStrategy]:
        """
        Obtiene las estrategias de monetización disponibles para un canal en una plataforma.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma (youtube, tiktok, etc.)
            
        Returns:
            Lista de estrategias disponibles
        """
        available_strategies = []
        
        try:
            # Verificar si tenemos acceso al adaptador de plataforma
            if platform not in self.platform_adapters:
                logger.warning(f"No hay adaptador disponible para la plataforma {platform}")
                return []
            
            # Obtener métricas del canal si tenemos acceso al gestor de análisis
            channel_metrics = {}
            if self.analysis_manager:
                channel_metrics = self.analysis_manager.get_channel_metrics(channel_id, platform)
            
            # Verificar cada estrategia según los umbrales y disponibilidad en la plataforma
            for strategy in MonetizationStrategy:
                # Verificar si la estrategia está disponible en la plataforma
                platform_config = self.platform_configs.get(platform, {})
                if strategy.name not in platform_config and platform != "all":
                    continue
                
                # Verificar si cumple con los umbrales mínimos
                threshold = self.activation_thresholds.get(strategy.name, {})
                meets_threshold = True
                
                for metric, min_value in threshold.items():
                    if metric in channel_metrics and channel_metrics[metric] < min_value:
                        meets_threshold = False
                        break
                
                if meets_threshold:
                    available_strategies.append(strategy)
            
            return available_strategies
        except Exception as e:
            logger.error(f"Error al obtener estrategias disponibles para {channel_id} en {platform}: {str(e)}")
            return []
    
    def activate_strategy(self, channel_id: str, platform: str, 
                         strategy: MonetizationStrategy, 
                         config: Dict[str, Any] = None) -> bool:
        """
        Activa una estrategia de monetización para un canal.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma (youtube, tiktok, etc.)
            strategy: Estrategia a activar
            config: Configuración específica para la estrategia
            
        Returns:
            True si se activó correctamente, False en caso contrario
        """
        with self.lock:
            try:
                # Verificar si la estrategia está disponible
                available_strategies = self.get_available_strategies(channel_id, platform)
                if strategy not in available_strategies:
                    logger.warning(f"Estrategia {strategy.name} no disponible para {channel_id} en {platform}")
                    return False
                
                # Verificar cumplimiento normativo si tenemos acceso al gestor
                if self.compliance_manager:
                    compliance_check = self.compliance_manager.check_monetization_compliance(
                        channel_id, platform, strategy.name)
                    if not compliance_check.get('compliant', False):
                        logger.warning(f"No cumple requisitos normativos: {compliance_check.get('reason', 'Desconocido')}")
                        return False
                
                # Configuración específica para la estrategia
                strategy_config = config or {}
                
                # Activar la estrategia a través del adaptador de plataforma
                adapter = self.platform_adapters.get(platform)
                if not adapter:
                    logger.error(f"No hay adaptador disponible para {platform}")
                    return False
                
                # Intentar activar la estrategia en la plataforma
                activation_result = adapter.activate_monetization(
                    channel_id=channel_id,
                    strategy=strategy.name,
                    config=strategy_config
                )
                
                if not activation_result.get('success', False):
                    logger.error(f"Error al activar {strategy.name}: {activation_result.get('error', 'Desconocido')}")
                    self.monetization_status[channel_id][strategy.name] = MonetizationStatus.FAILED
                    return False
                
                # Actualizar estado y configuración
                self.monetization_status[channel_id][strategy.name] = MonetizationStatus.ACTIVE
                self.channel_strategies[channel_id][strategy.name] = strategy_config
                
                logger.info(f"Estrategia {strategy.name} activada para {channel_id} en {platform}")
                return True
            except Exception as e:
                logger.error(f"Error al activar estrategia {strategy.name}: {str(e)}")
                self.monetization_status[channel_id][strategy.name] = MonetizationStatus.FAILED
                return False
    
    def deactivate_strategy(self, channel_id: str, platform: str, 
                           strategy: MonetizationStrategy) -> bool:
        """
        Desactiva una estrategia de monetización para un canal.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma (youtube, tiktok, etc.)
            strategy: Estrategia a desactivar
            
        Returns:
            True si se desactivó correctamente, False en caso contrario
        """
        with self.lock:
            try:
                # Verificar si la estrategia está activa
                current_status = self.monetization_status.get(channel_id, {}).get(strategy.name)
                if current_status != MonetizationStatus.ACTIVE:
                    logger.warning(f"Estrategia {strategy.name} no está activa para {channel_id}")
                    return False
                
                # Desactivar la estrategia a través del adaptador de plataforma
                adapter = self.platform_adapters.get(platform)
                if not adapter:
                    logger.error(f"No hay adaptador disponible para {platform}")
                    return False
                
                # Intentar desactivar la estrategia en la plataforma
                deactivation_result = adapter.deactivate_monetization(
                    channel_id=channel_id,
                    strategy=strategy.name
                )
                
                if not deactivation_result.get('success', False):
                    logger.error(f"Error al desactivar {strategy.name}: {deactivation_result.get('error', 'Desconocido')}")
                    return False
                
                # Actualizar estado
                self.monetization_status[channel_id][strategy.name] = MonetizationStatus.INACTIVE
                
                logger.info(f"Estrategia {strategy.name} desactivada para {channel_id} en {platform}")
                return True
            except Exception as e:
                logger.error(f"Error al desactivar estrategia {strategy.name}: {str(e)}")
                return False
    
    def get_strategy_status(self, channel_id: str, strategy: MonetizationStrategy = None) -> Dict[str, Any]:
        """
        Obtiene el estado de las estrategias de monetización para un canal.
        
        Args:
            channel_id: ID del canal
            strategy: Estrategia específica (opcional)
            
        Returns:
            Diccionario con el estado de las estrategias
        """
        try:
            channel_status = self.monetization_status.get(channel_id, {})
            
            if strategy:
                return {
                    'strategy': strategy.name,
                    'status': channel_status.get(strategy.name, MonetizationStatus.INACTIVE).name,
                    'config': self.channel_strategies.get(channel_id, {}).get(strategy.name, {})
                }
            else:
                result = {}
                for strat_name, status in channel_status.items():
                    result[strat_name] = {
                        'status': status.name,
                        'config': self.channel_strategies.get(channel_id, {}).get(strat_name, {})
                    }
                return result
        except Exception as e:
            logger.error(f"Error al obtener estado de monetización: {str(e)}")
            return {}
    
    def track_revenue(self, channel_id: str, platform: str, 
                     strategy: MonetizationStrategy, 
                     amount: float, 
                     timestamp: datetime.datetime = None,
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Registra ingresos generados por una estrategia de monetización.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma (youtube, tiktok, etc.)
            strategy: Estrategia que generó los ingresos
            amount: Cantidad de ingresos
            timestamp: Fecha y hora de los ingresos (opcional)
            metadata: Metadatos adicionales (opcional)
            
        Returns:
            True si se registró correctamente, False en caso contrario
        """
        with self.lock:
            try:
                # Usar timestamp actual si no se proporciona
                if timestamp is None:
                    timestamp = datetime.datetime.now()
                
                # Metadatos por defecto
                meta = metadata or {}
                meta['platform'] = platform
                
                # Registrar ingresos
                revenue_entry = {
                    'amount': amount,
                    'timestamp': timestamp.isoformat(),
                    'metadata': meta
                }
                
                self.revenue_history[channel_id][strategy.name].append(revenue_entry)
                
                # Actualizar métricas de rendimiento
                self._update_performance_metrics(channel_id, strategy)
                
                logger.debug(f"Ingresos registrados: {amount} para {channel_id} via {strategy.name}")
                return True
            except Exception as e:
                logger.error(f"Error al registrar ingresos: {str(e)}")
                return False
    
    def _update_performance_metrics(self, channel_id: str, strategy: MonetizationStrategy):
        """
        Actualiza las métricas de rendimiento para una estrategia.
        
        Args:
            channel_id: ID del canal
            strategy: Estrategia a actualizar
        """
        try:
            # Obtener historial de ingresos para la estrategia
            history = self.revenue_history.get(channel_id, {}).get(strategy.name, [])
            
            if not history:
                return
            
            # Calcular métricas básicas
            amounts = [entry['amount'] for entry in history]
            total_revenue = sum(amounts)
            avg_revenue = total_revenue / len(amounts) if amounts else 0
            
            # Calcular tendencia (últimos 7 días vs 7 días anteriores)
            now = datetime.datetime.now()
            last_week = [entry['amount'] for entry in history 
                        if (now - datetime.datetime.fromisoformat(entry['timestamp'])).days <= 7]
            previous_week = [entry['amount'] for entry in history 
                            if 7 < (now - datetime.datetime.fromisoformat(entry['timestamp'])).days <= 14]
            
            last_week_total = sum(last_week) if last_week else 0
            previous_week_total = sum(previous_week) if previous_week else 0
            
            trend = ((last_week_total - previous_week_total) / previous_week_total * 100 
                    if previous_week_total > 0 else 0)
            
            # Actualizar métricas
            self.performance_metrics[channel_id][strategy.name] = {
                'total_revenue': total_revenue,
                'average_revenue': avg_revenue,
                'last_update': now.isoformat(),
                'trend_percentage': trend,
                'entries_count': len(history)
            }
        except Exception as e:
            logger.error(f"Error al actualizar métricas de rendimiento: {str(e)}")
    
    def get_revenue_report(self, channel_id: str = None, 
                          platform: str = None,
                          strategy: MonetizationStrategy = None,
                          start_date: datetime.datetime = None,
                          end_date: datetime.datetime = None) -> Dict[str, Any]:
        """
        Genera un informe de ingresos filtrado por varios criterios.
        
        Args:
            channel_id: ID del canal (opcional)
            platform: Plataforma específica (opcional)
            strategy: Estrategia específica (opcional)
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            
        Returns:
            Informe de ingresos
        """
        try:
            # Establecer fechas predeterminadas si no se proporcionan
            if end_date is None:
                end_date = datetime.datetime.now()
            if start_date is None:
                start_date = end_date - datetime.timedelta(days=30)
            
            # Filtrar canales
            channels_to_report = [channel_id] if channel_id else list(self.revenue_history.keys())
            
            report = {
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_revenue': 0,
                'channels': {}
            }
            
            # Generar informe por canal
            for ch_id in channels_to_report:
                channel_report = self._generate_channel_report(
                    ch_id, platform, strategy, start_date, end_date)
                
                if channel_report['total_revenue'] > 0:
                    report['channels'][ch_id] = channel_report
                    report['total_revenue'] += channel_report['total_revenue']
            
            # Añadir resumen por estrategia
            strategy_summary = defaultdict(float)
            for ch_data in report['channels'].values():
                for strat, amount in ch_data['by_strategy'].items():
                    strategy_summary[strat] += amount
            
            report['by_strategy'] = dict(strategy_summary)
            
            # Añadir resumen por plataforma si se solicitó
            if platform is None:
                platform_summary = defaultdict(float)
                for ch_data in report['channels'].values():
                    for plat, amount in ch_data.get('by_platform', {}).items():
                        platform_summary[plat] += amount
                
                report['by_platform'] = dict(platform_summary)
            
            return report
        except Exception as e:
            logger.error(f"Error al generar informe de ingresos: {str(e)}")
            return {'error': str(e)}
    
    def _generate_channel_report(self, channel_id: str, 
                               platform: str = None,
                               strategy: MonetizationStrategy = None,
                               start_date: datetime.datetime = None,
                               end_date: datetime.datetime = None) -> Dict[str, Any]:
        """
        Genera un informe de ingresos para un canal específico.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma específica (opcional)
            strategy: Estrategia específica (opcional)
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Informe de ingresos del canal
        """
        channel_history = self.revenue_history.get(channel_id, {})
        
        # Filtrar por estrategia si se especifica
        strategies_to_report = [strategy.name] if strategy else channel_history.keys()
        
        # Inicializar informe del canal
        channel_report = {
            'total_revenue': 0,
            'by_strategy': {},
            'by_platform': defaultdict(float) if platform is None else {},
            'daily_revenue': defaultdict(float)
        }
        
        # Procesar cada estrategia
        for strat_name in strategies_to_report:
            if strat_name not in channel_history:
                continue
            
            strategy_entries = channel_history[strat_name]
            strategy_total = 0
            
            # Filtrar entradas por fecha y plataforma
            for entry in strategy_entries:
                entry_date = datetime.datetime.fromisoformat(entry['timestamp'])
                entry_platform = entry['metadata'].get('platform', 'unknown')
                
                # Verificar si está dentro del rango de fechas
                if start_date <= entry_date <= end_date:
                    # Verificar si coincide con la plataforma solicitada
                    if platform is None or entry_platform == platform:
                        amount = entry['amount']
                        strategy_total += amount
                        
                        # Actualizar totales por plataforma
                        if platform is None:
                            channel_report['by_platform'][entry_platform] += amount
                        
                        # Actualizar totales diarios
                        day_key = entry_date.strftime('%Y-%m-%d')
                        channel_report['daily_revenue'][day_key] += amount
            
            # Actualizar total de la estrategia si hay ingresos
            if strategy_total > 0:
                channel_report['by_strategy'][strat_name] = strategy_total
                channel_report['total_revenue'] += strategy_total
        
        # Convertir defaultdict a dict para la salida
        channel_report['daily_revenue'] = dict(channel_report['daily_revenue'])
        if platform is None:
            channel_report['by_platform'] = dict(channel_report['by_platform'])
        
        return channel_report
    
    def optimize_monetization(self, channel_id: str, platform: str = None) -> Dict[str, Any]:
        """
        Optimiza las estrategias de monetización para un canal.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma específica (opcional)
            
        Returns:
            Resultado de la optimización
        """
        try:
            # Obtener métricas del canal si tenemos acceso al gestor de análisis
            channel_metrics = {}
            if self.analysis_manager:
                channel_metrics = self.analysis_manager.get_channel_metrics(channel_id, platform)
            
            # Obtener estrategias activas
            active_strategies = {}
            for strat_name, status in self.monetization_status.get(channel_id, {}).items():
                if status == MonetizationStatus.ACTIVE:
                    active_strategies[strat_name] = self.performance_metrics.get(channel_id, {}).get(strat_name, {})
            
            # Si no hay estrategias activas, intentar activar las disponibles
            if not active_strategies and platform:
                available_strategies = self.get_available_strategies(channel_id, platform)
                for strategy in available_strategies:
                    self.activate_strategy(channel_id, platform, strategy)
                    
                # Actualizar estrategias activas
                for strat_name, status in self.monetization_status.get(channel_id, {}).items():
                    if status == MonetizationStatus.ACTIVE:
                        active_strategies[strat_name] = self.performance_metrics.get(channel_id, {}).get(strat_name, {})
            
            # Generar recomendaciones de optimización
            recommendations = []
            
            # Analizar rendimiento de cada estrategia
            for strat_name, metrics in active_strategies.items():
                # Verificar si hay suficientes datos para optimizar
                if metrics.get('entries_count', 0) < 5:
                    recommendations.append({
                        'strategy': strat_name,
                        'action': 'MONITOR',
                        'reason': 'Datos insuficientes para optimización',
                        'confidence': 0.5
                    })
                    continue
                
                # Verificar tendencia
                trend = metrics.get('trend_percentage', 0)
                
                if trend < -20:
                    # Tendencia muy negativa
                    recommendations.append({
                        'strategy': strat_name,
                        'action': 'REVIEW',
                        'reason': f'Caída significativa de ingresos ({trend:.1f}%)',
                        'confidence': 0.8
                    })
                elif trend < -10:
                    # Tendencia negativa
                    recommendations.append({
                        'strategy': strat_name,
                        'action': 'OPTIMIZE',
                        'reason': f'Disminución de ingresos ({trend:.1f}%)',
                        'confidence': 0.7
                    })
                elif trend > 20:
                    # Tendencia muy positiva
                    recommendations.append({
                        'strategy': strat_name,
                        'action': 'SCALE',
                        'reason': f'Crecimiento significativo ({trend:.1f}%)',
                        'confidence': 0.8
                    })
                elif trend > 10:
                    # Tendencia positiva
                    recommendations.append({
                        'strategy': strat_name,
                        'action': 'MAINTAIN',
                        'reason': f'Buen crecimiento ({trend:.1f}%)',
                        'confidence': 0.7
                    })
                else:
                    # Tendencia estable
                    recommendations.append({
                        'strategy': strat_name,
                        'action': 'MAINTAIN',
                        'reason': 'Rendimiento estable',
                        'confidence': 0.6
                    })
            
            # Verificar si hay estrategias disponibles no activadas
            if platform:
                available_strategies = self.get_available_strategies(channel_id, platform)
                for strategy in available_strategies:
                    if strategy.name not in active_strategies:
                        recommendations.append({
                            'strategy': strategy.name,
                            'action': 'ACTIVATE',
                            'reason': 'Estrategia disponible no utilizada',
                            'confidence': 0.7
                        })
            
            # Aplicar optimizaciones automáticas si está configurado
            auto_optimize = self.config.get('auto_optimize', False)
            applied_changes = []
            
            if auto_optimize:
                for rec in recommendations:
                    if rec['action'] == 'ACTIVATE' and rec['confidence'] > 0.7:
                        # Activar estrategia automáticamente
                        strategy_enum = MonetizationStrategy[rec['strategy']]
                        success = self.activate_strategy(channel_id, platform, strategy_enum)
                        if success:
                            applied_changes.append({
                                'strategy': rec['strategy'],
                                'action': 'ACTIVATED',
                                'success': True
                            })
                    
                    # Otras acciones automáticas podrían implementarse aquí
            
            return {
                'channel_id': channel_id,
                'platform': platform,
                'active_strategies': list(active_strategies.keys()),
                'recommendations': recommendations,
                'applied_changes': applied_changes,
                'timestamp': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error al optimizar monetización: {str(e)}")
            return {'error': str(e)}
    
    def get_monetization_opportunities(self, channel_id: str, platform: str = None) -> Dict[str, Any]:
        """
        Identifica oportunidades de monetización para un canal.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma específica (opcional)
            
        Returns:
            Oportunidades de monetización identificadas
        """
        try:
            # Obtener métricas del canal
            channel_metrics = {}
            if self.analysis_manager:
                channel_metrics = self.analysis_manager.get_channel_metrics(channel_id, platform)
            
            # Obtener estrategias activas
            active_strategies = set()
            for strat_name, status in self.monetization_status.get(channel_id, {}).items():
                if status == MonetizationStatus.ACTIVE:
                    active_strategies.add(strat_name)
            
            # Identificar oportunidades basadas en métricas y estrategias no utilizadas
            opportunities = []
            
            # Verificar cada estrategia
            for strategy in MonetizationStrategy:
                if strategy.name in active_strategies:
                    continue
                
                # Verificar si está cerca de cumplir los umbrales
                threshold = self.activation_thresholds.get(strategy.name, {})
                opportunity_score = 0
                reasons = []
                
                for metric, min_value in threshold.items():
                    if metric in channel_metrics:
                        current_value = channel_metrics[metric]
                        ratio = current_value / min_value if min_value > 0 else 0
                        
                                                if ratio >= 1:
                            # Ya cumple este umbral
                            opportunity_score += 0.2
                            reasons.append(f"Cumple umbral de {metric}: {current_value} >= {min_value}")
                        elif ratio >= 0.8:
                            # Está muy cerca de cumplir el umbral
                            opportunity_score += 0.15
                            reasons.append(f"Cerca de umbral de {metric}: {current_value} ({ratio*100:.1f}% del requerido)")
                        elif ratio >= 0.5:
                            # Está a medio camino
                            opportunity_score += 0.1
                            reasons.append(f"A medio camino del umbral de {metric}: {current_value} ({ratio*100:.1f}% del requerido)")
                        else:
                            # Está lejos de cumplir el umbral
                            opportunity_score += 0.05
                            reasons.append(f"Lejos del umbral de {metric}: {current_value} ({ratio*100:.1f}% del requerido)")
                
                # Verificar si la estrategia está disponible en la plataforma
                if platform:
                    platform_config = self.platform_configs.get(platform, {})
                    if strategy.name not in platform_config and platform != "all":
                        # No disponible en esta plataforma
                        continue
                
                # Añadir a oportunidades si tiene un puntaje mínimo
                if opportunity_score > 0.1:
                    opportunities.append({
                        'strategy': strategy.name,
                        'score': opportunity_score,
                        'reasons': reasons,
                        'requirements': threshold,
                        'current_metrics': {k: channel_metrics.get(k, 0) for k in threshold.keys()}
                    })
            
            # Ordenar oportunidades por puntaje
            opportunities = sorted(opportunities, key=lambda x: x['score'], reverse=True)
            
            return {
                'channel_id': channel_id,
                'platform': platform,
                'timestamp': datetime.datetime.now().isoformat(),
                'opportunities': opportunities
            }
        except Exception as e:
            logger.error(f"Error al identificar oportunidades de monetización: {str(e)}")
            return {'error': str(e)}
    
    def _get_audience_data(self, channel_id: str, platform: str = None, time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Obtiene datos simulados de audiencia para pruebas.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Datos de audiencia simulados
        """
        # Simulación de datos demográficos
        demographics = {
            "age_groups": {
                "13-17": random.uniform(0.05, 0.15),
                "18-24": random.uniform(0.2, 0.35),
                "25-34": random.uniform(0.25, 0.4),
                "35-44": random.uniform(0.1, 0.25),
                "45-54": random.uniform(0.05, 0.15),
                "55+": random.uniform(0.02, 0.1)
            },
            "gender": {
                "male": random.uniform(0.4, 0.6),
                "female": random.uniform(0.4, 0.6),
                "other": random.uniform(0.01, 0.05)
            },
            "locations": {
                "United States": random.uniform(0.2, 0.4),
                "India": random.uniform(0.1, 0.2),
                "Brazil": random.uniform(0.05, 0.15),
                "United Kingdom": random.uniform(0.05, 0.1),
                "Germany": random.uniform(0.03, 0.08),
                "Canada": random.uniform(0.03, 0.08),
                "Australia": random.uniform(0.02, 0.06),
                "Mexico": random.uniform(0.02, 0.06),
                "Other": random.uniform(0.1, 0.2)
            },
            "languages": {
                "English": random.uniform(0.5, 0.7),
                "Spanish": random.uniform(0.1, 0.2),
                "Portuguese": random.uniform(0.05, 0.15),
                "Hindi": random.uniform(0.05, 0.15),
                "German": random.uniform(0.03, 0.08),
                "French": random.uniform(0.03, 0.08),
                "Other": random.uniform(0.05, 0.15)
            },
            "devices": {
                "Mobile": random.uniform(0.6, 0.8),
                "Desktop": random.uniform(0.15, 0.35),
                "Tablet": random.uniform(0.05, 0.15),
                "TV": random.uniform(0.02, 0.1),
                "Other": random.uniform(0.01, 0.05)
            }
        }
        
        # Simulación de datos de crecimiento
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        growth_data = []
        followers = random.randint(1000, 50000)
        
        for i in range(30):
            current_date = start_date + datetime.timedelta(days=i)
            new_followers = max(0, int(random.gauss(followers * 0.02, followers * 0.01)))
            unfollowers = max(0, int(random.gauss(new_followers * 0.3, new_followers * 0.2)))
            followers += (new_followers - unfollowers)
            
            growth_data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "followers": followers,
                "new_followers": new_followers,
                "unfollowers": unfollowers
            })
        
        # Simulación de datos de retención
        retention_data = {
            "average_retention": {
                "day_1": random.uniform(0.5, 0.8),
                "day_3": random.uniform(0.3, 0.6),
                "day_7": random.uniform(0.2, 0.5),
                "day_14": random.uniform(0.15, 0.4),
                "day_30": random.uniform(0.1, 0.3)
            },
            "cohorts": [
                {
                    "cohort_date": (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
                    "initial_size": random.randint(100, 1000),
                    "retention": {
                        "day_1": random.uniform(0.5, 0.8),
                        "day_3": random.uniform(0.3, 0.6),
                        "day_7": random.uniform(0.2, 0.5),
                        "day_14": random.uniform(0.15, 0.4),
                        "day_30": random.uniform(0.1, 0.3)
                    }
                },
                {
                    "cohort_date": (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d"),
                    "initial_size": random.randint(100, 1000),
                    "retention": {
                        "day_1": random.uniform(0.5, 0.8),
                        "day_3": random.uniform(0.3, 0.6),
                        "day_7": random.uniform(0.2, 0.5),
                        "day_14": random.uniform(0.15, 0.4)
                    }
                }
            ]
        }
        
        # Simulación de intereses
        interests = {
            "categories": {
                "Technology": random.uniform(0.1, 0.5),
                "Entertainment": random.uniform(0.1, 0.5),
                "Gaming": random.uniform(0.1, 0.5),
                "Education": random.uniform(0.1, 0.5),
                "Lifestyle": random.uniform(0.1, 0.5),
                "Sports": random.uniform(0.1, 0.5),
                "Finance": random.uniform(0.1, 0.5),
                "Health": random.uniform(0.1, 0.5)
            },
            "topics": [
                {"name": "Artificial Intelligence", "interest_score": random.uniform(0.1, 1.0)},
                {"name": "Mobile Gaming", "interest_score": random.uniform(0.1, 1.0)},
                {"name": "Personal Finance", "interest_score": random.uniform(0.1, 1.0)},
                {"name": "Fitness", "interest_score": random.uniform(0.1, 1.0)},
                {"name": "Cooking", "interest_score": random.uniform(0.1, 1.0)},
                {"name": "Travel", "interest_score": random.uniform(0.1, 1.0)},
                {"name": "Photography", "interest_score": random.uniform(0.1, 1.0)},
                {"name": "Music", "interest_score": random.uniform(0.1, 1.0)},
                {"name": "Fashion", "interest_score": random.uniform(0.1, 1.0)},
                {"name": "Movies", "interest_score": random.uniform(0.1, 1.0)}
            ]
        }
        
        return {
            "channel_id": channel_id,
            "platform": platform if platform else "all",
            "demographics": demographics,
            "growth": growth_data,
            "retention": retention_data,
            "interests": interests
        }
    
    def _get_monetization_data(self, channel_id: str, platform: str = None, 
                              time_period: str = "last_30_days") -> Dict[str, Any]:
        """
        Obtiene datos simulados de monetización para pruebas.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma específica (opcional)
            time_period: Período de tiempo a analizar
            
        Returns:
            Dict[str, Any]: Datos de monetización simulados
        """
        # Determinar plataformas a simular
        platforms = [platform] if platform else ["youtube", "tiktok", "instagram"]
        
        # Simulación de ingresos por estrategia
        strategies = {strat.name: {} for strat in MonetizationStrategy}
        
        # Fecha de inicio para la simulación
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        
        # Generar datos para cada estrategia
        for strategy_name in strategies.keys():
            # Determinar si la estrategia está activa
            is_active = random.random() > 0.5
            
            if not is_active:
                strategies[strategy_name] = {
                    "status": "INACTIVE",
                    "revenue": [],
                    "total": 0,
                    "average_daily": 0
                }
                continue
            
            # Generar ingresos diarios
            daily_revenue = []
            base_amount = random.uniform(1.0, 50.0)
            total_revenue = 0
            
            for i in range(30):
                current_date = start_date + datetime.timedelta(days=i)
                
                # Simular variación diaria
                daily_amount = max(0, base_amount * random.uniform(0.5, 1.5))
                total_revenue += daily_amount
                
                daily_revenue.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "amount": daily_amount,
                    "platform": random.choice(platforms)
                })
            
            strategies[strategy_name] = {
                "status": "ACTIVE",
                "revenue": daily_revenue,
                "total": total_revenue,
                "average_daily": total_revenue / 30
            }
        
        # Calcular totales
        total_revenue = sum(strat["total"] for strat in strategies.values())
        average_daily = total_revenue / 30
        
        # Generar proyecciones
        projections = {
            "next_30_days": total_revenue * random.uniform(0.9, 1.2),
            "next_90_days": total_revenue * 3 * random.uniform(0.8, 1.3),
            "next_180_days": total_revenue * 6 * random.uniform(0.7, 1.4)
        }
        
        return {
            "channel_id": channel_id,
            "platform": platform if platform else "all",
            "time_period": time_period,
            "total_revenue": total_revenue,
            "average_daily": average_daily,
            "strategies": strategies,
            "projections": projections
        }
    
    def simulate_strategy_activation(self, channel_id: str, platform: str, 
                                   strategy: MonetizationStrategy) -> Dict[str, Any]:
        """
        Simula la activación de una estrategia para evaluar su potencial.
        
        Args:
            channel_id: ID del canal
            platform: Plataforma (youtube, tiktok, etc.)
            strategy: Estrategia a simular
            
        Returns:
            Dict[str, Any]: Resultados de la simulación
        """
        try:
            # Obtener métricas actuales del canal
            channel_metrics = {}
            if self.analysis_manager:
                channel_metrics = self.analysis_manager.get_channel_metrics(channel_id, platform)
            
            # Verificar si la estrategia es viable
            threshold = self.activation_thresholds.get(strategy.name, {})
            meets_threshold = True
            missing_requirements = []
            
            for metric, min_value in threshold.items():
                if metric in channel_metrics and channel_metrics[metric] < min_value:
                    meets_threshold = False
                    missing_requirements.append({
                        "metric": metric,
                        "required": min_value,
                        "current": channel_metrics.get(metric, 0),
                        "gap": min_value - channel_metrics.get(metric, 0)
                    })
            
            # Si no cumple requisitos, devolver plan de acción
            if not meets_threshold:
                return {
                    "channel_id": channel_id,
                    "platform": platform,
                    "strategy": strategy.name,
                    "viable": False,
                    "missing_requirements": missing_requirements,
                    "action_plan": self._generate_action_plan(missing_requirements)
                }
            
            # Simular ingresos potenciales
            base_revenue = random.uniform(10, 100)  # Ingresos base diarios
            
            # Ajustar según métricas del canal
            followers = channel_metrics.get("followers", 1000)
            engagement_rate = channel_metrics.get("engagement_rate", 0.02)
            
            # Factores de ajuste por estrategia
            strategy_factors = {
                "ADS": 0.8 + (engagement_rate * 10),
                "AFFILIATE": 1.2 + (engagement_rate * 15),
                "PRODUCTS": 1.5 + (engagement_rate * 20),
                "SUBSCRIPTIONS": 2.0 + (engagement_rate * 25),
                "SPONSORSHIPS": 2.5 + (engagement_rate * 30),
                "CREATOR_FUND": 0.5 + (followers / 100000),
                "NFT": 3.0 + (engagement_rate * 40),
                "TOKENIZATION": 2.0 + (engagement_rate * 35),
                "B2B_MARKETPLACE": 2.2 + (engagement_rate * 25),
                "DONATIONS": 0.7 + (engagement_rate * 20)
            }
            
            factor = strategy_factors.get(strategy.name, 1.0)
            
            # Calcular ingresos estimados
            daily_revenue = base_revenue * factor * (followers / 10000)
            monthly_revenue = daily_revenue * 30
            
            # Simular crecimiento
            growth_projections = {
                "30_days": monthly_revenue,
                "90_days": monthly_revenue * 1.2,
                "180_days": monthly_revenue * 1.5,
                "365_days": monthly_revenue * 2.0
            }
            
            # Generar recomendaciones
            recommendations = [
                f"Optimizar contenido para maximizar engagement y aumentar ingresos por {strategy.name}",
                f"Crear CTAs específicos para {strategy.name} para mejorar conversión",
                f"Analizar competidores que utilizan {strategy.name} para identificar mejores prácticas"
            ]
            
            # Añadir recomendaciones específicas por estrategia
            if strategy.name == "ADS":
                recommendations.append("Crear contenido más largo para aumentar tiempo de visualización y oportunidades de anuncios")
            elif strategy.name == "AFFILIATE":
                recommendations.append("Integrar productos relevantes naturalmente en el contenido para aumentar conversiones")
            elif strategy.name == "PRODUCTS":
                recommendations.append("Desarrollar productos digitales de bajo costo y alto margen como primeros lanzamientos")
            elif strategy.name == "SUBSCRIPTIONS":
                recommendations.append("Ofrecer contenido exclusivo de alto valor para incentivar suscripciones")
            
            return {
                "channel_id": channel_id,
                "platform": platform,
                "strategy": strategy.name,
                "viable": True,
                "estimated_revenue": {
                    "daily": daily_revenue,
                    "monthly": monthly_revenue,
                    "projections": growth_projections
                },
                "implementation_cost": {
                    "setup": random.uniform(50, 500),
                    "monthly": random.uniform(20, 200)
                },
                "roi": {
                    "30_days": (monthly_revenue - random.uniform(20, 200)) / random.uniform(50, 500),
                    "90_days": (monthly_revenue * 3 * 1.2 - random.uniform(20, 200) * 3) / random.uniform(50, 500),
                    "180_days": (monthly_revenue * 6 * 1.5 - random.uniform(20, 200) * 6) / random.uniform(50, 500)
                },
                "recommendations": recommendations
            }
        except Exception as e:
            logger.error(f"Error al simular activación de estrategia: {str(e)}")
            return {"error": str(e)}
    
    def _generate_action_plan(self, missing_requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Genera un plan de acción para cumplir con los requisitos faltantes.
        
        Args:
            missing_requirements: Lista de requisitos faltantes
            
        Returns:
            Lista de acciones recomendadas
        """
        action_plan = []
        
        for req in missing_requirements:
            metric = req["metric"]
            gap = req["gap"]
            
            if metric == "followers":
                action_plan.append({
                    "action": "Aumentar seguidores",
                    "target": f"+{gap} seguidores",
                    "strategies": [
                        "Colaborar con otros creadores para exposición cruzada",
                        "Implementar estrategia de hashtags optimizada",
                        "Aumentar frecuencia de publicación en horarios óptimos",
                        "Crear contenido viral con tendencias actuales"
                    ]
                })
            elif metric == "views":
                action_plan.append({
                    "action": "Aumentar visualizaciones",
                    "target": f"+{gap} visualizaciones",
                    "strategies": [
                        "Optimizar títulos y miniaturas para mayor CTR",
                        "Mejorar retención con hooks efectivos",
                        "Promocionar contenido en comunidades relevantes",
                        "Crear series de contenido para aumentar visualizaciones por usuario"
                    ]
                })
            elif metric == "engagement_rate":
                action_plan.append({
                    "action": "Mejorar tasa de engagement",
                    "target": f"+{gap*100:.2f}% de engagement",
                    "strategies": [
                        "Incluir llamados a la acción claros y efectivos",
                        "Hacer preguntas a la audiencia para fomentar comentarios",
                        "Responder activamente a comentarios para crear comunidad",
                        "Crear contenido que genere debate o reacciones emocionales"
                    ]
                })
            elif metric == "retention_rate":
                action_plan.append({
                    "action": "Mejorar retención de audiencia",
                    "target": f"+{gap*100:.2f}% de retención",
                    "strategies": [
                        "Crear contenido serializado que fomente el regreso",
                        "Establecer horario consistente de publicación",
                        "Mejorar calidad de producción y narrativa",
                        "Analizar puntos de abandono y optimizar estructura de contenido"
                    ]
                })
            elif metric == "content_quality":
                action_plan.append({
                    "action": "Mejorar calidad de contenido",
                    "target": f"+{gap*100:.2f}% en calidad",
                    "strategies": [
                        "Invertir en mejor equipamiento de producción",
                        "Mejorar guiones y estructura narrativa",
                        "Optimizar edición y efectos visuales",
                        "Estudiar contenido exitoso en el nicho para identificar elementos de calidad"
                    ]
                })
        
        return action_plan