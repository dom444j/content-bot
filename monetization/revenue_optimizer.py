"""
Módulo de optimización de ingresos.

Este módulo analiza y optimiza las estrategias de monetización para maximizar
los ingresos en todas las plataformas y canales.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
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
logger = logging.getLogger('revenue_optimizer')

class RevenueOptimizer:
    """
    Optimizador de ingresos para maximizar el retorno de inversión en todas las plataformas.
    """
    
    def __init__(self, config_path: str = 'config/monetization.json'):
        """
        Inicializa el optimizador de ingresos.
        
        Args:
            config_path: Ruta al archivo de configuración de monetización
        """
        self.config_path = config_path
        self.load_config()
        
        # Inicializar métricas de ingresos
        self.revenue_metrics = {
            'adsense': {'rpm': 0.0, 'views': 0, 'revenue': 0.0},
            'tiktok_creator_fund': {'rpm': 0.0, 'views': 0, 'revenue': 0.0},
            'instagram_bonuses': {'rpm': 0.0, 'views': 0, 'revenue': 0.0},
            'affiliate': {'clicks': 0, 'conversions': 0, 'revenue': 0.0},
            'products': {'sales': 0, 'revenue': 0.0},
            'nfts': {'sales': 0, 'revenue': 0.0},
            'memberships': {'subscribers': 0, 'revenue': 0.0},
            'sponsorships': {'deals': 0, 'revenue': 0.0},
            'b2b': {'deals': 0, 'revenue': 0.0},
            'tokens': {'holders': 0, 'revenue': 0.0}
        }
        
        # Historial de ingresos por día
        self.revenue_history = defaultdict(lambda: defaultdict(float))
        
        # Estrategias de monetización por canal
        self.channel_strategies = {}
        
        # Cargar datos históricos si existen
        self.load_revenue_data()
    
    def load_config(self) -> None:
        """Carga la configuración de monetización desde el archivo JSON."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info("Configuración de monetización cargada correctamente")
            else:
                # Configuración por defecto
                self.config = {
                    "strategies": {
                        "adsense": {
                            "enabled": True,
                            "priority": 5,
                            "min_rpm": 0.5,
                            "max_rpm": 20.0,
                            "avg_rpm": 3.5
                        },
                        "tiktok_creator_fund": {
                            "enabled": True,
                            "priority": 3,
                            "min_rpm": 0.02,
                            "max_rpm": 0.04,
                            "avg_rpm": 0.03
                        },
                        "instagram_bonuses": {
                            "enabled": True,
                            "priority": 3,
                            "min_rpm": 0.05,
                            "max_rpm": 0.2,
                            "avg_rpm": 0.1
                        },
                        "affiliate": {
                            "enabled": True,
                            "priority": 4,
                            "commission_rate": 0.15,
                            "conversion_rate": 0.02,
                            "avg_order_value": 50.0
                        },
                        "products": {
                            "enabled": False,
                            "priority": 4,
                            "margin": 0.7,
                            "conversion_rate": 0.01
                        },
                        "nfts": {
                            "enabled": False,
                            "priority": 2,
                            "avg_price": 100.0,
                            "royalty_rate": 0.1
                        },
                        "memberships": {
                            "enabled": False,
                            "priority": 3,
                            "monthly_fee": 5.0,
                            "retention_rate": 0.7
                        },
                        "sponsorships": {
                            "enabled": False,
                            "priority": 4,
                            "min_deal": 100.0,
                            "max_deal": 1000.0
                        },
                        "b2b": {
                            "enabled": False,
                            "priority": 5,
                            "min_deal": 500.0,
                            "max_deal": 2000.0
                        },
                        "tokens": {
                            "enabled": False,
                            "priority": 1,
                            "token_value": 1.0,
                            "revenue_share": 0.5
                        }
                    },
                    "channel_types": {
                        "finanzas": {
                            "adsense_rpm": 8.0,
                            "affiliate_conversion": 0.04,
                            "product_conversion": 0.02,
                            "recommended_strategies": ["adsense", "affiliate", "products", "b2b"]
                        },
                        "tecnología": {
                            "adsense_rpm": 6.0,
                            "affiliate_conversion": 0.03,
                            "product_conversion": 0.015,
                            "recommended_strategies": ["adsense", "affiliate", "sponsorships", "b2b"]
                        },
                        "gaming": {
                            "adsense_rpm": 4.0,
                            "affiliate_conversion": 0.025,
                            "product_conversion": 0.02,
                            "recommended_strategies": ["adsense", "memberships", "sponsorships", "nfts"]
                        },
                        "salud": {
                            "adsense_rpm": 5.0,
                            "affiliate_conversion": 0.035,
                            "product_conversion": 0.025,
                            "recommended_strategies": ["adsense", "affiliate", "products", "memberships"]
                        },
                        "humor": {
                            "adsense_rpm": 3.0,
                            "affiliate_conversion": 0.015,
                            "product_conversion": 0.01,
                            "recommended_strategies": ["adsense", "sponsorships", "memberships", "nfts"]
                        }
                    },
                    "cta_optimization": {
                        "timing": {
                            "early": {"start": 0, "end": 3, "effectiveness": 0.6},
                            "middle": {"start": 4, "end": 8, "effectiveness": 0.9},
                            "end": {"start": -5, "end": 0, "effectiveness": 0.7}
                        },
                        "types": {
                            "subscribe": {"effectiveness": 0.8, "revenue_impact": 0.3},
                            "affiliate": {"effectiveness": 0.6, "revenue_impact": 0.7},
                            "product": {"effectiveness": 0.5, "revenue_impact": 0.8},
                            "membership": {"effectiveness": 0.4, "revenue_impact": 0.9}
                        }
                    },
                    "thresholds": {
                        "min_subscribers_for_memberships": 1000,
                        "min_subscribers_for_sponsorships": 5000,
                        "min_subscribers_for_products": 2000,
                        "min_subscribers_for_b2b": 10000,
                        "min_subscribers_for_tokens": 20000
                    }
                }
                
                # Guardar configuración por defecto
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2)
                logger.info("Creada configuración de monetización por defecto")
        except Exception as e:
            logger.error(f"Error al cargar la configuración de monetización: {e}")
            self.config = {"strategies": {}, "channel_types": {}, "cta_optimization": {}, "thresholds": {}}
    
    def load_revenue_data(self) -> None:
        """Carga los datos históricos de ingresos si existen."""
        revenue_data_path = 'data/revenue_history.json'
        try:
            if os.path.exists(revenue_data_path):
                with open(revenue_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Convertir datos a formato interno
                    for date_str, sources in data.get('daily', {}).items():
                        for source, amount in sources.items():
                            self.revenue_history[date_str][source] = float(amount)
                    
                    # Cargar métricas actuales
                    if 'current_metrics' in data:
                        self.revenue_metrics = data['current_metrics']
                    
                    # Cargar estrategias por canal
                    if 'channel_strategies' in data:
                        self.channel_strategies = data['channel_strategies']
                    
                logger.info("Datos históricos de ingresos cargados correctamente")
        except Exception as e:
            logger.error(f"Error al cargar datos históricos de ingresos: {e}")
    
    def save_revenue_data(self) -> None:
        """Guarda los datos históricos de ingresos."""
        revenue_data_path = 'data/revenue_history.json'
        try:
            os.makedirs(os.path.dirname(revenue_data_path), exist_ok=True)
            
            # Preparar datos para guardar
            data = {
                'daily': dict(self.revenue_history),
                'current_metrics': self.revenue_metrics,
                'channel_strategies': self.channel_strategies,
                'last_update': datetime.now().isoformat()
            }
            
            with open(revenue_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info("Datos de ingresos guardados correctamente")
        except Exception as e:
            logger.error(f"Error al guardar datos de ingresos: {e}")
    
    def update_revenue(self, source: str, amount: float, metrics: Dict[str, Any] = None) -> None:
        """
        Actualiza los ingresos para una fuente específica.
        
        Args:
            source: Fuente de ingresos (adsense, affiliate, etc.)
            amount: Cantidad de ingresos
            metrics: Métricas adicionales (vistas, clics, etc.)
        """
        # Actualizar historial diario
        today = datetime.now().strftime('%Y-%m-%d')
        self.revenue_history[today][source] += amount
        
        # Actualizar métricas específicas
        if source in self.revenue_metrics:
            self.revenue_metrics[source]['revenue'] += amount
            
            if metrics:
                for key, value in metrics.items():
                    if key in self.revenue_metrics[source]:
                        self.revenue_metrics[source][key] += value
        
        # Guardar datos actualizados
        self.save_revenue_data()
        
        logger.info(f"Ingresos actualizados: {source}, +${amount:.2f}")
    
    def get_total_revenue(self, days: int = 30) -> Dict[str, Any]:
        """
        Obtiene los ingresos totales para un período específico.
        
        Args:
            days: Número de días para calcular los ingresos
            
        Returns:
            Diccionario con ingresos totales y desglose por fuente
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filtrar datos por rango de fechas
        filtered_data = {}
        for date_str, sources in self.revenue_history.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            if start_date <= date <= end_date:
                filtered_data[date_str] = sources
        
        # Calcular totales por fuente
        totals_by_source = defaultdict(float)
        for date_str, sources in filtered_data.items():
            for source, amount in sources.items():
                totals_by_source[source] += amount
        
        # Calcular total general
        total = sum(totals_by_source.values())
        
        # Calcular porcentajes
        percentages = {}
        if total > 0:
            for source, amount in totals_by_source.items():
                percentages[source] = (amount / total) * 100
        
        # Crear resultado
        result = {
            'total': total,
            'by_source': dict(totals_by_source),
            'percentages': percentages,
            'period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'days': days
            }
        }
        
        return result
    
    def get_revenue_trend(self, days: int = 30, source: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene la tendencia de ingresos para un período específico.
        
        Args:
            days: Número de días para calcular la tendencia
            source: Fuente específica (opcional)
            
        Returns:
            Diccionario con tendencia de ingresos
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generar lista de fechas
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        # Inicializar datos de tendencia
        trend_data = {date: 0.0 for date in date_range}
        
        # Llenar datos de tendencia
        for date_str in date_range:
            if date_str in self.revenue_history:
                if source:
                    # Solo una fuente específica
                    trend_data[date_str] = self.revenue_history[date_str].get(source, 0.0)
                else:
                    # Todas las fuentes
                    trend_data[date_str] = sum(self.revenue_history[date_str].values())
        
        # Calcular métricas de tendencia
        values = list(trend_data.values())
        
        if len(values) > 1:
            # Calcular cambio porcentual
            first_week = sum(values[:7]) if len(values) >= 7 else sum(values[:len(values)//2])
            last_week = sum(values[-7:]) if len(values) >= 7 else sum(values[len(values)//2:])
            
            if first_week > 0:
                percent_change = ((last_week - first_week) / first_week) * 100
            else:
                percent_change = 0.0
            
            # Calcular tendencia (pendiente de la línea)
            x = np.arange(len(values))
            if len(x) > 1:
                slope, _ = np.polyfit(x, values, 1)
            else:
                slope = 0.0
            
            trend_direction = "up" if slope > 0 else "down" if slope < 0 else "stable"
        else:
            percent_change = 0.0
            trend_direction = "stable"
        
        # Crear resultado
        result = {
            'data': trend_data,
            'metrics': {
                'total': sum(values),
                'average_daily': sum(values) / len(values) if values else 0,
                'percent_change': percent_change,
                'trend_direction': trend_direction
            },
            'period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'days': days
            }
        }
        
        return result
    
    def optimize_monetization_strategy(self, channel_id: str, channel_type: str, 
                                      subscribers: int, avg_views: int) -> Dict[str, Any]:
        """
        Optimiza la estrategia de monetización para un canal específico.
        
        Args:
            channel_id: Identificador del canal
            channel_type: Tipo de canal (finanzas, tecnología, etc.)
            subscribers: Número de suscriptores
            avg_views: Promedio de vistas por video
            
        Returns:
            Estrategia de monetización optimizada
        """
        logger.info(f"Optimizando estrategia para canal {channel_id} ({channel_type})")
        
        # Obtener configuración para el tipo de canal
        channel_config = self.config.get('channel_types', {}).get(channel_type, {})
        if not channel_config:
            logger.warning(f"No hay configuración para el tipo de canal: {channel_type}")
            channel_config = {
                "adsense_rpm": 3.0,
                "affiliate_conversion": 0.02,
                "product_conversion": 0.01,
                "recommended_strategies": ["adsense"]
            }
        
        # Verificar umbrales para habilitar estrategias
        thresholds = self.config.get('thresholds', {})
        
        enabled_strategies = []
        
        # Siempre habilitar anuncios y fondos de creadores
        enabled_strategies.extend(["adsense", "tiktok_creator_fund", "instagram_bonuses"])
        
        # Verificar otras estrategias basadas en umbrales
        if subscribers >= thresholds.get('min_subscribers_for_memberships', 1000):
            enabled_strategies.append("memberships")
        
        if subscribers >= thresholds.get('min_subscribers_for_sponsorships', 5000):
            enabled_strategies.append("sponsorships")
        
        if subscribers >= thresholds.get('min_subscribers_for_products', 2000):
            enabled_strategies.append("products")
        
        if subscribers >= thresholds.get('min_subscribers_for_b2b', 10000):
            enabled_strategies.append("b2b")
        
        if subscribers >= thresholds.get('min_subscribers_for_tokens', 20000):
            enabled_strategies.append("tokens")
        
        # Siempre habilitar afiliados
        enabled_strategies.append("affiliate")
        
        # Calcular ingresos proyectados para cada estrategia
        projected_revenue = {}
        
        # AdSense / YouTube
        adsense_rpm = channel_config.get('adsense_rpm', 3.0)
        projected_revenue['adsense'] = (avg_views / 1000) * adsense_rpm
        
        # TikTok Creator Fund
        tiktok_rpm = self.config.get('strategies', {}).get('tiktok_creator_fund', {}).get('avg_rpm', 0.03)
        projected_revenue['tiktok_creator_fund'] = (avg_views / 1000) * tiktok_rpm
        
        # Instagram Bonuses
        instagram_rpm = self.config.get('strategies', {}).get('instagram_bonuses', {}).get('avg_rpm', 0.1)
        projected_revenue['instagram_bonuses'] = (avg_views / 1000) * instagram_rpm
        
        # Afiliados
        affiliate_config = self.config.get('strategies', {}).get('affiliate', {})
        affiliate_conversion = channel_config.get('affiliate_conversion', 0.02)
        avg_order_value = affiliate_config.get('avg_order_value', 50.0)
        commission_rate = affiliate_config.get('commission_rate', 0.15)
        
        projected_revenue['affiliate'] = avg_views * affiliate_conversion * avg_order_value * commission_rate
        
        # Productos propios
        if "products" in enabled_strategies:
            product_config = self.config.get('strategies', {}).get('products', {})
            product_conversion = channel_config.get('product_conversion', 0.01)
            avg_product_price = 50.0  # Precio promedio de producto
            margin = product_config.get('margin', 0.7)
            
            projected_revenue['products'] = avg_views * product_conversion * avg_product_price * margin
        
        # Membresías
        if "memberships" in enabled_strategies:
            membership_config = self.config.get('strategies', {}).get('memberships', {})
            monthly_fee = membership_config.get('monthly_fee', 5.0)
            retention_rate = membership_config.get('retention_rate', 0.7)
            conversion_rate = 0.01  # 1% de suscriptores se convierten en miembros
            
            projected_revenue['memberships'] = subscribers * conversion_rate * monthly_fee * retention_rate
        
        # Patrocinios
        if "sponsorships" in enabled_strategies:
            sponsorship_config = self.config.get('strategies', {}).get('sponsorships', {})
            min_deal = sponsorship_config.get('min_deal', 100.0)
            max_deal = sponsorship_config.get('max_deal', 1000.0)
            
            # Escalar según suscriptores
            deal_value = min_deal + (max_deal - min_deal) * min(1.0, subscribers / 100000)
            deals_per_month = 0.5  # Promedio de 1 cada 2 meses
            
            projected_revenue['sponsorships'] = deal_value * deals_per_month
        
        # B2B
        if "b2b" in enabled_strategies:
            b2b_config = self.config.get('strategies', {}).get('b2b', {})
            min_deal = b2b_config.get('min_deal', 500.0)
            max_deal = b2b_config.get('max_deal', 2000.0)
            
            # Escalar según suscriptores
            deal_value = min_deal + (max_deal - min_deal) * min(1.0, subscribers / 200000)
            deals_per_month = 0.25  # Promedio de 1 cada 4 meses
            
            projected_revenue['b2b'] = deal_value * deals_per_month
        
        # Tokens
        if "tokens" in enabled_strategies:
            token_config = self.config.get('strategies', {}).get('tokens', {})
            token_value = token_config.get('token_value', 1.0)
            revenue_share = token_config.get('revenue_share', 0.5)
            token_holders = subscribers * 0.005  # 0.5% de suscriptores compran tokens
            
            projected_revenue['tokens'] = token_holders * token_value * revenue_share
        
        # Ordenar estrategias por ingresos proyectados
        sorted_strategies = sorted(
            [(strategy, revenue) for strategy, revenue in projected_revenue.items() if strategy in enabled_strategies],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Crear estrategia optimizada
        optimized_strategy = {
            'channel_id': channel_id,
            'channel_type': channel_type,
            'subscribers': subscribers,
            'avg_views': avg_views,
            'enabled_strategies': enabled_strategies,
            'prioritized_strategies': [s[0] for s in sorted_strategies],
            'projected_revenue': {s[0]: s[1] for s in sorted_strategies},
            'total_projected_monthly': sum(s[1] for s in sorted_strategies),
            'recommended_cta_types': self._get_recommended_cta_types(sorted_strategies),
            'recommended_cta_timing': self._get_recommended_cta_timing(),
            'last_updated': datetime.now().isoformat()
        }
        
        # Guardar estrategia para el canal
        self.channel_strategies[channel_id] = optimized_strategy
        self.save_revenue_data()
        
        return optimized_strategy
    
    def _get_recommended_cta_types(self, sorted_strategies: List[Tuple[str, float]]) -> List[str]:
        """
        Obtiene los tipos de CTA recomendados basados en las estrategias priorizadas.
        
        Args:
            sorted_strategies: Lista de estrategias ordenadas por ingresos proyectados
            
        Returns:
            Lista de tipos de CTA recomendados
        """
        cta_mapping = {
            'adsense': 'subscribe',
            'tiktok_creator_fund': 'subscribe',
            'instagram_bonuses': 'subscribe',
            'affiliate': 'affiliate',
            'products': 'product',
            'memberships': 'membership',
            'sponsorships': 'subscribe',
            'b2b': 'subscribe',
            'tokens': 'token'
        }
        
        # Obtener los 3 mejores tipos de CTA
        top_strategies = sorted_strategies[:3]
        cta_types = [cta_mapping.get(strategy, 'subscribe') for strategy, _ in top_strategies]
        
        # Eliminar duplicados manteniendo el orden
        unique_cta_types = []
        for cta in cta_types:
            if cta not in unique_cta_types:
                unique_cta_types.append(cta)
        
        return unique_cta_types
    
    def _get_recommended_cta_timing(self) -> Dict[str, Any]:
        """
        Obtiene el timing recomendado para CTAs basado en la configuración.
        
        Returns:
            Diccionario con timing recomendado
        """
        cta_timing = self.config.get('cta_optimization', {}).get('timing', {})
        
        # Encontrar el timing más efectivo
        best_timing = max(cta_timing.items(), key=lambda x: x[1].get('effectiveness', 0))
        
        return {
            'position': best_timing[0],
            'start_second': best_timing[1].get('start'),
            'end_second': best_timing[1].get('end'),
            'effectiveness': best_timing[1].get('effectiveness')
        }
    
    def get_channel_strategy(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene la estrategia de monetización para un canal específico.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Estrategia de monetización o None si no existe
        """
        return self.channel_strategies.get(channel_id)
    
    def get_all_channel_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Obtiene todas las estrategias de monetización.
        
        Returns:
            Diccionario con todas las estrategias de monetización
        """
        return self.channel_strategies
    
    def get_revenue_metrics(self) -> Dict[str, Any]:
        """
        Obtiene las métricas actuales de ingresos.
        
        Returns:
            Diccionario con métricas de ingresos
        """
        return self.revenue_metrics
    
    def get_revenue_by_source(self, days: int = 30) -> Dict[str, float]:
        """
        Obtiene los ingresos por fuente para un período específico.
        
        Args:
            days: Número de días para calcular los ingresos
            
        Returns:
            Diccionario con ingresos por fuente
        """
        revenue_data = self.get_total_revenue(days)
        return revenue_data.get('by_source', {})
    
    def get_best_performing_strategy(self, days: int = 30) -> Tuple[str, float]:
        """
        Obtiene la estrategia de monetización con mejor desempeño.
        
        Args:
            days: Número de días para calcular el desempeño
            
        Returns:
            Tupla con la estrategia y el monto de ingresos
        """
        revenue_by_source = self.get_revenue_by_source(days)
        
        if not revenue_by_source:
            return ("none", 0.0)
        
        best_strategy = max(revenue_by_source.items(), key=lambda x: x[1])
        return best_strategy
    
    def get_roi_by_channel(self, channel_costs: Dict[str, float], days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Calcula el ROI por canal basado en los costos proporcionados.
        
        Args:
            channel_costs: Diccionario con costos por canal
            days: Número de días para calcular el ROI
            
        Returns:
            Diccionario con ROI por canal
        """
        roi_data = {}
        
        for channel_id, strategy in self.channel_strategies.items():
            # Obtener ingresos proyectados mensuales
            monthly_revenue = strategy.get('total_projected_monthly', 0)
            
            # Obtener costo mensual
            monthly_cost = channel_costs.get(channel_id, 0)
            
            if monthly_cost > 0:
                # Calcular ROI
                roi = ((monthly_revenue - monthly_cost) / monthly_cost) * 100
            else:
                roi = 0
            
            roi_data[channel_id] = {
                'monthly_revenue': monthly_revenue,
                'monthly_cost': monthly_cost,
                'roi': roi,
                'profitable': roi > 0
            }
        
        return roi_data
    
    def get_cta_recommendations(self, channel_id: str, video_duration: int) -> Dict[str, Any]:
        """
        Obtiene recomendaciones de CTA para un video específico.
        
        Args:
            channel_id: Identificador del canal
            video_duration: Duración del video en segundos
            
        Returns:
            Diccionario con recomendaciones de CTA
        """
        strategy = self.get_channel_strategy(channel_id)
        
        if not strategy:
            # Estrategia por defecto
            return {
                'types': ['subscribe'],
                'timing': {
                    'position': 'middle',
                                        'seconds': min(6, video_duration // 2)
                }
            }
        
        # Obtener recomendaciones basadas en la estrategia del canal
        cta_types = strategy.get('recommended_cta_types', ['subscribe'])
        cta_timing = strategy.get('recommended_cta_timing', {})
        
        # Calcular segundos específicos basados en la duración del video
        position = cta_timing.get('position', 'middle')
        
        if position == 'early':
            seconds = min(3, video_duration // 10)
        elif position == 'end':
            seconds = max(video_duration - 10, video_duration * 0.9)
        else:  # middle
            seconds = video_duration // 2
        
        # Crear recomendaciones
        recommendations = {
            'types': cta_types[:2],  # Limitar a 2 tipos de CTA
            'timing': {
                'position': position,
                'seconds': int(seconds)
            },
            'message_templates': self._get_cta_message_templates(cta_types[:1]),
            'effectiveness': cta_timing.get('effectiveness', 0.7)
        }
        
        return recommendations
    
    def _get_cta_message_templates(self, cta_types: List[str]) -> Dict[str, str]:
        """
        Obtiene plantillas de mensajes para tipos de CTA específicos.
        
        Args:
            cta_types: Lista de tipos de CTA
            
        Returns:
            Diccionario con plantillas de mensajes
        """
        templates = {
            'subscribe': [
                "Si te gustó este video, no olvides suscribirte al canal",
                "Suscríbete para más contenido como este",
                "Haz clic en el botón de suscripción para no perderte ningún video"
            ],
            'affiliate': [
                "Encuentra todos los productos mencionados en la descripción",
                "Usa mi código de descuento para obtener un 10% de descuento",
                "Visita el enlace en la descripción para más información"
            ],
            'product': [
                "Consigue mi curso/producto en el enlace de la descripción",
                "Mi libro/guía está disponible ahora, encuentra el enlace abajo",
                "Lanza especial: 20% de descuento en mi producto esta semana"
            ],
            'membership': [
                "Únete a la membresía para acceder a contenido exclusivo",
                "Los miembros reciben videos anticipados y contenido adicional",
                "Apoya este canal uniéndote como miembro"
            ],
            'token': [
                "Adquiere tokens para participar en decisiones del canal",
                "Los poseedores de tokens reciben beneficios exclusivos",
                "Invierte en el futuro de este canal con nuestros tokens"
            ]
        }
        
        result = {}
        for cta_type in cta_types:
            if cta_type in templates:
                # Seleccionar aleatoriamente una plantilla
                import random
                result[cta_type] = random.choice(templates[cta_type])
        
        return result
    
    def analyze_revenue_opportunities(self, channel_id: str) -> Dict[str, Any]:
        """
        Analiza oportunidades de ingresos para un canal específico.
        
        Args:
            channel_id: Identificador del canal
            
        Returns:
            Diccionario con oportunidades de ingresos
        """
        strategy = self.get_channel_strategy(channel_id)
        
        if not strategy:
            return {
                'channel_id': channel_id,
                'status': 'no_strategy',
                'message': 'No hay estrategia de monetización para este canal',
                'opportunities': []
            }
        
        # Obtener estrategias habilitadas y priorizadas
        enabled = set(strategy.get('enabled_strategies', []))
        prioritized = strategy.get('prioritized_strategies', [])
        
        # Identificar estrategias no habilitadas que podrían ser oportunidades
        all_strategies = set(self.config.get('strategies', {}).keys())
        potential_strategies = all_strategies - enabled
        
        # Verificar umbrales para estrategias potenciales
        subscribers = strategy.get('subscribers', 0)
        thresholds = self.config.get('thresholds', {})
        
        opportunities = []
        
        # Verificar cada estrategia potencial
        for strat in potential_strategies:
            opportunity = {
                'strategy': strat,
                'status': 'not_eligible',
                'threshold': 0,
                'current': subscribers,
                'gap': 0,
                'estimated_revenue': 0.0,
                'difficulty': 'high'
            }
            
            # Verificar elegibilidad basada en umbrales
            if strat == 'memberships':
                threshold = thresholds.get('min_subscribers_for_memberships', 1000)
                opportunity['threshold'] = threshold
                opportunity['gap'] = max(0, threshold - subscribers)
                
                if subscribers >= threshold:
                    opportunity['status'] = 'eligible'
                    
                    # Estimar ingresos
                    monthly_fee = self.config.get('strategies', {}).get('memberships', {}).get('monthly_fee', 5.0)
                    conversion_rate = 0.01  # 1% de suscriptores
                    retention_rate = self.config.get('strategies', {}).get('memberships', {}).get('retention_rate', 0.7)
                    
                    opportunity['estimated_revenue'] = subscribers * conversion_rate * monthly_fee * retention_rate
                    opportunity['difficulty'] = 'medium'
            
            elif strat == 'sponsorships':
                threshold = thresholds.get('min_subscribers_for_sponsorships', 5000)
                opportunity['threshold'] = threshold
                opportunity['gap'] = max(0, threshold - subscribers)
                
                if subscribers >= threshold:
                    opportunity['status'] = 'eligible'
                    
                    # Estimar ingresos
                    sponsorship_config = self.config.get('strategies', {}).get('sponsorships', {})
                    min_deal = sponsorship_config.get('min_deal', 100.0)
                    max_deal = sponsorship_config.get('max_deal', 1000.0)
                    
                    deal_value = min_deal + (max_deal - min_deal) * min(1.0, subscribers / 100000)
                    deals_per_month = 0.5  # Promedio de 1 cada 2 meses
                    
                    opportunity['estimated_revenue'] = deal_value * deals_per_month
                    opportunity['difficulty'] = 'medium'
            
            elif strat == 'products':
                threshold = thresholds.get('min_subscribers_for_products', 2000)
                opportunity['threshold'] = threshold
                opportunity['gap'] = max(0, threshold - subscribers)
                
                if subscribers >= threshold:
                    opportunity['status'] = 'eligible'
                    
                    # Estimar ingresos
                    avg_views = strategy.get('avg_views', 0)
                    product_config = self.config.get('strategies', {}).get('products', {})
                    channel_type = strategy.get('channel_type', 'general')
                    channel_config = self.config.get('channel_types', {}).get(channel_type, {})
                    
                    product_conversion = channel_config.get('product_conversion', 0.01)
                    avg_product_price = 50.0
                    margin = product_config.get('margin', 0.7)
                    
                    opportunity['estimated_revenue'] = avg_views * product_conversion * avg_product_price * margin
                    opportunity['difficulty'] = 'high'
            
            elif strat == 'b2b':
                threshold = thresholds.get('min_subscribers_for_b2b', 10000)
                opportunity['threshold'] = threshold
                opportunity['gap'] = max(0, threshold - subscribers)
                
                if subscribers >= threshold:
                    opportunity['status'] = 'eligible'
                    
                    # Estimar ingresos
                    b2b_config = self.config.get('strategies', {}).get('b2b', {})
                    min_deal = b2b_config.get('min_deal', 500.0)
                    max_deal = b2b_config.get('max_deal', 2000.0)
                    
                    deal_value = min_deal + (max_deal - min_deal) * min(1.0, subscribers / 200000)
                    deals_per_month = 0.25  # Promedio de 1 cada 4 meses
                    
                    opportunity['estimated_revenue'] = deal_value * deals_per_month
                    opportunity['difficulty'] = 'high'
            
            elif strat == 'tokens':
                threshold = thresholds.get('min_subscribers_for_tokens', 20000)
                opportunity['threshold'] = threshold
                opportunity['gap'] = max(0, threshold - subscribers)
                
                if subscribers >= threshold:
                    opportunity['status'] = 'eligible'
                    
                    # Estimar ingresos
                    token_config = self.config.get('strategies', {}).get('tokens', {})
                    token_value = token_config.get('token_value', 1.0)
                    revenue_share = token_config.get('revenue_share', 0.5)
                    token_holders = subscribers * 0.005  # 0.5% de suscriptores
                    
                    opportunity['estimated_revenue'] = token_holders * token_value * revenue_share
                    opportunity['difficulty'] = 'very_high'
            
            # Añadir oportunidad si es elegible o está cerca
            if opportunity['status'] == 'eligible' or opportunity['gap'] < subscribers * 0.5:
                opportunities.append(opportunity)
        
        # Ordenar oportunidades por ingresos estimados
        opportunities.sort(key=lambda x: x['estimated_revenue'], reverse=True)
        
        # Crear resultado
        result = {
            'channel_id': channel_id,
            'status': 'analyzed',
            'current_strategies': len(enabled),
            'current_monthly_revenue': strategy.get('total_projected_monthly', 0),
            'opportunities': opportunities,
            'potential_additional_revenue': sum(o['estimated_revenue'] for o in opportunities),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def get_revenue_forecast(self, channel_id: str, months: int = 6) -> Dict[str, Any]:
        """
        Genera un pronóstico de ingresos para un canal específico.
        
        Args:
            channel_id: Identificador del canal
            months: Número de meses para el pronóstico
            
        Returns:
            Diccionario con pronóstico de ingresos
        """
        strategy = self.get_channel_strategy(channel_id)
        
        if not strategy:
            return {
                'channel_id': channel_id,
                'status': 'no_strategy',
                'message': 'No hay estrategia de monetización para este canal',
                'forecast': []
            }
        
        # Obtener ingresos mensuales actuales
        current_monthly = strategy.get('total_projected_monthly', 0)
        
        # Estimar tasa de crecimiento mensual (entre 5% y 15%)
        import random
        monthly_growth_rate = random.uniform(0.05, 0.15)
        
        # Generar pronóstico
        forecast = []
        
        for month in range(1, months + 1):
            # Calcular ingresos para el mes
            monthly_revenue = current_monthly * (1 + monthly_growth_rate) ** (month - 1)
            
            # Añadir variabilidad
            variability = random.uniform(-0.1, 0.1)
            monthly_revenue = monthly_revenue * (1 + variability)
            
            # Crear punto de pronóstico
            forecast_point = {
                'month': month,
                'revenue': round(monthly_revenue, 2),
                'growth_from_previous': round(monthly_growth_rate * 100, 1) if month == 1 else round(((monthly_revenue / forecast[-1]['revenue']) - 1) * 100, 1)
            }
            
            forecast.append(forecast_point)
        
        # Crear resultado
        result = {
            'channel_id': channel_id,
            'current_monthly_revenue': current_monthly,
            'forecast_months': months,
            'total_forecasted_revenue': round(sum(point['revenue'] for point in forecast), 2),
            'average_monthly_growth': round(monthly_growth_rate * 100, 1),
            'forecast': forecast,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def optimize_pricing(self, product_type: str, channel_id: str, 
                        current_price: float, current_conversion: float) -> Dict[str, Any]:
        """
        Optimiza el precio de un producto para maximizar ingresos.
        
        Args:
            product_type: Tipo de producto (course, ebook, membership, etc.)
            channel_id: Identificador del canal
            current_price: Precio actual
            current_conversion: Tasa de conversión actual
            
        Returns:
            Diccionario con recomendaciones de precio
        """
        # Obtener estrategia del canal
        strategy = self.get_channel_strategy(channel_id)
        
        if not strategy:
            return {
                'product_type': product_type,
                'status': 'no_strategy',
                'message': 'No hay estrategia de monetización para este canal',
                'recommendation': 'maintain_current'
            }
        
        # Definir elasticidad de precio por tipo de producto
        elasticity = {
            'course': -1.2,  # Más sensible al precio
            'ebook': -1.5,
            'membership': -0.8,  # Menos sensible al precio
            'physical_product': -1.3,
            'consultation': -0.7,
            'software': -0.9
        }
        
        # Usar elasticidad por defecto si no está definida
        price_elasticity = elasticity.get(product_type, -1.0)
        
        # Calcular conversiones estimadas para diferentes precios
        price_points = []
        
        # Probar precios desde -30% hasta +50% del precio actual
        for percent_change in range(-30, 51, 10):
            new_price = current_price * (1 + percent_change / 100)
            
            # Calcular nueva conversión basada en elasticidad
            # Fórmula: % cambio en conversión = elasticidad * % cambio en precio
            conversion_change = price_elasticity * (percent_change / 100)
            new_conversion = current_conversion * (1 + conversion_change)
            
            # Calcular ingresos
            revenue = new_price * new_conversion
            
            price_points.append({
                'price': round(new_price, 2),
                'conversion_rate': round(new_conversion, 4),
                'revenue_index': round(revenue / (current_price * current_conversion), 2),
                'percent_change': percent_change
            })
        
        # Encontrar el precio óptimo (mayor índice de ingresos)
        optimal_price_point = max(price_points, key=lambda x: x['revenue_index'])
        
        # Determinar recomendación
        if optimal_price_point['percent_change'] > 0:
            recommendation = 'increase_price'
            message = f"Aumentar el precio en {optimal_price_point['percent_change']}% para maximizar ingresos"
        elif optimal_price_point['percent_change'] < 0:
            recommendation = 'decrease_price'
            message = f"Reducir el precio en {abs(optimal_price_point['percent_change'])}% para maximizar ingresos"
        else:
            recommendation = 'maintain_current'
            message = "Mantener el precio actual"
        
        # Crear resultado
        result = {
            'product_type': product_type,
            'current_price': current_price,
            'current_conversion_rate': current_conversion,
            'price_elasticity': price_elasticity,
            'optimal_price': optimal_price_point['price'],
            'estimated_conversion_rate': optimal_price_point['conversion_rate'],
            'revenue_improvement': f"{(optimal_price_point['revenue_index'] - 1) * 100:.1f}%",
            'recommendation': recommendation,
            'message': message,
            'price_points': price_points,
            'timestamp': datetime.now().isoformat()
        }
        
        return result


if __name__ == "__main__":
    # Ejemplo de uso
    optimizer = RevenueOptimizer()
    
    # Simular actualización de ingresos
    optimizer.update_revenue('adsense', 25.50, {'views': 10000, 'rpm': 2.55})
    optimizer.update_revenue('affiliate', 42.75, {'clicks': 150, 'conversions': 5})
    
    # Obtener ingresos totales
    total_revenue = optimizer.get_total_revenue(days=30)
    print(f"Ingresos totales (30 días): ${total_revenue['total']:.2f}")
    print("Desglose por fuente:")
    for source, amount in total_revenue['by_source'].items():
        print(f"- {source}: ${amount:.2f} ({total_revenue['percentages'].get(source, 0):.1f}%)")
    
    # Optimizar estrategia para un canal
    channel_strategy = optimizer.optimize_monetization_strategy(
        channel_id="channel123",
        channel_type="tecnología",
        subscribers=8500,
        avg_views=5000
    )
    
    print(f"\nEstrategia optimizada para canal:")
    print(f"Ingresos mensuales proyectados: ${channel_strategy['total_projected_monthly']:.2f}")
    print(f"Estrategias priorizadas: {', '.join(channel_strategy['prioritized_strategies'][:3])}")
    
    # Obtener recomendaciones de CTA
    cta_recommendations = optimizer.get_cta_recommendations("channel123", 600)
    print(f"\nRecomendaciones de CTA:")
    print(f"Tipos: {', '.join(cta_recommendations['types'])}")
    print(f"Timing: {cta_recommendations['timing']['position']} ({cta_recommendations['timing']['seconds']} segundos)")
    
    # Analizar oportunidades de ingresos
    opportunities = optimizer.analyze_revenue_opportunities("channel123")
    print(f"\nOportunidades de ingresos adicionales:")
    for opp in opportunities['opportunities']:
        if opp['status'] == 'eligible':
            print(f"- {opp['strategy']}: ${opp['estimated_revenue']:.2f}/mes (Dificultad: {opp['difficulty']})")
    
    # Generar pronóstico de ingresos
    forecast = optimizer.get_revenue_forecast("channel123", months=6)
    print(f"\nPronóstico de ingresos para 6 meses:")
    print(f"Total: ${forecast['total_forecasted_revenue']:.2f}")
    print(f"Crecimiento mensual promedio: {forecast['average_monthly_growth']}%")
    
    # Optimizar precio de un producto
    pricing = optimizer.optimize_pricing("course", "channel123", 99.99, 0.02)
    print(f"\nOptimización de precio para curso:")
    print(f"Precio actual: ${pricing['current_price']:.2f}")
    print(f"Precio óptimo: ${pricing['optimal_price']:.2f}")
    print(f"Mejora de ingresos estimada: {pricing['revenue_improvement']}")
    print(f"Recomendación: {pricing['message']}")