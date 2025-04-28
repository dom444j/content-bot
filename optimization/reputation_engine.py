"""
Motor de reputación para evaluar y puntuar CTAs basado en su efectividad y recepción.
Este módulo asigna puntuaciones a diferentes elementos del contenido para optimizar
futuras creaciones y maximizar el engagement y la conversión.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import json
import os
from sklearn.preprocessing import MinMaxScaler

class ReputationEngine:
    """
    Motor de reputación que evalúa y puntúa CTAs y otros elementos del contenido.
    
    Capacidades:
    - Puntuación de CTAs basada en métricas de engagement
    - Evaluación de efectividad de elementos visuales
    - Seguimiento histórico de rendimiento
    - Recomendaciones para optimización
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Inicializa el motor de reputación.
        
        Args:
            data_path: Ruta opcional al archivo de datos de reputación
        """
        self.logger = logging.getLogger(__name__)
        
        # Configurar ruta de datos
        self.data_path = data_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'reputation_data.json'
        )
        
        # Cargar datos existentes o crear nuevos
        self.reputation_data = self._load_data()
        
        # Configurar escaladores para normalización
        self.scalers = {
            'cta': MinMaxScaler(feature_range=(0, 100)),
            'visual': MinMaxScaler(feature_range=(0, 100)),
            'audio': MinMaxScaler(feature_range=(0, 100)),
            'content': MinMaxScaler(feature_range=(0, 100))
        }
        
        # Pesos para diferentes métricas
        self.metric_weights = {
            'cta': {
                'click_through_rate': 0.4,
                'conversion_rate': 0.3,
                'engagement_rate': 0.2,
                'retention_post_cta': 0.1
            },
            'visual': {
                'watch_time': 0.3,
                'engagement_rate': 0.3,
                'shares': 0.2,
                'saves': 0.2
            },
            'audio': {
                'watch_time': 0.4,
                'retention_rate': 0.3,
                'engagement_rate': 0.3
            },
            'content': {
                'watch_time': 0.25,
                'engagement_rate': 0.25,
                'shares': 0.2,
                'follows_gained': 0.15,
                'sentiment_score': 0.15
            }
        }
    
    def _load_data(self) -> Dict[str, Any]:
        """
        Carga datos de reputación existentes o crea una estructura nueva.
        
        Returns:
            Datos de reputación
        """
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Crear estructura de datos inicial
                return {
                    'cta_reputation': {},
                    'visual_reputation': {},
                    'audio_reputation': {},
                    'content_reputation': {},
                    'historical_data': [],
                    'last_updated': datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Error al cargar datos de reputación: {str(e)}")
            # Devolver estructura vacía en caso de error
            return {
                'cta_reputation': {},
                'visual_reputation': {},
                'audio_reputation': {},
                'content_reputation': {},
                'historical_data': [],
                'last_updated': datetime.now().isoformat()
            }
    
    def _save_data(self) -> bool:
        """
        Guarda los datos de reputación en disco.
        
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            
            # Actualizar timestamp
            self.reputation_data['last_updated'] = datetime.now().isoformat()
            
            # Guardar datos
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.reputation_data, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar datos de reputación: {str(e)}")
            return False
    
    def score_cta(self, cta_id: str, metrics: Dict[str, float]) -> float:
        """
        Calcula la puntuación de reputación para un CTA específico.
        
        Args:
            cta_id: Identificador único del CTA
            metrics: Diccionario con métricas de rendimiento
            
        Returns:
            Puntuación de reputación (0-100)
        """
        try:
            # Verificar métricas mínimas requeridas
            required_metrics = ['click_through_rate', 'engagement_rate']
            if not all(metric in metrics for metric in required_metrics):
                self.logger.warning(f"Faltan métricas requeridas para evaluar CTA {cta_id}")
                return 0.0
            
            # Calcular puntuación ponderada
            score = 0.0
            weights_sum = 0.0
            
            for metric, weight in self.metric_weights['cta'].items():
                if metric in metrics:
                    score += metrics[metric] * weight
                    weights_sum += weight
            
            # Normalizar por pesos utilizados
            if weights_sum > 0:
                score = score / weights_sum * 100
            else:
                score = 0.0
            
            # Limitar a rango 0-100
            score = max(0.0, min(100.0, score))
            
            # Guardar puntuación en datos de reputación
            if cta_id not in self.reputation_data['cta_reputation']:
                self.reputation_data['cta_reputation'][cta_id] = {
                    'scores': [],
                    'average_score': 0.0,
                    'count': 0,
                    'first_seen': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
            
            # Actualizar datos
            self.reputation_data['cta_reputation'][cta_id]['scores'].append({
                'score': score,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            # Recalcular promedio
            scores = [entry['score'] for entry in self.reputation_data['cta_reputation'][cta_id]['scores']]
            self.reputation_data['cta_reputation'][cta_id]['average_score'] = sum(scores) / len(scores)
            self.reputation_data['cta_reputation'][cta_id]['count'] = len(scores)
            self.reputation_data['cta_reputation'][cta_id]['last_updated'] = datetime.now().isoformat()
            
            # Guardar datos
            self._save_data()
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error al calcular puntuación de CTA {cta_id}: {str(e)}")
            return 0.0
    
    def score_visual(self, visual_id: str, metrics: Dict[str, float]) -> float:
        """
        Calcula la puntuación de reputación para un elemento visual.
        
        Args:
            visual_id: Identificador único del elemento visual
            metrics: Diccionario con métricas de rendimiento
            
        Returns:
            Puntuación de reputación (0-100)
        """
        try:
            # Verificar métricas mínimas requeridas
            required_metrics = ['watch_time', 'engagement_rate']
            if not all(metric in metrics for metric in required_metrics):
                self.logger.warning(f"Faltan métricas requeridas para evaluar visual {visual_id}")
                return 0.0
            
            # Calcular puntuación ponderada
            score = 0.0
            weights_sum = 0.0
            
            for metric, weight in self.metric_weights['visual'].items():
                if metric in metrics:
                    score += metrics[metric] * weight
                    weights_sum += weight
            
            # Normalizar por pesos utilizados
            if weights_sum > 0:
                score = score / weights_sum * 100
            else:
                score = 0.0
            
            # Limitar a rango 0-100
            score = max(0.0, min(100.0, score))
            
            # Guardar puntuación en datos de reputación
            if visual_id not in self.reputation_data['visual_reputation']:
                self.reputation_data['visual_reputation'][visual_id] = {
                    'scores': [],
                    'average_score': 0.0,
                    'count': 0,
                    'first_seen': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat()
                }
            
            # Actualizar datos
            self.reputation_data['visual_reputation'][visual_id]['scores'].append({
                'score': score,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            # Recalcular promedio
            scores = [entry['score'] for entry in self.reputation_data['visual_reputation'][visual_id]['scores']]
            self.reputation_data['visual_reputation'][visual_id]['average_score'] = sum(scores) / len(scores)
            self.reputation_data['visual_reputation'][visual_id]['count'] = len(scores)
            self.reputation_data['visual_reputation'][visual_id]['last_updated'] = datetime.now().isoformat()
            
            # Guardar datos
            self._save_data()
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error al calcular puntuación visual {visual_id}: {str(e)}")
            return 0.0
    
    def get_cta_reputation(self, cta_id: str) -> Dict[str, Any]:
        """
        Obtiene la información de reputación para un CTA específico.
        
        Args:
            cta_id: Identificador único del CTA
            
        Returns:
            Información de reputación
        """
        if cta_id in self.reputation_data['cta_reputation']:
            return self.reputation_data['cta_reputation'][cta_id]
        else:
            return {
                'average_score': 0.0,
                'count': 0,
                'error': 'CTA no encontrado'
            }
    
    def get_top_ctas(self, limit: int = 10, min_count: int = 3) -> List[Dict[str, Any]]:
        """
        Obtiene los CTAs con mejor puntuación.
        
        Args:
            limit: Número máximo de CTAs a devolver
            min_count: Número mínimo de evaluaciones para considerar un CTA
            
        Returns:
            Lista de CTAs con mejor puntuación
        """
        try:
            # Filtrar CTAs con suficientes evaluaciones
            qualified_ctas = [
                {
                    'cta_id': cta_id,
                    'average_score': data['average_score'],
                    'count': data['count'],
                    'last_updated': data['last_updated']
                }
                for cta_id, data in self.reputation_data['cta_reputation'].items()
                if data['count'] >= min_count
            ]
            
            # Ordenar por puntuación promedio
            sorted_ctas = sorted(
                qualified_ctas,
                key=lambda x: x['average_score'],
                reverse=True
            )
            
            # Limitar resultados
            return sorted_ctas[:limit]
            
        except Exception as e:
            self.logger.error(f"Error al obtener top CTAs: {str(e)}")
            return []
    
    def analyze_cta_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Analiza tendencias en la efectividad de CTAs durante un período.
        
        Args:
            days: Número de días a analizar
            
        Returns:
            Análisis de tendencias
        """
        try:
            # Calcular fecha límite
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Recopilar datos de CTAs en el período
            cta_data = {}
            
            for cta_id, data in self.reputation_data['cta_reputation'].items():
                # Filtrar puntuaciones dentro del período
                recent_scores = [
                    entry for entry in data['scores']
                    if entry['timestamp'] >= cutoff_date
                ]
                
                if recent_scores:
                    # Calcular promedio reciente
                    recent_avg = sum(entry['score'] for entry in recent_scores) / len(recent_scores)
                    
                    # Calcular tendencia (comparación con promedio histórico)
                    trend = recent_avg - data['average_score']
                    
                    cta_data[cta_id] = {
                        'recent_average': recent_avg,
                        'historical_average': data['average_score'],
                        'trend': trend,
                        'count_recent': len(recent_scores),
                        'count_total': data['count']
                    }
            
            # Identificar CTAs con tendencia positiva y negativa
            trending_up = sorted(
                [(cta_id, data) for cta_id, data in cta_data.items() if data['trend'] > 0],
                key=lambda x: x[1]['trend'],
                reverse=True
            )
            
            trending_down = sorted(
                [(cta_id, data) for cta_id, data in cta_data.items() if data['trend'] < 0],
                key=lambda x: x[1]['trend']
            )
            
            return {
                'trending_up': [{'cta_id': cta_id, **data} for cta_id, data in trending_up[:5]],
                'trending_down': [{'cta_id': cta_id, **data} for cta_id, data in trending_down[:5]],
                'analysis_period_days': days,
                'total_ctas_analyzed': len(cta_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error al analizar tendencias de CTAs: {str(e)}")
            return {'error': str(e)}
    
    def recommend_ctas(self, platform: str, content_type: str, audience_segment: str = None) -> List[str]:
        """
        Recomienda CTAs efectivos basados en plataforma, tipo de contenido y segmento de audiencia.
        
        Args:
            platform: Plataforma (youtube, tiktok, instagram, etc.)
            content_type: Tipo de contenido (tutorial, entretenimiento, etc.)
            audience_segment: Segmento de audiencia opcional
            
        Returns:
            Lista de IDs de CTAs recomendados
        """
        try:
            # Filtrar CTAs con metadatos relevantes
            relevant_ctas = []
            
            for cta_id, data in self.reputation_data['cta_reputation'].items():
                # Verificar si hay suficientes datos
                if data['count'] < 3:
                    continue
                
                # Verificar si hay metadatos en las puntuaciones
                has_relevant_metadata = False
                
                for score_entry in data['scores']:
                    if 'metrics' in score_entry and 'metadata' in score_entry:
                        metadata = score_entry['metadata']
                        
                        # Verificar coincidencia de plataforma y tipo de contenido
                        platform_match = metadata.get('platform') == platform
                        content_match = metadata.get('content_type') == content_type
                        
                        # Verificar coincidencia de segmento si se especificó
                        segment_match = True
                        if audience_segment:
                            segment_match = metadata.get('audience_segment') == audience_segment
                        
                        if platform_match and content_match and segment_match:
                            has_relevant_metadata = True
                            break
                
                if has_relevant_metadata:
                    relevant_ctas.append({
                        'cta_id': cta_id,
                        'average_score': data['average_score'],
                        'count': data['count']
                    })
            
            # Ordenar por puntuación
            sorted_ctas = sorted(
                relevant_ctas,
                key=lambda x: x['average_score'],
                reverse=True
            )
            
            # Devolver IDs de los mejores CTAs
            return [cta['cta_id'] for cta in sorted_ctas[:5]]
            
        except Exception as e:
            self.logger.error(f"Error al recomendar CTAs: {str(e)}")
            return []
    
    def update_reputation_from_analytics(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actualiza puntuaciones de reputación basadas en datos de analytics.
        
        Args:
            analytics_data: Datos de analytics con métricas y metadatos
            
        Returns:
            Resumen de actualizaciones
        """
        try:
            updates = {
                'cta_updates': 0,
                'visual_updates': 0,
                'content_updates': 0,
                'audio_updates': 0
            }
            
            # Procesar CTAs
            if 'ctas' in analytics_data:
                for cta_entry in analytics_data['ctas']:
                    if 'cta_id' in cta_entry and 'metrics' in cta_entry:
                        self.score_cta(cta_entry['cta_id'], cta_entry['metrics'])
                        updates['cta_updates'] += 1
            
            # Procesar visuales
            if 'visuals' in analytics_data:
                for visual_entry in analytics_data['visuals']:
                    if 'visual_id' in visual_entry and 'metrics' in visual_entry:
                        self.score_visual(visual_entry['visual_id'], visual_entry['metrics'])
                        updates['visual_updates'] += 1
            
            # Procesar contenido
            if 'content' in analytics_data:
                for content_entry in analytics_data['content']:
                    if 'content_id' in content_entry and 'metrics' in content_entry:
                        # Implementar lógica similar a score_cta para contenido
                        updates['content_updates'] += 1
            
            # Procesar audio
            if 'audio' in analytics_data:
                for audio_entry in analytics_data['audio']:
                    if 'audio_id' in audio_entry and 'metrics' in audio_entry:
                        # Implementar lógica similar a score_cta para audio
                        updates['audio_updates'] += 1
            
            # Guardar datos actualizados
            self._save_data()
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Error al actualizar reputación desde analytics: {str(e)}")
            return {'error': str(e)}
    
    def get_reputation_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del estado actual del sistema de reputación.
        
        Returns:
            Resumen de reputación
        """
        try:
            # Contar elementos
            cta_count = len(self.reputation_data['cta_reputation'])
            visual_count = len(self.reputation_data['visual_reputation'])
            audio_count = len(self.reputation_data['audio_reputation'])
            content_count = len(self.reputation_data['content_reputation'])
            
            # Calcular promedios globales
            cta_avg = 0.0
            if cta_count > 0:
                cta_avg = sum(data['average_score'] for data in self.reputation_data['cta_reputation'].values()) / cta_count
            
            visual_avg = 0.0
            if visual_count > 0:
                visual_avg = sum(data['average_score'] for data in self.reputation_data['visual_reputation'].values()) / visual_count
            
            # Obtener top CTAs
            top_ctas = self.get_top_ctas(limit=5)
            
            return {
                'total_elements': {
                    'ctas': cta_count,
                    'visuals': visual_count,
                    'audio': audio_count,
                    'content': content_count
                },
                'average_scores': {
                    'ctas': cta_avg,
                    'visuals': visual_avg
                },
                'top_ctas': top_ctas,
                'last_updated': self.reputation_data['last_updated']
            }
            
        except Exception as e:
            self.logger.error(f"Error al obtener resumen de reputación: {str(e)}")
            return {'error': str(e)}