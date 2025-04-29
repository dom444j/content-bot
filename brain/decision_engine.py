"""
Decision Engine - Motor de decisiones y auto-mejoras

Este módulo implementa algoritmos avanzados de aprendizaje por refuerzo:
- Contextual bandits (LinUCB) para optimización de CTAs, visuales y voces
- Auto-mejoras basadas en métricas históricas y en tiempo real
- Redistribución de tráfico optimizada por ROI
- Sistema de reintentos automáticos para robustez
- Integración profunda con KnowledgeBase para aprendizaje continuo
- Evaluación continua y optimización avanzada del rendimiento
"""

import os
import sys
import json
import logging
import random
import datetime
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import time
from functools import wraps
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pickle
import bson
from pymongo.binary import Binary
import scipy.stats as stats
from datetime import timedelta

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase
from utils.config_loader import get_config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(context)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'decision_engine.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DecisionEngine')

class DecisionEngineError(Exception):
    """Excepción personalizada para errores del motor de decisiones"""
    pass

class ContextValidationError(DecisionEngineError):
    """Excepción para errores de validación de contexto"""
    pass

def retry_on_failure(func):
    """Decorador para reintentos automáticos en operaciones críticas"""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, DecisionEngineError)),
        before_sleep=lambda retry_state: logger.warning(
            f"Reintentando {func.__name__} (intento {retry_state.attempt_number})..."
        )
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class LinUCBBandit:
    """Implementación de bandit contextual basado en LinUCB"""
    
    def __init__(self, feature_dim: int, alpha: float = 1.0):
        """
        Inicializa el bandit LinUCB
        
        Args:
            feature_dim: Dimensión del vector de características
            alpha: Parámetro de exploración
        """
        self.alpha = alpha
        self.feature_dim = feature_dim
        # Matriz A para cada acción (inicialmente identidad)
        self.A = {}
        # Vector b para cada acción (inicialmente ceros)
        self.b = {}
        # Theta estimado para cada acción
        self.theta = {}
    
    def add_action(self, action: str):
        """Añade una nueva acción al bandit"""
        if action not in self.A:
            self.A[action] = np.identity(self.feature_dim)
            self.b[action] = np.zeros(self.feature_dim)
            self.theta[action] = np.zeros(self.feature_dim)
    
    def select_action(self, context: np.ndarray) -> str:
        """
        Selecciona la mejor acción usando LinUCB
        
        Args:
            context: Vector de características del contexto
            
        Returns:
            Acción seleccionada
        """
        if not self.A:
            return None
            
        scores = {}
        for action in self.A:
            A_inv = np.linalg.inv(self.A[action])
            self.theta[action] = A_inv @ self.b[action]
            mean = context @ self.theta[action]
            variance = np.sqrt(context.T @ A_inv @ context)
            scores[action] = mean + self.alpha * variance
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def update(self, action: str, context: np.ndarray, reward: float):
        """
        Actualiza los parámetros del bandit
        
        Args:
            action: Acción tomada
            context: Vector de características
            reward: Recompensa obtenida
        """
        if action in self.A:
            context = context.reshape(-1, 1)
            self.A[action] += context @ context.T
            self.b[action] += reward * context.flatten()

class DecisionEngine:
    """
    Motor de decisiones avanzado con aprendizaje por refuerzo
    y auto-mejoras optimizadas
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DecisionEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el motor de decisiones si aún no está inicializado"""
        if self._initialized:
            return
            
        logger.info("Inicializando Decision Engine...", extra={'context': 'init'})
        
        # Cargar base de conocimiento
        self.kb = KnowledgeBase()
        
        # Cargar configuración de estrategia
        self.strategy_file = os.path.join('config', 'strategy.json')
        self.strategy = self._load_strategy()
        
        # Configuración de bandits contextuales (LinUCB)
        self.bandits_config = self.strategy.get('optimization_strategies', {}).get('contextual_bandits', {})
        self.exploration_alpha = self.bandits_config.get('exploration_alpha', 1.0)
        self.feature_dim = self.bandits_config.get('feature_dim', 20)
        self.learning_rate = self.bandits_config.get('learning_rate', 0.1)
        self.reward_metrics = self.bandits_config.get('reward_metrics', {
            'ctr': 0.4,
            'conversion_rate': 0.3,
            'engagement_rate': 0.2,
            'retention_rate': 0.1
        })
        
        # Estado de los bandits (LinUCB)
        self.cta_bandits: Dict[str, LinUCBBandit] = {}
        self.visual_bandits: Dict[str, LinUCBBandit] = {}
        self.voice_bandits: Dict[str, LinUCBBandit] = {}
        
        # Configuración de redistribución de tráfico
        self.redistribution_config = self.strategy.get('optimization_strategies', {}).get('traffic_redistribution', {})
        self.ctr_threshold = self.redistribution_config.get('ctr_threshold', 0.05)
        self.roi_threshold = self.redistribution_config.get('roi_threshold', 50)
        
        # Configuración de métricas avanzadas
        self.metrics_config = self.strategy.get('metrics', {
            'confidence_threshold': 0.95,
            'min_samples': 100,
            'performance_window_days': 7
        })
        
        # Cache para decisiones recientes
        self.decision_cache: Dict[str, Dict] = {}
        self.cache_timeout = 3600  # 1 hora en segundos
        
        # Configuración para evaluación continua
        self.evaluation_interval = self.strategy.get('evaluation', {}).get('interval_hours', 6)
        self.last_evaluation_time = datetime.datetime.now() - timedelta(hours=self.evaluation_interval + 1)
        self.performance_history = {}
        
        # Cargar datos históricos
        self._load_bandit_state()
        
        # Programar evaluación continua
        self._schedule_continuous_evaluation()
        
        self._initialized = True
        logger.info("Decision Engine inicializado correctamente", extra={'context': 'init'})
    
    @retry_on_failure
    def _load_strategy(self) -> Dict[str, Any]:
        """Carga la configuración de estrategia desde el archivo JSON"""
        try:
            config = get_config(self.strategy_file)
            if config:
                return config
            logger.warning(f"Archivo de estrategia no encontrado: {self.strategy_file}")
            return self._create_default_strategy()
        except Exception as e:
            logger.error(f"Error al cargar estrategia: {str(e)}", extra={'context': 'strategy_load'})
            raise DecisionEngineError(f"Error al cargar estrategia: {str(e)}")
    
    def _create_default_strategy(self) -> Dict[str, Any]:
        """Crea una configuración de estrategia por defecto"""
        default_strategy = {
            "optimization_strategies": {
                "contextual_bandits": {
                    "exploration_alpha": 1.0,
                    "feature_dim": 20,
                    "learning_rate": 0.1,
                    "reward_metrics": {
                        "ctr": 0.4,
                        "conversion_rate": 0.3,
                        "engagement_rate": 0.2,
                        "retention_rate": 0.1
                    }
                },
                "traffic_redistribution": {
                    "ctr_threshold": 0.05,
                    "roi_threshold": 50,
                    "redistribution_interval_days": 7
                }
            },
            "metrics": {
                "confidence_threshold": 0.95,
                "min_samples": 100,
                "performance_window_days": 7
            },
            "evaluation": {
                "interval_hours": 6
            },
            "cta_strategies": {
                "timing": {
                    "early": {"start": 0, "end": 3},
                    "middle": {"start": 4, "end": 8},
                    "late": {"start": 8, "end": 15}
                },
                "types": ["question", "challenge", "offer", "curiosity", "social_proof"]
            },
            "visual_strategies": {
                "styles": ["minimalist", "vibrant", "dramatic", "playful", "professional"],
                "color_schemes": ["warm", "cool", "neutral", "high_contrast", "pastel"]
            },
            "voice_strategies": {
                "tones": ["enthusiastic", "calm", "authoritative", "friendly", "mysterious"],
                "pacing": ["fast", "moderate", "slow", "dynamic"]
            }
        }
        
        os.makedirs(os.path.dirname(self.strategy_file), exist_ok=True)
        with open(self.strategy_file, 'w', encoding='utf-8') as f:
            json.dump(default_strategy, f, indent=4)
        
        return default_strategy
    
    @retry_on_failure
    def _load_bandit_state(self):
        """Carga el estado de los bandits desde la base de conocimiento"""
        try:
            bandit_state = self.kb.get_from_mongodb('bandits', {'type': 'state'})
            if bandit_state:
                for bandit_type in ['cta_bandits', 'visual_bandits', 'voice_bandits']:
                    bandit_data = bandit_state.get(bandit_type, {})
                    for key, data in bandit_data.items():
                        bandit = LinUCBBandit(feature_dim=self.feature_dim, alpha=self.exploration_alpha)
                        for action, params in data.items():
                            bandit.add_action(action)
                            
                            # Deserializar matrices y vectores usando el formato optimizado
                            if 'A_binary' in params:
                                # Usar deserialización binaria si está disponible
                                bandit.A[action] = pickle.loads(params['A_binary'])
                            else:
                                # Fallback a formato JSON para compatibilidad
                                bandit.A[action] = np.array(params.get('A', np.identity(self.feature_dim)))
                                
                            if 'b_binary' in params:
                                # Usar deserialización binaria si está disponible
                                bandit.b[action] = pickle.loads(params['b_binary'])
                            else:
                                # Fallback a formato JSON para compatibilidad
                                bandit.b[action] = np.array(params.get('b', np.zeros(self.feature_dim)))
                                
                        getattr(self, bandit_type)[key] = bandit
                logger.info("Estado de bandits cargado correctamente", extra={'context': 'bandit_load'})
            else:
                logger.info("No se encontró estado previo de bandits", extra={'context': 'bandit_load'})
        except Exception as e:
            logger.error(f"Error al cargar estado de bandits: {str(e)}", extra={'context': 'bandit_load'})
            raise DecisionEngineError(f"Error al cargar estado de bandits: {str(e)}")
    
    @retry_on_failure
    def _save_bandit_state(self):
        """Guarda el estado actual de los bandits en la base de conocimiento usando serialización binaria"""
        try:
            bandit_state = {
                'type': 'state',
                'cta_bandits': {
                    key: {
                        action: {
                            # Serializar matrices y vectores en formato binario para eficiencia
                            'A_binary': Binary(pickle.dumps(bandit.A[action], protocol=4)),
                            'b_binary': Binary(pickle.dumps(bandit.b[action], protocol=4)),
                            # Mantener versiones JSON para compatibilidad y depuración
                            'A_shape': bandit.A[action].shape,
                            'b_shape': bandit.b[action].shape,
                            'samples': int(np.sum(bandit.A[action].diagonal()))
                        } for action in bandit.A
                    } for key, bandit in self.cta_bandits.items()
                },
                'visual_bandits': {
                    key: {
                        action: {
                            'A_binary': Binary(pickle.dumps(bandit.A[action], protocol=4)),
                            'b_binary': Binary(pickle.dumps(bandit.b[action], protocol=4)),
                            'A_shape': bandit.A[action].shape,
                            'b_shape': bandit.b[action].shape,
                            'samples': int(np.sum(bandit.A[action].diagonal()))
                        } for action in bandit.A
                    } for key, bandit in self.visual_bandits.items()
                },
                'voice_bandits': {
                    key: {
                        action: {
                            'A_binary': Binary(pickle.dumps(bandit.A[action], protocol=4)),
                            'b_binary': Binary(pickle.dumps(bandit.b[action], protocol=4)),
                            'A_shape': bandit.A[action].shape,
                            'b_shape': bandit.b[action].shape,
                            'samples': int(np.sum(bandit.A[action].diagonal()))
                        } for action in bandit.A
                    } for key, bandit in self.voice_bandits.items()
                },
                'last_updated': datetime.datetime.now().isoformat()
            }
            self.kb.save_to_mongodb('bandits', bandit_state, {'type': 'state'})
            logger.info("Estado de bandits guardado correctamente usando serialización binaria", 
                       extra={'context': 'bandit_save'})
        except Exception as e:
            logger.error(f"Error al guardar estado de bandits: {str(e)}", extra={'context': 'bandit_save'})
            raise DecisionEngineError(f"Error al guardar estado de bandits: {str(e)}")
    
    def _validate_context(self, context: Dict[str, Any]) -> None:
        """Valida la integridad del contexto"""
        required_keys = ['niche', 'platform', 'audience']
        for key in required_keys:
            if key not in context:
                raise ContextValidationError(f"Clave requerida '{key}' no encontrada en el contexto")
        
        if not isinstance(context['audience'], dict):
            raise ContextValidationError("El campo 'audience' debe ser un diccionario")
        
        audience_required = ['age_group', 'engagement_level', 'retention_rate']
        for key in audience_required:
            if key not in context['audience']:
                raise ContextValidationError(f"Clave requerida '{key}' no encontrada en audience")
    
    def _get_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extrae y normaliza características del contexto para los bandits
        
        Args:
            context: Diccionario con información contextual
            Returns:
            Vector de características normalizado
        """
        self._validate_context(context)
        features = []
        
        # Características de plataforma (one-hot encoding)
        platforms = ['youtube', 'tiktok', 'instagram', 'threads', 'bluesky', 'x']
        platform = context.get('platform', '').lower()
        for p in platforms:
            features.append(1.0 if p == platform else 0.0)
        
        # Características de nicho (one-hot encoding)
        niches = ['finance', 'health', 'technology', 'gaming', 'humor', 'education']
        niche = context.get('niche', '').lower()
        for n in niches:
            features.append(1.0 if n == niche else 0.0)
        
        # Características de audiencia
        audience = context.get('audience', {})
        features.append(audience.get('age_group', 0) / 5.0)
        features.append(audience.get('engagement_level', 0) / 10.0)
        features.append(audience.get('retention_rate', 0) / 100.0)
        
        # Características temporales
        current_time = datetime.datetime.now()
        features.append(current_time.hour / 24.0)
        features.append(current_time.weekday() / 6.0)
        
        # Características adicionales (tendencias)
        trend_score = context.get('trend_score', 0.5)
        features.append(trend_score / 1.0)
        
        # Asegurar dimensión fija
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim], dtype=np.float32)
    
    def _calculate_reward(self, metrics: Dict[str, float]) -> float:
        """Calcula la recompensa ponderada basada en métricas"""
        reward = 0.0
        for metric_name, weight in self.reward_metrics.items():
            if metric_name in metrics:
                reward += weight * metrics[metric_name]
        return max(0.0, min(1.0, reward))
    
    def _initialize_bandit(self, bandit_key: str, actions: List[str], bandit_dict: Dict) -> LinUCBBandit:
        """Inicializa un nuevo bandit LinUCB"""
        bandit = LinUCBBandit(feature_dim=self.feature_dim, alpha=self.exploration_alpha)
        for action in actions:
            bandit.add_action(action)
        bandit_dict[bandit_key] = bandit
        return bandit
    
    def select_cta_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selecciona la mejor estrategia de CTA usando LinUCB
        
        Args:
            context: Información contextual
            Returns:
            Estrategia de CTA seleccionada
        """
        niche = context.get('niche', 'general')
        platform = context.get('platform', 'general')
        bandit_key = f"{niche}_{platform}"
        
        cta_strategies = self.strategy.get('cta_strategies', {})
        timing_options = list(cta_strategies.get('timing', {}).keys())
        type_options = cta_strategies.get('types', [])
        actions = [f"{timing}_{cta_type}" for timing in timing_options for cta_type in type_options]
        
        if bandit_key not in self.cta_bandits:
            self._initialize_bandit(bandit_key, actions, self.cta_bandits)
        
        context_features = self._get_context_features(context)
        selected_action = self.cta_bandits[bandit_key].select_action(context_features)
        
        if selected_action:
            timing, cta_type = selected_action.split('_', 1)
            timing_details = cta_strategies.get('timing', {}).get(timing, {})
            
            decision = {
                'timing': timing,
                'start_time': timing_details.get('start', 0),
                'end_time': timing_details.get('end', 10),
                'type': cta_type,
                'bandit_key': bandit_key,
                'action': selected_action,
                'confidence': self._calculate_action_confidence(bandit_key, selected_action, 'cta')
            }
            self.decision_cache[f"cta_{bandit_key}_{time.time()}"] = decision
            return decision
        else:
            return {
                'timing': 'middle',
                'start_time': 4,
                'end_time': 8,
                'type': 'question',
                'bandit_key': bandit_key,
                'action': 'middle_question',
                'confidence': 0.5
            }
    
    def select_visual_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selecciona la mejor estrategia visual usando LinUCB
        
        Args:
            context: Información contextual
            Returns:
            Estrategia visual seleccionada
        """
        niche = context.get('niche', 'general')
        platform = context.get('platform', 'general')
        bandit_key = f"{niche}_{platform}"
        
        visual_strategies = self.strategy.get('visual_strategies', {})
        style_options = visual_strategies.get('styles', [])
        color_options = visual_strategies.get('color_schemes', [])
        actions = [f"{style}_{color}" for style in style_options for color in color_options]
        
        if bandit_key not in self.visual_bandits:
            self._initialize_bandit(bandit_key, actions, self.visual_bandits)
        
        context_features = self._get_context_features(context)
        selected_action = self.visual_bandits[bandit_key].select_action(context_features)
        
        if selected_action:
            style, color_scheme = selected_action.split('_', 1)
            decision = {
                'style': style,
                'color_scheme': color_scheme,
                'bandit_key': bandit_key,
                'action': selected_action,
                'confidence': self._calculate_action_confidence(bandit_key, selected_action, 'visual')
            }
            self.decision_cache[f"visual_{bandit_key}_{time.time()}"] = decision
            return decision
        else:
            return {
                'style': 'vibrant',
                'color_scheme': 'warm',
                'bandit_key': bandit_key,
                'action': 'vibrant_warm',
                'confidence': 0.5
            }
    
    def select_voice_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selecciona la mejor estrategia de voz usando LinUCB
        
        Args:
            context: Información contextual
            Returns:
            Estrategia de voz seleccionada
        """
        niche = context.get('niche', 'general')
        character = context.get('character', 'general')
        bandit_key = f"{niche}_{character}"
        
        voice_strategies = self.strategy.get('voice_strategies', {})
        tone_options = voice_strategies.get('tones', [])
        pacing_options = voice_strategies.get('pacing', [])
        actions = [f"{tone}_{pacing}" for tone in tone_options for pacing in pacing_options]
        
        if bandit_key not in self.voice_bandits:
            self._initialize_bandit(bandit_key, actions, self.voice_bandits)
        
        context_features = self._get_context_features(context)
        selected_action = self.voice_bandits[bandit_key].select_action(context_features)
        
        if selected_action:
            tone, pacing = selected_action.split('_', 1)
            decision = {
                'tone': tone,
                'pacing': pacing,
                'bandit_key': bandit_key,
                'action': selected_action,
                'confidence': self._calculate_action_confidence(bandit_key, selected_action, 'voice')
            }
            self.decision_cache[f"voice_{bandit_key}_{time.time()}"] = decision
            return decision
        else:
            return {
                'tone': 'enthusiastic',
                'pacing': 'dynamic',
                'bandit_key': bandit_key,
                'action': 'enthusiastic_dynamic',
                'confidence': 0.5
            }
    
    def _calculate_action_confidence(self, bandit_key: str, action: str, bandit_type: str) -> float:
        """Calcula la confianza en la acción seleccionada"""
        bandit_dict = getattr(self, f"{bandit_type}_bandits")
        if bandit_key not in bandit_dict:
            return 0.5
        
        bandit = bandit_dict[bandit_key]
        if action not in bandit.A:
            return 0.5
        
        # Usar la varianza para estimar la confianza
        context = np.ones(self.feature_dim)  # Contexto dummy para simplificar
        A_inv = np.linalg.inv(bandit.A[action])
        variance = np.sqrt(context.T @ A_inv @ context)
        confidence = 1.0 - min(variance, 1.0)
        return max(0.5, confidence)
    
    @retry_on_failure
    def update_cta_strategy(self, bandit_key: str, action: str, metrics: Dict[str, float], context: Dict[str, Any]):
        """
        Actualiza la estrategia de CTA basada en métricas
        
        Args:
            bandit_key: Clave del bandit
            action: Acción tomada
            metrics: Métricas de rendimiento
            context: Contexto original
        """
        reward = self._calculate_reward(metrics)
        context_features = self._get_context_features(context)
        
        if bandit_key in self.cta_bandits:
            self.cta_bandits[bandit_key].update(action, context_features, reward)
        
        self._save_bandit_state()
        self.kb.update_cta_performance(f"cta_{bandit_key}_{action}", metrics)
        
        logger.info(
            f"Actualizada estrategia CTA {action} con recompensa {reward:.4f}",
            extra={'context': f'cta_update_{bandit_key}'}
        )
    
    @retry_on_failure
    def update_visual_strategy(self, bandit_key: str, action: str, metrics: Dict[str, float], context: Dict[str, Any]):
        """
        Actualiza la estrategia visual basada en métricas
        
        Args:
            bandit_key: Clave del bandit
            action: Acción tomada
            metrics: Métricas de rendimiento
            context: Contexto original
        """
        reward = self._calculate_reward(metrics)
        context_features = self._get_context_features(context)
        
        if bandit_key in self.visual_bandits:
            self.visual_bandits[bandit_key].update(action, context_features, reward)
        
        self._save_bandit_state()
        
        logger.info(
            f"Actualizada estrategia visual {action} con recompensa {reward:.4f}",
            extra={'context': f'visual_update_{bandit_key}'}
        )
    
    @retry_on_failure
    def update_voice_strategy(self, bandit_key: str, action: str, metrics: Dict[str, float], context: Dict[str, Any]):
        """
        Actualiza la estrategia de voz basada en métricas
        
        Args:
            bandit_key: Clave del bandit
            action: Acción tomada
            metrics: Métricas de rendimiento
            context: Contexto original
        """
        reward = self._calculate_reward(metrics)
        context_features = self._get_context_features(context)
        
        if bandit_key in self.voice_bandits:
            self.voice_bandits[bandit_key].update(action, context_features, reward)
        
        self._save_bandit_state()
        
        logger.info(
            f"Actualizada estrategia de voz {action} con recompensa {reward:.4f}",
            extra={'context': f'voice_update_{bandit_key}'}
        )
    
    def should_redistribute_traffic(self, channel_metrics: Dict[str, Dict[str, float]]) -> bool:
        """
        Determina si se debe redistribuir el tráfico entre canales
        
        Args:
            channel_metrics: Métricas por canal
            Returns:
            True si se debe redistribuir, False en caso contrario
        """
        low_ctr_channels = []
        high_roi_channels = []
        
        for channel, metrics in channel_metrics.items():
            if metrics.get('ctr', 0) < self.ctr_threshold:
                low_ctr_channels.append(channel)
            if metrics.get('roi', 0) > self.roi_threshold:
                high_roi_channels.append(channel)
        
        result = len(low_ctr_channels) > 0 and len(high_roi_channels) > 0
        logger.info(
            f"Redistribución de tráfico requerida: {result}",
            extra={'context': 'traffic_redistribution_check'}
        )
        return result
    
    def get_traffic_redistribution_plan(self, channel_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Genera un plan de redistribución de tráfico
        
        Args:
            channel_metrics: Métricas por canal
            Returns:
            Factores de redistribución por canal
        """
        low_ctr_channels = {
            channel: metrics.get('ctr', 0)
            for channel, metrics in channel_metrics.items()
            if metrics.get('ctr', 0) < self.ctr_threshold
        }
        
        high_roi_channels = {
            channel: metrics.get('roi', 0)
            for channel, metrics in channel_metrics.items()
            if metrics.get('roi', 0) > self.roi_threshold
        }
        
        redistribution_plan = {channel: 1.0 for channel in channel_metrics}
        total_reduction = 0.0
        
        for channel, ctr in low_ctr_channels.items():
            reduction_factor = 1.0 - (self.ctr_threshold - ctr) / self.ctr_threshold
            reduction_factor = max(0.2, reduction_factor)
            redistribution_plan[channel] = reduction_factor
            total_reduction += (1.0 - reduction_factor)
        
        if high_roi_channels and total_reduction > 0:
            total_roi = sum(high_roi_channels.values())
            for channel, roi in high_roi_channels.items():
                increase_factor = 1.0 + (total_reduction * roi / total_roi)
                redistribution_plan[channel] = increase_factor
        
        logger.info(
            f"Plan de redistribución generado: {redistribution_plan}",
            extra={'context': 'traffic_redistribution_plan'}
        )
        return redistribution_plan
    
    def make_content_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma una decisión completa sobre la estrategia de contenido
        
        Args:
            context: Información contextual
            Returns:
            Decisión completa con estrategias
        """
        try:
            self._validate_context(context)
            cta_strategy = self.select_cta_strategy(context)
            visual_strategy = self.select_visual_strategy(context)
            voice_strategy = self.select_voice_strategy(context)
            
            decision = {
                'cta': cta_strategy,
                'visual': visual_strategy,
                'voice': voice_strategy,
                'timestamp': datetime.datetime.now().isoformat(),
                'context': context,
                'decision_id': f"decision_{context.get('niche', 'general')}_{int(time.time())}"
            }
            
            self.kb.save_to_mongodb('decisions', decision)
            logger.info(
                f"Decisión de contenido generada para {context.get('niche')} en {context.get('platform')}",
                extra={'context': f'decision_{decision['decision_id']}'}
            )
            
            return decision
        except Exception as e:
            logger.error(
                f"Error al generar decisión de contenido: {str(e)}",
                extra={'context': 'content_decision'}
            )
            raise DecisionEngineError(f"Error al generar decisión: {str(e)}")
    
    def optimize_strategy_in_real_time(self, content_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimiza la estrategia en tiempo real
        
        Args:
            content_id: Identificador del contenido
            metrics: Métricas actuales
            Returns:
            Ajustes recomendados
        """
        try:
            decision = self.kb.get_from_mongodb('decisions', {'decision_id': content_id})
            if not decision:
                logger.warning(
                    f"No se encontró decisión para contenido {content_id}",
                    extra={'context': 'realtime_optimization'}
                )
                return {}
            
            adjustments = {}
            ctr = metrics.get('ctr', 0)
            conversion_rate = metrics.get('conversion_rate', 0)
            
            if conversion_rate < 0.05:
                cta_strategy = decision.get('cta', {})
                current_timing = cta_strategy.get('timing')
                if current_timing == 'early':
                    adjustments['cta_timing'] = 'middle'
                elif current_timing == 'late':
                    adjustments['cta_timing'] = 'middle'
                
                current_type = cta_strategy.get('type')
                if current_type == 'question':
                    adjustments['cta_type'] = 'challenge'
                elif current_type == 'offer':
                    adjustments['cta_type'] = 'curiosity'
            
            if ctr < 0.03:
                visual_strategy = decision.get('visual', {})
                current_style = visual_strategy.get('style')
                if current_style == 'minimalist':
                    adjustments['visual_style'] = 'vibrant'
                elif current_style == 'professional':
                    adjustments['visual_style'] = 'dramatic'
                
                current_color = visual_strategy.get('color_scheme')
                if current_color == 'neutral':
                    adjustments['visual_color'] = 'high_contrast'
                elif current_color == 'cool':
                    adjustments['visual_color'] = 'warm'
            
            if adjustments:
                logger.info(
                    f"Ajustes en tiempo real generados para contenido {content_id}: {adjustments}",
                    extra={'context': 'realtime_optimization'}
                )
            
            return adjustments
        except Exception as e:
            logger.error(
                f"Error en optimización en tiempo real: {str(e)}",
                extra={'context': 'realtime_optimization'}
            )
            return {}
    
    def _schedule_continuous_evaluation(self):
        """Programa la evaluación continua de los bandits"""
        # Esta función sería llamada por un scheduler externo
        # Aquí solo verificamos si es momento de realizar la evaluación
        current_time = datetime.datetime.now()
        hours_since_last = (current_time - self.last_evaluation_time).total_seconds() / 3600
        
        if hours_since_last >= self.evaluation_interval:
            logger.info(f"Iniciando evaluación continua de bandits (última: {hours_since_last:.1f}h atrás)",
                       extra={'context': 'continuous_evaluation'})
            self.run_continuous_evaluation()
            self.last_evaluation_time = current_time
    
    def run_continuous_evaluation(self):
        """Ejecuta una evaluación completa del rendimiento de todos los bandits"""
        try:
            # 1. Evaluar bandits de CTA
            cta_performance = self._evaluate_bandit_group('cta_bandits')
            
            # 2. Evaluar bandits visuales
            visual_performance = self._evaluate_bandit_group('visual_bandits')
            
            # 3. Evaluar bandits de voz
            voice_performance = self._evaluate_bandit_group('voice_bandits')
            
            # 4. Calcular métricas agregadas
            aggregated_metrics = self._calculate_aggregated_metrics(
                cta_performance, visual_performance, voice_performance)
            
            # 5. Guardar resultados de evaluación
            evaluation_result = {
                'timestamp': datetime.datetime.now().isoformat(),
                'cta_performance': cta_performance,
                'visual_performance': visual_performance,
                'voice_performance': voice_performance,
                'aggregated_metrics': aggregated_metrics
            }
            
            # Guardar en la base de conocimiento
            self.kb.save_to_mongodb('bandit_evaluations', evaluation_result)
            
            # Actualizar historial de rendimiento
            self.performance_history[datetime.datetime.now().isoformat()] = aggregated_metrics
            
            # Limpiar historial antiguo (mantener solo últimos 30 días)
            self._clean_old_performance_history()
            
            # 6. Generar recomendaciones de optimización
            recommendations = self._generate_optimization_recommendations(evaluation_result)
            
            logger.info(f"Evaluación continua completada. Métricas agregadas: {aggregated_metrics}",
                       extra={'context': 'continuous_evaluation'})
            
            return evaluation_result, recommendations
            
        except Exception as e:
            logger.error(f"Error en evaluación continua: {str(e)}", 
                        extra={'context': 'continuous_evaluation'})
            return None, None
    
    def _evaluate_bandit_group(self, bandit_type: str) -> Dict[str, Dict]:
        """Evalúa el rendimiento de un grupo de bandits"""
        bandit_dict = getattr(self, bandit_type)
        performance = {}
        
        for bandit_key, bandit in bandit_dict.items():
            # Obtener datos históricos de rendimiento
            historical_data = self._get_historical_performance_data(bandit_type, bandit_key)
            
            # Calcular métricas por acción
            action_metrics = {}
            for action in bandit.A:
                # Obtener datos específicos de esta acción
                action_data = [d for d in historical_data if d.get('action') == action]
                
                if not action_data:
                    continue
                
                # Calcular métricas
                metrics = {
                    'ctr': self._calculate_mean_metric([d.get('metrics', {}).get('ctr', 0) for d in action_data]),
                    'conversion_rate': self._calculate_mean_metric([d.get('metrics', {}).get('conversion_rate', 0) for d in action_data]),
                    'engagement_rate': self._calculate_mean_metric([d.get('metrics', {}).get('engagement_rate', 0) for d in action_data]),
                    'retention_rate': self._calculate_mean_metric([d.get('metrics', {}).get('retention_rate', 0) for d in action_data]),
                }
                
                # Calcular recompensa acumulada
                cumulative_reward = sum(self._calculate_reward(d.get('metrics', {})) for d in action_data)
                
                # Calcular intervalo de confianza
                if len(action_data) >= 5:  # Mínimo de muestras para estadísticas significativas
                    ctr_values = [d.get('metrics', {}).get('ctr', 0) for d in action_data]
                    ci_low, ci_high = self._calculate_confidence_interval(ctr_values)
                    confidence_interval = {'low': ci_low, 'high': ci_high}
                else:
                    confidence_interval = {'low': 0, 'high': 0}
                
                # Calcular tendencia (últimos 7 días vs anteriores)
                trend = self._calculate_metric_trend(action_data, 'ctr')
                
                # Guardar métricas de esta acción
                action_metrics[action] = {
                    'metrics': metrics,
                    'cumulative_reward': cumulative_reward,
                    'samples': len(action_data),
                    'confidence_interval': confidence_interval,
                    'trend': trend,
                    'last_updated': datetime.datetime.now().isoformat()
                }
            
            # Identificar la mejor acción basada en recompensa acumulada
            if action_metrics:
                best_action = max(action_metrics.items(), 
                                 key=lambda x: x[1]['cumulative_reward'])
                
                performance[bandit_key] = {
                    'action_metrics': action_metrics,
                    'best_action': best_action[0],
                    'best_action_reward': best_action[1]['cumulative_reward'],
                    'total_samples': sum(m['samples'] for m in action_metrics.values()),
                    'evaluation_time': datetime.datetime.now().isoformat()
                }
        
        return performance
    
    def _get_historical_performance_data(self, bandit_type: str, bandit_key: str) -> List[Dict]:
        """Obtiene datos históricos de rendimiento para un bandit específico"""
        # Determinar el tipo de colección según el tipo de bandit
        collection_prefix = bandit_type.split('_')[0]  # 'cta', 'visual', 'voice'
        
        # Obtener datos de los últimos 30 días
        start_date = datetime.datetime.now() - datetime.timedelta(days=30)
        
        # Consultar la base de conocimiento
        query = {
            'bandit_key': bandit_key,
            'timestamp': {'$gte': start_date.isoformat()}
        }
        
        historical_data = self.kb.get_all_from_mongodb(f'{collection_prefix}_performance', query)
        return historical_data if historical_data else []
    
    def _calculate_mean_metric(self, values: List[float]) -> float:
        """Calcula la media de una métrica, manejando listas vacías"""
        return sum(values) / len(values) if values else 0
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calcula el intervalo de confianza para una lista de valores"""
        if not values or len(values) < 2:
            return 0, 0
            
        mean = np.mean(values)
        sem = stats.sem(values)
        ci = stats.t.interval(confidence, len(values)-1, loc=mean, scale=sem)
        return ci[0], ci[1]
    
    def _calculate_metric_trend(self, data: List[Dict], metric_name: str) -> float:
        """Calcula la tendencia de una métrica (cambio porcentual)"""
        if not data or len(data) < 2:
            return 0
            
        # Ordenar por timestamp
        sorted_data = sorted(data, key=lambda x: x.get('timestamp', ''))
        
        # Dividir en dos períodos (reciente y anterior)
        mid_point = len(sorted_data) // 2
        recent_data = sorted_data[mid_point:]
        previous_data = sorted_data[:mid_point]
        
        if not recent_data or not previous_data:
            return 0
            
        # Calcular medias
        recent_mean = np.mean([d.get('metrics', {}).get(metric_name, 0) for d in recent_data])
        previous_mean = np.mean([d.get('metrics', {}).get(metric_name, 0) for d in previous_data])
        
        # Calcular cambio porcentual
        if previous_mean == 0:
            return 0
        
        return (recent_mean - previous_mean) / previous_mean
    
    def _calculate_aggregated_metrics(self, cta_perf: Dict, visual_perf: Dict, voice_perf: Dict) -> Dict:
        """Calcula métricas agregadas de todos los bandits"""
        # Contar total de muestras
        total_samples = 0
        for perf_dict in [cta_perf, visual_perf, voice_perf]:
            for bandit_key, metrics in perf_dict.items():
                total_samples += metrics.get('total_samples', 0)
        
        # Calcular CTR promedio ponderado
        weighted_ctr = 0
        total_weight = 0
        
        for perf_dict in [cta_perf, visual_perf, voice_perf]:
            for bandit_key, metrics in perf_dict.items():
                for action, action_metrics in metrics.get('action_metrics', {}).items():
                    samples = action_metrics.get('samples', 0)
                    ctr = action_metrics.get('metrics', {}).get('ctr', 0)
                    weighted_ctr += ctr * samples
                    total_weight += samples
        
        avg_ctr = weighted_ctr / total_weight if total_weight > 0 else 0
        
        # Calcular otras métricas agregadas
        return {
            'avg_ctr': avg_ctr,
            'total_samples': total_samples,
            'cta_bandits_count': len(cta_perf),
            'visual_bandits_count': len(visual_perf),
            'voice_bandits_count': len(voice_perf),
            'evaluation_time': datetime.datetime.now().isoformat()
        }
    
    def _clean_old_performance_history(self):
        """Limpia entradas antiguas del historial de rendimiento"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=30)
        cutoff_str = cutoff_date.isoformat()
        
        # Eliminar entradas antiguas
        self.performance_history = {
            ts: metrics for ts, metrics in self.performance_history.items()
            if ts >= cutoff_str
        }
    
    def _generate_optimization_recommendations(self, evaluation_result: Dict) -> Dict[str, List[Dict]]:
        """Genera recomendaciones de optimización basadas en la evaluación"""
        recommendations = {
            'cta': [],
            'visual': [],
            'voice': []
        }
        
        # Analizar bandits de CTA
        for bandit_key, metrics in evaluation_result.get('cta_performance', {}).items():
            action_metrics = metrics.get('action_metrics', {})
            
            # Identificar acciones de bajo rendimiento
            for action, action_data in action_metrics.items():
                ctr = action_data.get('metrics', {}).get('ctr', 0)
                samples = action_data.get('samples', 0)
                trend = action_data.get('trend', 0)
                
                # Si tiene suficientes muestras, CTR bajo y tendencia negativa
                if samples >= 50 and ctr < 0.03 and trend < 0:
                    recommendations['cta'].append({
                        'bandit_key': bandit_key,
                        'action': action,
                        'issue': 'low_ctr_negative_trend',
                        'current_ctr': ctr,
                        'samples': samples,
                        'recommendation': 'Considerar reemplazar esta estrategia de CTA o ajustar timing'
                    })
        
        # Analizar bandits visuales
        for bandit_key, metrics in evaluation_result.get('visual_performance', {}).items():
            action_metrics = metrics.get('action_metrics', {})
            
            # Identificar acciones de bajo rendimiento
            for action, action_data in action_metrics.items():
                engagement = action_data.get('metrics', {}).get('engagement_rate', 0)
                samples = action_data.get('samples', 0)
                
                # Si tiene suficientes muestras y engagement bajo
                if samples >= 50 and engagement < 0.2:
                    recommendations['visual'].append({
                        'bandit_key': bandit_key,
                        'action': action,
                        'issue': 'low_engagement',
                        'current_engagement': engagement,
                        'samples': samples,
                        'recommendation': 'Considerar un estilo visual más impactante o contrastante'
                    })
        
        # Analizar bandits de voz
        for bandit_key, metrics in evaluation_result.get('voice_performance', {}).items():
            action_metrics = metrics.get('action_metrics', {})
            
            # Identificar acciones de bajo rendimiento
            for action, action_data in action_metrics.items():
                retention = action_data.get('metrics', {}).get('retention_rate', 0)
                samples = action_data.get('samples', 0)
                
                # Si tiene suficientes muestras y retención baja
                if samples >= 50 and retention < 0.4:
                    recommendations['voice'].append({
                        'bandit_key': bandit_key,
                        'action': action,
                        'issue': 'low_retention',
                        'current_retention': retention,
                        'samples': samples,
                        'recommendation': 'Considerar un tono de voz más dinámico o cambiar el ritmo'
                    })
        
        return recommendations
    
    def evaluate_bandit_performance(self, bandit_type: str, bandit_key: str) -> Dict[str, Any]:
        """
        Evalúa el rendimiento de un bandit específico con métricas avanzadas
        
        Args:
            bandit_type: Tipo de bandit (cta, visual, voice)
            bandit_key: Clave del bandit
            Returns:
            Informe de rendimiento detallado
        """
        bandit_dict = getattr(self, f"{bandit_type}_bandits")
        if bandit_key not in bandit_dict:
            return {'status': 'not_found'}
        
        bandit = bandit_dict[bandit_key]
        
        # Obtener datos históricos
        historical_data = self._get_historical_performance_data(f"{bandit_type}_bandits", bandit_key)
        
        # Preparar informe básico
        report = {
            'bandit_key': bandit_key,
            'bandit_type': bandit_type,
            'actions': {},
            'timestamp': datetime.datetime.now().isoformat(),
            'historical_performance': {}
        }
        
        # Analizar cada acción
        for action in bandit.A:
            # Estimación actual del modelo
            context = np.ones(self.feature_dim)
            A_inv = np.linalg.inv(bandit.A[action])
            mean = context @ bandit.theta[action]
            variance = np.sqrt(context.T @ A_inv @ context)
            
            # Datos históricos de esta acción
            action_data = [d for d in historical_data if d.get('action') == action]
            
            # Métricas históricas
            if action_data:
                ctr_values = [d.get('metrics', {}).get('ctr', 0) for d in action_data]
                conversion_values = [d.get('metrics', {}).get('conversion_rate', 0) for d in action_data]
                
                # Calcular intervalos de confianza
                ctr_ci = self._calculate_confidence_interval(ctr_values) if len(ctr_values) >= 5 else (0, 0)
                conversion_ci = self._calculate_confidence_interval(conversion_values) if len(conversion_values) >= 5 else (0, 0)
                
                # Calcular tendencias
                ctr_trend = self._calculate_metric_trend(action_data, 'ctr')
                conversion_trend = self._calculate_metric_trend(action_data, 'conversion_rate')
                
                historical_metrics = {
                    'ctr': {
                        'mean': self._calculate_mean_metric(ctr_values),
                        'confidence_interval': {'low': ctr_ci[0], 'high': ctr_ci[1]},
                        'trend': ctr_trend,
                        'samples': len(ctr_values)
                    },
                    'conversion_rate': {
                        'mean': self._calculate_mean_metric(conversion_values),
                        'confidence_interval': {'low': conversion_ci[0], 'high': conversion_ci[1]},
                        'trend': conversion_trend,
                        'samples': len(conversion_values)
                    }
                }
            else:
                historical_metrics = {}
            
            # Añadir al informe
            report['actions'][action] = {
                'model_estimation': {
                    'estimated_reward': float(mean),
                    'confidence': 1.0 - min(float(variance), 1.0),
                    'samples': int(np.sum(bandit.A[action].diagonal()))
                },
                'historical_metrics': historical_metrics
            }
        
        # Análisis de rendimiento global
        if historical_data:
            # Agrupar por día para ver tendencia temporal
            daily_performance = {}
            for entry in historical_data:
                date_str = entry.get('timestamp', '').split('T')[0]
                if date_str not in daily_performance:
                    daily_performance[date_str] = []
                daily_performance[date_str].append(entry)
            
            # Calcular métricas diarias
            for date, entries in daily_performance.items():
                ctr_values = [e.get('metrics', {}).get('ctr', 0) for e in entries]
                report['historical_performance'][date] = {
                    'avg_ctr': self._calculate_mean_metric(ctr_values),
                    'samples': len(entries)
                }
        
        # Añadir recomendaciones
        report['recommendations'] = []
        
        # Identificar acciones de bajo rendimiento
        for action, metrics in report['actions'].items():
            historical = metrics.get('historical_metrics', {})
            ctr_data = historical.get('ctr', {})
            
            if ctr_data and ctr_data.get('samples', 0) >= 50:
                ctr_mean = ctr_data.get('mean', 0)
                ctr_trend = ctr_data.get('trend', 0)
                
                if ctr_mean < 0.03:
                    report['recommendations'].append({
                        'action': action,
                        'issue': 'low_ctr',
                        'severity': 'high' if ctr_mean < 0.01 else 'medium',
                        'recommendation': 'Considerar reemplazar esta estrategia o ajustar parámetros'
                    })
                
                if ctr_trend < -0.1:
                    report['recommendations'].append({
                        'action': action,
                        'issue': 'negative_trend',
                        'severity': 'high' if ctr_trend < -0.2 else 'medium',
                        'recommendation': 'Investigar causas de la tendencia negativa'
                    })
        
        return report

if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    engine = DecisionEngine()
    
    test_context = {
        'niche': 'finance',
        'platform': 'youtube',
        'character': 'finance_expert',
        'audience': {
            'age_group': 3,
            'engagement_level': 7,
            'retention_rate': 65
        },
        'trend_score': 0.8
    }
    
    print("\nPrueba de selección de estrategias:")
    cta_strategy = engine.select_cta_strategy(test_context)
    print(f"Estrategia CTA: {cta_strategy}")
    
    visual_strategy = engine.select_visual_strategy(test_context)
    print(f"Estrategia visual: {visual_strategy}")
    
    voice_strategy = engine.select_voice_strategy(test_context)
    print(f"Estrategia de voz: {voice_strategy}")
    
    print("\nPrueba de decisión completa:")
    decision = engine.make_content_decision(test_context)
    print(f"Decisión completa: {json.dumps(decision, indent=2)}")
    
    print("\nPrueba de actualización de estrategias:")
    test_metrics = {
        'ctr': 0.08,
        'conversion_rate': 0.12,
        'engagement_rate': 0.35,
        'retention_rate': 0.70
    }
    
    engine.update_cta_strategy(cta_strategy['bandit_key'], cta_strategy['action'], test_metrics, test_context)
    engine.update_visual_strategy(visual_strategy['bandit_key'], visual_strategy['action'], test_metrics, test_context)
    engine.update_voice_strategy(voice_strategy['bandit_key'], voice_strategy['action'], test_metrics, test_context)
    
    print("\nPrueba de redistribución de tráfico:")
    test_channel_metrics = {
        'youtube_finance': {'ctr': 0.02, 'roi': 30},
        'tiktok_finance': {'ctr': 0.08, 'roi': 120},
        'instagram_finance': {'ctr': 0.04, 'roi': 60}
    }
    
    should_redistribute = engine.should_redistribute_traffic(test_channel_metrics)
    print(f"¿Redistribuir tráfico? {should_redistribute}")
    
    if should_redistribute:
        redistribution_plan = engine.get_traffic_redistribution_plan(test_channel_metrics)
        print(f"Plan de redistribución: {redistribution_plan}")
    
    print("\nPrueba de optimización en tiempo real:")
    engine.kb.save_to_mongodb('decisions', decision)
    test_realtime_metrics = {'ctr': 0.02, 'conversion_rate': 0.03}
    adjustments = engine.optimize_strategy_in_real_time(decision['decision_id'], test_realtime_metrics)
    print(f"Ajustes recomendados: {adjustments}")
    
    print("\nPrueba de evaluación de bandit:")
    report = engine.evaluate_bandit_performance('cta', cta_strategy['bandit_key'])
    print(f"Informe de rendimiento: {json.dumps(report, indent=2)}")
    
    print("\nPrueba de evaluación continua:")
    evaluation_result, recommendations = engine.run_continuous_evaluation()
    if evaluation_result:
        print(f"Resultados de evaluación continua: {json.dumps(evaluation_result, indent=2)}")
        print(f"Recomendaciones: {json.dumps(recommendations, indent=2)}")
    
    print("\nPruebas completadas con éxito")