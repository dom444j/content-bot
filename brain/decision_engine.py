"""
Decision Engine - Motor de decisiones y auto-mejoras

Este módulo implementa algoritmos de aprendizaje por refuerzo:
- Contextual bandits para optimización de CTAs
- Auto-mejoras para visuales y voces
- Redistribución de tráfico basada en ROI
- Optimización en tiempo real de estrategias
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

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'decision_engine.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DecisionEngine')

class DecisionEngine:
    """
    Motor de decisiones que implementa algoritmos de aprendizaje por refuerzo
    para optimizar CTAs, visuales, voces y estrategias de monetización.
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
            
        logger.info("Inicializando Decision Engine...")
        
        # Cargar base de conocimiento
        self.kb = KnowledgeBase()
        
        # Cargar configuración de estrategia
        self.strategy_file = os.path.join('config', 'strategy.json')
        self.strategy = self._load_strategy()
        
        # Configuración de bandits contextuales
        self.bandits_config = self.strategy.get('optimization_strategies', {}).get('contextual_bandits', {})
        self.exploration_rate = self.bandits_config.get('exploration_rate', 0.2)
        self.learning_rate = self.bandits_config.get('learning_rate', 0.1)
        self.reward_metrics = self.bandits_config.get('reward_metrics', {})
        
        # Estado de los bandits
        self.cta_bandits = {}  # Por nicho y plataforma
        self.visual_bandits = {}  # Por nicho y plataforma
        self.voice_bandits = {}  # Por nicho y personaje
        
        # Configuración de redistribución de tráfico
        self.redistribution_config = self.strategy.get('optimization_strategies', {}).get('traffic_redistribution', {})
        self.ctr_threshold = self.redistribution_config.get('ctr_threshold', 0.05)
        self.roi_threshold = self.redistribution_config.get('roi_threshold', 50)
        
        # Cargar datos históricos si existen
        self._load_bandit_state()
        
        self._initialized = True
        logger.info("Decision Engine inicializado correctamente")
    
    def _load_strategy(self) -> Dict[str, Any]:
        """Carga la configuración de estrategia desde el archivo JSON"""
        try:
            if os.path.exists(self.strategy_file):
                with open(self.strategy_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de estrategia no encontrado: {self.strategy_file}")
                return self._create_default_strategy()
        except Exception as e:
            logger.error(f"Error al cargar estrategia: {str(e)}")
            return self._create_default_strategy()
    
    def _create_default_strategy(self) -> Dict[str, Any]:
        """Crea una configuración de estrategia por defecto"""
        default_strategy = {
            "optimization_strategies": {
                "contextual_bandits": {
                    "exploration_rate": 0.2,
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
        
        # Guardar estrategia por defecto
        os.makedirs(os.path.dirname(self.strategy_file), exist_ok=True)
        with open(self.strategy_file, 'w', encoding='utf-8') as f:
            json.dump(default_strategy, f, indent=4)
        
        return default_strategy
    
    def _load_bandit_state(self):
        """Carga el estado de los bandits desde la base de conocimiento"""
        try:
            bandit_state = self.kb.get_bandit_state()
            if bandit_state:
                self.cta_bandits = bandit_state.get('cta_bandits', {})
                self.visual_bandits = bandit_state.get('visual_bandits', {})
                self.voice_bandits = bandit_state.get('voice_bandits', {})
                logger.info("Estado de bandits cargado correctamente")
            else:
                logger.info("No se encontró estado previo de bandits, se usarán valores iniciales")
        except Exception as e:
            logger.error(f"Error al cargar estado de bandits: {str(e)}")
    
    def _save_bandit_state(self):
        """Guarda el estado actual de los bandits en la base de conocimiento"""
        try:
            bandit_state = {
                'cta_bandits': self.cta_bandits,
                'visual_bandits': self.visual_bandits,
                'voice_bandits': self.voice_bandits,
                'last_updated': datetime.datetime.now().isoformat()
            }
            self.kb.save_bandit_state(bandit_state)
            logger.info("Estado de bandits guardado correctamente")
        except Exception as e:
            logger.error(f"Error al guardar estado de bandits: {str(e)}")
    
    def _get_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extrae y normaliza características del contexto para los bandits
        
        Args:
            context: Diccionario con información contextual (nicho, plataforma, audiencia, etc.)
            
        Returns:
            Vector de características normalizado
        """
        # Extraer características relevantes
        features = []
        
        # Características de plataforma (one-hot encoding)
        platforms = ['youtube', 'tiktok', 'instagram', 'threads', 'bluesky']
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
        features.append(audience.get('age_group', 0) / 5.0)  # Normalizado (0-5)
        features.append(audience.get('engagement_level', 0) / 10.0)  # Normalizado (0-10)
        features.append(audience.get('retention_rate', 0) / 100.0)  # Normalizado (0-100%)
        
        # Características temporales
        current_hour = datetime.datetime.now().hour
        features.append(current_hour / 24.0)  # Hora normalizada (0-1)
        current_day = datetime.datetime.now().weekday()
        features.append(current_day / 6.0)  # Día de la semana normalizado (0-1)
        
        # Convertir a array numpy y asegurar que sea float32
        return np.array(features, dtype=np.float32)
    
    def _select_action_with_exploration(self, action_values: Dict[str, float]) -> str:
        """
        Selecciona una acción usando epsilon-greedy
        
        Args:
            action_values: Diccionario de valores estimados para cada acción
            
        Returns:
            La acción seleccionada
        """
        if not action_values:
            return None
            
        # Exploración aleatoria
        if random.random() < self.exploration_rate:
            return random.choice(list(action_values.keys()))
        
        # Explotación (seleccionar la mejor acción)
        return max(action_values.items(), key=lambda x: x[1])[0]
    
    def _update_action_value(self, bandit_key: str, action: str, reward: float, bandits: Dict):
        """
        Actualiza el valor estimado de una acción usando aprendizaje por refuerzo
        
        Args:
            bandit_key: Clave del bandit (combinación de contexto)
            action: La acción tomada
            reward: La recompensa obtenida
            bandits: Diccionario de bandits a actualizar
        """
        if bandit_key not in bandits:
            bandits[bandit_key] = {}
        
        if action not in bandits[bandit_key]:
            bandits[bandit_key][action] = {
                'value': 0.0,
                'count': 0
            }
        
        # Actualizar valor usando regla de actualización incremental
        current = bandits[bandit_key][action]
        current['count'] += 1
        current['value'] += self.learning_rate * (reward - current['value'])
    
    def select_cta_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selecciona la mejor estrategia de CTA para el contexto dado
        
        Args:
            context: Información contextual (nicho, plataforma, audiencia)
            
        Returns:
            Estrategia de CTA seleccionada
        """
        # Crear clave para el bandit basada en nicho y plataforma
        niche = context.get('niche', 'general')
        platform = context.get('platform', 'general')
        bandit_key = f"{niche}_{platform}"
        
        # Obtener estrategias de CTA disponibles
        cta_strategies = self.strategy.get('cta_strategies', {})
        timing_options = list(cta_strategies.get('timing', {}).keys())
        type_options = cta_strategies.get('types', [])
        
        # Inicializar valores si es necesario
        if bandit_key not in self.cta_bandits:
            self.cta_bandits[bandit_key] = {}
            
            # Inicializar valores para todas las combinaciones
            for timing in timing_options:
                for cta_type in type_options:
                    action_key = f"{timing}_{cta_type}"
                    self.cta_bandits[bandit_key][action_key] = {
                        'value': 0.5,  # Valor inicial optimista
                        'count': 0
                    }
        
        # Extraer valores actuales
        action_values = {k: v['value'] for k, v in self.cta_bandits[bandit_key].items()}
        
        # Seleccionar acción con exploración
        selected_action = self._select_action_with_exploration(action_values)
        
        if selected_action:
            # Separar timing y tipo
            timing, cta_type = selected_action.split('_', 1)
            
            # Obtener detalles de timing
            timing_details = cta_strategies.get('timing', {}).get(timing, {})
            
            return {
                'timing': timing,
                'start_time': timing_details.get('start', 0),
                'end_time': timing_details.get('end', 10),
                'type': cta_type,
                'bandit_key': bandit_key,
                'action': selected_action
            }
        else:
            # Estrategia por defecto si no hay acciones disponibles
            return {
                'timing': 'middle',
                'start_time': 4,
                'end_time': 8,
                'type': 'question',
                'bandit_key': bandit_key,
                'action': 'middle_question'
            }
    
    def select_visual_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selecciona la mejor estrategia visual para el contexto dado
        
        Args:
            context: Información contextual (nicho, plataforma, audiencia)
            
        Returns:
            Estrategia visual seleccionada
        """
        # Crear clave para el bandit basada en nicho y plataforma
        niche = context.get('niche', 'general')
        platform = context.get('platform', 'general')
        bandit_key = f"{niche}_{platform}"
        
        # Obtener estrategias visuales disponibles
        visual_strategies = self.strategy.get('visual_strategies', {})
        style_options = visual_strategies.get('styles', [])
        color_options = visual_strategies.get('color_schemes', [])
        
        # Inicializar valores si es necesario
        if bandit_key not in self.visual_bandits:
            self.visual_bandits[bandit_key] = {}
            
            # Inicializar valores para todas las combinaciones
            for style in style_options:
                for color in color_options:
                    action_key = f"{style}_{color}"
                    self.visual_bandits[bandit_key][action_key] = {
                        'value': 0.5,  # Valor inicial optimista
                        'count': 0
                    }
        
        # Extraer valores actuales
        action_values = {k: v['value'] for k, v in self.visual_bandits[bandit_key].items()}
        
        # Seleccionar acción con exploración
        selected_action = self._select_action_with_exploration(action_values)
        
        if selected_action:
            # Separar estilo y esquema de color
            style, color_scheme = selected_action.split('_', 1)
            
            return {
                'style': style,
                'color_scheme': color_scheme,
                'bandit_key': bandit_key,
                'action': selected_action
            }
        else:
            # Estrategia por defecto si no hay acciones disponibles
            return {
                'style': 'vibrant',
                'color_scheme': 'warm',
                'bandit_key': bandit_key,
                'action': 'vibrant_warm'
            }
    
    def select_voice_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Selecciona la mejor estrategia de voz para el contexto dado
        
        Args:
            context: Información contextual (nicho, personaje, audiencia)
            
        Returns:
            Estrategia de voz seleccionada
        """
        # Crear clave para el bandit basada en nicho y personaje
        niche = context.get('niche', 'general')
        character = context.get('character', 'general')
        bandit_key = f"{niche}_{character}"
        
        # Obtener estrategias de voz disponibles
        voice_strategies = self.strategy.get('voice_strategies', {})
        tone_options = voice_strategies.get('tones', [])
        pacing_options = voice_strategies.get('pacing', [])
        
        # Inicializar valores si es necesario
        if bandit_key not in self.voice_bandits:
            self.voice_bandits[bandit_key] = {}
            
            # Inicializar valores para todas las combinaciones
            for tone in tone_options:
                for pacing in pacing_options:
                    action_key = f"{tone}_{pacing}"
                    self.voice_bandits[bandit_key][action_key] = {
                        'value': 0.5,  # Valor inicial optimista
                        'count': 0
                    }
        
        # Extraer valores actuales
        action_values = {k: v['value'] for k, v in self.voice_bandits[bandit_key].items()}
        
        # Seleccionar acción con exploración
        selected_action = self._select_action_with_exploration(action_values)
        
        if selected_action:
            # Separar tono y ritmo
            tone, pacing = selected_action.split('_', 1)
            
            return {
                'tone': tone,
                'pacing': pacing,
                'bandit_key': bandit_key,
                'action': selected_action
            }
        else:
            # Estrategia por defecto si no hay acciones disponibles
            return {
                'tone': 'enthusiastic',
                'pacing': 'dynamic',
                'bandit_key': bandit_key,
                'action': 'enthusiastic_dynamic'
            }
    
    def update_cta_strategy(self, bandit_key: str, action: str, metrics: Dict[str, float]):
        """
        Actualiza la estrategia de CTA basada en métricas de rendimiento
        
        Args:
            bandit_key: Clave del bandit (combinación de nicho y plataforma)
            action: La acción tomada (combinación de timing y tipo)
            metrics: Métricas de rendimiento (CTR, tasa de conversión, etc.)
        """
        # Calcular recompensa ponderada
        reward = 0.0
        for metric_name, weight in self.reward_metrics.items():
            if metric_name in metrics:
                reward += weight * metrics[metric_name]
        
        # Normalizar recompensa al rango [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        # Actualizar valor de la acción
        self._update_action_value(bandit_key, action, reward, self.cta_bandits)
        
        # Guardar estado actualizado
        self._save_bandit_state()
        
        logger.info(f"Actualizada estrategia CTA {action} con recompensa {reward:.4f}")
    
    def update_visual_strategy(self, bandit_key: str, action: str, metrics: Dict[str, float]):
        """
        Actualiza la estrategia visual basada en métricas de rendimiento
        
        Args:
            bandit_key: Clave del bandit (combinación de nicho y plataforma)
            action: La acción tomada (combinación de estilo y esquema de color)
            metrics: Métricas de rendimiento (CTR, tasa de conversión, etc.)
        """
        # Calcular recompensa ponderada
        reward = 0.0
        for metric_name, weight in self.reward_metrics.items():
            if metric_name in metrics:
                reward += weight * metrics[metric_name]
        
        # Normalizar recompensa al rango [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        # Actualizar valor de la acción
        self._update_action_value(bandit_key, action, reward, self.visual_bandits)
        
        # Guardar estado actualizado
        self._save_bandit_state()
        
        logger.info(f"Actualizada estrategia visual {action} con recompensa {reward:.4f}")
    
    def update_voice_strategy(self, bandit_key: str, action: str, metrics: Dict[str, float]):
        """
        Actualiza la estrategia de voz basada en métricas de rendimiento
        
        Args:
            bandit_key: Clave del bandit (combinación de nicho y personaje)
            action: La acción tomada (combinación de tono y ritmo)
            metrics: Métricas de rendimiento (retención, engagement, etc.)
        """
        # Calcular recompensa ponderada
        reward = 0.0
        for metric_name, weight in self.reward_metrics.items():
            if metric_name in metrics:
                reward += weight * metrics[metric_name]
        
        # Normalizar recompensa al rango [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        # Actualizar valor de la acción
        self._update_action_value(bandit_key, action, reward, self.voice_bandits)
        
        # Guardar estado actualizado
        self._save_bandit_state()
        
        logger.info(f"Actualizada estrategia de voz {action} con recompensa {reward:.4f}")
    
    def should_redistribute_traffic(self, channel_metrics: Dict[str, Dict[str, float]]) -> bool:
        """
        Determina si se debe redistribuir el tráfico entre canales
        
        Args:
            channel_metrics: Métricas por canal (CTR, ROI, etc.)
            
        Returns:
            True si se debe redistribuir, False en caso contrario
        """
        # Verificar si hay canales con bajo CTR
        low_ctr_channels = []
        for channel, metrics in channel_metrics.items():
            if metrics.get('ctr', 0) < self.ctr_threshold:
                low_ctr_channels.append(channel)
        
        # Verificar si hay canales con alto ROI
        high_roi_channels = []
        for channel, metrics in channel_metrics.items():
            if metrics.get('roi', 0) > self.roi_threshold:
                high_roi_channels.append(channel)
        
        # Redistribuir si hay canales con bajo CTR y canales con alto ROI
        return len(low_ctr_channels) > 0 and len(high_roi_channels) > 0
    
    def get_traffic_redistribution_plan(self, channel_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Genera un plan de redistribución de tráfico entre canales
        
        Args:
            channel_metrics: Métricas por canal (CTR, ROI, etc.)
            
        Returns:
            Diccionario con los factores de redistribución por canal
        """
        # Identificar canales con bajo CTR
        low_ctr_channels = {}
        for channel, metrics in channel_metrics.items():
            ctr = metrics.get('ctr', 0)
            if ctr < self.ctr_threshold:
                low_ctr_channels[channel] = ctr
        
        # Identificar canales con alto ROI
        high_roi_channels = {}
        for channel, metrics in channel_metrics.items():
            roi = metrics.get('roi', 0)
            if roi > self.roi_threshold:
                high_roi_channels[channel] = roi
        
        # Calcular factores de redistribución
        redistribution_plan = {}
        
        # Inicializar todos los canales con factor 1.0 (sin cambios)
        for channel in channel_metrics:
            redistribution_plan[channel] = 1.0
        
        # Reducir inversión en canales de bajo CTR
        total_reduction = 0.0
        for channel, ctr in low_ctr_channels.items():
            # Reducir proporcionalmente a la diferencia con el umbral
            reduction_factor = 1.0 - (self.ctr_threshold - ctr) / self.ctr_threshold
            reduction_factor = max(0.2, reduction_factor)  # No reducir más del 80%
            
            redistribution_plan[channel] = reduction_factor
            total_reduction += (1.0 - reduction_factor)
        
        # Aumentar inversión en canales de alto ROI
        if high_roi_channels and total_reduction > 0:
            # Normalizar ROIs para distribución
            total_roi = sum(high_roi_channels.values())
            for channel, roi in high_roi_channels.items():
                # Aumentar proporcionalmente al ROI
                increase_factor = 1.0 + (total_reduction * roi / total_roi)
                redistribution_plan[channel] = increase_factor
        
        logger.info(f"Plan de redistribución generado: {redistribution_plan}")
        return redistribution_plan
    
    def make_content_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma una decisión completa sobre la estrategia de contenido
        
        Args:
            context: Información contextual completa
            
        Returns:
            Decisión completa con estrategias de CTA, visuales y voz
        """
        # Seleccionar estrategias individuales
        cta_strategy = self.select_cta_strategy(context)
        visual_strategy = self.select_visual_strategy(context)
        voice_strategy = self.select_voice_strategy(context)
        
        # Combinar en una decisión completa
        decision = {
            'cta': cta_strategy,
            'visual': visual_strategy,
            'voice': voice_strategy,
            'timestamp': datetime.datetime.now().isoformat(),
            'context': context
        }
        
        # Registrar decisión
        logger.info(f"Decisión de contenido generada para {context.get('niche')} en {context.get('platform')}")
        
        return decision
    
    def optimize_strategy_in_real_time(self, content_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Optimiza la estrategia en tiempo real basada en métricas actuales
        
        Args:
            content_id: Identificador del contenido
            metrics: Métricas actuales (CTR, conversión, etc.)
            
        Returns:
            Ajustes recomendados a la estrategia
        """
        # Obtener decisión original
        original_decision = self.kb.get_content_decision(content_id)
        if not original_decision:
            logger.warning(f"No se encontró decisión original para contenido {content_id}")
            return {}
        
        # Verificar si es necesario ajustar
        ctr = metrics.get('ctr', 0)
        conversion_rate = metrics.get('conversion_rate', 0)
        
        adjustments = {}
        
        # Ajustar CTA si la conversión es baja
        if conversion_rate < 0.05:  # Umbral de 5%
            cta_strategy = original_decision.get('cta', {})
            current_timing = cta_strategy.get('timing')
            
            # Sugerir cambio de timing
            if current_timing == 'early':
                adjustments['cta_timing'] = 'middle'
            elif current_timing == 'late':
                adjustments['cta_timing'] = 'middle'
            
            # Sugerir cambio de tipo
            current_type = cta_strategy.get('type')
            if current_type == 'question':
                adjustments['cta_type'] = 'challenge'
            elif current_type == 'offer':
                adjustments['cta_type'] = 'curiosity'
        
        # Ajustar visuales si el CTR es bajo
        if ctr < 0.03:  # Umbral de 3%
            visual_strategy = original_decision.get('visual', {})
            current_style = visual_strategy.get('style')
            
            # Sugerir cambio de estilo visual
            if current_style == 'minimalist':
                adjustments['visual_style'] = 'vibrant'
            elif current_style == 'professional':
                adjustments['visual_style'] = 'dramatic'
            
            # Sugerir cambio de esquema de color
            current_color = visual_strategy.get('color_scheme')
            if current_color == 'neutral':
                adjustments['visual_color'] = 'high_contrast'
            elif current_color == 'cool':
                adjustments['visual_color'] = 'warm'
        
        # Registrar ajustes
        if adjustments:
            logger.info(f"Ajustes en tiempo real generados para contenido {content_id}: {adjustments}")
        
        return adjustments

# Función para obtener la instancia del motor de decisiones
def get_decision_engine():
    return DecisionEngine()

# Si se ejecuta como script principal, realizar pruebas
if __name__ == "__main__":
    # Crear directorio de logs si no existe
    os.makedirs('logs', exist_ok=True)
    
    # Probar el motor de decisiones
    engine = DecisionEngine()
    
    # Contexto de ejemplo
    test_context = {
        'niche': 'finance',
        'platform': 'youtube',
        'character': 'finance_expert',
        'audience': {
                        'age_group': 3,  # 25-34
            'engagement_level': 7,
            'retention_rate': 65
        }
    }
    
    # Probar selección de estrategias
    print("\nPrueba de selección de estrategias:")
    cta_strategy = engine.select_cta_strategy(test_context)
    print(f"Estrategia CTA seleccionada: {cta_strategy}")
    
    visual_strategy = engine.select_visual_strategy(test_context)
    print(f"Estrategia visual seleccionada: {visual_strategy}")
    
    voice_strategy = engine.select_voice_strategy(test_context)
    print(f"Estrategia de voz seleccionada: {voice_strategy}")
    
    # Probar decisión completa
    print("\nPrueba de decisión completa:")
    decision = engine.make_content_decision(test_context)
    print(f"Decisión completa: {json.dumps(decision, indent=2)}")
    
    # Probar actualización de estrategias
    print("\nPrueba de actualización de estrategias:")
    test_metrics = {
        'ctr': 0.08,
        'conversion_rate': 0.12,
        'engagement_rate': 0.35,
        'retention_rate': 0.70
    }
    
    engine.update_cta_strategy(cta_strategy['bandit_key'], cta_strategy['action'], test_metrics)
    engine.update_visual_strategy(visual_strategy['bandit_key'], visual_strategy['action'], test_metrics)
    engine.update_voice_strategy(voice_strategy['bandit_key'], voice_strategy['action'], test_metrics)
    
    # Probar redistribución de tráfico
    print("\nPrueba de redistribución de tráfico:")
    test_channel_metrics = {
        'youtube_finance': {
            'ctr': 0.02,
            'roi': 30
        },
        'tiktok_finance': {
            'ctr': 0.08,
            'roi': 120
        },
        'instagram_finance': {
            'ctr': 0.04,
            'roi': 60
        }
    }
    
    should_redistribute = engine.should_redistribute_traffic(test_channel_metrics)
    print(f"¿Debe redistribuir tráfico? {should_redistribute}")
    
    if should_redistribute:
        redistribution_plan = engine.get_traffic_redistribution_plan(test_channel_metrics)
        print(f"Plan de redistribución: {redistribution_plan}")
    
    # Probar optimización en tiempo real
    print("\nPrueba de optimización en tiempo real:")
    # Simular que la base de conocimiento tiene la decisión original
    engine.kb.save_content_decision("test_content_123", decision)
    
    test_realtime_metrics = {
        'ctr': 0.02,
        'conversion_rate': 0.03
    }
    
    adjustments = engine.optimize_strategy_in_real_time("test_content_123", test_realtime_metrics)
    print(f"Ajustes recomendados: {adjustments}")
    
    print("\nPruebas completadas con éxito")