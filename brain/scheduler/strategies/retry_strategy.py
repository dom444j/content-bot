"""
Estrategia de reintentos para el sistema de planificación.

Este módulo implementa diferentes políticas de reintento para tareas
que fallan durante su ejecución, permitiendo recuperación automática
y resiliente ante fallos transitorios.
"""

import logging
import time
import random
import math
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta

from ..core.task_model import Task, TaskStatus
from .base_strategy import BaseStrategy

logger = logging.getLogger('Scheduler.Strategies.Retry')

class RetryPolicy:
    """
    Define una política de reintentos con diferentes estrategias.
    
    Atributos:
        max_retries: Número máximo de reintentos permitidos
        retry_delay: Tiempo base entre reintentos (segundos)
        backoff_factor: Factor de incremento para backoff exponencial
        jitter: Cantidad de aleatoriedad a añadir (0-1)
        max_delay: Tiempo máximo entre reintentos (segundos)
        retry_on_exceptions: Lista de excepciones que activan reintentos
        retry_on_status_codes: Lista de códigos HTTP que activan reintentos
    """
    
    def __init__(self, 
                max_retries: int = 3, 
                retry_delay: float = 1.0,
                backoff_factor: float = 2.0,
                jitter: float = 0.1,
                max_delay: float = 60.0,
                retry_on_exceptions: List[str] = None,
                retry_on_status_codes: List[int] = None):
        """
        Inicializa una política de reintentos.
        
        Args:
            max_retries: Número máximo de reintentos permitidos
            retry_delay: Tiempo base entre reintentos (segundos)
            backoff_factor: Factor de incremento para backoff exponencial
            jitter: Cantidad de aleatoriedad a añadir (0-1)
            max_delay: Tiempo máximo entre reintentos (segundos)
            retry_on_exceptions: Lista de excepciones que activan reintentos
            retry_on_status_codes: Lista de códigos HTTP que activan reintentos
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.max_delay = max_delay
        self.retry_on_exceptions = retry_on_exceptions or []
        self.retry_on_status_codes = retry_on_status_codes or []
        
        # Validación básica
        if max_retries < 0:
            raise ValueError("El número máximo de reintentos no puede ser negativo")
        if retry_delay < 0:
            raise ValueError("El tiempo de espera entre reintentos no puede ser negativo")
        if backoff_factor <= 0:
            raise ValueError("El factor de backoff debe ser mayor que cero")
        if not 0 <= jitter <= 1:
            raise ValueError("El jitter debe estar entre 0 y 1")
        if max_delay < retry_delay:
            raise ValueError("El tiempo máximo de espera debe ser mayor o igual al tiempo base")
    
    def calculate_next_retry_delay(self, attempt: int) -> float:
        """
        Calcula el tiempo de espera para el siguiente reintento.
        
        Args:
            attempt: Número de intento actual (comenzando en 1)
            
        Returns:
            Tiempo de espera en segundos
        """
        if attempt <= 0:
            return 0
        
        # Calcular delay base según la estrategia de backoff
        if self.backoff_factor > 1:
            # Backoff exponencial
            delay = self.retry_delay * (self.backoff_factor ** (attempt - 1))
        else:
            # Backoff lineal
            delay = self.retry_delay * attempt
        
        # Aplicar límite máximo
        delay = min(delay, self.max_delay)
        
        # Aplicar jitter para evitar sincronización
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)  # Asegurar que no sea negativo
    
    def should_retry(self, task: Task, error: str = None, status_code: int = None) -> bool:
        """
        Determina si una tarea debe ser reintentada basándose en el error.
        
        Args:
            task: Tarea que falló
            error: Mensaje de error o excepción
            status_code: Código de estado HTTP (si aplica)
            
        Returns:
            True si la tarea debe reintentarse, False en caso contrario
        """
        # Verificar número de intentos
        current_attempts = task.metadata.get('retry_count', 0)
        if current_attempts >= self.max_retries:
            return False
        
        # Si no hay criterios específicos, siempre reintentar
        if not self.retry_on_exceptions and not self.retry_on_status_codes:
            return True
        
        # Verificar por tipo de excepción
        if error and self.retry_on_exceptions:
            for exception_name in self.retry_on_exceptions:
                if exception_name in error:
                    return True
        
        # Verificar por código de estado HTTP
        if status_code and self.retry_on_status_codes:
            if status_code in self.retry_on_status_codes:
                return True
        
        # Si hay criterios pero no coinciden, no reintentar
        if self.retry_on_exceptions or self.retry_on_status_codes:
            return False
        
        # Por defecto, reintentar
        return True


class RetryStrategy(BaseStrategy):
    """
    Estrategia que implementa políticas de reintento para tareas fallidas.
    
    Características:
    - Múltiples políticas por tipo de tarea
    - Backoff exponencial con jitter
    - Reintentos selectivos por tipo de error
    - Límites configurables de intentos
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa la estrategia de reintentos.
        
        Args:
            config: Configuración específica para la estrategia
        """
        super().__init__(config)
        
        # Políticas de reintento por tipo de tarea
        self.policies: Dict[str, RetryPolicy] = {}
        
        # Política por defecto
        self.default_policy = RetryPolicy(
            max_retries=3,
            retry_delay=2.0,
            backoff_factor=2.0,
            jitter=0.1,
            max_delay=60.0
        )
        
        # Cargar políticas desde configuración
        self._load_policies_from_config()
        
        logger.info(f"RetryStrategy inicializada con {len(self.policies)} políticas específicas")
    
    def _load_policies_from_config(self) -> None:
        """
        Carga políticas de reintento desde la configuración.
        """
        if not self.config or 'policies' not in self.config:
            self._set_default_policies()
            return
        
        # Cargar política por defecto si está definida
        if 'default' in self.config:
            default_config = self.config['default']
            try:
                self.default_policy = RetryPolicy(
                    max_retries=default_config.get('max_retries', 3),
                    retry_delay=default_config.get('retry_delay', 2.0),
                    backoff_factor=default_config.get('backoff_factor', 2.0),
                    jitter=default_config.get('jitter', 0.1),
                    max_delay=default_config.get('max_delay', 60.0),
                    retry_on_exceptions=default_config.get('retry_on_exceptions'),
                    retry_on_status_codes=default_config.get('retry_on_status_codes')
                )
                logger.debug("Política por defecto cargada desde configuración")
            except ValueError as e:
                logger.error(f"Error al cargar política por defecto: {str(e)}")
        
        # Cargar políticas específicas
        for policy_config in self.config.get('policies', []):
            try:
                task_type = policy_config.get('task_type')
                if not task_type:
                    logger.warning("Política sin tipo de tarea especificado, ignorando")
                    continue
                
                policy = RetryPolicy(
                    max_retries=policy_config.get('max_retries', 3),
                    retry_delay=policy_config.get('retry_delay', 2.0),
                    backoff_factor=policy_config.get('backoff_factor', 2.0),
                    jitter=policy_config.get('jitter', 0.1),
                    max_delay=policy_config.get('max_delay', 60.0),
                    retry_on_exceptions=policy_config.get('retry_on_exceptions'),
                    retry_on_status_codes=policy_config.get('retry_on_status_codes')
                )
                
                self.policies[task_type] = policy
                logger.debug(f"Política cargada para tipo de tarea '{task_type}'")
            
            except (KeyError, ValueError) as e:
                logger.error(f"Error al cargar política de reintento: {str(e)}")
    
    def _set_default_policies(self) -> None:
        """
        Configura políticas predeterminadas para tipos comunes de tareas.
        """
        # Políticas para APIs externas (más restrictivas)
        api_policy = RetryPolicy(
            max_retries=5,
            retry_delay=1.0,
            backoff_factor=2.0,
            jitter=0.2,
            max_delay=30.0,
            retry_on_exceptions=["ConnectionError", "Timeout", "RequestException"],
            retry_on_status_codes=[429, 500, 502, 503, 504]
        )
        
        # Política para tareas de procesamiento (más agresiva)
        processing_policy = RetryPolicy(
            max_retries=3,
            retry_delay=5.0,
            backoff_factor=1.5,
            jitter=0.1,
            max_delay=60.0,
            retry_on_exceptions=["ProcessingError", "ResourceError"]
        )
        
        # Política para tareas de análisis (menos agresiva)
        analysis_policy = RetryPolicy(
            max_retries=2,
            retry_delay=10.0,
            backoff_factor=1.0,
            jitter=0.05,
            max_delay=30.0
        )
        
        # Registrar políticas
        self.policies.update({
            # APIs externas
            "youtube_upload": api_policy,
            "tiktok_upload": api_policy,
            "instagram_upload": api_policy,
            "api_request": api_policy,
            
            # Procesamiento
            "video_rendering": processing_policy,
            "content_creation": processing_policy,
            "image_generation": processing_policy,
            
            # Análisis
            "analytics": analysis_policy,
            "trend_analysis": analysis_policy,
            "performance_report": analysis_policy
        })
    
    def get_policy(self, task: Task) -> RetryPolicy:
        """
        Obtiene la política de reintentos aplicable a una tarea.
        
        Args:
            task: Tarea a evaluar
            
        Returns:
            Política de reintentos aplicable
        """
        # Buscar política específica por tipo de tarea
        if task.task_type in self.policies:
            return self.policies[task.task_type]
        
        # Buscar política por categoría si está definida
        category = task.metadata.get('category')
        if category and category in self.policies:
            return self.policies[category]
        
        # Usar política por defecto
        return self.default_policy
    
    def should_apply(self, task: Task, context: Dict[str, Any] = None) -> bool:
        """
        Determina si la estrategia debe aplicarse a una tarea.
        
        Args:
            task: Tarea a evaluar
            context: Contexto adicional
            
        Returns:
            True si la estrategia debe aplicarse
        """
        # Solo aplicar a tareas fallidas
        if context and 'error' in context:
            return True
        
        # O a tareas que ya tienen intentos previos
        if 'retry_count' in task.metadata and task.metadata['retry_count'] > 0:
            return True
        
        return False
    
    def apply(self, task: Task, context: Dict[str, Any] = None) -> Task:
        """
        Aplica la estrategia de reintentos a una tarea.
        
        Args:
            task: Tarea a procesar
            context: Contexto adicional (debe incluir 'error' para tareas fallidas)
            
        Returns:
            Tarea modificada con información de reintento
        """
        context = context or {}
        error = context.get('error')
        status_code = None
        
        # Extraer código de estado si está disponible
        if error and 'status_code' in context:
            status_code = context['status_code']
        elif error and 'HTTP Error' in error:
            # Intentar extraer código de estado del mensaje de error
            try:
                status_code = int(error.split('HTTP Error')[1].split(':')[0].strip())
            except (IndexError, ValueError):
                pass
        
        # Obtener política aplicable
        policy = self.get_policy(task)
        
        # Incrementar contador de reintentos
        current_attempts = task.metadata.get('retry_count', 0)
        task.metadata['retry_count'] = current_attempts + 1
        
        # Verificar si debemos reintentar
        if not policy.should_retry(task, error, status_code):
            logger.info(f"Tarea {task.task_id} ({task.task_type}) ha alcanzado el límite de reintentos "
                       f"({current_attempts}/{policy.max_retries})")
            
            # Marcar como fallida permanentemente
            task.metadata['retry_exhausted'] = True
            return task
        
        # Calcular tiempo de espera para el próximo intento
        next_attempt = current_attempts + 1
        delay = policy.calculate_next_retry_delay(next_attempt)
        
        # Calcular próximo tiempo de ejecución
        next_execution_time = datetime.now() + timedelta(seconds=delay)
        task.scheduled_time = next_execution_time
        
        # Registrar información de reintento
        task.metadata['last_error'] = error
        task.metadata['last_status_code'] = status_code
        task.metadata['retry_delay'] = delay
        task.metadata['retry_scheduled_at'] = datetime.now().isoformat()
        
        logger.info(f"Programando reintento {next_attempt}/{policy.max_retries} para tarea "
                   f"{task.task_id} ({task.task_type}) en {delay:.2f}s")
        
        return task
    
    def add_policy(self, task_type: str, policy: RetryPolicy) -> None:
        """
        Añade o actualiza una política de reintentos.
        
        Args:
            task_type: Tipo de tarea
            policy: Política a aplicar
        """
        self.policies[task_type] = policy
        logger.debug(f"Política de reintentos actualizada para '{task_type}'")
    
    def remove_policy(self, task_type: str) -> bool:
        """
        Elimina una política de reintentos.
        
        Args:
            task_type: Tipo de tarea
            
        Returns:
            True si se eliminó, False si no existía
        """
        if task_type in self.policies:
            del self.policies[task_type]
            logger.debug(f"Política de reintentos eliminada para '{task_type}'")
            return True
        return False
    
    def get_all_policies(self) -> Dict[str, RetryPolicy]:
        """
        Obtiene todas las políticas de reintentos.
        
        Returns:
            Diccionario con todas las políticas
        """
        result = {'default': self.default_policy}
        result.update(self.policies)
        return result