"""
Estrategia de control de tasas de ejecución para el sistema de planificación.

Este módulo implementa mecanismos para limitar la frecuencia de ejecución
de tareas según diversos criterios como tipo de tarea, canal, API, etc.
Ayuda a prevenir limitaciones de API, sobrecarga de recursos y distribuir
la carga de manera uniforme.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading

from ..core.task_model import Task, TaskStatus
from .base_strategy import BaseStrategy

logger = logging.getLogger('Scheduler.Strategies.Throttling')

class RateLimiter:
    """
    Implementa un limitador de tasa con ventana deslizante.
    
    Atributos:
        max_requests: Número máximo de solicitudes permitidas en el período
        time_window: Ventana de tiempo en segundos
        window_type: Tipo de ventana ('fixed' o 'sliding')
    """
    
    def __init__(self, 
                max_requests: int, 
                time_window: float,
                window_type: str = 'sliding'):
        """
        Inicializa un limitador de tasa.
        
        Args:
            max_requests: Número máximo de solicitudes permitidas
            time_window: Ventana de tiempo en segundos
            window_type: Tipo de ventana ('fixed' o 'sliding')
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.window_type = window_type
        
        # Para ventana deslizante, mantener historial de timestamps
        self.request_history = deque()
        
        # Para ventana fija, mantener contadores por intervalo
        self.fixed_window_count = 0
        self.current_window_start = time.time()
        
        # Lock para operaciones thread-safe
        self.lock = threading.RLock()
    
    def can_proceed(self) -> bool:
        """
        Verifica si una nueva solicitud puede proceder según los límites.
        
        Returns:
            True si la solicitud puede proceder, False en caso contrario
        """
        with self.lock:
            current_time = time.time()
            
            if self.window_type == 'sliding':
                # Eliminar timestamps antiguos fuera de la ventana
                while self.request_history and current_time - self.request_history[0] > self.time_window:
                    self.request_history.popleft()
                
                # Verificar si estamos dentro del límite
                if len(self.request_history) < self.max_requests:
                    return True
                
                return False
            
            else:  # Ventana fija
                # Verificar si estamos en una nueva ventana
                if current_time - self.current_window_start > self.time_window:
                    # Reiniciar para nueva ventana
                    self.current_window_start = current_time
                    self.fixed_window_count = 0
                
                # Verificar si estamos dentro del límite
                if self.fixed_window_count < self.max_requests:
                    return True
                
                return False
    
    def register_request(self) -> None:
        """
        Registra una nueva solicitud.
        """
        with self.lock:
            current_time = time.time()
            
            if self.window_type == 'sliding':
                self.request_history.append(current_time)
            else:  # Ventana fija
                # Verificar si estamos en una nueva ventana
                if current_time - self.current_window_start > self.time_window:
                    # Reiniciar para nueva ventana
                    self.current_window_start = current_time
                    self.fixed_window_count = 0
                
                self.fixed_window_count += 1
    
    def get_wait_time(self) -> float:
        """
        Calcula el tiempo de espera recomendado antes del próximo intento.
        
        Returns:
            Tiempo de espera en segundos
        """
        with self.lock:
            current_time = time.time()
            
            if self.window_type == 'sliding':
                if not self.request_history:
                    return 0
                
                # Si estamos al límite, calcular tiempo hasta que expire la solicitud más antigua
                if len(self.request_history) >= self.max_requests:
                    oldest_request = self.request_history[0]
                    return max(0, (oldest_request + self.time_window) - current_time)
                
                return 0
            
            else:  # Ventana fija
                # Si estamos al límite, calcular tiempo hasta la siguiente ventana
                if self.fixed_window_count >= self.max_requests:
                    next_window = self.current_window_start + self.time_window
                    return max(0, next_window - current_time)
                
                return 0
    
    def get_current_usage(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el uso actual.
        
        Returns:
            Diccionario con información de uso
        """
        with self.lock:
            current_time = time.time()
            
            if self.window_type == 'sliding':
                # Limpiar historial antiguo
                while self.request_history and current_time - self.request_history[0] > self.time_window:
                    self.request_history.popleft()
                
                return {
                    'current_requests': len(self.request_history),
                    'max_requests': self.max_requests,
                    'usage_percent': (len(self.request_history) / self.max_requests) * 100 if self.max_requests > 0 else 0,
                    'window_type': 'sliding',
                    'time_window': self.time_window
                }
            
            else:  # Ventana fija
                # Verificar si estamos en una nueva ventana
                if current_time - self.current_window_start > self.time_window:
                    window_elapsed = self.time_window  # Ventana completa
                    window_remaining = 0
                else:
                    window_elapsed = current_time - self.current_window_start
                    window_remaining = self.time_window - window_elapsed
                
                return {
                    'current_requests': self.fixed_window_count,
                    'max_requests': self.max_requests,
                    'usage_percent': (self.fixed_window_count / self.max_requests) * 100 if self.max_requests > 0 else 0,
                    'window_type': 'fixed',
                    'time_window': self.time_window,
                    'window_elapsed': window_elapsed,
                    'window_remaining': window_remaining
                }


class ThrottlingStrategy(BaseStrategy):
    """
    Estrategia que implementa control de tasas de ejecución para tareas.
    
    Características:
    - Limitación por tipo de tarea
    - Limitación por canal o plataforma
    - Limitación por API o recurso externo
    - Ventanas deslizantes y fijas
    - Distribución uniforme de carga
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa la estrategia de control de tasas.
        
        Args:
            config: Configuración específica para la estrategia
        """
        super().__init__(config)
        
        # Limitadores por tipo de tarea
        self.task_type_limiters: Dict[str, RateLimiter] = {}
        
        # Limitadores por canal/plataforma
        self.channel_limiters: Dict[str, RateLimiter] = {}
        
        # Limitadores por API/recurso
        self.api_limiters: Dict[str, RateLimiter] = {}
        
        # Limitador global (todas las tareas)
        self.global_limiter: Optional[RateLimiter] = None
        
        # Cargar limitadores desde configuración
        self._load_limiters_from_config()
        
        # Si no hay limitadores configurados, usar valores predeterminados
        if not self.task_type_limiters and not self.channel_limiters and not self.api_limiters and not self.global_limiter:
            self._set_default_limiters()
        
        logger.info(f"ThrottlingStrategy inicializada con {len(self.task_type_limiters)} limitadores por tipo, "
                   f"{len(self.channel_limiters)} por canal, {len(self.api_limiters)} por API "
                   f"y {'un' if self.global_limiter else 'ningún'} limitador global")
    
    def _load_limiters_from_config(self) -> None:
        """
        Carga limitadores desde la configuración.
        """
        if not self.config:
            return
        
        # Cargar limitador global
        if 'global' in self.config:
            global_config = self.config['global']
            try:
                self.global_limiter = RateLimiter(
                    max_requests=global_config.get('max_requests', 100),
                    time_window=global_config.get('time_window', 60),
                    window_type=global_config.get('window_type', 'sliding')
                )
                logger.debug("Limitador global cargado desde configuración")
            except (KeyError, ValueError) as e:
                logger.error(f"Error al cargar limitador global: {str(e)}")
        
        # Cargar limitadores por tipo de tarea
        for limiter_config in self.config.get('task_type_limiters', []):
            try:
                task_type = limiter_config.get('task_type')
                if not task_type:
                    logger.warning("Limitador sin tipo de tarea especificado, ignorando")
                    continue
                
                limiter = RateLimiter(
                    max_requests=limiter_config.get('max_requests', 10),
                    time_window=limiter_config.get('time_window', 60),
                    window_type=limiter_config.get('window_type', 'sliding')
                )
                
                self.task_type_limiters[task_type] = limiter
                logger.debug(f"Limitador cargado para tipo de tarea '{task_type}'")
            
            except (KeyError, ValueError) as e:
                logger.error(f"Error al cargar limitador por tipo: {str(e)}")
        
        # Cargar limitadores por canal
        for limiter_config in self.config.get('channel_limiters', []):
            try:
                channel_id = limiter_config.get('channel_id')
                if not channel_id:
                    logger.warning("Limitador sin canal especificado, ignorando")
                    continue
                
                limiter = RateLimiter(
                    max_requests=limiter_config.get('max_requests', 5),
                    time_window=limiter_config.get('time_window', 60),
                    window_type=limiter_config.get('window_type', 'sliding')
                )
                
                self.channel_limiters[channel_id] = limiter
                logger.debug(f"Limitador cargado para canal '{channel_id}'")
            
            except (KeyError, ValueError) as e:
                logger.error(f"Error al cargar limitador por canal: {str(e)}")
        
        # Cargar limitadores por API
        for limiter_config in self.config.get('api_limiters', []):
            try:
                api_name = limiter_config.get('api_name')
                if not api_name:
                    logger.warning("Limitador sin API especificada, ignorando")
                    continue
                
                limiter = RateLimiter(
                    max_requests=limiter_config.get('max_requests', 20),
                    time_window=limiter_config.get('time_window', 60),
                    window_type=limiter_config.get('window_type', 'sliding')
                )
                
                self.api_limiters[api_name] = limiter
                logger.debug(f"Limitador cargado para API '{api_name}'")
            
            except (KeyError, ValueError) as e:
                logger.error(f"Error al cargar limitador por API: {str(e)}")
    
    def _set_default_limiters(self) -> None:
        """
        Configura limitadores predeterminados.
        """
        # Limitador global (todas las tareas)
        self.global_limiter = RateLimiter(
            max_requests=100,  # 100 tareas por minuto como máximo
            time_window=60,    # Ventana de 1 minuto
            window_type='sliding'
        )
        
        # Limitadores por tipo de tarea
        self.task_type_limiters = {
            # Publicaciones (más restrictivas)
            "publish": RateLimiter(
                max_requests=10,   # 10 publicaciones por hora
                time_window=3600,  # Ventana de 1 hora
                window_type='sliding'
            ),
            
            # Análisis (menos restrictivo)
            "analysis": RateLimiter(
                max_requests=20,   # 20 análisis por minuto
                time_window=60,    # Ventana de 1 minuto
                window_type='sliding'
            ),
            
            # Creación de contenido (equilibrado)
            "content_creation": RateLimiter(
                max_requests=30,   # 30 tareas por hora
                time_window=3600,  # Ventana de 1 hora
                window_type='sliding'
            )
        }
        
        # Limitadores por API (para APIs externas)
        self.api_limiters = {
            # YouTube API (muy restrictiva)
            "youtube_api": RateLimiter(
                max_requests=5,    # 5 solicitudes por minuto
                time_window=60,    # Ventana de 1 minuto
                window_type='sliding'
            ),
            
            # TikTok API
            "tiktok_api": RateLimiter(
                max_requests=10,   # 10 solicitudes por minuto
                time_window=60,    # Ventana de 1 minuto
                window_type='sliding'
            ),
            
            # Instagram API
            "instagram_api": RateLimiter(
                max_requests=15,   # 15 solicitudes por hora
                time_window=3600,  # Ventana de 1 hora
                window_type='sliding'
            ),
            
            # OpenAI API
            "openai_api": RateLimiter(
                max_requests=20,   # 20 solicitudes por minuto
                time_window=60,    # Ventana de 1 minuto
                window_type='sliding'
            )
        }
    
    def should_apply(self, task: Task, context: Dict[str, Any] = None) -> bool:
        """
        Determina si la estrategia debe aplicarse a una tarea.
        
        Args:
            task: Tarea a evaluar
            context: Contexto adicional
            
        Returns:
            True si la estrategia debe aplicarse
        """
        # Solo aplicar a tareas que están a punto de ejecutarse
        if task.status != TaskStatus.SCHEDULED:
            return False
        
        # No aplicar a tareas que explícitamente lo deshabilitan
        if task.metadata.get('skip_throttling', False):
            return True
        
        return True
    
    def apply(self, task: Task, context: Dict[str, Any] = None) -> Task:
        """
        Aplica la estrategia de control de tasas a una tarea.
        
        Args:
            task: Tarea a procesar
            context: Contexto adicional
            
        Returns:
            Tarea posiblemente reprogramada según límites
        """
        context = context or {}
        
        # Verificar si podemos proceder con la tarea
        can_proceed, wait_time, limiter_info = self._check_rate_limits(task)
        
        if can_proceed:
            # Registrar la ejecución en los limitadores aplicables
            self._register_execution(task)
            return task
        
        # No podemos proceder, reprogramar la tarea
        new_execution_time = datetime.now() + timedelta(seconds=wait_time)
        
        # Actualizar tiempo programado
        original_time = task.scheduled_time
        task.scheduled_time = new_execution_time
        
        # Registrar información en metadatos
        if 'throttling_history' not in task.metadata:
            task.metadata['throttling_history'] = []
        
        task.metadata['throttling_history'].append({
            'timestamp': datetime.now().isoformat(),
            'original_time': original_time.isoformat() if original_time else None,
            'new_time': new_execution_time.isoformat(),
            'wait_time': wait_time,
            'limiter_type': limiter_info.get('type'),
            'limiter_name': limiter_info.get('name')
        })
        
        logger.info(f"Tarea {task.task_id} ({task.task_type}) reprogramada por límite de tasa "
                   f"({limiter_info.get('type')}: {limiter_info.get('name')}). "
                   f"Nueva ejecución en {wait_time:.2f}s")
        
        return task
    
    def _check_rate_limits(self, task: Task) -> Tuple[bool, float, Dict[str, str]]:
        """
        Verifica todos los limitadores aplicables a una tarea.
        
        Args:
            task: Tarea a verificar
            
        Returns:
            Tupla (puede_proceder, tiempo_espera, info_limitador)
        """
        # Lista para almacenar resultados de todos los limitadores aplicables
        limiters_to_check = []
        
        # Verificar limitador global
        if self.global_limiter:
            limiters_to_check.append(('global', 'global', self.global_limiter))
        
        # Verificar limitador por tipo de tarea
        if task.task_type in self.task_type_limiters:
            limiters_to_check.append(('task_type', task.task_type, self.task_type_limiters[task.task_type]))
        
        # Verificar limitador por canal
        channel_id = task.data.get('channel_id')
        if channel_id and channel_id in self.channel_limiters:
            limiters_to_check.append(('channel', channel_id, self.channel_limiters[channel_id]))
        
        # Verificar limitador por API
        api_name = task.data.get('api_name') or task.metadata.get('api_name')
        if api_name and api_name in self.api_limiters:
            limiters_to_check.append(('api', api_name, self.api_limiters[api_name]))
        
        # Si no hay limitadores aplicables, siempre proceder
        if not limiters_to_check:
            return True, 0, {'type': None, 'name': None}
        
        # Verificar todos los limitadores aplicables
        max_wait_time = 0
        blocking_limiter = {'type': None, 'name': None}
        
        for limiter_type, limiter_name, limiter in limiters_to_check:
            if not limiter.can_proceed():
                wait_time = limiter.get_wait_time()
                
                # Actualizar tiempo máximo de espera
                if wait_time > max_wait_time:
                    max_wait_time = wait_time
                    blocking_limiter = {'type': limiter_type, 'name': limiter_name}
        
        # Si hay tiempo de espera, no podemos proceder
        if max_wait_time > 0:
            return False, max_wait_time, blocking_limiter
        
        # Todos los limitadores permiten proceder
        return True, 0, {'type': None, 'name': None}
    
    def _register_execution(self, task: Task) -> None:
        """
        Registra la ejecución en todos los limitadores aplicables.
        
        Args:
            task: Tarea que se va a ejecutar
        """
        # Registrar en limitador global
        if self.global_limiter:
            self.global_limiter.register_request()
        
        # Registrar en limitador por tipo de tarea
        if task.task_type in self.task_type_limiters:
            self.task_type_limiters[task.task_type].register_request()
        
        # Registrar en limitador por canal
        channel_id = task.data.get('channel_id')
        if channel_id and channel_id in self.channel_limiters:
            self.channel_limiters[channel_id].register_request()
        
        # Registrar en limitador por API
        api_name = task.data.get('api_name') or task.metadata.get('api_name')
        if api_name and api_name in self.api_limiters:
            self.api_limiters[api_name].register_request()
    
    def add_task_type_limiter(self, task_type: str, max_requests: int, time_window: float, window_type: str = 'sliding') -> None:
        """
        Añade o actualiza un limitador por tipo de tarea.
        
        Args:
            task_type: Tipo de tarea
            max_requests: Número máximo de solicitudes
            time_window: Ventana de tiempo en segundos
            window_type: Tipo de ventana ('fixed' o 'sliding')
        """
        self.task_type_limiters[task_type] = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            window_type=window_type
        )
        logger.debug(f"Limitador actualizado para tipo de tarea '{task_type}'")
    
    def add_channel_limiter(self, channel_id: str, max_requests: int, time_window: float, window_type: str = 'sliding') -> None:
        """
        Añade o actualiza un limitador por canal.
        
        Args:
            channel_id: ID del canal
            max_requests: Número máximo de solicitudes
            time_window: Ventana de tiempo en segundos
            window_type: Tipo de ventana ('fixed' o 'sliding')
        """
        self.channel_limiters[channel_id] = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            window_type=window_type
        )
        logger.debug(f"Limitador actualizado para canal '{channel_id}'")
    
    def add_api_limiter(self, api_name: str, max_requests: int, time_window: float, window_type: str = 'sliding') -> None:
        """
        Añade o actualiza un limitador por API.
        
        Args:
            api_name: Nombre de la API
            max_requests: Número máximo de solicitudes
            time_window: Ventana de tiempo en segundos
            window_type: Tipo de ventana ('fixed' o 'sliding')
        """
        self.api_limiters[api_name] = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            window_type=window_type
        )
        logger.debug(f"Limitador actualizado para API '{api_name}'")
    
    def set_global_limiter(self, max_requests: int, time_window: float, window_type: str = 'sliding') -> None:
        """
        Establece o actualiza el limitador global.
        
        Args:
            max_requests: Número máximo de solicitudes
            time_window: Ventana de tiempo en segundos
            window_type: Tipo de ventana ('fixed' o 'sliding')
        """
        self.global_limiter = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            window_type=window_type
        )
        logger.debug(f"Limitador global actualizado: {max_requests} solicitudes / {time_window}s ({window_type})")
    
    def remove_task_type_limiter(self, task_type: str) -> bool:
        """
        Elimina un limitador por tipo de tarea.
        
        Args:
            task_type: Tipo de tarea
            
        Returns:
            True si se eliminó, False si no existía
        """
        if task_type in self.task_type_limiters:
            del self.task_type_limiters[task_type]
            logger.debug(f"Limitador eliminado para tipo de tarea '{task_type}'")
            return True
        return False
    
    def remove_channel_limiter(self, channel_id: str) -> bool:
        """
        Elimina un limitador por canal.
        
        Args:
            channel_id: ID del canal
            
        Returns:
            True si se eliminó, False si no existía
        """
        if channel_id in self.channel_limiters:
            del self.channel_limiters[channel_id]
            logger.debug(f"Limitador eliminado para canal '{channel_id}'")
            return True
        return False
    
    def remove_api_limiter(self, api_name: str) -> bool:
        """
        Elimina un limitador por API.
        
        Args:
            api_name: Nombre de la API
            
        Returns:
            True si se eliminó, False si no existía
        """
        if api_name in self.api_limiters:
            del self.api_limiters[api_name]
            logger.debug(f"Limitador eliminado para API '{api_name}'")
            return True
        return False
    
    def remove_global_limiter(self) -> bool:
        """
        Elimina el limitador global.
        
        Returns:
            True si se eliminó, False si no existía
        """
        if self.global_limiter:
            self.global_limiter = None
            logger.debug("Limitador global eliminado")
            return True
        return False
    
    def get_limiter_status(self, limiter_type: str, limiter_name: str = None) -> Dict[str, Any]:
        """
        Obtiene el estado actual de un limitador específico.
        
        Args:
            limiter_type: Tipo de limitador ('global', 'task_type', 'channel', 'api')
            limiter_name: Nombre del limitador (no necesario para 'global')
            
        Returns:
            Diccionario con información de estado o None si no existe
        """
        limiter = None
        
        if limiter_type == 'global':
            limiter = self.global_limiter
        elif limiter_type == 'task_type' and limiter_name:
            limiter = self.task_type_limiters.get(limiter_name)
        elif limiter_type == 'channel' and limiter_name:
            limiter = self.channel_limiters.get(limiter_name)
        elif limiter_type == 'api' and limiter_name:
            limiter = self.api_limiters.get(limiter_name)
        
        if not limiter:
            return None
        
        # Obtener información de uso
        usage_info = limiter.get_current_usage()
        
        # Añadir información adicional
        usage_info.update({
            'limiter_type': limiter_type,
            'limiter_name': limiter_name,
            'can_proceed': limiter.can_proceed(),
            'wait_time': limiter.get_wait_time()
        })
        
        return usage_info
    
    def get_all_limiters_status(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Obtiene el estado de todos los limitadores.
        
        Returns:
            Diccionario con estados de todos los limitadores
        """
        result = {
            'global': [],
            'task_type': [],
            'channel': [],
            'api': []
        }
        
        # Limitador global
        if self.global_limiter:
            global_status = self.get_limiter_status('global')
            if global_status:
                result['global'].append(global_status)
        
        # Limitadores por tipo de tarea
        for task_type in self.task_type_limiters:
            status = self.get_limiter_status('task_type', task_type)
            if status:
                result['task_type'].append(status)
        
        # Limitadores por canal
        for channel_id in self.channel_limiters:
            status = self.get_limiter_status('channel', channel_id)
            if status:
                result['channel'].append(status)
        
        # Limitadores por API
        for api_name in self.api_limiters:
            status = self.get_limiter_status('api', api_name)
            if status:
                result['api'].append(status)
        
        return result
    
    def reset_all_limiters(self) -> None:
        """
        Reinicia todos los limitadores a sus valores predeterminados.
        """
        # Eliminar todos los limitadores actuales
        self.task_type_limiters = {}
        self.channel_limiters = {}
        self.api_limiters = {}
        self.global_limiter = None
        
        # Configurar limitadores predeterminados
        self._set_default_limiters()
        
        logger.info("Todos los limitadores han sido reiniciados a valores predeterminados")
    
    def export_config(self) -> Dict[str, Any]:
        """
        Exporta la configuración actual de limitadores.
        
        Returns:
            Diccionario con configuración exportable
        """
        config = {
            'task_type_limiters': [],
            'channel_limiters': [],
            'api_limiters': []
        }
        
        # Exportar limitador global
        if self.global_limiter:
            global_usage = self.global_limiter.get_current_usage()
            config['global'] = {
                'max_requests': self.global_limiter.max_requests,
                'time_window': self.global_limiter.time_window,
                'window_type': self.global_limiter.window_type
            }
        
        # Exportar limitadores por tipo de tarea
        for task_type, limiter in self.task_type_limiters.items():
            config['task_type_limiters'].append({
                'task_type': task_type,
                'max_requests': limiter.max_requests,
                'time_window': limiter.time_window,
                'window_type': limiter.window_type
            })
        
        # Exportar limitadores por canal
        for channel_id, limiter in self.channel_limiters.items():
            config['channel_limiters'].append({
                'channel_id': channel_id,
                'max_requests': limiter.max_requests,
                'time_window': limiter.time_window,
                'window_type': limiter.window_type
            })
        
        # Exportar limitadores por API
        for api_name, limiter in self.api_limiters.items():
            config['api_limiters'].append({
                'api_name': api_name,
                'max_requests': limiter.max_requests,
                'time_window': limiter.time_window,
                'window_type': limiter.window_type
            })
        
        return config