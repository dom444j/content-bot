"""
Clase base para estrategias del sistema de planificación.

Este módulo define la interfaz común para todas las estrategias
que pueden aplicarse a las tareas del planificador.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union

from ..core.task_model import Task

logger = logging.getLogger('Scheduler.Strategies.Base')

class BaseStrategy(ABC):
    """
    Clase base abstracta para todas las estrategias del planificador.
    
    Las estrategias implementan comportamientos configurables que pueden
    aplicarse a las tareas, como políticas de reintento, priorización,
    control de tasas, etc.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa la estrategia con configuración opcional.
        
        Args:
            config: Diccionario de configuración para la estrategia
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        logger.debug(f"Inicializando estrategia {self.name}")
    
    @abstractmethod
    def apply(self, task: Task, context: Dict[str, Any] = None) -> Task:
        """
        Aplica la estrategia a una tarea.
        
        Args:
            task: Tarea a la que aplicar la estrategia
            context: Contexto adicional para la aplicación de la estrategia
            
        Returns:
            Tarea modificada según la estrategia
        """
        pass
    
    def should_apply(self, task: Task, context: Dict[str, Any] = None) -> bool:
        """
        Determina si la estrategia debe aplicarse a una tarea específica.
        
        Args:
            task: Tarea a evaluar
            context: Contexto adicional para la decisión
            
        Returns:
            True si la estrategia debe aplicarse, False en caso contrario
        """
        # Por defecto, aplicar a todas las tareas
        # Las subclases pueden sobrescribir este método para implementar
        # lógica específica de aplicabilidad
        return True
    
    def get_name(self) -> str:
        """
        Obtiene el nombre de la estrategia.
        
        Returns:
            Nombre de la estrategia
        """
        return self.name
    
    def get_config(self) -> Dict[str, Any]:
        """
        Obtiene la configuración actual de la estrategia.
        
        Returns:
            Diccionario de configuración
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Actualiza la configuración de la estrategia.
        
        Args:
            new_config: Nueva configuración a aplicar
        """
        self.config.update(new_config)
        logger.debug(f"Configuración actualizada para estrategia {self.name}")