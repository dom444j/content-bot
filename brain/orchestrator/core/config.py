"""
Config - Gestión de configuración para el Orchestrator

Este módulo proporciona funcionalidades para cargar, validar y acceder a la
configuración del sistema Orchestrator desde diferentes fuentes (archivos JSON,
variables de entorno, etc.).
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

# Configuración de logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Gestor de configuración para el Orchestrator.
    
    Esta clase se encarga de cargar, validar y proporcionar acceso a la
    configuración del sistema desde diferentes fuentes, con soporte para
    valores por defecto y validación de esquemas.
    """
    
    def __init__(self, config_path: str = "config/strategy.json"):
        """
        Inicializa el gestor de configuración.
        
        Args:
            config_path: Ruta al archivo de configuración principal
        """
        self.config_path = config_path
        self.config = {}
        self.env_prefix = "ORCHESTRATOR_"
        self.load_config()
        logger.info(f"ConfigManager inicializado con archivo {config_path}")
    
    def load_config(self) -> None:
        """
        Carga la configuración desde el archivo y las variables de entorno.
        """
        try:
            # Cargar desde archivo
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Configuración cargada desde {self.config_path}")
            else:
                logger.warning(f"Archivo de configuración {self.config_path} no encontrado")
                self.config = {}
            
            # Sobrescribir con variables de entorno
            self._load_from_env()
            
            # Validar configuración
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            # Usar configuración por defecto en caso de error
            self.config = self._get_default_config()
    
    def _load_from_env(self) -> None:
        """
        Carga configuración desde variables de entorno.
        
        Las variables deben tener el prefijo ORCHESTRATOR_ y usar guiones bajos
        para representar la jerarquía (ej: ORCHESTRATOR_SYSTEM_NUM_WORKERS).
        """
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Quitar prefijo y convertir a estructura jerárquica
                config_key = key[len(self.env_prefix):].lower()
                path = config_key.split('_')
                
                # Convertir valor según tipo
                try:
                    # Intentar convertir a número si es posible
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
                        value = float(value)
                    elif value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                except (ValueError, AttributeError):
                    pass
                
                # Actualizar configuración
                self._set_nested_value(self.config, path, value)
                logger.debug(f"Configuración actualizada desde variable de entorno: {key}")
    
    def _set_nested_value(self, config: Dict[str, Any], path: List[str], value: Any) -> None:
        """
        Establece un valor en una estructura anidada.
        
        Args:
            config: Diccionario de configuración
            path: Lista de claves que representan la ruta al valor
            value: Valor a establecer
        """
        if len(path) == 1:
            config[path[0]] = value
            return
        
        key = path[0]
        if key not in config:
            config[key] = {}
        
        self._set_nested_value(config[key], path[1:], value)
    
    def _validate_config(self) -> None:
        """
        Valida la configuración cargada.
        
        Verifica que los valores requeridos estén presentes y sean del tipo correcto.
        """
        required_keys = [
            ("system", "num_workers", int),
            ("system", "max_retries", int),
            ("system", "retry_delay", (int, float)),
        ]
        
        for path in required_keys:
            value = self.get(path[:-1], path[-1])
            if value is None:
                logger.warning(f"Valor requerido no encontrado: {'.'.join(path[:-1])}.{path[-1]}")
                # Establecer valor por defecto
                default_value = self._get_default_value(path)
                self._set_nested_value(self.config, path[:-1] + [path[-1]], default_value)
            elif not isinstance(value, path[-1]):
                logger.warning(f"Tipo incorrecto para {'.'.join(path[:-1])}.{path[-1]}: "
                              f"esperado {path[-1]}, obtenido {type(value)}")
                # Intentar convertir o usar valor por defecto
                try:
                    if isinstance(path[-1], tuple):
                        # Si se aceptan múltiples tipos, intentar convertir al primero
                        converted_value = path[-1][0](value)
                    else:
                        converted_value = path[-1](value)
                    self._set_nested_value(self.config, path[:-1] + [path[-1]], converted_value)
                except (ValueError, TypeError):
                    default_value = self._get_default_value(path)
                    self._set_nested_value(self.config, path[:-1] + [path[-1]], default_value)
    
    def _get_default_value(self, path: tuple) -> Any:
        """
        Obtiene el valor por defecto para una ruta de configuración.
        
        Args:
            path: Tupla que representa la ruta al valor
            
        Returns:
            Any: Valor por defecto
        """
        defaults = {
            ("system", "num_workers", int): 5,
            ("system", "max_retries", int): 3,
            ("system", "retry_delay", (int, float)): 5.0,
            ("system", "log_level", str): "INFO",
            ("system", "persistence_enabled", bool): True,
        }
        
        return defaults.get(path, None)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Obtiene una configuración por defecto.
        
        Returns:
            Dict[str, Any]: Configuración por defecto
        """
        return {
            "system": {
                "num_workers": 5,
                "max_retries": 3,
                "retry_delay": 5.0,
                "log_level": "INFO",
                "persistence_enabled": True,
            },
            "content": {
                "max_length": 1000,
                "default_language": "es",
            },
            "publish": {
                "default_platforms": ["youtube", "tiktok"],
                "schedule_ahead_days": 7,
            },
            "analysis": {
                "metrics_window_days": 30,
                "min_data_points": 10,
            },
            "monetization": {
                "min_followers": 1000,
                "min_engagement_rate": 0.02,
            },
        }
    
    def get(self, *args) -> Any:
        """
        Obtiene un valor de configuración.
        
        Se puede llamar de dos formas:
        - get("section", "key"): para acceder a config["section"]["key"]
        - get("section.key"): para acceder a config["section"]["key"]
        
        Args:
            *args: Ruta al valor, como argumentos separados o como string con puntos
            
        Returns:
            Any: Valor de configuración o None si no existe
        """
        if len(args) == 1 and isinstance(args[0], str) and "." in args[0]:
            # Formato "section.key"
            path = args[0].split(".")
        else:
            # Formato "section", "key"
            path = args
        
        # Navegar por la estructura
        current = self.config
        for part in path:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def set(self, path: str, value: Any) -> None:
        """
        Establece un valor de configuración.
        
        Args:
            path: Ruta al valor, usando puntos como separadores
            value: Valor a establecer
        """
        parts = path.split(".")
        self._set_nested_value(self.config, parts, value)
        logger.debug(f"Configuración actualizada: {path} = {value}")
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Guarda la configuración actual en un archivo.
        
        Args:
            path: Ruta al archivo (opcional, usa self.config_path por defecto)
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        try:
            save_path = path or self.config_path
            
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuración guardada en {save_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error al guardar configuración: {str(e)}")
            return False
    
    def reload(self) -> None:
        """
        Recarga la configuración desde el archivo.
        """
        self.load_config()
        logger.info("Configuración recargada")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Obtiene toda la configuración.
        
        Returns:
            Dict[str, Any]: Configuración completa
        """
        return self.config.copy()