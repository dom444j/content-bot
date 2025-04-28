"""
Base Model

Clase base para todos los modelos predictivos del sistema.
Proporciona funcionalidad común como entrenamiento, evaluación,
persistencia y carga de modelos.
"""

import os
import json
import pickle
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/predictive_models.log"),
        logging.StreamHandler()
    ]
)

class BaseModel(ABC):
    """
    Clase base abstracta para modelos predictivos
    """
    
    def __init__(self, model_name: str, model_type: str, 
                 features: List[str], target: str,
                 model_params: Dict[str, Any] = None,
                 model_dir: str = "data/models"):
        """
        Inicializa el modelo base
        
        Args:
            model_name: Nombre único del modelo
            model_type: Tipo de modelo (ej. 'regression', 'classification')
            features: Lista de características de entrada
            target: Variable objetivo
            model_params: Parámetros específicos del modelo
            model_dir: Directorio para guardar/cargar modelos
        """
        self.model_name = model_name
        self.model_type = model_type
        self.features = features
        self.target = target
        self.model_params = model_params or {}
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.metrics = {}
        self.training_history = []
        self.last_trained = None
        self.version = "1.0.0"
        
        # Crear directorio si no existe
        os.makedirs(model_dir, exist_ok=True)
        
        # Logger específico para este modelo
        self.logger = logging.getLogger(f"PredictiveModel.{model_name}")
    
    @abstractmethod
    def build_model(self) -> Any:
        """
        Construye el modelo con los parámetros especificados
        
        Returns:
            Modelo construido
        """
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesa los datos antes del entrenamiento o predicción
        
        Args:
            data: DataFrame con los datos
            
        Returns:
            Tuple con (X, y) donde X son las características y y es el objetivo
        """
        # Verificar que todas las características estén presentes
        missing_features = [f for f in self.features if f not in data.columns]
        if missing_features:
            self.logger.warning(f"Características faltantes: {missing_features}")
            raise ValueError(f"Faltan características en los datos: {missing_features}")
        
        # Verificar que el objetivo esté presente
        if self.target not in data.columns:
            self.logger.warning(f"Objetivo faltante: {self.target}")
            raise ValueError(f"Falta el objetivo en los datos: {self.target}")
        
        # Extraer características y objetivo
        X = data[self.features].copy()
        y = data[self.target].copy()
        
        # Manejar valores nulos
        if X.isnull().any().any():
            self.logger.warning(f"Valores nulos encontrados en las características. Rellenando con la media.")
            X = X.fillna(X.mean())
        
        if y.isnull().any():
            self.logger.warning(f"Valores nulos encontrados en el objetivo. Rellenando con la media.")
            y = y.fillna(y.mean())
        
        return X, y
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2, 
              random_state: int = 42) -> Dict[str, Any]:
        """
        Entrena el modelo con los datos proporcionados
        
        Args:
            data: DataFrame con los datos
            test_size: Proporción de datos para prueba
            random_state: Semilla para reproducibilidad
            
        Returns:
            Métricas de evaluación
        """
        self.logger.info(f"Iniciando entrenamiento del modelo {self.model_name}")
        
        # Preprocesar datos
        X, y = self.preprocess_data(data)
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Construir modelo si no existe
        if self.model is None:
            self.model = self.build_model()
        
        # Entrenar modelo
        self.logger.info(f"Entrenando modelo con {len(X_train)} muestras")
        self.model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = self.model.predict(X_test)
        
        # Calcular métricas
        metrics = self._calculate_metrics(y_test, y_pred)
        self.metrics = metrics
        
        # Calcular importancia de características si está disponible
        self._calculate_feature_importance()
        
        # Registrar entrenamiento
        self.last_trained = datetime.datetime.now().isoformat()
        self.training_history.append({
            "date": self.last_trained,
            "samples": len(X),
            "metrics": metrics
        })
        
        # Guardar modelo
        self.save()
        
        self.logger.info(f"Entrenamiento completado. Métricas: {metrics}")
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas de evaluación
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            
        Returns:
            Diccionario con métricas
        """
        metrics = {}
        
        if self.model_type == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        elif self.model_type == 'classification':
            # Para clasificación, implementar métricas adecuadas
            pass
        
        return metrics
    
    def _calculate_feature_importance(self) -> None:
        """
        Calcula la importancia de las características si está disponible
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.features, self.model.feature_importances_))
            elif hasattr(self.model, 'coef_'):
                if self.model.coef_.ndim == 1:
                    self.feature_importance = dict(zip(self.features, np.abs(self.model.coef_)))
                else:
                    # Para modelos multiclase
                    self.feature_importance = dict(zip(self.features, np.mean(np.abs(self.model.coef_), axis=0)))
        except Exception as e:
            self.logger.warning(f"No se pudo calcular la importancia de características: {str(e)}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado
        
        Args:
            data: DataFrame con los datos
            
        Returns:
            Array con predicciones
        """
        if self.model is None:
            self.logger.error("El modelo no está entrenado")
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Si solo se proporcionan las características
        if all(f in data.columns for f in self.features):
            X = data[self.features].copy()
        else:
            # Intentar preprocesar los datos completos
            X, _ = self.preprocess_data(data)
        
        # Manejar valores nulos
        if X.isnull().any().any():
            self.logger.warning(f"Valores nulos encontrados en las características. Rellenando con la media.")
            X = X.fillna(X.mean())
        
        # Realizar predicción
        return self.model.predict(X)
    
    def save(self, filepath: str = None) -> str:
        """
        Guarda el modelo en disco
        
        Args:
            filepath: Ruta donde guardar el modelo. Si es None, se usa el directorio por defecto.
            
        Returns:
            Ruta donde se guardó el modelo
        """
        if self.model is None:
            self.logger.error("No hay modelo para guardar")
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")
        
        if filepath is None:
            # Crear nombre de archivo basado en el nombre del modelo y la fecha
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_{timestamp}.pkl"
            filepath = os.path.join(self.model_dir, filename)
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar modelo
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features,
                'target': self.target,
                'model_type': self.model_type,
                'model_params': self.model_params,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history,
                'last_trained': self.last_trained,
                'version': self.version
            }, f)
        
        # Guardar metadatos en formato JSON
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_name': self.model_name,
                'model_type': self.model_type,
                'features': self.features,
                'target': self.target,
                'model_params': self.model_params,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'training_history': self.training_history,
                'last_trained': self.last_trained,
                'version': self.version,
                'filepath': filepath
            }, f, indent=4, default=str)
        
        self.logger.info(f"Modelo guardado en {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseModel':
        """
        Carga un modelo desde disco
        
        Args:
            filepath: Ruta del modelo guardado
            
        Returns:
            Instancia del modelo cargado
        """
        logger = logging.getLogger(f"PredictiveModel.Loader")
        
        if not os.path.exists(filepath):
            logger.error(f"Archivo no encontrado: {filepath}")
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Crear instancia
            instance = cls(
                model_name=data.get('model_name', 'loaded_model'),
                model_type=data.get('model_type', 'unknown'),
                features=data.get('features', []),
                target=data.get('target', ''),
                model_params=data.get('model_params', {})
            )
            
            # Restaurar estado
            instance.model = data.get('model')
            instance.scaler = data.get('scaler')
            instance.metrics = data.get('metrics', {})
            instance.feature_importance = data.get('feature_importance', {})
            instance.training_history = data.get('training_history', [])
            instance.last_trained = data.get('last_trained')
            instance.version = data.get('version', '1.0.0')
            
            logger.info(f"Modelo cargado desde {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Obtiene la importancia de las características
        
        Returns:
            Diccionario con la importancia de cada característica
        """
        if self.feature_importance is None:
            self.logger.warning("La importancia de características no está disponible")
            return {}
        
        # Ordenar por importancia
        return dict(sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtiene información del modelo
        
        Returns:
            Diccionario con información del modelo
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'features': self.features,
            'target': self.target,
            'model_params': self.model_params,
            'metrics': self.metrics,
            'feature_importance': self.get_feature_importance(),
            'training_history': self.training_history,
            'last_trained': self.last_trained,
            'version': self.version
        }
    
    def update_model(self, new_data: pd.DataFrame, 
                     retrain: bool = False) -> Dict[str, Any]:
        """
        Actualiza el modelo con nuevos datos
        
        Args:
            new_data: DataFrame con nuevos datos
            retrain: Si es True, se reentrenará el modelo desde cero
            
        Returns:
            Métricas de evaluación
        """
        if retrain or self.model is None:
            # Reentrenar desde cero
            return self.train(new_data)
        else:
            # Actualizar modelo existente (si es compatible)
            try:
                X, y = self.preprocess_data(new_data)
                
                # Verificar si el modelo soporta partial_fit
                if hasattr(self.model, 'partial_fit'):
                    self.logger.info(f"Actualizando modelo con {len(X)} nuevas muestras")
                    self.model.partial_fit(X, y)
                    
                    # Evaluar con los nuevos datos
                    y_pred = self.model.predict(X)
                    metrics = self._calculate_metrics(y, y_pred)
                    
                    # Actualizar métricas e historial
                    self.metrics = metrics
                    self._calculate_feature_importance()
                    
                    self.last_trained = datetime.datetime.now().isoformat()
                    self.training_history.append({
                        "date": self.last_trained,
                        "samples": len(X),
                        "metrics": metrics,
                        "update_type": "incremental"
                    })
                    
                    # Guardar modelo actualizado
                    self.save()
                    
                    self.logger.info(f"Actualización completada. Métricas: {metrics}")
                    return metrics
                else:
                    self.logger.warning("El modelo no soporta actualización incremental. Reentrenando.")
                    return self.train(new_data)
            except Exception as e:
                self.logger.error(f"Error al actualizar modelo: {str(e)}")
                self.logger.warning("Reentrenando modelo desde cero debido al error.")
                return self.train(new_data)