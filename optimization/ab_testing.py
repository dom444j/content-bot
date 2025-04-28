"""
A/B Testing Module

Este módulo implementa un sistema de pruebas A/B para optimizar diferentes aspectos
del contenido, como CTAs, timing, visuales y voces. Permite realizar experimentos
controlados y analizar los resultados para mejorar continuamente el rendimiento.
"""

import os
import sys
import json
import random
import logging
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ab_testing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ABTesting")

class ABTest:
    """
    Clase para gestionar pruebas A/B
    """
    
    def __init__(self, test_name: str, variants: List[Dict[str, Any]], 
                 metrics: List[str], config_path: str = "config/ab_testing.json"):
        """
        Inicializa una prueba A/B
        
        Args:
            test_name: Nombre único de la prueba
            variants: Lista de variantes a probar (cada una es un diccionario con parámetros)
            metrics: Lista de métricas a medir
            config_path: Ruta al archivo de configuración
        """
        self.test_id = str(uuid.uuid4())
        self.test_name = test_name
        self.variants = variants
        self.metrics = metrics
        self.config_path = config_path
        self.start_date = datetime.datetime.now()
        self.end_date = None
        self.status = "created"
        self.results = {}
        self.sample_size = self._calculate_sample_size()
        self.config = self._load_config()
        
        # Registrar la prueba
        self._save_test()
        logger.info(f"Prueba A/B '{test_name}' creada con ID {self.test_id}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Carga la configuración desde el archivo
        
        Returns:
            Configuración de pruebas A/B
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Configuración por defecto
                default_config = {
                    "min_sample_size": 100,
                    "confidence_level": 0.95,
                    "min_detectable_effect": 0.1,
                    "max_test_duration_days": 14,
                    "auto_stop_on_significance": True
                }
                
                # Crear directorio si no existe
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # Guardar configuración por defecto
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4)
                
                return default_config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {
                "min_sample_size": 100,
                "confidence_level": 0.95,
                "min_detectable_effect": 0.1,
                "max_test_duration_days": 14,
                "auto_stop_on_significance": True
            }
    
    def _calculate_sample_size(self) -> int:
        """
        Calcula el tamaño de muestra necesario para la prueba
        
        Returns:
            Tamaño de muestra recomendado
        """
        # Valores por defecto
        baseline_conversion = 0.1  # 10% de conversión base
        min_detectable_effect = 0.1  # Detectar cambios del 10%
        confidence_level = 0.95
        power = 0.8
        
        # Cálculo básico para prueba de proporciones
        # Fórmula simplificada para prueba de dos colas
        z_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        z_beta = stats.norm.ppf(power)
        
        p1 = baseline_conversion
        p2 = baseline_conversion * (1 + min_detectable_effect)
        p_avg = (p1 + p2) / 2
        
        sample_size = (2 * p_avg * (1 - p_avg) * (z_alpha + z_beta)**2) / (p1 - p2)**2
        
        # Redondear hacia arriba y multiplicar por 2 (para dos variantes)
        sample_size = int(np.ceil(sample_size)) * 2
        
        # Asegurar un mínimo razonable
        min_sample = 100
        return max(sample_size, min_sample)
    
    def _save_test(self) -> None:
        """
        Guarda la información de la prueba en un archivo JSON
        """
        try:
            # Crear directorio si no existe
            os.makedirs("data/ab_tests", exist_ok=True)
            
            # Preparar datos para guardar
            test_data = {
                "test_id": self.test_id,
                "test_name": self.test_name,
                "variants": self.variants,
                "metrics": self.metrics,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "status": self.status,
                "results": self.results,
                "sample_size": self.sample_size
            }
            
            # Guardar en archivo
            file_path = f"data/ab_tests/{self.test_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=4)
            
            logger.info(f"Prueba A/B guardada en {file_path}")
        except Exception as e:
            logger.error(f"Error al guardar prueba A/B: {str(e)}")
    
    def start(self) -> Dict[str, Any]:
        """
        Inicia la prueba A/B
        
        Returns:
            Información sobre la prueba iniciada
        """
        try:
            if self.status != "created":
                return {"status": "error", "message": f"La prueba ya está en estado {self.status}"}
            
            self.status = "running"
            self._save_test()
            
            logger.info(f"Prueba A/B '{self.test_name}' iniciada")
            return {
                "status": "success",
                "message": f"Prueba A/B '{self.test_name}' iniciada",
                "test_id": self.test_id,
                "variants": self.variants,
                "sample_size": self.sample_size
            }
        except Exception as e:
            logger.error(f"Error al iniciar prueba A/B: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def stop(self) -> Dict[str, Any]:
        """
        Detiene la prueba A/B
        
        Returns:
            Información sobre la prueba detenida
        """
        try:
            if self.status != "running":
                return {"status": "error", "message": f"La prueba no está en ejecución (estado actual: {self.status})"}
            
            self.status = "stopped"
            self.end_date = datetime.datetime.now()
            self._save_test()
            
            logger.info(f"Prueba A/B '{self.test_name}' detenida")
            return {
                "status": "success",
                "message": f"Prueba A/B '{self.test_name}' detenida",
                "test_id": self.test_id,
                "duration_days": (self.end_date - self.start_date).days
            }
        except Exception as e:
            logger.error(f"Error al detener prueba A/B: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def assign_variant(self, user_id: str) -> Dict[str, Any]:
        """
        Asigna una variante a un usuario
        
        Args:
            user_id: Identificador único del usuario
            
        Returns:
            Variante asignada
        """
        try:
            if self.status != "running":
                return {"status": "error", "message": f"La prueba no está en ejecución (estado actual: {self.status})"}
            
            # Usar hash del user_id para asignación determinística
            # Esto asegura que un usuario siempre vea la misma variante
            hash_value = hash(f"{user_id}_{self.test_id}")
            variant_index = hash_value % len(self.variants)
            
            # Obtener la variante
            variant = self.variants[variant_index]
            
            # Registrar asignación
            self._record_assignment(user_id, variant)
            
            return {
                "status": "success",
                "variant": variant,
                "variant_id": variant.get("id", str(variant_index)),
                "test_id": self.test_id
            }
        except Exception as e:
            logger.error(f"Error al asignar variante: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _record_assignment(self, user_id: str, variant: Dict[str, Any]) -> None:
        """
        Registra la asignación de una variante a un usuario
        
        Args:
            user_id: Identificador único del usuario
            variant: Variante asignada
        """
        try:
            # Crear directorio si no existe
            os.makedirs("data/ab_tests/assignments", exist_ok=True)
            
            # Preparar datos para guardar
            assignment_data = {
                "test_id": self.test_id,
                "user_id": user_id,
                "variant_id": variant.get("id", str(self.variants.index(variant))),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Guardar en archivo
            file_path = f"data/ab_tests/assignments/{self.test_id}.jsonl"
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(assignment_data) + "\n")
        except Exception as e:
            logger.error(f"Error al registrar asignación: {str(e)}")
    
    def record_conversion(self, user_id: str, metric: str, value: Any) -> Dict[str, Any]:
        """
        Registra una conversión para una métrica
        
        Args:
            user_id: Identificador único del usuario
            metric: Nombre de la métrica
            value: Valor de la conversión
            
        Returns:
            Información sobre la conversión registrada
        """
        try:
            if self.status != "running":
                return {"status": "error", "message": f"La prueba no está en ejecución (estado actual: {self.status})"}
            
            if metric not in self.metrics:
                return {"status": "error", "message": f"Métrica '{metric}' no definida para esta prueba"}
            
            # Crear directorio si no existe
            os.makedirs("data/ab_tests/conversions", exist_ok=True)
            
            # Preparar datos para guardar
            conversion_data = {
                "test_id": self.test_id,
                "user_id": user_id,
                "metric": metric,
                "value": value,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Guardar en archivo
            file_path = f"data/ab_tests/conversions/{self.test_id}.jsonl"
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(conversion_data) + "\n")
            
            # Verificar si debemos analizar resultados
            self._check_for_analysis()
            
            return {
                "status": "success",
                "message": f"Conversión registrada para métrica '{metric}'",
                "test_id": self.test_id
            }
        except Exception as e:
            logger.error(f"Error al registrar conversión: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _check_for_analysis(self) -> None:
        """
        Verifica si es momento de analizar los resultados
        """
        try:
            # Obtener número de conversiones
            file_path = f"data/ab_tests/conversions/{self.test_id}.jsonl"
            if not os.path.exists(file_path):
                return
            
            # Contar líneas en el archivo
            with open(file_path, 'r', encoding='utf-8') as f:
                num_conversions = sum(1 for _ in f)
            
            # Verificar si hemos alcanzado el tamaño de muestra
            if num_conversions >= self.sample_size:
                logger.info(f"Alcanzado tamaño de muestra para prueba '{self.test_name}'. Analizando resultados...")
                self.analyze_results()
                
                # Detener automáticamente si está configurado
                if self.config.get("auto_stop_on_significance", True) and self.results.get("has_significant_results", False):
                    logger.info(f"Deteniendo prueba '{self.test_name}' automáticamente debido a resultados significativos")
                    self.stop()
            
            # Verificar si hemos excedido la duración máxima
            max_duration = self.config.get("max_test_duration_days", 14)
            current_duration = (datetime.datetime.now() - self.start_date).days
            
            if current_duration >= max_duration:
                logger.info(f"Alcanzada duración máxima para prueba '{self.test_name}'. Analizando resultados...")
                self.analyze_results()
                self.stop()
        except Exception as e:
            logger.error(f"Error al verificar análisis: {str(e)}")
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analiza los resultados de la prueba
        
        Returns:
            Resultados del análisis
        """
        try:
            # Cargar asignaciones
            assignments_path = f"data/ab_tests/assignments/{self.test_id}.jsonl"
            if not os.path.exists(assignments_path):
                return {"status": "error", "message": "No hay asignaciones registradas"}
            
            assignments_df = pd.read_json(assignments_path, lines=True)
            
            # Cargar conversiones
            conversions_path = f"data/ab_tests/conversions/{self.test_id}.jsonl"
            if not os.path.exists(conversions_path):
                return {"status": "error", "message": "No hay conversiones registradas"}
            
            conversions_df = pd.read_json(conversions_path, lines=True)
            
            # Unir datos
            data = pd.merge(assignments_df, conversions_df, on=["test_id", "user_id"], how="left")
            
            # Inicializar resultados
            results = {
                "metrics": {},
                "has_significant_results": False,
                "winning_variant": None,
                "sample_size": len(assignments_df),
                "conversions": len(conversions_df)
            }
            
            # Analizar cada métrica
            for metric in self.metrics:
                metric_data = data[data["metric"] == metric]
                
                if len(metric_data) == 0:
                    results["metrics"][metric] = {
                        "status": "no_data",
                        "message": f"No hay datos para la métrica '{metric}'"
                    }
                    continue
                
                # Calcular estadísticas por variante
                variant_stats = {}
                
                for variant in self.variants:
                    variant_id = variant.get("id", str(self.variants.index(variant)))
                    variant_data = metric_data[metric_data["variant_id"] == variant_id]
                    
                    # Calcular conversiones
                    total_users = len(assignments_df[assignments_df["variant_id"] == variant_id])
                    converted_users = len(variant_data)
                    
                    if total_users == 0:
                        conversion_rate = 0
                    else:
                        conversion_rate = converted_users / total_users
                    
                    # Guardar estadísticas
                    variant_stats[variant_id] = {
                        "total_users": total_users,
                        "converted_users": converted_users,
                        "conversion_rate": conversion_rate,
                        "average_value": variant_data["value"].mean() if "value" in variant_data else None
                    }
                
                # Realizar prueba estadística
                if len(variant_stats) >= 2:
                    # Obtener las dos primeras variantes para comparación
                    variant_ids = list(variant_stats.keys())
                    variant_a = variant_stats[variant_ids[0]]
                    variant_b = variant_stats[variant_ids[1]]
                    
                    # Prueba de proporciones para tasas de conversión
                    success_a = variant_a["converted_users"]
                    total_a = variant_a["total_users"]
                    success_b = variant_b["converted_users"]
                    total_b = variant_b["total_users"]
                    
                    # Evitar división por cero
                    if total_a == 0 or total_b == 0:
                        p_value = 1.0
                    else:
                        # Prueba de proporciones
                        stat, p_value = stats.proportions_ztest(
                            [success_a, success_b],
                            [total_a, total_b]
                        )
                    
                    # Determinar significancia
                    alpha = 1 - self.config.get("confidence_level", 0.95)
                    is_significant = p_value < alpha
                    
                    # Determinar ganador
                    if is_significant:
                        if variant_a["conversion_rate"] > variant_b["conversion_rate"]:
                            winner = variant_ids[0]
                            improvement = (variant_a["conversion_rate"] - variant_b["conversion_rate"]) / variant_b["conversion_rate"]
                        else:
                            winner = variant_ids[1]
                            improvement = (variant_b["conversion_rate"] - variant_a["conversion_rate"]) / variant_a["conversion_rate"]
                        
                        results["has_significant_results"] = True
                        results["winning_variant"] = winner
                    else:
                        winner = None
                        improvement = 0
                    
                    # Guardar resultados de la prueba
                    results["metrics"][metric] = {
                        "variant_stats": variant_stats,
                        "p_value": p_value,
                        "is_significant": is_significant,
                        "winner": winner,
                        "improvement": improvement
                    }
                else:
                    results["metrics"][metric] = {
                        "status": "insufficient_variants",
                        "message": "Se necesitan al menos dos variantes para comparar",
                        "variant_stats": variant_stats
                    }
            
            # Guardar resultados
            self.results = results
            self._save_test()
            
            logger.info(f"Análisis completado para prueba '{self.test_name}'")
            return {
                "status": "success",
                "results": results
            }
        except Exception as e:
            logger.error(f"Error al analizar resultados: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    def load_test(test_id: str) -> 'ABTest':
        """
        Carga una prueba A/B desde un archivo
        
        Args:
            test_id: ID de la prueba
            
        Returns:
            Instancia de ABTest
        """
        try:
            file_path = f"data/ab_tests/{test_id}.json"
            
            if not os.path.exists(file_path):
                logger.error(f"Prueba A/B con ID {test_id} no encontrada")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # Crear instancia
            test = ABTest(
                test_name=test_data["test_name"],
                variants=test_data["variants"],
                metrics=test_data["metrics"]
            )
            
            # Restaurar estado
            test.test_id = test_data["test_id"]
            test.start_date = datetime.datetime.fromisoformat(test_data["start_date"])
            test.end_date = datetime.datetime.fromisoformat(test_data["end_date"]) if test_data["end_date"] else None
            test.status = test_data["status"]
            test.results = test_data["results"]
            test.sample_size = test_data["sample_size"]
            
            logger.info(f"Prueba A/B '{test.test_name}' cargada desde {file_path}")
            return test
        except Exception as e:
            logger.error(f"Error al cargar prueba A/B: {str(e)}")
            return None
    
    @staticmethod
    def list_tests() -> List[Dict[str, Any]]:
        """
        Lista todas las pruebas A/B
        
        Returns:
            Lista de pruebas
        """
        try:
            tests = []
            
            # Crear directorio si no existe
            os.makedirs("data/ab_tests", exist_ok=True)
            
            # Listar archivos
            for file_name in os.listdir("data/ab_tests"):
                if file_name.endswith(".json"):
                    file_path = os.path.join("data/ab_tests", file_name)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        test_data = json.load(f)
                    
                    # Extraer información básica
                    tests.append({
                        "test_id": test_data["test_id"],
                        "test_name": test_data["test_name"],
                        "status": test_data["status"],
                        "start_date": test_data["start_date"],
                        "end_date": test_data["end_date"],
                        "metrics": test_data["metrics"],
                        "has_significant_results": test_data["results"].get("has_significant_results", False) if "results" in test_data else False,
                        "winning_variant": test_data["results"].get("winning_variant") if "results" in test_data else None
                    })
            
            return tests
        except Exception as e:
            logger.error(f"Error al listar pruebas A/B: {str(e)}")
            return []


class ABTestManager:
    """
    Gestor de pruebas A/B
    """
    
    def __init__(self, config_path: str = "config/ab_testing.json"):
        """
        Inicializa el gestor de pruebas A/B
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path
        self.active_tests = {}
        self._load_active_tests()
    
    def _load_active_tests(self) -> None:
        """
        Carga las pruebas activas
        """
        try:
            tests = ABTest.list_tests()
            
            for test_info in tests:
                if test_info["status"] == "running":
                    test_id = test_info["test_id"]
                    test = ABTest.load_test(test_id)
                    
                    if test:
                        self.active_tests[test_id] = test
            
            logger.info(f"Cargadas {len(self.active_tests)} pruebas activas")
        except Exception as e:
            logger.error(f"Error al cargar pruebas activas: {str(e)}")
    
    def create_test(self, test_name: str, variants: List[Dict[str, Any]], 
                   metrics: List[str]) -> Dict[str, Any]:
        """
        Crea una nueva prueba A/B
        
        Args:
            test_name: Nombre único de la prueba
            variants: Lista de variantes a probar
            metrics: Lista de métricas a medir
            
        Returns:
            Información sobre la prueba creada
        """
        try:
            # Validar parámetros
            if not test_name:
                return {"status": "error", "message": "El nombre de la prueba es obligatorio"}
            
            if len(variants) < 2:
                return {"status": "error", "message": "Se necesitan al menos dos variantes"}
            
            if not metrics:
                return {"status": "error", "message": "Se necesita al menos una métrica"}
            
            # Crear prueba
            test = ABTest(test_name, variants, metrics, self.config_path)
            
            return {
                "status": "success",
                "message": f"Prueba A/B '{test_name}' creada correctamente",
                "test_id": test.test_id
            }
        except Exception as e:
            logger.error(f"Error al crear prueba A/B: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def start_test(self, test_id: str) -> Dict[str, Any]:
        """
        Inicia una prueba A/B
        
        Args:
            test_id: ID de la prueba
            
        Returns:
            Información sobre la prueba iniciada
        """
        try:
            # Cargar prueba
            test = ABTest.load_test(test_id)
            
            if not test:
                return {"status": "error", "message": f"Prueba A/B con ID {test_id} no encontrada"}
            
            # Iniciar prueba
            result = test.start()
            
            if result["status"] == "success":
                # Agregar a pruebas activas
                self.active_tests[test_id] = test
            
            return result
        except Exception as e:
            logger.error(f"Error al iniciar prueba A/B: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def stop_test(self, test_id: str) -> Dict[str, Any]:
        """
        Detiene una prueba A/B
        
        Args:
            test_id: ID de la prueba
            
        Returns:
            Información sobre la prueba detenida
        """
        try:
            # Verificar si está en pruebas activas
            if test_id in self.active_tests:
                test = self.active_tests[test_id]
            else:
                # Cargar prueba
                test = ABTest.load_test(test_id)
            
            if not test:
                return {"status": "error", "message": f"Prueba A/B con ID {test_id} no encontrada"}
            
            # Detener prueba
            result = test.stop()
            
            if result["status"] == "success":
                # Eliminar de pruebas activas
                if test_id in self.active_tests:
                    del self.active_tests[test_id]
            
            return result
        except Exception as e:
            logger.error(f"Error al detener prueba A/B: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_variant(self, test_id: str, user_id: str) -> Dict[str, Any]:
        """
        Obtiene la variante para un usuario
        
        Args:
            test_id: ID de la prueba
            user_id: ID del usuario
            
        Returns:
            Variante asignada
        """
        try:
            # Verificar si está en pruebas activas
            if test_id in self.active_tests:
                test = self.active_tests[test_id]
            else:
                # Cargar prueba
                test = ABTest.load_test(test_id)
            
            if not test:
                return {"status": "error", "message": f"Prueba A/B con ID {test_id} no encontrada"}
            
            # Obtener variante
            return test.assign_variant(user_id)
        except Exception as e:
            logger.error(f"Error al obtener variante: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def record_conversion(self, test_id: str, user_id: str, metric: str, value: Any) -> Dict[str, Any]:
        """
        Registra una conversión
        
        Args:
            test_id: ID de la prueba
            user_id: ID del usuario
            metric: Nombre de la métrica
            value: Valor de la conversión
            
        Returns:
            Información sobre la conversión registrada
        """
        try:
            # Verificar si está en pruebas activas
            if test_id in self.active_tests:
                test = self.active_tests[test_id]
            else:
                # Cargar prueba
                test = ABTest.load_test(test_id)
            
            if not test:
                return {"status": "error", "message": f"Prueba A/B con ID {test_id} no encontrada"}
            
            # Registrar conversión
            return test.record_conversion(user_id, metric, value)
        except Exception as e:
            logger.error(f"Error al registrar conversión: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_results(self, test_id: str) -> Dict[str, Any]:
        """
        Obtiene los resultados de una prueba
        
        Args:
            test_id: ID de la prueba
            
        Returns:
            Resultados de la prueba
        """
        try:
            # Cargar prueba
            test = ABTest.load_test(test_id)
            
            if not test:
                return {"status": "error", "message": f"Prueba A/B con ID {test_id} no encontrada"}
            
            # Analizar resultados
            return test.analyze_results()
        except Exception as e:
            logger.error(f"Error al obtener resultados: {str(e)}")
                        return {"status": "error", "message": str(e)}
    
    def list_tests(self) -> Dict[str, Any]:
        """
        Lista todas las pruebas A/B
        
        Returns:
            Lista de pruebas
        """
        try:
            tests = ABTest.list_tests()
            
            return {
                "status": "success",
                "tests": tests
            }
        except Exception as e:
            logger.error(f"Error al listar pruebas A/B: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_test_info(self, test_id: str) -> Dict[str, Any]:
        """
        Obtiene información detallada de una prueba
        
        Args:
            test_id: ID de la prueba
            
        Returns:
            Información de la prueba
        """
        try:
            # Cargar prueba
            test = ABTest.load_test(test_id)
            
            if not test:
                return {"status": "error", "message": f"Prueba A/B con ID {test_id} no encontrada"}
            
            # Preparar información
            test_info = {
                "test_id": test.test_id,
                "test_name": test.test_name,
                "variants": test.variants,
                "metrics": test.metrics,
                "start_date": test.start_date.isoformat(),
                "end_date": test.end_date.isoformat() if test.end_date else None,
                "status": test.status,
                "sample_size": test.sample_size,
                "results": test.results
            }
            
            return {
                "status": "success",
                "test": test_info
            }
        except Exception as e:
            logger.error(f"Error al obtener información de prueba: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def delete_test(self, test_id: str) -> Dict[str, Any]:
        """
        Elimina una prueba A/B
        
        Args:
            test_id: ID de la prueba
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar si la prueba existe
            file_path = f"data/ab_tests/{test_id}.json"
            
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"Prueba A/B con ID {test_id} no encontrada"}
            
            # Verificar si está activa
            if test_id in self.active_tests:
                return {"status": "error", "message": f"No se puede eliminar una prueba en ejecución. Detenga la prueba primero."}
            
            # Eliminar archivos
            os.remove(file_path)
            
            # Eliminar asignaciones
            assignments_path = f"data/ab_tests/assignments/{test_id}.jsonl"
            if os.path.exists(assignments_path):
                os.remove(assignments_path)
            
            # Eliminar conversiones
            conversions_path = f"data/ab_tests/conversions/{test_id}.jsonl"
            if os.path.exists(conversions_path):
                os.remove(conversions_path)
            
            logger.info(f"Prueba A/B con ID {test_id} eliminada")
            return {
                "status": "success",
                "message": f"Prueba A/B eliminada correctamente"
            }
        except Exception as e:
            logger.error(f"Error al eliminar prueba A/B: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_recommendations(self, test_id: str) -> Dict[str, Any]:
        """
        Obtiene recomendaciones basadas en los resultados de una prueba
        
        Args:
            test_id: ID de la prueba
            
        Returns:
            Recomendaciones
        """
        try:
            # Cargar prueba
            test = ABTest.load_test(test_id)
            
            if not test:
                return {"status": "error", "message": f"Prueba A/B con ID {test_id} no encontrada"}
            
            # Verificar si hay resultados
            if not test.results or "metrics" not in test.results:
                return {"status": "error", "message": "La prueba no tiene resultados analizados"}
            
            # Preparar recomendaciones
            recommendations = []
            
            # Verificar si hay un ganador claro
            if test.results.get("has_significant_results", False):
                winning_variant_id = test.results.get("winning_variant")
                
                # Buscar la variante ganadora
                winning_variant = None
                for variant in test.variants:
                    variant_id = variant.get("id", str(test.variants.index(variant)))
                    if variant_id == winning_variant_id:
                        winning_variant = variant
                        break
                
                if winning_variant:
                    # Generar recomendación principal
                    recommendations.append({
                        "type": "primary",
                        "message": f"Implementar la variante ganadora '{winning_variant.get('name', winning_variant_id)}'",
                        "confidence": "alta",
                        "variant": winning_variant
                    })
                    
                    # Analizar métricas específicas
                    for metric, metric_results in test.results["metrics"].items():
                        if isinstance(metric_results, dict) and metric_results.get("is_significant", False):
                            improvement = metric_results.get("improvement", 0)
                            
                            if improvement > 0:
                                recommendations.append({
                                    "type": "metric",
                                    "metric": metric,
                                    "message": f"La variante ganadora mejoró la métrica '{metric}' en un {improvement:.2%}",
                                    "improvement": improvement
                                })
            else:
                # No hay resultados significativos
                recommendations.append({
                    "type": "primary",
                    "message": "No hay diferencias significativas entre las variantes",
                    "confidence": "baja",
                    "suggestion": "Considere ejecutar la prueba por más tiempo o con un tamaño de muestra mayor"
                })
                
                # Verificar si hay tendencias
                for metric, metric_results in test.results["metrics"].items():
                    if isinstance(metric_results, dict) and "variant_stats" in metric_results:
                        variant_stats = metric_results["variant_stats"]
                        
                        # Encontrar la variante con mejor tasa de conversión
                        best_variant_id = None
                        best_rate = -1
                        
                        for variant_id, stats in variant_stats.items():
                            if stats.get("conversion_rate", 0) > best_rate:
                                best_rate = stats.get("conversion_rate", 0)
                                best_variant_id = variant_id
                        
                        if best_variant_id:
                            recommendations.append({
                                "type": "trend",
                                "metric": metric,
                                "message": f"La variante '{best_variant_id}' muestra una tendencia positiva para la métrica '{metric}'",
                                "confidence": "media"
                            })
            
            return {
                "status": "success",
                "recommendations": recommendations
            }
        except Exception as e:
            logger.error(f"Error al generar recomendaciones: {str(e)}")
            return {"status": "error", "message": str(e)}


# Ejemplo de uso
if __name__ == "__main__":
    # Crear gestor de pruebas
    manager = ABTestManager()
    
    # Crear una prueba
    variants = [
        {"id": "A", "name": "Control", "cta_text": "Comprar ahora", "cta_color": "#3498db"},
        {"id": "B", "name": "Variante B", "cta_text": "Añadir al carrito", "cta_color": "#e74c3c"}
    ]
    
    metrics = ["click_rate", "conversion_rate", "revenue"]
    
    result = manager.create_test("Prueba CTA Botón", variants, metrics)
    print(result)
    
    if result["status"] == "success":
        test_id = result["test_id"]
        
        # Iniciar prueba
        manager.start_test(test_id)
        
        # Simular asignaciones y conversiones
        for i in range(200):
            user_id = f"user_{i}"
            
            # Asignar variante
            variant_result = manager.get_variant(test_id, user_id)
            
            if variant_result["status"] == "success":
                variant_id = variant_result["variant_id"]
                
                # Simular conversión con probabilidad diferente según variante
                if variant_id == "A":
                    if random.random() < 0.1:  # 10% de conversión
                        manager.record_conversion(test_id, user_id, "click_rate", 1)
                    if random.random() < 0.05:  # 5% de conversión
                        manager.record_conversion(test_id, user_id, "conversion_rate", 1)
                        manager.record_conversion(test_id, user_id, "revenue", random.randint(10, 50))
                else:  # Variante B
                    if random.random() < 0.15:  # 15% de conversión
                        manager.record_conversion(test_id, user_id, "click_rate", 1)
                    if random.random() < 0.08:  # 8% de conversión
                        manager.record_conversion(test_id, user_id, "conversion_rate", 1)
                        manager.record_conversion(test_id, user_id, "revenue", random.randint(10, 60))
        
        # Obtener resultados
        results = manager.get_results(test_id)
        print(json.dumps(results, indent=2))
        
        # Obtener recomendaciones
        recommendations = manager.get_recommendations(test_id)
        print(json.dumps(recommendations, indent=2))
        
        # Detener prueba
        manager.stop_test(test_id)