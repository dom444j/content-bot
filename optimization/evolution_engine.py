import logging
import random
import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime, timedelta
import copy
from collections import defaultdict

class EvolutionEngine:
    """
    Motor de evolución genética para optimizar CTAs, voces, visuales y otros elementos
    
    Este módulo implementa algoritmos genéticos para evolucionar y mejorar
    automáticamente diferentes aspectos del contenido, generando variantes
    optimizadas basadas en el rendimiento histórico.
    """
    
    def __init__(self, config_path: str = "../config/evolution_config.json",
                 data_path: str = "../data/evolution_data.json",
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 elitism_count: int = 2):
        """
        Inicializa el motor de evolución
        
        Args:
            config_path: Ruta al archivo de configuración
            data_path: Ruta al archivo de datos de evolución
            population_size: Tamaño de la población para algoritmos genéticos
            generations: Número de generaciones a evolucionar
            mutation_rate: Tasa de mutación (0-1)
            crossover_rate: Tasa de cruce (0-1)
            elitism_count: Número de individuos élite a preservar
        """
        self.logger = logging.getLogger("EvolutionEngine")
        self.config_path = config_path
        self.data_path = data_path
        
        # Parámetros de evolución
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        
        # Cargar configuración y datos
        self._load_config()
        self._load_data()
        
        # Historial de evolución
        self.evolution_history = defaultdict(list)
        
        # Registro de experimentos
        self.experiments = {}
        
        # Funciones de fitness por tipo de contenido
        self.fitness_functions = {
            "cta": self._calculate_cta_fitness,
            "voice": self._calculate_voice_fitness,
            "visual": self._calculate_visual_fitness,
            "thumbnail": self._calculate_thumbnail_fitness,
            "script": self._calculate_script_fitness,
            "title": self._calculate_title_fitness
        }
        
        # Funciones de mutación por tipo de contenido
        self.mutation_functions = {
            "cta": self._mutate_cta,
            "voice": self._mutate_voice,
            "visual": self._mutate_visual,
            "thumbnail": self._mutate_thumbnail,
            "script": self._mutate_script,
            "title": self._mutate_title
        }
        
        # Funciones de cruce por tipo de contenido
        self.crossover_functions = {
            "cta": self._crossover_cta,
            "voice": self._crossover_voice,
            "visual": self._crossover_visual,
            "thumbnail": self._crossover_thumbnail,
            "script": self._crossover_script,
            "title": self._crossover_title
        }
    
    def _load_config(self) -> None:
        """Carga la configuración de evolución"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"Configuración cargada desde {self.config_path}")
            else:
                self.logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                self.config = self._create_default_config()
                self._save_config()
        except Exception as e:
            self.logger.error(f"Error al cargar configuración: {str(e)}")
            self.config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Crea una configuración predeterminada"""
        return {
            "evolution_settings": {
                "population_size": self.population_size,
                "generations": self.generations,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "elitism_count": self.elitism_count
            },
            "content_types": {
                "cta": {
                    "enabled": True,
                    "weight": 1.0,
                    "metrics": ["click_rate", "retention_rate", "conversion_rate"]
                },
                "voice": {
                    "enabled": True,
                    "weight": 0.8,
                    "metrics": ["engagement_rate", "retention_rate", "sentiment_score"]
                },
                "visual": {
                    "enabled": True,
                    "weight": 0.9,
                    "metrics": ["engagement_rate", "click_rate", "watch_time"]
                },
                "thumbnail": {
                    "enabled": True,
                    "weight": 1.0,
                    "metrics": ["click_rate", "impression_ctr"]
                },
                "script": {
                    "enabled": True,
                    "weight": 0.7,
                    "metrics": ["retention_rate", "engagement_rate", "sentiment_score"]
                },
                "title": {
                    "enabled": True,
                    "weight": 1.0,
                    "metrics": ["click_rate", "impression_ctr", "search_rank"]
                }
            },
            "fitness_weights": {
                "click_rate": 1.0,
                "retention_rate": 0.8,
                "conversion_rate": 1.2,
                "engagement_rate": 0.9,
                "sentiment_score": 0.7,
                "watch_time": 0.8,
                "impression_ctr": 1.0,
                "search_rank": 0.6
            },
            "evolution_schedule": {
                "cta": "daily",
                "voice": "weekly",
                "visual": "weekly",
                "thumbnail": "daily",
                "script": "weekly",
                "title": "daily"
            }
        }
    
    def _save_config(self) -> None:
        """Guarda la configuración actual"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            self.logger.info(f"Configuración guardada en {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error al guardar configuración: {str(e)}")
    
    def _load_data(self) -> None:
        """Carga los datos de evolución"""
        try:
            if os.path.exists(self.data_path):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                self.logger.info(f"Datos cargados desde {self.data_path}")
            else:
                self.logger.warning(f"Archivo de datos no encontrado: {self.data_path}")
                self.data = {
                    "populations": {},
                    "history": {},
                    "experiments": {},
                    "performance": {}
                }
        except Exception as e:
            self.logger.error(f"Error al cargar datos: {str(e)}")
            self.data = {
                "populations": {},
                "history": {},
                "experiments": {},
                "performance": {}
            }
    
    def _save_data(self) -> None:
        """Guarda los datos de evolución"""
        try:
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4)
            self.logger.info(f"Datos guardados en {self.data_path}")
        except Exception as e:
            self.logger.error(f"Error al guardar datos: {str(e)}")
    
    def evolve_content(self, content_type: str, initial_population: Optional[List[Dict[str, Any]]] = None,
                      fitness_data: Optional[Dict[str, Dict[str, float]]] = None,
                      custom_fitness_function: Optional[Callable] = None,
                      experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Evoluciona una población de contenido utilizando algoritmos genéticos
        
        Args:
            content_type: Tipo de contenido a evolucionar (cta, voice, visual, etc.)
            initial_population: Población inicial (opcional)
            fitness_data: Datos de fitness para evaluación (opcional)
            custom_fitness_function: Función personalizada de fitness (opcional)
            experiment_id: ID de experimento para seguimiento (opcional)
            
        Returns:
            Resultado de la evolución con la mejor solución y estadísticas
        """
        if content_type not in self.fitness_functions:
            self.logger.error(f"Tipo de contenido no soportado: {content_type}")
            return {"success": False, "error": f"Tipo de contenido no soportado: {content_type}"}
        
        try:
            # Crear ID de experimento si no se proporciona
            if experiment_id is None:
                experiment_id = f"{content_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Inicializar población
            population = self._initialize_population(content_type, initial_population)
            
            # Seleccionar función de fitness
            fitness_function = custom_fitness_function if custom_fitness_function else self.fitness_functions[content_type]
            
            # Inicializar registro de experimento
            self.experiments[experiment_id] = {
                "content_type": content_type,
                "start_time": datetime.now().isoformat(),
                "generations": [],
                "best_fitness": 0,
                "best_individual": None,
                "settings": {
                    "population_size": len(population),
                    "generations": self.generations,
                    "mutation_rate": self.mutation_rate,
                    "crossover_rate": self.crossover_rate,
                    "elitism_count": self.elitism_count
                }
            }
            
            # Evolucionar por generaciones
            best_individual = None
            best_fitness = 0
            
            for generation in range(self.generations):
                self.logger.info(f"Evolucionando {content_type} - Generación {generation+1}/{self.generations}")
                
                # Evaluar fitness de la población
                fitness_scores = self._evaluate_population(population, fitness_function, fitness_data)
                
                # Encontrar el mejor individuo
                generation_best_idx = np.argmax(fitness_scores)
                generation_best = population[generation_best_idx]
                generation_best_fitness = fitness_scores[generation_best_idx]
                
                # Actualizar mejor global si es necesario
                if generation_best_fitness > best_fitness:
                    best_fitness = generation_best_fitness
                    best_individual = copy.deepcopy(generation_best)
                
                # Registrar estadísticas de generación
                generation_stats = {
                    "generation": generation + 1,
                    "best_fitness": generation_best_fitness,
                    "avg_fitness": np.mean(fitness_scores),
                    "best_individual": generation_best,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.experiments[experiment_id]["generations"].append(generation_stats)
                
                # Verificar si es la última generación
                if generation == self.generations - 1:
                    break
                
                # Crear nueva población
                population = self._create_new_population(population, fitness_scores, content_type)
            
            # Actualizar mejor resultado en el experimento
            self.experiments[experiment_id]["best_fitness"] = best_fitness
            self.experiments[experiment_id]["best_individual"] = best_individual
            self.experiments[experiment_id]["end_time"] = datetime.now().isoformat()
            
            # Guardar en historial
            self.evolution_history[content_type].append({
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "best_fitness": best_fitness,
                "best_individual": best_individual
            })
            
            # Actualizar datos
            self.data["experiments"][experiment_id] = self.experiments[experiment_id]
            self.data["populations"][content_type] = population
            self.data["history"][content_type] = self.evolution_history[content_type]
            self._save_data()
            
            # Preparar resultado
            result = {
                "success": True,
                "experiment_id": experiment_id,
                "content_type": content_type,
                "best_individual": best_individual,
                "best_fitness": best_fitness,
                "generations": len(self.experiments[experiment_id]["generations"]),
                "population_size": len(population),
                "evolution_time": (
                    datetime.fromisoformat(self.experiments[experiment_id]["end_time"]) - 
                    datetime.fromisoformat(self.experiments[experiment_id]["start_time"])
                ).total_seconds()
            }
            
            self.logger.info(f"Evolución completada para {content_type}. Mejor fitness: {best_fitness:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error en evolución de {content_type}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _initialize_population(self, content_type: str, initial_population: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Inicializa una población para evolución
        
        Args:
            content_type: Tipo de contenido
            initial_population: Población inicial opcional
            
        Returns:
            Población inicializada
        """
        # Si se proporciona población inicial, usarla
        if initial_population and len(initial_population) > 0:
            if len(initial_population) < self.population_size:
                # Completar hasta el tamaño deseado
                additional_needed = self.population_size - len(initial_population)
                for _ in range(additional_needed):
                    # Clonar y mutar individuos existentes
                    template = random.choice(initial_population)
                    new_individual = copy.deepcopy(template)
                    new_individual = self.mutation_functions[content_type](new_individual, rate=0.5)
                    initial_population.append(new_individual)
            return initial_population[:self.population_size]
        
        # Usar población existente si está disponible
        if content_type in self.data["populations"] and len(self.data["populations"][content_type]) > 0:
            existing_population = self.data["populations"][content_type]
            if len(existing_population) < self.population_size:
                # Completar hasta el tamaño deseado
                additional_needed = self.population_size - len(existing_population)
                for _ in range(additional_needed):
                    # Clonar y mutar individuos existentes
                    template = random.choice(existing_population)
                    new_individual = copy.deepcopy(template)
                    new_individual = self.mutation_functions[content_type](new_individual, rate=0.5)
                    existing_population.append(new_individual)
            return existing_population[:self.population_size]
        
        # Crear población desde cero
        self.logger.info(f"Creando nueva población para {content_type}")
        population = []
        
        # Generar individuos según el tipo de contenido
        for _ in range(self.population_size):
            if content_type == "cta":
                individual = self._generate_random_cta()
            elif content_type == "voice":
                individual = self._generate_random_voice()
            elif content_type == "visual":
                individual = self._generate_random_visual()
            elif content_type == "thumbnail":
                individual = self._generate_random_thumbnail()
            elif content_type == "script":
                individual = self._generate_random_script()
            elif content_type == "title":
                individual = self._generate_random_title()
            else:
                individual = {"type": content_type, "data": {}}
            
            population.append(individual)
        
        return population
    
    def _evaluate_population(self, population: List[Dict[str, Any]], 
                            fitness_function: Callable, 
                            fitness_data: Optional[Dict[str, Dict[str, float]]] = None) -> List[float]:
        """
        Evalúa el fitness de una población
        
        Args:
            population: Lista de individuos
            fitness_function: Función para evaluar fitness
            fitness_data: Datos de rendimiento para evaluación
            
        Returns:
            Lista de puntuaciones de fitness
        """
        fitness_scores = []
        
        for individual in population:
            # Si hay datos de fitness, usarlos
            if fitness_data and individual.get("id") in fitness_data:
                fitness = fitness_function(individual, fitness_data[individual["id"]])
            else:
                # Usar función de fitness sin datos específicos
                fitness = fitness_function(individual)
            
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def _create_new_population(self, population: List[Dict[str, Any]], 
                              fitness_scores: List[float],
                              content_type: str) -> List[Dict[str, Any]]:
        """
        Crea una nueva población mediante selección, cruce y mutación
        
        Args:
            population: Población actual
            fitness_scores: Puntuaciones de fitness
            content_type: Tipo de contenido
            
        Returns:
            Nueva población
        """
        # Normalizar fitness para selección
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            # Si todos tienen fitness 0, usar probabilidad uniforme
            selection_probs = [1.0 / len(population) for _ in population]
        else:
            selection_probs = [f / total_fitness for f in fitness_scores]
        
        # Preservar élites
        elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
        new_population = [copy.deepcopy(population[i]) for i in elite_indices]
        
        # Completar el resto de la población
        while len(new_population) < self.population_size:
            # Selección
            if random.random() < self.crossover_rate:
                # Cruce
                parent1_idx = np.random.choice(len(population), p=selection_probs)
                parent2_idx = np.random.choice(len(population), p=selection_probs)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Evitar cruce con el mismo individuo
                attempts = 0
                while parent1_idx == parent2_idx and attempts < 5:
                    parent2_idx = np.random.choice(len(population), p=selection_probs)
                    parent2 = population[parent2_idx]
                    attempts += 1
                
                # Realizar cruce
                child = self.crossover_functions[content_type](parent1, parent2)
            else:
                # Clonación
                parent_idx = np.random.choice(len(population), p=selection_probs)
                child = copy.deepcopy(population[parent_idx])
            
            # Mutación
            if random.random() < self.mutation_rate:
                child = self.mutation_functions[content_type](child)
            
            # Añadir a nueva población
            new_population.append(child)
        
        return new_population
    
    # Funciones de fitness
    def _calculate_cta_fitness(self, cta: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> float:
        """Calcula fitness para CTAs"""
        if metrics:
            # Usar métricas reales si están disponibles
            weights = self.config["fitness_weights"]
            score = 0.0
            
            for metric, value in metrics.items():
                if metric in weights:
                    score += value * weights[metric]
            
            return score
        else:
            # Estimación heurística si no hay métricas
            score = 0.0
            
            # Longitud óptima (entre 5 y 15 palabras)
            text = cta.get("text", "")
            words = text.split()
            word_count = len(words)
            
            if 5 <= word_count <= 15:
                score += 0.3
            elif 3 <= word_count < 5 or 15 < word_count <= 20:
                score += 0.2
            else:
                score += 0.1
            
            # Presencia de verbos de acción
            action_verbs = ["sigue", "comenta", "comparte", "dale", "descubre", "aprende", 
                           "únete", "descarga", "visita", "prueba", "obtén", "consigue"]
            
            if any(verb in text.lower() for verb in action_verbs):
                score += 0.2
            
            # Presencia de emojis (moderada)
            emoji_count = sum(1 for char in text if ord(char) > 127)
            if 1 <= emoji_count <= 3:
                score += 0.2
            elif emoji_count > 3:
                score += 0.1
            
            # Urgencia o exclusividad
            urgency_terms = ["ahora", "hoy", "ya", "inmediatamente", "limitado", 
                            "exclusivo", "único", "nunca", "última", "especial"]
            
            if any(term in text.lower() for term in urgency_terms):
                score += 0.2
            
            # Claridad y concisión
            avg_word_length = sum(len(word) for word in words) / max(1, word_count)
            if avg_word_length <= 6:
                score += 0.1
            
            return score
    
    def _calculate_voice_fitness(self, voice: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> float:
        """Calcula fitness para voces"""
        if metrics:
            # Usar métricas reales si están disponibles
            weights = self.config["fitness_weights"]
            score = 0.0
            
            for metric, value in metrics.items():
                if metric in weights:
                    score += value * weights[metric]
            
            return score
        else:
            # Estimación heurística si no hay métricas
            score = 0.0
            
            # Características de voz
            pitch = voice.get("pitch", 0.5)
            speed = voice.get("speed", 0.5)
            clarity = voice.get("clarity", 0.5)
            emotion = voice.get("emotion", 0.5)
            
            # Pitch moderado (ni muy agudo ni muy grave)
            if 0.3 <= pitch <= 0.7:
                score += 0.2
            else:
                score += 0.1
            
            # Velocidad adecuada (ni muy lenta ni muy rápida)
            if 0.4 <= speed <= 0.6:
                score += 0.2
            else:
                score += 0.1
            
            # Alta claridad
            score += clarity * 0.3
            
            # Emoción adecuada
            score += emotion * 0.3
            
            return score
    
    def _calculate_visual_fitness(self, visual: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> float:
        """Calcula fitness para visuales"""
        if metrics:
            # Usar métricas reales si están disponibles
            weights = self.config["fitness_weights"]
            score = 0.0
            
            for metric, value in metrics.items():
                if metric in weights:
                    score += value * weights[metric]
            
            return score
        else:
            # Estimación heurística si no hay métricas
            score = 0.0
            
            # Características visuales
            brightness = visual.get("brightness", 0.5)
            contrast = visual.get("contrast", 0.5)
            saturation = visual.get("saturation", 0.5)
            complexity = visual.get("complexity", 0.5)
            
            # Brillo adecuado (ni muy oscuro ni muy brillante)
            if 0.4 <= brightness <= 0.7:
                score += 0.2
            else:
                score += 0.1
            
            # Contraste alto
            score += contrast * 0.3
            
            # Saturación moderada
            if 0.4 <= saturation <= 0.7:
                score += 0.2
            else:
                score += 0.1
            
            # Complejidad óptima (ni muy simple ni muy compleja)
            if 0.3 <= complexity <= 0.7:
                score += 0.3
            else:
                score += 0.1
            
            return score
    
    def _calculate_thumbnail_fitness(self, thumbnail: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> float:
        """Calcula fitness para miniaturas"""
        if metrics:
            # Usar métricas reales si están disponibles
            weights = self.config["fitness_weights"]
            score = 0.0
            
            for metric, value in metrics.items():
                if metric in weights:
                    score += value * weights[metric]
            
            return score
        else:
            # Estimación heurística si no hay métricas
            score = 0.0
            
            # Características de miniatura
            has_text = thumbnail.get("has_text", False)
            text_size = thumbnail.get("text_size", 0.5)
            face_count = thumbnail.get("face_count", 0)
            brightness = thumbnail.get("brightness", 0.5)
            contrast = thumbnail.get("contrast", 0.5)
            
            # Presencia de texto
            if has_text:
                score += 0.2
                
                # Tamaño de texto adecuado
                if text_size >= 0.6:
                    score += 0.2
                else:
                    score += 0.1
            
            # Presencia de rostros (1-2 es óptimo)
            if 1 <= face_count <= 2:
                score += 0.3
            elif face_count > 2:
                score += 0.1
            
            # Alto contraste
            score += contrast * 0.2
            
            # Brillo adecuado
            if 0.4 <= brightness <= 0.7:
                score += 0.1
            
            return score
    
    def _calculate_script_fitness(self, script: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> float:
        """Calcula fitness para guiones"""
        if metrics:
            # Usar métricas reales si están disponibles
            weights = self.config["fitness_weights"]
            score = 0.0
            
            for metric, value in metrics.items():
                if metric in weights:
                    score += value * weights[metric]
            
            return score
        else:
            # Estimación heurística si no hay métricas
            score = 0.0
            
            # Características del guión
            duration = script.get("duration", 60)  # en segundos
            hook_strength = script.get("hook_strength", 0.5)
            cta_count = script.get("cta_count", 0)
            question_count = script.get("question_count", 0)
            
            # Duración óptima (15-60 segundos para shorts/reels)
            if 15 <= duration <= 60:
                score += 0.3
            elif 60 < duration <= 180:
                score += 0.2
            else:
                score += 0.1
            
            # Gancho fuerte
            score += hook_strength * 0.3
            
            # CTAs (1-2 es óptimo)
            if 1 <= cta_count <= 2:
                score += 0.2
            elif cta_count > 2:
                score += 0.1
            
            # Preguntas para engagement (1-2 es óptimo)
            if 1 <= question_count <= 2:
                score += 0.2
            elif question_count > 2:
                score += 0.1
            
            return score
    
    def _calculate_title_fitness(self, title: Dict[str, Any], metrics: Optional[Dict[str, float]] = None) -> float:
        """Calcula fitness para títulos"""
        if metrics:
            # Usar métricas reales si están disponibles
            weights = self.config["fitness_weights"]
            score = 0.0
            
            for metric, value in metrics.items():
                if metric in weights:
                    score += value * weights[metric]
            
            return score
        else:
            # Estimación heurística si no hay métricas
            score = 0.0
            
            # Características del título
            text = title.get("text", "")
            words = text.split()
            word_count = len(words)
            has_number = any(char.isdigit() for char in text)
            has_question = "?" in text
            
            # Longitud óptima (5-10 palabras)
            if 5 <= word_count <= 10:
                score += 0.3
            elif 3 <= word_count < 5 or 10 < word_count <= 15:
                score += 0.2
            else:
                score += 0.1
            
            # Presencia de números
            if has_number:
                score += 0.2
            
            # Presencia de pregunta
            if has_question:
                score += 0.2
            
            # Palabras de alto impacto
            impact_words = ["increíble", "sorprendente", "impactante", "secreto", "revelado", 
                           "mejor", "peor", "nunca", "siempre", "fácil", "rápido", "gratis"]
            
            if any(word.lower() in impact_words for word in words):
                score += 0.2
            
            # Capitalización adecuada (no todo en mayúsculas)
            if not text.isupper() and text[0].isupper():
                score += 0.1
            
            return score
    
    # Funciones de generación aleatoria
        def _generate_random_cta(self) -> Dict[str, Any]:
        """Genera un CTA aleatorio"""
        cta_templates = [
            "¡Dale like y suscríbete para más contenido!",
            "Comenta abajo qué piensas sobre esto",
            "Comparte este video con alguien que necesite verlo",
            "¿Te gustó? ¡Dale like y comparte!",
            "Suscríbete para no perderte nuestros próximos videos",
            "Síguenos para más consejos como este",
            "¡No olvides activar las notificaciones!",
            "Descubre más en nuestro perfil",
            "¡Guarda este video para más tarde!",
            "¿Quieres más contenido? ¡Comenta abajo!"
        ]
        
        # Seleccionar texto base
        text = random.choice(cta_templates)
        
        # Posiblemente añadir emoji
        emojis = ["✅", "👍", "🔥", "⭐", "💯", "🚀", "💪", "👇", "❤️", "🎯"]
        if random.random() < 0.7:  # 70% de probabilidad de añadir emoji
            emoji_count = random.randint(1, 3)
            selected_emojis = random.sample(emojis, emoji_count)
            text = text + " " + "".join(selected_emojis)
        
        # Posiblemente añadir urgencia
        urgency_phrases = [
            " ¡Ahora!",
            " ¡Hoy mismo!",
            " ¡No esperes más!",
            " ¡Oferta limitada!",
            " ¡No te lo pierdas!"
        ]
        if random.random() < 0.3:  # 30% de probabilidad de añadir urgencia
            text = text + random.choice(urgency_phrases)
        
        return {
            "id": f"cta_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "cta",
            "text": text,
            "position": random.choice(["end", "middle", "beginning"]),
            "duration": random.randint(5, 15),
            "created_at": datetime.now().isoformat()
        }
    
    def _generate_random_voice(self) -> Dict[str, Any]:
        """Genera una configuración de voz aleatoria"""
        voice_types = ["natural", "energética", "calmada", "profesional", "amigable", "seria"]
        
        return {
            "id": f"voice_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "voice",
            "voice_type": random.choice(voice_types),
            "pitch": round(random.uniform(0.3, 0.7), 2),
            "speed": round(random.uniform(0.4, 0.6), 2),
            "clarity": round(random.uniform(0.5, 0.9), 2),
            "emotion": round(random.uniform(0.4, 0.8), 2),
            "created_at": datetime.now().isoformat()
        }
    
    def _generate_random_visual(self) -> Dict[str, Any]:
        """Genera una configuración visual aleatoria"""
        visual_styles = ["minimalista", "colorido", "oscuro", "brillante", "contrastante", "suave"]
        transitions = ["corte", "fade", "deslizamiento", "zoom", "ninguna"]
        
        return {
            "id": f"visual_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "visual",
            "style": random.choice(visual_styles),
            "brightness": round(random.uniform(0.4, 0.7), 2),
            "contrast": round(random.uniform(0.5, 0.8), 2),
            "saturation": round(random.uniform(0.4, 0.7), 2),
            "complexity": round(random.uniform(0.3, 0.7), 2),
            "transition": random.choice(transitions),
            "created_at": datetime.now().isoformat()
        }
    
    def _generate_random_thumbnail(self) -> Dict[str, Any]:
        """Genera una configuración de miniatura aleatoria"""
        text_options = ["pregunta", "número", "afirmación impactante", "ninguno"]
        
        has_text = random.choice(text_options) != "ninguno"
        text_type = random.choice(text_options) if has_text else "ninguno"
        
        return {
            "id": f"thumbnail_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "thumbnail",
            "has_text": has_text,
            "text_type": text_type,
            "text_size": round(random.uniform(0.5, 0.9), 2) if has_text else 0,
            "face_count": random.randint(0, 3),
            "brightness": round(random.uniform(0.4, 0.7), 2),
            "contrast": round(random.uniform(0.5, 0.8), 2),
            "border": random.choice([True, False]),
            "created_at": datetime.now().isoformat()
        }
    
    def _generate_random_script(self) -> Dict[str, Any]:
        """Genera una configuración de guión aleatoria"""
        hook_types = ["pregunta", "dato sorprendente", "problema común", "historia", "controversia"]
        structure_types = ["problema-solución", "lista", "tutorial", "narrativa", "comparación"]
        
        return {
            "id": f"script_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "script",
            "hook_type": random.choice(hook_types),
            "hook_strength": round(random.uniform(0.5, 0.9), 2),
            "structure": random.choice(structure_types),
            "duration": random.randint(15, 120),
            "cta_count": random.randint(1, 3),
            "question_count": random.randint(0, 3),
            "created_at": datetime.now().isoformat()
        }
    
    def _generate_random_title(self) -> Dict[str, Any]:
        """Genera un título aleatorio"""
        title_templates = [
            "Los {number} secretos para {topic} que nadie te contó",
            "{number} formas de {action} en solo {timeframe}",
            "Cómo {action} sin {negative_outcome}",
            "¿Por qué deberías {action} ahora mismo?",
            "La verdad sobre {topic} que te sorprenderá",
            "{topic}: lo que necesitas saber en {year}",
            "Descubre cómo {action} como un profesional",
            "{action} vs {alternative_action}: ¿Cuál es mejor?",
            "El método probado para {action} rápidamente",
            "Lo que nadie te dice sobre {topic}"
        ]
        
        topics = ["marketing digital", "redes sociales", "productividad", "finanzas personales", 
                 "bienestar", "tecnología", "emprendimiento", "desarrollo personal"]
        
        actions = ["aumentar seguidores", "ganar dinero", "mejorar tu contenido", "optimizar tu tiempo", 
                  "crecer en redes", "aprender", "invertir", "transformar tu vida"]
        
        timeframes = ["7 días", "24 horas", "un mes", "minutos", "una semana"]
        
        negative_outcomes = ["fracasar", "perder tiempo", "gastar dinero", "estresarte", "complicarte"]
        
        alternative_actions = ["contenido orgánico", "publicidad pagada", "outsourcing", "automatización"]
        
        # Seleccionar plantilla
        template = random.choice(title_templates)
        
        # Reemplazar variables
        title = template
        if "{number}" in title:
            title = title.replace("{number}", str(random.randint(3, 10)))
        if "{topic}" in title:
            title = title.replace("{topic}", random.choice(topics))
        if "{action}" in title:
            title = title.replace("{action}", random.choice(actions))
        if "{timeframe}" in title:
            title = title.replace("{timeframe}", random.choice(timeframes))
        if "{negative_outcome}" in title:
            title = title.replace("{negative_outcome}", random.choice(negative_outcomes))
        if "{alternative_action}" in title:
            title = title.replace("{alternative_action}", random.choice(alternative_actions))
        if "{year}" in title:
            title = title.replace("{year}", str(datetime.now().year))
        
        return {
            "id": f"title_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "title",
            "text": title,
            "has_number": any(char.isdigit() for char in title),
            "has_question": "?" in title,
            "length": len(title.split()),
            "created_at": datetime.now().isoformat()
        }
    
    # Funciones de mutación
    def _mutate_cta(self, cta: Dict[str, Any], rate: float = None) -> Dict[str, Any]:
        """Muta un CTA"""
        if rate is None:
            rate = self.mutation_rate
        
        result = copy.deepcopy(cta)
        
        # Preservar ID original si existe
        if "id" not in result:
            result["id"] = f"cta_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Actualizar timestamp
        result["created_at"] = datetime.now().isoformat()
        
        # Posibles mutaciones
        mutations = []
        
        # 1. Cambiar texto completamente
        if random.random() < rate * 0.3:
            new_cta = self._generate_random_cta()
            result["text"] = new_cta["text"]
            mutations.append("text_replaced")
        
        # 2. Añadir/quitar/cambiar emoji
        if random.random() < rate * 0.5:
            text = result["text"]
            emojis = ["✅", "👍", "🔥", "⭐", "💯", "🚀", "💪", "👇", "❤️", "🎯"]
            
            # Contar emojis actuales
            emoji_count = sum(1 for char in text if ord(char) > 127)
            
            if emoji_count == 0:
                # Añadir emoji
                emoji_count = random.randint(1, 2)
                selected_emojis = random.sample(emojis, emoji_count)
                result["text"] = text + " " + "".join(selected_emojis)
                mutations.append("emoji_added")
            elif emoji_count > 2 or random.random() < 0.5:
                # Quitar emojis
                result["text"] = ''.join(char for char in text if ord(char) <= 127).strip()
                mutations.append("emoji_removed")
            else:
                # Cambiar emojis
                text_without_emoji = ''.join(char for char in text if ord(char) <= 127).strip()
                emoji_count = random.randint(1, 2)
                selected_emojis = random.sample(emojis, emoji_count)
                result["text"] = text_without_emoji + " " + "".join(selected_emojis)
                mutations.append("emoji_changed")
        
        # 3. Cambiar posición
        if random.random() < rate * 0.4:
            positions = ["end", "middle", "beginning"]
            positions.remove(result["position"])
            result["position"] = random.choice(positions)
            mutations.append("position_changed")
        
        # 4. Cambiar duración
        if random.random() < rate * 0.4:
            current_duration = result["duration"]
            new_duration = max(3, min(20, current_duration + random.randint(-5, 5)))
            result["duration"] = new_duration
            mutations.append("duration_changed")
        
        # Registrar mutaciones
        result["mutations"] = mutations
        
        return result
    
    def _mutate_voice(self, voice: Dict[str, Any], rate: float = None) -> Dict[str, Any]:
        """Muta una configuración de voz"""
        if rate is None:
            rate = self.mutation_rate
        
        result = copy.deepcopy(voice)
        
        # Preservar ID original si existe
        if "id" not in result:
            result["id"] = f"voice_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Actualizar timestamp
        result["created_at"] = datetime.now().isoformat()
        
        # Posibles mutaciones
        mutations = []
        
        # 1. Cambiar tipo de voz
        if random.random() < rate * 0.3:
            voice_types = ["natural", "energética", "calmada", "profesional", "amigable", "seria"]
            voice_types.remove(result["voice_type"])
            result["voice_type"] = random.choice(voice_types)
            mutations.append("voice_type_changed")
        
        # 2. Ajustar pitch
        if random.random() < rate * 0.5:
            current_pitch = result["pitch"]
            delta = random.uniform(-0.2, 0.2)
            result["pitch"] = max(0.1, min(0.9, current_pitch + delta))
            result["pitch"] = round(result["pitch"], 2)
            mutations.append("pitch_adjusted")
        
        # 3. Ajustar velocidad
        if random.random() < rate * 0.5:
            current_speed = result["speed"]
            delta = random.uniform(-0.15, 0.15)
            result["speed"] = max(0.2, min(0.8, current_speed + delta))
            result["speed"] = round(result["speed"], 2)
            mutations.append("speed_adjusted")
        
        # 4. Ajustar claridad
        if random.random() < rate * 0.4:
            current_clarity = result["clarity"]
            delta = random.uniform(-0.15, 0.15)
            result["clarity"] = max(0.3, min(1.0, current_clarity + delta))
            result["clarity"] = round(result["clarity"], 2)
            mutations.append("clarity_adjusted")
        
        # 5. Ajustar emoción
        if random.random() < rate * 0.4:
            current_emotion = result["emotion"]
            delta = random.uniform(-0.2, 0.2)
            result["emotion"] = max(0.2, min(1.0, current_emotion + delta))
            result["emotion"] = round(result["emotion"], 2)
            mutations.append("emotion_adjusted")
        
        # Registrar mutaciones
        result["mutations"] = mutations
        
        return result
    
    def _mutate_visual(self, visual: Dict[str, Any], rate: float = None) -> Dict[str, Any]:
        """Muta una configuración visual"""
        if rate is None:
            rate = self.mutation_rate
        
        result = copy.deepcopy(visual)
        
        # Preservar ID original si existe
        if "id" not in result:
            result["id"] = f"visual_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Actualizar timestamp
        result["created_at"] = datetime.now().isoformat()
        
        # Posibles mutaciones
        mutations = []
        
        # 1. Cambiar estilo
        if random.random() < rate * 0.3:
            visual_styles = ["minimalista", "colorido", "oscuro", "brillante", "contrastante", "suave"]
            visual_styles.remove(result["style"])
            result["style"] = random.choice(visual_styles)
            mutations.append("style_changed")
        
        # 2. Ajustar brillo
        if random.random() < rate * 0.5:
            current_brightness = result["brightness"]
            delta = random.uniform(-0.15, 0.15)
            result["brightness"] = max(0.2, min(0.8, current_brightness + delta))
            result["brightness"] = round(result["brightness"], 2)
            mutations.append("brightness_adjusted")
        
        # 3. Ajustar contraste
        if random.random() < rate * 0.5:
            current_contrast = result["contrast"]
            delta = random.uniform(-0.15, 0.15)
            result["contrast"] = max(0.3, min(0.9, current_contrast + delta))
            result["contrast"] = round(result["contrast"], 2)
            mutations.append("contrast_adjusted")
        
        # 4. Ajustar saturación
        if random.random() < rate * 0.4:
            current_saturation = result["saturation"]
            delta = random.uniform(-0.15, 0.15)
            result["saturation"] = max(0.2, min(0.8, current_saturation + delta))
            result["saturation"] = round(result["saturation"], 2)
            mutations.append("saturation_adjusted")
        
        # 5. Ajustar complejidad
        if random.random() < rate * 0.4:
            current_complexity = result["complexity"]
            delta = random.uniform(-0.15, 0.15)
            result["complexity"] = max(0.1, min(0.9, current_complexity + delta))
            result["complexity"] = round(result["complexity"], 2)
            mutations.append("complexity_adjusted")
        
        # 6. Cambiar transición
        if random.random() < rate * 0.3:
            transitions = ["corte", "fade", "deslizamiento", "zoom", "ninguna"]
            transitions.remove(result["transition"])
            result["transition"] = random.choice(transitions)
            mutations.append("transition_changed")
        
        # Registrar mutaciones
        result["mutations"] = mutations
        
        return result
    
    def _mutate_thumbnail(self, thumbnail: Dict[str, Any], rate: float = None) -> Dict[str, Any]:
        """Muta una configuración de miniatura"""
        if rate is None:
            rate = self.mutation_rate
        
        result = copy.deepcopy(thumbnail)
        
        # Preservar ID original si existe
        if "id" not in result:
            result["id"] = f"thumbnail_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Actualizar timestamp
        result["created_at"] = datetime.now().isoformat()
        
        # Posibles mutaciones
        mutations = []
        
        # 1. Cambiar presencia de texto
        if random.random() < rate * 0.3:
            result["has_text"] = not result["has_text"]
            
            if result["has_text"]:
                text_options = ["pregunta", "número", "afirmación impactante"]
                result["text_type"] = random.choice(text_options)
                result["text_size"] = round(random.uniform(0.5, 0.9), 2)
                mutations.append("text_added")
            else:
                result["text_type"] = "ninguno"
                result["text_size"] = 0
                mutations.append("text_removed")
        
        # 2. Cambiar tipo de texto (si tiene texto)
        elif result["has_text"] and random.random() < rate * 0.4:
            text_options = ["pregunta", "número", "afirmación impactante"]
            text_options.remove(result["text_type"])
            result["text_type"] = random.choice(text_options)
            mutations.append("text_type_changed")
        
        # 3. Ajustar tamaño de texto (si tiene texto)
        if result["has_text"] and random.random() < rate * 0.5:
            current_size = result["text_size"]
            delta = random.uniform(-0.15, 0.15)
            result["text_size"] = max(0.3, min(1.0, current_size + delta))
            result["text_size"] = round(result["text_size"], 2)
            mutations.append("text_size_adjusted")
        
        # 4. Cambiar número de rostros
        if random.random() < rate * 0.4:
            current_faces = result["face_count"]
            if current_faces == 0:
                result["face_count"] = random.randint(1, 2)
                mutations.append("faces_added")
            elif current_faces >= 2:
                result["face_count"] = random.randint(0, 1)
                mutations.append("faces_reduced")
            else:
                result["face_count"] = random.choice([0, 2])
                mutations.append("face_count_changed")
        
        # 5. Ajustar brillo
        if random.random() < rate * 0.4:
            current_brightness = result["brightness"]
            delta = random.uniform(-0.15, 0.15)
            result["brightness"] = max(0.2, min(0.8, current_brightness + delta))
            result["brightness"] = round(result["brightness"], 2)
            mutations.append("brightness_adjusted")
        
        # 6. Ajustar contraste
        if random.random() < rate * 0.4:
            current_contrast = result["contrast"]
            delta = random.uniform(-0.15, 0.15)
            result["contrast"] = max(0.3, min(0.9, current_contrast + delta))
            result["contrast"] = round(result["contrast"], 2)
            mutations.append("contrast_adjusted")
        
        # 7. Cambiar borde
        if random.random() < rate * 0.3:
            result["border"] = not result["border"]
            mutations.append("border_toggled")
        
        # Registrar mutaciones
        result["mutations"] = mutations
        
        return result
    
    def _mutate_script(self, script: Dict[str, Any], rate: float = None) -> Dict[str, Any]:
        """Muta una configuración de guión"""
        if rate is None:
            rate = self.mutation_rate
        
        result = copy.deepcopy(script)
        
        # Preservar ID original si existe
        if "id" not in result:
            result["id"] = f"script_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Actualizar timestamp
        result["created_at"] = datetime.now().isoformat()
        
        # Posibles mutaciones
        mutations = []
        
        # 1. Cambiar tipo de gancho
        if random.random() < rate * 0.4:
            hook_types = ["pregunta", "dato sorprendente", "problema común", "historia", "controversia"]
            hook_types.remove(result["hook_type"])
            result["hook_type"] = random.choice(hook_types)
            mutations.append("hook_type_changed")
        
        # 2. Ajustar fuerza del gancho
        if random.random() < rate * 0.5:
            current_strength = result["hook_strength"]
            delta = random.uniform(-0.15, 0.15)
            result["hook_strength"] = max(0.3, min(1.0, current_strength + delta))
            result["hook_strength"] = round(result["hook_strength"], 2)
            mutations.append("hook_strength_adjusted")
        
        # 3. Cambiar estructura
        if random.random() < rate * 0.3:
            structure_types = ["problema-solución", "lista", "tutorial", "narrativa", "comparación"]
            structure_types.remove(result["structure"])
            result["structure"] = random.choice(structure_types)
            mutations.append("structure_changed")
        
        # 4. Ajustar duración
        if random.random() < rate * 0.6:
            current_duration = result["duration"]
            if current_duration < 30:
                # Para videos cortos, pequeños ajustes
                delta = random.randint(-10, 15)
            else:
                # Para videos más largos, ajustes mayores
                delta = random.randint(-30, 30)
            
            result["duration"] = max(10, current_duration + delta)
            mutations.append("duration_adjusted")
        
        # 5. Ajustar número de CTAs
        if random.random() < rate * 0.4:
            current_ctas = result["cta_count"]
            if current_ctas >= 2:
                result["cta_count"] = random.randint(1, current_ctas - 1)
                mutations.append("cta_count_reduced")
            else:
                result["cta_count"] = random.randint(current_ctas + 1, 3)
                mutations.append("cta_count_increased")
        
        # 6. Ajustar número de preguntas
        if random.random() < rate * 0.4:
            current_questions = result["question_count"]
            if current_questions >= 2:
                result["question_count"] = random.randint(0, current_questions - 1)
                mutations.append("question_count_reduced")
            else:
                result["question_count"] = random.randint(current_questions + 1, 3)
                mutations.append("question_count_increased")
        
        # Registrar mutaciones
        result["mutations"] = mutations
        
        return result
    
    def _mutate_title(self, title: Dict[str, Any], rate: float = None) -> Dict[str, Any]:
        """Muta un título"""
        if rate is None:
            rate = self.mutation_rate
        
        result = copy.deepcopy(title)
        
        # Preservar ID original si existe
        if "id" not in result:
            result["id"] = f"title_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
        
        # Actualizar timestamp
        result["created_at"] = datetime.now().isoformat()
        
        # Posibles mutaciones
        mutations = []
        
        # 1. Cambiar título completamente
        if random.random() < rate * 0.3:
            new_title = self._generate_random_title()
            result["text"] = new_title["text"]
            result["has_number"] = new_title["has_number"]
            result["has_question"] = new_title["has_question"]
            result["length"] = new_title["length"]
            mutations.append("title_replaced")
            return result
        
        # 2. Añadir/quitar número
        if random.random() < rate * 0.4:
            text = result["text"]
            if not result["has_number"]:
                # Añadir número
                if "mejores" in text.lower():
                    text = text.replace("mejores", f"{random.randint(3, 10)} mejores")
                elif "formas" in text.lower():
                    text = text.replace("formas", f"{random.randint(3, 10)} formas")
                elif "consejos" in text.lower():
                    text = text.replace("consejos", f"{random.randint(3, 10)} consejos")
                elif "razones" in text.lower():
                    text = text.replace("razones", f"{random.randint(3, 10)} razones")
                elif "secretos" in text.lower():
                    text = text.replace("secretos", f"{random.randint(3, 10)} secretos")
                else:
                    # Añadir al principio
                    text = f"{random.randint(3, 10)} " + text
                
                result["has_number"] = True
                mutations.append("number_added")
            else:
                # Cambiar número
                words = text.split()
                for i, word in enumerate(words):
                    if word.isdigit():
                        words[i] = str(random.randint(3, 10))
                        mutations.append("number_changed")
                        break
                text = " ".join(words)
            
            result["text"] = text
        
        # 3. Añadir/quitar pregunta
        if random.random() < rate * 0.4:
            text = result["text"]
            if not result["has_question"]:
                # Convertir a pregunta
                if text.startswith("Cómo"):
                    text = "¿" + text + "?"
                elif text.startswith("Por qué"):
                    text = "¿" + text + "?"
                else:
                    text = "¿" + text + "?"
                
                result["has_question"] = True
                mutations.append("question_added")
            elif result["has_question"]:
                # Quitar pregunta
                text = text.replace("¿", "").replace("?", "")
                result["has_question"] = False
                mutations.append("question_removed")
            
            result["text"] = text
        
        # 4. Añadir palabra de impacto
        if random.random() < rate * 0.5:
            impact_words = ["increíble", "sorprendente", "impactante", "secreto", "revelado", 
                           "mejor", "peor", "nunca", "siempre", "fácil", "rápido", "gratis"]
            
            text = result["text"]
            words = text.split()
            
            # Insertar palabra de impacto en posición aleatoria
            insert_pos = random.randint(0, len(words))
            impact_word = random.choice(impact_words)
            
            words.insert(insert_pos, impact_word)
            result["text"] = " ".join(words)
            result["length"] = len(words)
            mutations.append("impact_word_added")
        
        # Registrar mutaciones
        result["mutations"] = mutations
        
        return result
    
    # Funciones de cruce
        def _crossover_cta(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza cruce entre dos CTAs"""
        child = {
            "id": f"cta_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "cta",
            "created_at": datetime.now().isoformat()
        }
        
        # Heredar texto de uno de los padres
        if random.random() < 0.5:
            child["text"] = parent1["text"]
        else:
            child["text"] = parent2["text"]
        
        # Heredar posición
        if random.random() < 0.5:
            child["position"] = parent1["position"]
        else:
            child["position"] = parent2["position"]
        
        # Heredar duración (posible promedio)
        if random.random() < 0.7:
            # Herencia directa
            if random.random() < 0.5:
                child["duration"] = parent1["duration"]
            else:
                child["duration"] = parent2["duration"]
        else:
            # Promedio de ambos padres
            child["duration"] = round((parent1["duration"] + parent2["duration"]) / 2)
        
        return child
    
    def _crossover_voice(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza cruce entre dos configuraciones de voz"""
        child = {
            "id": f"voice_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "voice",
            "created_at": datetime.now().isoformat()
        }
        
        # Heredar tipo de voz
        if random.random() < 0.5:
            child["voice_type"] = parent1["voice_type"]
        else:
            child["voice_type"] = parent2["voice_type"]
        
        # Heredar o promediar parámetros numéricos
        for param in ["pitch", "speed", "clarity", "emotion"]:
            if random.random() < 0.7:
                # Herencia directa
                if random.random() < 0.5:
                    child[param] = parent1[param]
                else:
                    child[param] = parent2[param]
            else:
                # Promedio de ambos padres
                child[param] = round((parent1[param] + parent2[param]) / 2, 2)
        
        return child
    
    def _crossover_visual(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza cruce entre dos configuraciones visuales"""
        child = {
            "id": f"visual_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "visual",
            "created_at": datetime.now().isoformat()
        }
        
        # Heredar estilo
        if random.random() < 0.5:
            child["style"] = parent1["style"]
        else:
            child["style"] = parent2["style"]
        
        # Heredar transición
        if random.random() < 0.5:
            child["transition"] = parent1["transition"]
        else:
            child["transition"] = parent2["transition"]
        
        # Heredar o promediar parámetros numéricos
        for param in ["brightness", "contrast", "saturation", "complexity"]:
            if random.random() < 0.7:
                # Herencia directa
                if random.random() < 0.5:
                    child[param] = parent1[param]
                else:
                    child[param] = parent2[param]
            else:
                # Promedio de ambos padres
                child[param] = round((parent1[param] + parent2[param]) / 2, 2)
        
        return child
    
    def _crossover_thumbnail(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza cruce entre dos configuraciones de miniatura"""
        child = {
            "id": f"thumbnail_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "thumbnail",
            "created_at": datetime.now().isoformat()
        }
        
        # Heredar presencia de texto
        if random.random() < 0.5:
            child["has_text"] = parent1["has_text"]
            if child["has_text"]:
                child["text_type"] = parent1["text_type"]
                child["text_size"] = parent1["text_size"]
            else:
                child["text_type"] = "ninguno"
                child["text_size"] = 0
        else:
            child["has_text"] = parent2["has_text"]
            if child["has_text"]:
                child["text_type"] = parent2["text_type"]
                child["text_size"] = parent2["text_size"]
            else:
                child["text_type"] = "ninguno"
                child["text_size"] = 0
        
        # Heredar número de rostros
        if random.random() < 0.5:
            child["face_count"] = parent1["face_count"]
        else:
            child["face_count"] = parent2["face_count"]
        
        # Heredar borde
        if random.random() < 0.5:
            child["border"] = parent1["border"]
        else:
            child["border"] = parent2["border"]
        
        # Heredar o promediar parámetros numéricos
        for param in ["brightness", "contrast"]:
            if random.random() < 0.7:
                # Herencia directa
                if random.random() < 0.5:
                    child[param] = parent1[param]
                else:
                    child[param] = parent2[param]
            else:
                # Promedio de ambos padres
                child[param] = round((parent1[param] + parent2[param]) / 2, 2)
        
        return child
    
    def _crossover_script(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza cruce entre dos configuraciones de guión"""
        child = {
            "id": f"script_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "script",
            "created_at": datetime.now().isoformat()
        }
        
        # Heredar tipo de gancho
        if random.random() < 0.5:
            child["hook_type"] = parent1["hook_type"]
        else:
            child["hook_type"] = parent2["hook_type"]
        
        # Heredar estructura
        if random.random() < 0.5:
            child["structure"] = parent1["structure"]
        else:
            child["structure"] = parent2["structure"]
        
        # Heredar o promediar fuerza del gancho
        if random.random() < 0.7:
            # Herencia directa
            if random.random() < 0.5:
                child["hook_strength"] = parent1["hook_strength"]
            else:
                child["hook_strength"] = parent2["hook_strength"]
        else:
            # Promedio de ambos padres
            child["hook_strength"] = round((parent1["hook_strength"] + parent2["hook_strength"]) / 2, 2)
        
        # Heredar o promediar duración
        if random.random() < 0.7:
            # Herencia directa
            if random.random() < 0.5:
                child["duration"] = parent1["duration"]
            else:
                child["duration"] = parent2["duration"]
        else:
            # Promedio de ambos padres
            child["duration"] = round((parent1["duration"] + parent2["duration"]) / 2)
        
        # Heredar número de CTAs
        if random.random() < 0.5:
            child["cta_count"] = parent1["cta_count"]
        else:
            child["cta_count"] = parent2["cta_count"]
        
        # Heredar número de preguntas
        if random.random() < 0.5:
            child["question_count"] = parent1["question_count"]
        else:
            child["question_count"] = parent2["question_count"]
        
        return child
    
    def _crossover_title(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza cruce entre dos títulos"""
        child = {
            "id": f"title_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "type": "title",
            "created_at": datetime.now().isoformat()
        }
        
        # Estrategia 1: Tomar texto completo de uno de los padres (70% de probabilidad)
        if random.random() < 0.7:
            if random.random() < 0.5:
                child["text"] = parent1["text"]
                child["has_number"] = parent1["has_number"]
                child["has_question"] = parent1["has_question"]
                child["length"] = parent1["length"]
            else:
                child["text"] = parent2["text"]
                child["has_number"] = parent2["has_number"]
                child["has_question"] = parent2["has_question"]
                child["length"] = parent2["length"]
        else:
            # Estrategia 2: Combinar partes de ambos padres (30% de probabilidad)
            # Dividir textos en partes
            p1_words = parent1["text"].split()
            p2_words = parent2["text"].split()
            
            # Punto de corte para parent1
            cut_point1 = random.randint(1, max(1, len(p1_words) - 1))
            
            # Crear nuevo texto
            new_text = " ".join(p1_words[:cut_point1] + p2_words[cut_point1:])
            
            # Actualizar propiedades
            child["text"] = new_text
            child["has_number"] = any(char.isdigit() for char in new_text)
            child["has_question"] = "?" in new_text
            child["length"] = len(new_text.split())
        
        return child