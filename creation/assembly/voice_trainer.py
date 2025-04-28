"""
Voice Trainer - Entrenador de voces personalizadas

Este módulo gestiona el entrenamiento de modelos de voz personalizados:
- Entrenamiento de modelos XTTS (Coqui TTS)
- Entrenamiento de modelos RVC (Retrieval-based Voice Conversion)
- Gestión de muestras de voz y modelos entrenados
- Integración con servicios de entrenamiento (local o en la nube)
"""

import os
import sys
import json
import logging
import time
import random
import shutil
import numpy as np
import soundfile as sf
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'voice_trainer.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('VoiceTrainer')

class VoiceTrainer:
    """
    Clase para entrenar y gestionar modelos de voz personalizados.
    """

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.config = self._load_config()
        self.models = self._load_models()
        self.training_jobs = {}
        logger.info("VoiceTrainer initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración de entrenamiento de voces"""
        try:
            # Intentar cargar desde la base de conocimiento
            config = self.knowledge_base.get_voice_training_config()
            if not config:
                # Configuración por defecto
                config = {
                    "models_directory": "models/voice",
                    "samples_directory": "datasets/voice_samples",
                    "xtts": {
                        "enabled": True,
                        "local": True,
                        "model_path": "models/xtts_base",
                        "min_samples": 3,
                        "recommended_samples": 10,
                        "sample_min_duration": 5,
                        "sample_max_duration": 30,
                        "training_steps": 5000,
                        "batch_size": 4,
                        "learning_rate": 0.0001
                    },
                    "rvc": {
                        "enabled": True,
                        "local": True,
                        "model_path": "models/rvc_base",
                        "min_samples": 5,
                        "recommended_samples": 20,
                        "sample_min_duration": 10,
                        "sample_max_duration": 60,
                        "training_epochs": 100,
                        "batch_size": 8,
                        "learning_rate": 0.0002
                    },
                    "cloud_services": {
                        "elevenlabs": {
                            "enabled": False,
                            "api_key": "",
                            "base_url": "https://api.elevenlabs.io/v1"
                        }
                    },
                    "max_concurrent_jobs": 1,
                    "auto_cleanup": True,
                    "cleanup_days": 30
                }
                # Guardar en la base de conocimiento
                self.knowledge_base.save_voice_training_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading voice training config: {str(e)}")
            # Retornar configuración mínima
            return {
                "models_directory": "models/voice",
                "samples_directory": "datasets/voice_samples",
                "xtts": {"enabled": True, "local": True},
                "rvc": {"enabled": True, "local": True},
                "max_concurrent_jobs": 1
            }

    def _load_models(self) -> Dict[str, Dict[str, Any]]:
        """Carga información sobre modelos de voz entrenados"""
        try:
            # Intentar cargar desde la base de conocimiento
            models = self.knowledge_base.get_voice_models()
            if not models:
                models = {}
            return models
        except Exception as e:
            logger.error(f"Error loading voice models: {str(e)}")
            return {}

    def create_voice_profile(self, name: str, description: str = "", 
                           character_id: str = None, 
                           model_type: str = "xtts") -> Dict[str, Any]:
        """
        Crea un nuevo perfil de voz
        
        Args:
            name: Nombre del perfil de voz
            description: Descripción del perfil
            character_id: ID del personaje asociado (opcional)
            model_type: Tipo de modelo (xtts, rvc)
            
        Returns:
            Diccionario con información del perfil creado
        """
        try:
            # Generar ID único
            voice_id = f"voice_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Crear estructura de directorios
            samples_dir = os.path.join(self.config["samples_directory"], voice_id)
            os.makedirs(samples_dir, exist_ok=True)
            
            # Crear perfil
            profile = {
                "id": voice_id,
                "name": name,
                "description": description,
                "character_id": character_id,
                "model_type": model_type,
                "created_at": time.time(),
                "updated_at": time.time(),
                "samples": [],
                "models": [],
                "status": "created",
                "training_history": []
            }
            
            # Guardar en la base de conocimiento
            self.knowledge_base.save_voice_profile(voice_id, profile)
            
            logger.info(f"Created voice profile: {voice_id} ({name})")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating voice profile: {str(e)}")
            raise

    def add_voice_sample(self, voice_id: str, sample_path: str, 
                       transcript: str = "", 
                       validate: bool = True) -> Dict[str, Any]:
        """
        Añade una muestra de voz a un perfil
        
        Args:
            voice_id: ID del perfil de voz
            sample_path: Ruta al archivo de audio de muestra
            transcript: Transcripción del audio (opcional)
            validate: Si se debe validar la calidad de la muestra
            
        Returns:
            Diccionario con información de la muestra añadida
        """
        try:
            # Obtener perfil
            profile = self.knowledge_base.get_voice_profile(voice_id)
            if not profile:
                raise ValueError(f"Voice profile not found: {voice_id}")
                
            # Validar muestra si se solicita
            if validate:
                validation = self._validate_voice_sample(sample_path, profile["model_type"])
                if not validation["valid"]:
                    raise ValueError(f"Invalid voice sample: {validation['reason']}")
                    
            # Copiar muestra al directorio del perfil
            samples_dir = os.path.join(self.config["samples_directory"], voice_id)
            os.makedirs(samples_dir, exist_ok=True)
            
            # Generar nombre único para la muestra
            sample_filename = f"sample_{len(profile['samples']) + 1}_{int(time.time())}.wav"
            sample_dest = os.path.join(samples_dir, sample_filename)
            
            # Convertir a formato WAV si es necesario
            if sample_path.lower().endswith(".wav"):
                shutil.copy(sample_path, sample_dest)
            else:
                self._convert_to_wav(sample_path, sample_dest)
                
            # Crear información de la muestra
            sample_info = {
                "id": f"sample_{int(time.time())}_{random.randint(1000, 9999)}",
                "filename": sample_filename,
                "path": sample_dest,
                "transcript": transcript,
                "duration": self._get_audio_duration(sample_dest),
                "added_at": time.time()
            }
            
            # Añadir a perfil
            profile["samples"].append(sample_info)
            profile["updated_at"] = time.time()
            
            # Actualizar estado si es la primera muestra
            if len(profile["samples"]) == 1:
                profile["status"] = "samples_added"
                
            # Guardar en la base de conocimiento
            self.knowledge_base.save_voice_profile(voice_id, profile)
            
            logger.info(f"Added voice sample to profile {voice_id}: {sample_filename}")
            return sample_info
            
        except Exception as e:
            logger.error(f"Error adding voice sample: {str(e)}")
            raise

    def _validate_voice_sample(self, sample_path: str, model_type: str) -> Dict[str, Any]:
        """
        Valida la calidad de una muestra de voz
        
        Args:
            sample_path: Ruta al archivo de audio
            model_type: Tipo de modelo (xtts, rvc)
            
        Returns:
            Diccionario con resultado de validación
        """
        try:
            # Verificar que el archivo existe
            if not os.path.exists(sample_path):
                return {"valid": False, "reason": "File does not exist"}
                
            # Obtener duración
            duration = self._get_audio_duration(sample_path)
            
            # Verificar duración mínima y máxima según tipo de modelo
            model_config = self.config.get(model_type, {})
            min_duration = model_config.get("sample_min_duration", 5)
            max_duration = model_config.get("sample_max_duration", 60)
            
            if duration < min_duration:
                return {
                    "valid": False, 
                    "reason": f"Sample too short ({duration:.1f}s), minimum is {min_duration}s"
                }
                
            if duration > max_duration:
                return {
                    "valid": False, 
                    "reason": f"Sample too long ({duration:.1f}s), maximum is {max_duration}s"
                }
                
            # Aquí se podrían añadir más validaciones:
            # - Calidad de audio (SNR)
            # - Presencia de voz
            # - Nivel de ruido
            # - etc.
                
            return {"valid": True, "duration": duration}
            
        except Exception as e:
            logger.error(f"Error validating voice sample: {str(e)}")
            return {"valid": False, "reason": str(e)}

    def train_voice_model(self, voice_id: str, 
                         model_name: str = None,
                         model_type: str = None,
                         training_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Inicia el entrenamiento de un modelo de voz
        
        Args:
            voice_id: ID del perfil de voz
            model_name: Nombre del modelo (opcional)
            model_type: Tipo de modelo (xtts, rvc) (opcional)
            training_params: Parámetros de entrenamiento (opcional)
            
        Returns:
            Diccionario con información del trabajo de entrenamiento
        """
        try:
            # Obtener perfil
            profile = self.knowledge_base.get_voice_profile(voice_id)
            if not profile:
                raise ValueError(f"Voice profile not found: {voice_id}")
                
            # Verificar si hay suficientes muestras
            if not profile["samples"]:
                raise ValueError("No voice samples available for training")
                
            # Usar tipo de modelo del perfil si no se especifica
            if not model_type:
                model_type = profile.get("model_type", "xtts")
                
            # Verificar configuración del tipo de modelo
            model_config = self.config.get(model_type, {})
            if not model_config.get("enabled", False):
                raise ValueError(f"Model type {model_type} is not enabled")
                
            # Verificar número mínimo de muestras
            min_samples = model_config.get("min_samples", 3)
            if len(profile["samples"]) < min_samples:
                raise ValueError(
                    f"Not enough samples. Have {len(profile['samples'])}, need at least {min_samples}"
                )
                
            # Generar nombre de modelo si no se proporciona
            if not model_name:
                model_name = f"{profile['name']}_{model_type}_{int(time.time())}"
                
            # Crear directorio para el modelo
            model_dir = os.path.join(self.config["models_directory"], voice_id, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # Generar ID único para el trabajo
            job_id = f"job_{int(time.time())}_{random.randint(1000, 9999)}"
            
            # Combinar parámetros de entrenamiento
            default_params = self._get_default_training_params(model_type)
            params = default_params.copy()
            if training_params:
                params.update(training_params)
                
            # Crear información del trabajo
            job_info = {
                "id": job_id,
                "voice_id": voice_id,
                "model_name": model_name,
                "model_type": model_type,
                "model_dir": model_dir,
                "params": params,
                "status": "queued",
                "progress": 0.0,
                "created_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "error": None
            }
            
            # Añadir a la lista de trabajos
            self.training_jobs[job_id] = job_info
            
            # Actualizar perfil
            profile["status"] = "training_queued"
            profile["training_history"].append({
                "job_id": job_id,
                "model_name": model_name,
                "model_type": model_type,
                "status": "queued",
                "created_at": time.time()
            })
            self.knowledge_base.save_voice_profile(voice_id, profile)
            
            # Iniciar entrenamiento en segundo plano
            self._schedule_training_job(job_id)
            
            logger.info(f"Queued voice model training: {job_id} for profile {voice_id}")
            return job_info
            
        except Exception as e:
            logger.error(f"Error training voice model: {str(e)}")
            raise

    def _get_default_training_params(self, model_type: str) -> Dict[str, Any]:
        """Obtiene parámetros de entrenamiento por defecto según tipo de modelo"""
        if model_type == "xtts":
            return {
                "training_steps": self.config.get("xtts", {}).get("training_steps", 5000),
                "batch_size": self.config.get("xtts", {}).get("batch_size", 4),
                "learning_rate": self.config.get("xtts", {}).get("learning_rate", 0.0001)
            }
        elif model_type == "rvc":
            return {
                "training_epochs": self.config.get("rvc", {}).get("training_epochs", 100),
                "batch_size": self.config.get("rvc", {}).get("batch_size", 8),
                "learning_rate": self.config.get("rvc", {}).get("learning_rate", 0.0002)
            }
        else:
            return {}

    def _schedule_training_job(self, job_id: str) -> None:
        """
        Programa un trabajo de entrenamiento
        
        En una implementación real, esto iniciaría un thread o proceso
        para ejecutar el entrenamiento en segundo plano.
        """
        # Verificar si hay capacidad para iniciar el trabajo
        active_jobs = sum(1 for job in self.training_jobs.values() 
                         if job["status"] in ["running", "preparing"])
                         
        max_jobs = self.config.get("max_concurrent_jobs", 1)
        
        if active_jobs >= max_jobs:
            logger.info(f"Job {job_id} queued, waiting for capacity")
            return
            
        # Simular inicio de entrenamiento en segundo plano
        # En una implementación real, aquí se iniciaría un thread o proceso
        job = self.training_jobs[job_id]
        job["status"] = "preparing"
        
        # Simular entrenamiento (en implementación real, esto sería asíncrono)
        self._run_training_job(job_id)

    def _run_training_job(self, job_id: str) -> None:
        """
        Ejecuta un trabajo de entrenamiento
        
        En una implementación real, esto se ejecutaría en un thread o proceso separado.
        """
        try:
            job = self.training_jobs[job_id]
            voice_id = job["voice_id"]
            model_type = job["model_type"]
            
            # Actualizar estado
            job["status"] = "running"
            job["started_at"] = time.time()
            
            # Actualizar perfil
            profile = self.knowledge_base.get_voice_profile(voice_id)
            if profile:
                profile["status"] = "training"
                for history in profile["training_history"]:
                    if history["job_id"] == job_id:
                        history["status"] = "running"
                        history["started_at"] = job["started_at"]
                self.knowledge_base.save_voice_profile(voice_id, profile)
            
            logger.info(f"Started training job {job_id} for voice {voice_id}")
            
            # Simular entrenamiento
            # En una implementación real, aquí se llamaría al código de entrenamiento
            if model_type == "xtts":
                self._train_xtts_model(job_id)
            elif model_type == "rvc":
                self._train_rvc_model(job_id)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Actualizar estado al completar
            job["status"] = "completed"
            job["progress"] = 100.0
            job["completed_at"] = time.time()
            
            # Crear información del modelo
            model_info = self._create_model_info(job)
            
            # Actualizar perfil
            profile = self.knowledge_base.get_voice_profile(voice_id)
            if profile:
                profile["status"] = "trained"
                profile["models"].append(model_info)
                for history in profile["training_history"]:
                    if history["job_id"] == job_id:
                        history["status"] = "completed"
                        history["completed_at"] = job["completed_at"]
                self.knowledge_base.save_voice_profile(voice_id, profile)
                
            # Actualizar modelos
            self.models[model_info["id"]] = model_info
            self.knowledge_base.save_voice_model(model_info["id"], model_info)
            
            logger.info(f"Completed training job {job_id} for voice {voice_id}")
            
            # Programar siguiente trabajo en cola
            self._schedule_next_job()
            
        except Exception as e:
            # Manejar error
            logger.error(f"Error in training job {job_id}: {str(e)}")
            
            job = self.training_jobs.get(job_id)
            if job:
                job["status"] = "failed"
                job["error"] = str(e)
                job["completed_at"] = time.time()
                
                # Actualizar perfil
                voice_id = job["voice_id"]
                profile = self.knowledge_base.get_voice_profile(voice_id)
                if profile:
                    profile["status"] = "training_failed"
                    for history in profile["training_history"]:
                        if history["job_id"] == job_id:
                            history["status"] = "failed"
                            history["error"] = str(e)
                            history["completed_at"] = job["completed_at"]
                    self.knowledge_base.save_voice_profile(voice_id, profile)
                    
            # Programar siguiente trabajo en cola
            self._schedule_next_job()

    def _schedule_next_job(self) -> None:
        """Programa el siguiente trabajo en cola"""
        # Buscar trabajos en cola
        queued_jobs = [job_id for job_id, job in self.training_jobs.items() 
                      if job["status"] == "queued"]
                      
        if not queued_jobs:
            return
            
        # Ordenar por tiempo de creación (primero los más antiguos)
        queued_jobs.sort(key=lambda job_id: self.training_jobs[job_id]["created_at"])
        
        # Programar el primero
        self._schedule_training_job(queued_jobs[0])

    def _train_xtts_model(self, job_id: str) -> None:
        """
        Entrena un modelo XTTS
        
        En una implementación real, aquí se llamaría a la biblioteca de XTTS.
        """
        job = self.training_jobs[job_id]
        voice_id = job["voice_id"]
        
        # Obtener perfil y muestras
        profile = self.knowledge_base.get_voice_profile(voice_id)
        if not profile or not profile["samples"]:
            raise ValueError("No samples available for training")
            
        # Simular progreso de entrenamiento
        steps = job["params"].get("training_steps", 5000)
        for step in range(0, steps + 1, 100):
            # Actualizar progreso
            progress = min(step / steps * 100, 99.9)
            job["progress"] = progress
            
            # Simular tiempo de entrenamiento
            time.sleep(0.1)  # En una implementación real, esto tomaría mucho más tiempo
            
            # Actualizar perfil periódicamente
            if step % 1000 == 0:
                profile = self.knowledge_base.get_voice_profile(voice_id)
                if profile:
                    for history in profile["training_history"]:
                        if history["job_id"] == job_id:
                            history["progress"] = progress
                    self.knowledge_base.save_voice_profile(voice_id, profile)
                    
        # Crear archivo de modelo simulado
        model_path = os.path.join(job["model_dir"], f"{job['model_name']}.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Crear archivo vacío como placeholder
        with open(model_path, "wb") as f:
            f.write(b"XTTS_MODEL_PLACEHOLDER")
            
        # Guardar configuración del modelo
        config_path = os.path.join(job["model_dir"], f"{job['model_name']}_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_type": "xtts",
                "name": job["model_name"],
                "params": job["params"],
                "created_at": time.time()
            }, f, indent=2)
            
        # Actualizar ruta del modelo en el trabajo
        job["model_path"] = model_path
        job["config_path"] = config_path

    def _train_rvc_model(self, job_id: str) -> None:
        """
        Entrena un modelo RVC
        
        En una implementación real, aquí se llamaría a la biblioteca de RVC.
        """
        job = self.training_jobs[job_id]
        voice_id = job["voice_id"]
        
        # Obtener perfil y muestras
        profile = self.knowledge_base.get_voice_profile(voice_id)
        if not profile or not profile["samples"]:
            raise ValueError("No samples available for training")
            
        # Simular progreso de entrenamiento
        epochs = job["params"].get("training_epochs", 100)
        for epoch in range(epochs + 1):
            # Actualizar progreso
            progress = min(epoch / epochs * 100, 99.9)
            job["progress"] = progress
            
            # Simular tiempo de entrenamiento
            time.sleep(0.1)  # En una implementación real, esto tomaría mucho más tiempo
            
            # Actualizar perfil periódicamente
            if epoch % 10 == 0:
                profile = self.knowledge_base.get_voice_profile(voice_id)
                if profile:
                    for history in profile["training_history"]:
                        if history["job_id"] == job_id:
                            history["progress"] = progress
                    self.knowledge_base.save_voice_profile(voice_id, profile)
                    
        # Crear archivo de modelo simulado
        model_path = os.path.join(job["model_dir"], f"{job['model_name']}.pth")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Crear archivo vacío como placeholder
        with open(model_path, "wb") as f:
            f.write(b"RVC_MODEL_PLACEHOLDER")
            
        # Guardar configuración del modelo
        config_path = os.path.join(job["model_dir"], f"{job['model_name']}_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "model_type": "rvc",
                "name": job["model_name"],
                "params": job["params"],
                "created_at": time.time()
            }, f, indent=2)
            
        # Actualizar ruta del modelo en el trabajo
        job["model_path"] = model_path
        job["config_path"] = config_path

    def _create_model_info(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Crea información del modelo entrenado"""
        model_id = f"model_{int(time.time())}_{random.randint(1000, 9999)}"
        
        return {
            "id": model_id,
            "name": job["model_name"],
            "voice_id": job["voice_id"],
            "model_type": job["model_type"],
            "model_path": job.get("model_path", ""),
            "config_path": job.get("config_path", ""),
            "params": job["params"],
            "created_at": time.time(),
            "job_id": job["id"],
            "status": "active"
        }

    def get_voice_profile(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un perfil de voz por su ID
        
        Args:
            voice_id: ID del perfil de voz
            
        Returns:
            Diccionario con el perfil o None si no existe
        """
        try:
            return self.knowledge_base.get_voice_profile(voice_id)
        except Exception as e:
            logger.error(f"Error getting voice profile: {str(e)}")
            return None

    def get_voice_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un modelo de voz por su ID
        
        Args:
            model_id: ID del modelo de voz
            
        Returns:
            Diccionario con el modelo o None si no existe
        """
        try:
            # Intentar obtener del caché local
            if model_id in self.models:
                return self.models[model_id]
                
            # Intentar obtener de la base de conocimiento
            model = self.knowledge_base.get_voice_model(model_id)
            if model:
                self.models[model_id] = model
            return model
        except Exception as e:
            logger.error(f"Error getting voice model: {str(e)}")
            return None

    def list_voice_profiles(self, character_id: str = None) -> List[Dict[str, Any]]:
        """
        Lista perfiles de voz disponibles
        
        Args:
            character_id: Filtrar por ID de personaje (opcional)
            
        Returns:
            Lista de perfiles de voz
        """
        try:
            profiles = self.knowledge_base.list_voice_profiles()
            
            # Filtrar por personaje si se especifica
            if character_id:
                profiles = [p for p in profiles if p.get("character_id") == character_id]
                
            return profiles
        except Exception as e:
            logger.error(f"Error listing voice profiles: {str(e)}")
            return []

    def list_voice_models(self, voice_id: str = None, 
                        model_type: str = None) -> List[Dict[str, Any]]:
        """
        Lista modelos de voz disponibles
        
        Args:
            voice_id: Filtrar por ID de perfil de voz (opcional)
            model_type: Filtrar por tipo de modelo (opcional)
            
        Returns:
            Lista de modelos de voz
        """
        try:
            models = list(self.models.values())
            
            # Si no hay modelos en caché, cargar desde la base de conocimiento
            if not models:
                models = self.knowledge_base.list_voice_models()
                
                # Actualizar caché
                for model in models:
                    self.models[model["id"]] = model
            
            # Filtrar por perfil si se especifica
            if voice_id:
                models = [m for m in models if m.get("voice_id") == voice_id]
                
            # Filtrar por tipo si se especifica
            if model_type:
                models = [m for m in models if m.get("model_type") == model_type]
                
            return models
        except Exception as e:
            logger.error(f"Error listing voice models: {str(e)}")
            return []

    def get_training_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un trabajo de entrenamiento
        
        Args:
            job_id: ID del trabajo
            
        Returns:
            Diccionario con información del trabajo o None si no existe
        """
        return self.training_jobs.get(job_id)

    def list_training_jobs(self, voice_id: str = None, 
                         status: str = None) -> List[Dict[str, Any]]:
        """
        Lista trabajos de entrenamiento
        
        Args:
            voice_id: Filtrar por ID de perfil de voz (opcional)
            status: Filtrar por estado (opcional)
            
        Returns:
            Lista de trabajos de entrenamiento
        """
        jobs = list(self.training_jobs.values())
        
        # Filtrar por perfil si se especifica
        if voice_id:
            jobs = [j for j in jobs if j.get("voice_id") == voice_id]
            
        # Filtrar por estado si se especifica
        if status:
            jobs = [j for j in jobs if j.get("status") == status]
            
        # Ordenar por tiempo de creación (más recientes primero)
        jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
        
        return jobs

        def delete_voice_profile(self, voice_id: str, 
                           delete_samples: bool = True,
                           delete_models: bool = False) -> bool:
        """
        Elimina un perfil de voz
        
        Args:
            voice_id: ID del perfil de voz
            delete_samples: Si se deben eliminar las muestras
            delete_models: Si se deben eliminar los modelos entrenados
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        try:
            # Obtener perfil
            profile = self.knowledge_base.get_voice_profile(voice_id)
            if not profile:
                logger.warning(f"Voice profile not found: {voice_id}")
                return False
                
            # Verificar si hay trabajos de entrenamiento activos
            active_jobs = [j for j in self.training_jobs.values() 
                          if j.get("voice_id") == voice_id and 
                          j.get("status") in ["queued", "preparing", "running"]]
                          
            if active_jobs:
                logger.warning(f"Cannot delete voice profile {voice_id} with active training jobs")
                return False
                
            # Eliminar muestras si se solicita
            if delete_samples:
                samples_dir = os.path.join(self.config["samples_directory"], voice_id)
                if os.path.exists(samples_dir):
                    try:
                        shutil.rmtree(samples_dir)
                        logger.info(f"Deleted samples directory for voice {voice_id}")
                    except Exception as e:
                        logger.error(f"Error deleting samples directory: {str(e)}")
                        
            # Eliminar modelos si se solicita
            if delete_models:
                # Obtener IDs de modelos asociados al perfil
                model_ids = [m["id"] for m in profile.get("models", [])]
                
                # Eliminar archivos de modelos
                models_dir = os.path.join(self.config["models_directory"], voice_id)
                if os.path.exists(models_dir):
                    try:
                        shutil.rmtree(models_dir)
                        logger.info(f"Deleted models directory for voice {voice_id}")
                    except Exception as e:
                        logger.error(f"Error deleting models directory: {str(e)}")
                        
                # Eliminar información de modelos de la base de conocimiento
                for model_id in model_ids:
                    try:
                        self.knowledge_base.delete_voice_model(model_id)
                        # Eliminar del caché local
                        if model_id in self.models:
                            del self.models[model_id]
                    except Exception as e:
                        logger.error(f"Error deleting model {model_id}: {str(e)}")
                        
            # Eliminar perfil de la base de conocimiento
            self.knowledge_base.delete_voice_profile(voice_id)
            
            logger.info(f"Deleted voice profile: {voice_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting voice profile: {str(e)}")
            return False
            
    def delete_voice_model(self, model_id: str, delete_files: bool = True) -> bool:
        """
        Elimina un modelo de voz
        
        Args:
            model_id: ID del modelo de voz
            delete_files: Si se deben eliminar los archivos del modelo
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        try:
            # Obtener modelo
            model = self.get_voice_model(model_id)
            if not model:
                logger.warning(f"Voice model not found: {model_id}")
                return False
                
            # Eliminar archivos si se solicita
            if delete_files and model.get("model_path") and os.path.exists(model["model_path"]):
                try:
                    # Eliminar archivo de modelo
                    os.remove(model["model_path"])
                    
                    # Eliminar archivo de configuración si existe
                    if model.get("config_path") and os.path.exists(model["config_path"]):
                        os.remove(model["config_path"])
                        
                    logger.info(f"Deleted model files for {model_id}")
                except Exception as e:
                    logger.error(f"Error deleting model files: {str(e)}")
                    
            # Actualizar perfil de voz
            voice_id = model.get("voice_id")
            if voice_id:
                profile = self.knowledge_base.get_voice_profile(voice_id)
                if profile:
                    # Eliminar modelo de la lista de modelos del perfil
                    profile["models"] = [m for m in profile.get("models", []) 
                                       if m.get("id") != model_id]
                    self.knowledge_base.save_voice_profile(voice_id, profile)
                    
            # Eliminar de la base de conocimiento
            self.knowledge_base.delete_voice_model(model_id)
            
            # Eliminar del caché local
            if model_id in self.models:
                del self.models[model_id]
                
            logger.info(f"Deleted voice model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting voice model: {str(e)}")
            return False
            
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Obtiene la duración de un archivo de audio en segundos
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            Duración en segundos
        """
        try:
            info = sf.info(audio_path)
            return info.duration
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            return 0.0
            
    def _convert_to_wav(self, input_path: str, output_path: str) -> bool:
        """
        Convierte un archivo de audio a formato WAV
        
        Args:
            input_path: Ruta al archivo de audio de entrada
            output_path: Ruta al archivo WAV de salida
            
        Returns:
            True si la conversión fue exitosa, False en caso contrario
        """
        try:
            # Leer audio con soundfile
            data, samplerate = sf.read(input_path)
            
            # Guardar como WAV
            sf.write(output_path, data, samplerate, subtype='PCM_16')
            
            return True
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            return False
            
    def generate_speech(self, text: str, model_id: str, 
                      output_path: str = None,
                      voice_settings: Dict[str, Any] = None) -> Optional[str]:
        """
        Genera voz a partir de texto usando un modelo entrenado
        
        Args:
            text: Texto a convertir en voz
            model_id: ID del modelo de voz a utilizar
            output_path: Ruta donde guardar el archivo de audio (opcional)
            voice_settings: Configuración adicional para la generación (opcional)
            
        Returns:
            Ruta al archivo de audio generado o None si hubo un error
        """
        try:
            # Obtener modelo
            model = self.get_voice_model(model_id)
            if not model:
                raise ValueError(f"Voice model not found: {model_id}")
                
            # Verificar que el modelo existe
            if not os.path.exists(model.get("model_path", "")):
                raise ValueError(f"Model file not found: {model.get('model_path')}")
                
            # Generar nombre de archivo si no se proporciona
            if not output_path:
                output_dir = os.path.join("outputs", "speech")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(
                    output_dir, 
                    f"speech_{int(time.time())}_{random.randint(1000, 9999)}.wav"
                )
                
            # Configuración por defecto
            settings = {
                "speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0
            }
            
            # Actualizar con configuración proporcionada
            if voice_settings:
                settings.update(voice_settings)
                
            # Generar voz según tipo de modelo
            model_type = model.get("model_type")
            
            if model_type == "xtts":
                return self._generate_speech_xtts(text, model, output_path, settings)
            elif model_type == "rvc":
                return self._generate_speech_rvc(text, model, output_path, settings)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return None
            
    def _generate_speech_xtts(self, text: str, model: Dict[str, Any], 
                            output_path: str, settings: Dict[str, Any]) -> str:
        """
        Genera voz usando un modelo XTTS
        
        En una implementación real, aquí se llamaría a la biblioteca de XTTS.
        """
        try:
            # Simular generación de voz
            logger.info(f"Generating speech with XTTS model: {model['name']}")
            
            # Crear archivo de audio simulado
            # En una implementación real, aquí se llamaría a la API de XTTS
            with open(output_path, "wb") as f:
                f.write(b"XTTS_AUDIO_PLACEHOLDER")
                
            logger.info(f"Generated speech saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech with XTTS: {str(e)}")
            raise
            
    def _generate_speech_rvc(self, text: str, model: Dict[str, Any], 
                           output_path: str, settings: Dict[str, Any]) -> str:
        """
        Genera voz usando un modelo RVC
        
        En una implementación real, aquí se llamaría a la biblioteca de RVC.
        """
        try:
            # Simular generación de voz
            logger.info(f"Generating speech with RVC model: {model['name']}")
            
            # Crear archivo de audio simulado
            # En una implementación real, aquí se llamaría a la API de RVC
            with open(output_path, "wb") as f:
                f.write(b"RVC_AUDIO_PLACEHOLDER")
                
            logger.info(f"Generated speech saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech with RVC: {str(e)}")
            raise
            
    def cleanup_old_jobs(self, days: int = None) -> int:
        """
        Limpia trabajos de entrenamiento antiguos
        
        Args:
            days: Número de días para considerar un trabajo como antiguo
                 (por defecto usa la configuración)
            
        Returns:
            Número de trabajos eliminados
        """
        try:
            # Usar configuración por defecto si no se especifica
            if days is None:
                days = self.config.get("cleanup_days", 30)
                
            # Calcular tiempo límite
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            # Identificar trabajos antiguos
            old_jobs = [
                job_id for job_id, job in self.training_jobs.items()
                if job.get("completed_at", 0) < cutoff_time or
                   job.get("created_at", 0) < cutoff_time
            ]
            
            # Eliminar trabajos
            for job_id in old_jobs:
                del self.training_jobs[job_id]
                
            logger.info(f"Cleaned up {len(old_jobs)} old training jobs")
            return len(old_jobs)
            
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {str(e)}")
            return 0