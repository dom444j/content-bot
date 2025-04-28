"""
Audio Engine - Generador de audio para contenido multimedia

Este módulo gestiona la creación de pistas de audio para videos:
- Generación de locuciones a partir de guiones
- Aplicación de efectos de sonido y música de fondo
- Procesamiento y optimización de audio
- Integración con servicios de voz (ElevenLabs, Piper TTS, XTTS/RVC)
"""

import os
import sys
import json
import logging
import random
import time
import requests
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
        logging.FileHandler(os.path.join('logs', 'audio_engine.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('AudioEngine')

class AudioEngine:
    """
    Clase para gestionar la generación y procesamiento de audio para contenido.
    """

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.config = self._load_config()
        self.cache = {}
        logger.info("AudioEngine initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración de servicios de audio"""
        try:
            # Intentar cargar desde la base de conocimiento
            config = self.knowledge_base.get_audio_services_config()
            if not config:
                # Configuración por defecto
                config = {
                    "default_service": "piper",
                    "services": {
                        "elevenlabs": {
                            "api_key": "",
                            "base_url": "https://api.elevenlabs.io/v1",
                            "active": False,
                            "free_characters_per_month": 10000,
                            "cost_per_month": 5
                        },
                        "piper": {
                            "model_path": "models/piper",
                            "active": True,
                            "local": True,
                            "cost_per_month": 0
                        },
                        "xtts": {
                            "model_path": "models/xtts",
                            "active": False,
                            "local": True,
                            "cost_per_month": 0
                        },
                        "rvc": {
                            "model_path": "models/rvc",
                            "active": False,
                            "local": True,
                            "cost_per_month": 0
                        }
                    },
                    "cache_enabled": True,
                    "cache_expiry_days": 30,
                    "default_format": "wav",
                    "default_sample_rate": 44100,
                    "background_music": {
                        "enabled": True,
                        "volume": 0.2,
                        "library_path": "datasets/audio/background_music"
                    },
                    "sound_effects": {
                        "enabled": True,
                        "volume": 0.5,
                        "library_path": "datasets/audio/sound_effects"
                    }
                }
                # Guardar en la base de conocimiento
                self.knowledge_base.save_audio_services_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading audio services config: {str(e)}")
            # Retornar configuración mínima
            return {
                "default_service": "piper",
                "services": {
                    "piper": {
                        "active": True,
                        "local": True
                    }
                },
                "cache_enabled": True
            }

    def generate_speech(self, text: str, voice_id: str = None, 
                       character_id: str = None, 
                       service: str = None) -> Dict[str, Any]:
        """
        Genera una locución a partir de texto
        
        Args:
            text: Texto a convertir en voz
            voice_id: ID de la voz a utilizar (opcional)
            character_id: ID del personaje (opcional)
            service: Servicio de generación a utilizar (opcional)
            
        Returns:
            Diccionario con información del audio generado
        """
        try:
            # Determinar servicio a utilizar
            if not service:
                service = self._select_service()
                
            # Determinar voz a utilizar
            if not voice_id and character_id:
                # Obtener voz asociada al personaje
                character = self.knowledge_base.get_character(character_id)
                if character and "voice_id" in character:
                    voice_id = character["voice_id"]
                    
            # Si aún no hay voice_id, usar uno por defecto
            if not voice_id:
                voice_id = self._get_default_voice_id(service)
                
            # Verificar caché
            cache_key = f"speech_{service}_{voice_id}_{hash(text)}"
            if self.config.get("cache_enabled", True) and cache_key in self.cache:
                logger.info(f"Using cached speech for text: {text[:20]}...")
                return self.cache[cache_key]
                
            # Generar audio según el servicio
            if service == "elevenlabs":
                audio_path = self._generate_with_elevenlabs(text, voice_id)
            elif service == "piper":
                audio_path = self._generate_with_piper(text, voice_id)
            elif service == "xtts":
                audio_path = self._generate_with_xtts(text, voice_id)
            elif service == "rvc":
                # RVC necesita audio base, generamos con Piper primero
                base_audio_path = self._generate_with_piper(text, self._get_default_voice_id("piper"))
                audio_path = self._process_with_rvc(base_audio_path, voice_id)
            else:
                # Servicio no implementado, usar piper
                logger.warning(f"Service {service} not implemented, falling back to piper")
                audio_path = self._generate_with_piper(text, self._get_default_voice_id("piper"))
                
            # Crear resultado
            result = {
                "path": audio_path,
                "text": text,
                "voice_id": voice_id,
                "service": service,
                "duration": self._get_audio_duration(audio_path),
                "created_at": time.time()
            }
            
            # Guardar en caché
            if self.config.get("cache_enabled", True):
                self.cache[cache_key] = result
                
            logger.info(f"Generated speech for text: {text[:20]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            # Retornar resultado vacío
            return {
                "path": "",
                "text": text,
                "voice_id": voice_id or "",
                "service": service or "",
                "duration": 0,
                "created_at": time.time(),
                "error": str(e)
            }

    def _select_service(self) -> str:
        """Selecciona el servicio de generación de voz a utilizar"""
        services = self.config.get("services", {})
        
        # Usar el servicio por defecto si está activo
        default_service = self.config.get("default_service", "piper")
        if default_service in services and services[default_service].get("active", False):
            return default_service
            
        # Buscar cualquier servicio activo
        for service_name, service_config in services.items():
            if service_config.get("active", False):
                return service_name
                
        # Si no hay servicios activos, usar piper (asumiendo local)
        logger.warning("No active voice services found, defaulting to piper")
        return "piper"

    def _get_default_voice_id(self, service: str) -> str:
        """Obtiene un ID de voz por defecto para el servicio especificado"""
        if service == "elevenlabs":
            return "21m00Tcm4TlvDq8ikWAM"  # Rachel (voz por defecto de ElevenLabs)
        elif service == "piper":
            return "en_US-lessac-medium"  # Voz en inglés de Piper
        elif service == "xtts":
            return "default"  # Voz por defecto de XTTS
        elif service == "rvc":
            return "default"  # Modelo por defecto de RVC
        else:
            return "default"

    def _generate_with_elevenlabs(self, text: str, voice_id: str) -> str:
        """
        Genera audio con ElevenLabs
        
        Args:
            text: Texto a convertir en voz
            voice_id: ID de la voz a utilizar
            
        Returns:
            Ruta al archivo de audio generado
        """
        try:
            # Obtener configuración de ElevenLabs
            service_config = self.config.get("services", {}).get("elevenlabs", {})
            api_key = service_config.get("api_key", "")
            base_url = service_config.get("base_url", "https://api.elevenlabs.io/v1")
            
            if not api_key:
                logger.error("ElevenLabs API key not configured")
                raise ValueError("ElevenLabs API key not configured")
                
            # Configurar headers
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            
            # Crear payload
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            # Realizar solicitud
            response = requests.post(
                f"{base_url}/text-to-speech/{voice_id}",
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                logger.error(f"Error from ElevenLabs API: {response.text}")
                raise ValueError(f"ElevenLabs API error: {response.text}")
                
            # Crear directorio para guardar el audio
            output_dir = os.path.join("data", "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo único
            filename = f"elevenlabs_{voice_id}_{int(time.time())}.mp3"
            output_path = os.path.join(output_dir, filename)
            
            # Guardar audio
            with open(output_path, "wb") as f:
                f.write(response.content)
                
            logger.info(f"Generated audio with ElevenLabs: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating with ElevenLabs: {str(e)}")
            raise

    def _generate_with_piper(self, text: str, voice_id: str) -> str:
        """
        Genera audio con Piper TTS
        
        Args:
            text: Texto a convertir en voz
            voice_id: ID de la voz a utilizar
            
        Returns:
            Ruta al archivo de audio generado
        """
        try:
            # Verificar si hay implementación local
            service_config = self.config.get("services", {}).get("piper", {})
            model_path = service_config.get("model_path", "models/piper")
            
            # Crear directorio para guardar el audio
            output_dir = os.path.join("data", "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo único
            filename = f"piper_{voice_id}_{int(time.time())}.wav"
            output_path = os.path.join(output_dir, filename)
            
            # Aquí iría la implementación real con la biblioteca de Piper
            # Por ahora, simular generación
            logger.info(f"Generating audio with Piper using voice: {voice_id}")
            
            # Simular generación (en implementación real, aquí se llamaría a Piper)
            # Crear un archivo de audio vacío como placeholder
            self._create_placeholder_audio(output_path, text)
            
            logger.info(f"Generated audio with Piper: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating with Piper: {str(e)}")
            raise

    def _generate_with_xtts(self, text: str, voice_id: str) -> str:
        """
        Genera audio con XTTS
        
        Args:
            text: Texto a convertir en voz
            voice_id: ID de la voz a utilizar (referencia a muestra de voz)
            
        Returns:
            Ruta al archivo de audio generado
        """
        try:
            # Verificar si hay implementación local
            service_config = self.config.get("services", {}).get("xtts", {})
            model_path = service_config.get("model_path", "models/xtts")
            
            # Crear directorio para guardar el audio
            output_dir = os.path.join("data", "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo único
            filename = f"xtts_{voice_id}_{int(time.time())}.wav"
            output_path = os.path.join(output_dir, filename)
            
            # Aquí iría la implementación real con la biblioteca de XTTS
            # Por ahora, simular generación
            logger.info(f"Generating audio with XTTS using voice: {voice_id}")
            
            # Simular generación (en implementación real, aquí se llamaría a XTTS)
            # Crear un archivo de audio vacío como placeholder
            self._create_placeholder_audio(output_path, text)
            
            logger.info(f"Generated audio with XTTS: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating with XTTS: {str(e)}")
            raise

    def _process_with_rvc(self, audio_path: str, voice_id: str) -> str:
        """
        Procesa un audio con RVC para cambiar la voz
        
        Args:
            audio_path: Ruta al archivo de audio a procesar
            voice_id: ID del modelo de voz RVC a utilizar
            
        Returns:
            Ruta al archivo de audio procesado
        """
        try:
            # Verificar si hay implementación local
            service_config = self.config.get("services", {}).get("rvc", {})
            model_path = service_config.get("model_path", "models/rvc")
            
            # Crear directorio para guardar el audio
            output_dir = os.path.join("data", "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo único
            filename = f"rvc_{voice_id}_{int(time.time())}.wav"
            output_path = os.path.join(output_dir, filename)
            
            # Aquí iría la implementación real con la biblioteca de RVC
            # Por ahora, simular procesamiento
            logger.info(f"Processing audio with RVC using voice model: {voice_id}")
            
            # Simular procesamiento (en implementación real, aquí se llamaría a RVC)
            # Copiar el archivo de audio original como placeholder
            import shutil
            shutil.copy(audio_path, output_path)
            
            logger.info(f"Processed audio with RVC: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing with RVC: {str(e)}")
            raise

    def add_background_music(self, audio_path: str, 
                           music_path: str = None, 
                           volume: float = None) -> str:
        """
        Añade música de fondo a un audio
        
        Args:
            audio_path: Ruta al archivo de audio principal
            music_path: Ruta al archivo de música (opcional)
            volume: Volumen de la música (0.0-1.0, opcional)
            
        Returns:
            Ruta al archivo de audio con música
        """
        try:
            # Verificar configuración
            bg_config = self.config.get("background_music", {})
            enabled = bg_config.get("enabled", True)
            
            if not enabled:
                logger.info("Background music disabled, returning original audio")
                return audio_path
                
            # Usar volumen de configuración si no se especifica
            if volume is None:
                volume = bg_config.get("volume", 0.2)
                
            # Seleccionar música aleatoria si no se especifica
            if not music_path:
                library_path = bg_config.get("library_path", "datasets/audio/background_music")
                music_files = self._get_audio_files_in_directory(library_path)
                
                if not music_files:
                    logger.warning(f"No background music files found in {library_path}")
                    return audio_path
                    
                music_path = random.choice(music_files)
                
            # Crear directorio para guardar el audio
            output_dir = os.path.join("data", "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo único
            filename = f"mixed_{os.path.basename(audio_path)}"
            output_path = os.path.join(output_dir, filename)
            
            # Aquí iría la implementación real de mezcla de audio
            # Por ahora, simular procesamiento
            logger.info(f"Adding background music to {audio_path} with volume {volume}")
            
            # Simular procesamiento (en implementación real, aquí se mezclarían los audios)
            # Copiar el archivo de audio original como placeholder
            import shutil
            shutil.copy(audio_path, output_path)
            
            logger.info(f"Added background music: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding background music: {str(e)}")
            return audio_path

    def add_sound_effect(self, audio_path: str, 
                        effect_path: str = None, 
                        position: float = 0.0,
                        volume: float = None) -> str:
        """
        Añade un efecto de sonido a un audio
        
        Args:
            audio_path: Ruta al archivo de audio principal
            effect_path: Ruta al archivo de efecto (opcional)
            position: Posición relativa en el audio (0.0-1.0)
            volume: Volumen del efecto (0.0-1.0, opcional)
            
        Returns:
            Ruta al archivo de audio con efecto
        """
        try:
            # Verificar configuración
            fx_config = self.config.get("sound_effects", {})
            enabled = fx_config.get("enabled", True)
            
            if not enabled:
                logger.info("Sound effects disabled, returning original audio")
                return audio_path
                
            # Usar volumen de configuración si no se especifica
            if volume is None:
                volume = fx_config.get("volume", 0.5)
                
            # Seleccionar efecto aleatorio si no se especifica
            if not effect_path:
                library_path = fx_config.get("library_path", "datasets/audio/sound_effects")
                effect_files = self._get_audio_files_in_directory(library_path)
                
                if not effect_files:
                    logger.warning(f"No sound effect files found in {library_path}")
                    return audio_path
                    
                effect_path = random.choice(effect_files)
                
            # Crear directorio para guardar el audio
            output_dir = os.path.join("data", "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo único
            filename = f"fx_{os.path.basename(audio_path)}"
            output_path = os.path.join(output_dir, filename)
            
            # Aquí iría la implementación real de adición de efectos
            # Por ahora, simular procesamiento
            logger.info(f"Adding sound effect to {audio_path} at position {position} with volume {volume}")
            
            # Simular procesamiento (en implementación real, aquí se añadiría el efecto)
            # Copiar el archivo de audio original como placeholder
            import shutil
            shutil.copy(audio_path, output_path)
            
            logger.info(f"Added sound effect: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding sound effect: {str(e)}")
            return audio_path

    def process_audio(self, audio_path: str, 
                     normalize: bool = True,
                     remove_silence: bool = True,
                     format: str = None) -> str:
        """
        Procesa un archivo de audio para optimizarlo
        
        Args:
            audio_path: Ruta al archivo de audio a procesar
            normalize: Si se debe normalizar el volumen
            remove_silence: Si se deben eliminar silencios
            format: Formato de salida (opcional)
            
        Returns:
            Ruta al archivo de audio procesado
        """
        try:
            # Usar formato de configuración si no se especifica
            if not format:
                format = self.config.get("default_format", "wav")
                
            # Crear directorio para guardar el audio
            output_dir = os.path.join("data", "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo único
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            filename = f"proc_{base_name}.{format}"
            output_path = os.path.join(output_dir, filename)
            
            # Aquí iría la implementación real de procesamiento de audio
            # Por ahora, simular procesamiento
            logger.info(f"Processing audio {audio_path} (normalize={normalize}, remove_silence={remove_silence})")
            
            # Simular procesamiento (en implementación real, aquí se procesaría el audio)
            # Copiar el archivo de audio original como placeholder
            import shutil
            shutil.copy(audio_path, output_path)
            
            logger.info(f"Processed audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return audio_path

    def concatenate_audio_files(self, audio_paths: List[str], 
                              crossfade: float = 0.0) -> str:
        """
        Concatena múltiples archivos de audio
        
        Args:
            audio_paths: Lista de rutas a archivos de audio
            crossfade: Duración del crossfade entre archivos (segundos)
            
        Returns:
            Ruta al archivo de audio concatenado
        """
        try:
            if not audio_paths:
                raise ValueError("No audio paths provided")
                
            # Crear directorio para guardar el audio
            output_dir = os.path.join("data", "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar nombre de archivo único
            filename = f"concat_{int(time.time())}.wav"
            output_path = os.path.join(output_dir, filename)
            
            # Aquí iría la implementación real de concatenación de audio
            # Por ahora, simular procesamiento
            logger.info(f"Concatenating {len(audio_paths)} audio files with crossfade={crossfade}s")
            
            # Simular procesamiento (en implementación real, aquí se concatenarían los audios)
            # Copiar el primer archivo de audio como placeholder
            import shutil
            shutil.copy(audio_paths[0], output_path)
            
            logger.info(f"Concatenated audio: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error concatenating audio files: {str(e)}")
            if audio_paths:
                return audio_paths[0]
            else:
                return ""

    def _get_audio_files_in_directory(self, directory: str) -> List[str]:
        """Obtiene una lista de archivos de audio en un directorio"""
        try:
            audio_extensions = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]
            audio_files = []
            
            if not os.path.exists(directory):
                logger.warning(f"Directory does not exist: {directory}")
                return []
                
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_files.append(os.path.join(root, file))
                        
            return audio_files
            
        except Exception as e:
            logger.error(f"Error getting audio files: {str(e)}")
            return []

    def _get_audio_duration(self, audio_path: str) -> float:
        """Obtiene la duración de un archivo de audio en segundos"""
        try:
            # Aquí iría la implementación real para obtener la duración
            # Por ahora, simular duración
            return 5.0  # Duración simulada de 5 segundos
            
        except Exception as e:
            logger.error(f"Error getting audio duration: {str(e)}")
            return 0.0

    def _create_placeholder_audio(self, path: str, text: str = "") -> None:
        """
        Crea un archivo de audio de placeholder
        
        Args:
            path: Ruta donde guardar el audio
            text: Texto asociado al audio (para logging)
        """
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Crear un archivo de audio vacío
            sample_rate = self.config.get("default_sample_rate", 44100)
            duration = min(len(text) * 0.1, 10.0)  # Aproximadamente 0.1s por carácter, máximo 10s
            duration = max(duration, 1.0)  # Mínimo 1s
            
            # Generar un tono simple como placeholder
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = 0.5 * np.sin(2 * np.pi * 440 * t)  # Tono A4 (440 Hz)
            
            # Guardar como archivo WAV
            sf.write(path, tone, sample_rate)
            
            logger.info(f"Created placeholder audio at {path}")
            
        except Exception as e:
            logger.error(f"Error creating placeholder audio: {str(e)}")

    def clear_cache(self) -> None:
        """Limpia la caché de audio"""
        self.cache = {}
        logger.info("Cleared audio cache")

    def get_service_status(self) -> Dict[str, bool]:
        """
        Obtiene el estado de los servicios de generación de audio
        
        Returns:
            Diccionario con el estado de cada servicio
        """
        services = self.config.get("services", {})
        status = {}
        
        for service_name, service_config in services.items():
            status[service_name] = service_config.get("active", False)
            
        return status

    def update_service_config(self, service_name: str, config_updates: Dict[str, Any]) -> bool:
        """
        Actualiza la configuración de un servicio
        
        Args:
            service_name: Nombre del servicio
            config_updates: Actualizaciones de configuración
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        try:
            services = self.config.get("services", {})
            
            if service_name not in services:
                logger.warning(f"Service not found: {service_name}")
                return False
                
            # Aplicar actualizaciones
            for key, value in config_updates.items():
                services[service_name][key] = value
                
            # Actualizar configuración
            self.config["services"] = services
            
            # Guardar en la base de conocimiento
            self.knowledge_base.save_audio_services_config(self.config)
            
            logger.info(f"Updated service config: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating service config: {str(e)}")
            return False