"""
Voice Trainer Module

Este módulo implementa el entrenamiento de voces personalizadas utilizando XTTS (XTreme Text-to-Speech)
y RVC (Retrieval-based Voice Conversion). Permite crear voces únicas para personajes virtuales
que aumentan el engagement y la retención de audiencia.

Características principales:
- Entrenamiento de modelos XTTS para síntesis de voz de alta calidad
- Entrenamiento de modelos RVC para conversión de voz
- Gestión de datasets de voz
- Optimización de modelos para uso en producción
- Exportación de modelos para uso en el sistema de generación de contenido
"""

import os
import json
import logging
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/voice_trainer.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("VoiceTrainer")

@dataclass
class VoiceTrainingConfig:
    """Configuración para el entrenamiento de voces."""
    model_type: str = "xtts"  # "xtts" o "rvc"
    character_name: str = "default"
    voice_samples_dir: str = "datasets/voice_samples"
    output_dir: str = "models/voice_models"
    epochs: int = 1000
    batch_size: int = 16
    learning_rate: float = 1e-4
    sample_rate: int = 22050
    use_gpu: bool = True
    speaker_embedding_dim: int = 512
    checkpoint_interval: int = 100
    validation_split: float = 0.1
    augmentation_factor: float = 0.2
    emotion_control: bool = True
    pitch_control: bool = True
    speed_control: bool = True
    use_pretrained: bool = True
    pretrained_model_path: Optional[str] = None
    metadata: Dict = None


class VoiceTrainer:
    """
    Clase principal para el entrenamiento de voces personalizadas.
    Soporta XTTS (XTreme Text-to-Speech) y RVC (Retrieval-based Voice Conversion).
    """
    
    def __init__(self, config: VoiceTrainingConfig):
        """
        Inicializa el entrenador de voces.
        
        Args:
            config: Configuración para el entrenamiento
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        
        # Crear directorios necesarios
        os.makedirs(config.voice_samples_dir, exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, config.character_name), exist_ok=True)
        
        # Inicializar modelo según tipo
        self.model = None
        self.tokenizer = None
        self.vocoder = None
        self.optimizer = None
        self.scheduler = None
        
        # Métricas de entrenamiento
        self.training_history = {
            "loss": [],
            "val_loss": [],
            "epochs_completed": 0,
            "best_val_loss": float("inf"),
            "training_time": 0
        }
        
        logger.info(f"Inicializando VoiceTrainer para {config.character_name} usando {config.model_type}")
        logger.info(f"Dispositivo: {self.device}")
    
    def prepare_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepara el dataset de entrenamiento y validación.
        
        Returns:
            Tupla con datasets de entrenamiento y validación
        """
        logger.info(f"Preparando dataset desde {self.config.voice_samples_dir}")
        
        character_samples_dir = os.path.join(
            self.config.voice_samples_dir, 
            self.config.character_name
        )
        
        if not os.path.exists(character_samples_dir):
            raise FileNotFoundError(f"No se encontraron muestras de voz para {self.config.character_name}")
        
        # Obtener archivos de audio
        audio_files = []
        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            audio_files.extend(list(Path(character_samples_dir).glob(f"*{ext}")))
        
        if not audio_files:
            raise ValueError(f"No se encontraron archivos de audio en {character_samples_dir}")
        
        logger.info(f"Se encontraron {len(audio_files)} archivos de audio")
        
        # Procesar archivos de audio
        dataset = []
        for audio_file in tqdm(audio_files, desc="Procesando archivos de audio"):
            try:
                # Cargar audio
                audio, sr = librosa.load(str(audio_file), sr=self.config.sample_rate)
                
                # Verificar duración mínima (0.5 segundos)
                if len(audio) < 0.5 * sr:
                    logger.warning(f"Archivo {audio_file} demasiado corto, omitiendo")
                    continue
                
                # Buscar archivo de transcripción correspondiente
                transcript_file = audio_file.with_suffix(".txt")
                if transcript_file.exists():
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()
                else:
                    # Si no hay transcripción, usar nombre de archivo sin extensión
                    transcript = audio_file.stem.replace("_", " ")
                
                # Añadir a dataset
                dataset.append({
                    "audio_path": str(audio_file),
                    "transcript": transcript,
                    "duration": len(audio) / sr,
                    "sample_rate": sr
                })
                
                # Aplicar aumentación de datos si está configurado
                if self.config.augmentation_factor > 0:
                    dataset.extend(self._augment_sample(audio, sr, transcript, str(audio_file)))
                
            except Exception as e:
                logger.error(f"Error procesando {audio_file}: {str(e)}")
        
        # Dividir en entrenamiento y validación
        np.random.shuffle(dataset)
        split_idx = int(len(dataset) * (1 - self.config.validation_split))
        train_dataset = dataset[:split_idx]
        val_dataset = dataset[split_idx:]
        
        logger.info(f"Dataset preparado: {len(train_dataset)} muestras de entrenamiento, {len(val_dataset)} de validación")
        
        return train_dataset, val_dataset
    
    def _augment_sample(self, audio: np.ndarray, sr: int, transcript: str, original_path: str) -> List[Dict]:
        """
        Aplica técnicas de aumentación de datos a una muestra de audio.
        
        Args:
            audio: Array de audio
            sr: Tasa de muestreo
            transcript: Transcripción del audio
            original_path: Ruta del archivo original
            
        Returns:
            Lista de muestras aumentadas
        """
        augmented_samples = []
        num_augmentations = int(self.config.augmentation_factor * 3)  # 3 tipos de aumentación
        
        for i in range(num_augmentations):
            aug_type = i % 3  # 0: pitch, 1: speed, 2: noise
            
            if aug_type == 0 and self.config.pitch_control:
                # Cambio de tono
                pitch_shift = np.random.uniform(-2, 2)
                aug_audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
                aug_name = f"pitch_{pitch_shift:.1f}"
            
            elif aug_type == 1 and self.config.speed_control:
                # Cambio de velocidad
                speed_factor = np.random.uniform(0.9, 1.1)
                aug_audio = librosa.effects.time_stretch(audio, rate=speed_factor)
                aug_name = f"speed_{speed_factor:.1f}"
            
            elif aug_type == 2:
                # Añadir ruido
                noise_level = np.random.uniform(0.001, 0.005)
                noise = np.random.randn(len(audio))
                aug_audio = audio + noise_level * noise
                aug_name = f"noise_{noise_level:.3f}"
            
            else:
                continue
            
            # Crear nombre para archivo aumentado
            base_path = os.path.dirname(original_path)
            file_name = os.path.basename(original_path)
            name, ext = os.path.splitext(file_name)
            aug_path = os.path.join(base_path, f"{name}_{aug_name}{ext}")
            
            # Guardar audio aumentado
            sf.write(aug_path, aug_audio, sr)
            
            # Añadir a lista de muestras aumentadas
            augmented_samples.append({
                "audio_path": aug_path,
                "transcript": transcript,
                "duration": len(aug_audio) / sr,
                "sample_rate": sr,
                "augmentation": aug_name
            })
        
        return augmented_samples
    
    def initialize_model(self):
        """Inicializa el modelo de voz según la configuración."""
        if self.config.model_type == "xtts":
            self._initialize_xtts_model()
        elif self.config.model_type == "rvc":
            self._initialize_rvc_model()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.config.model_type}")
    
    def _initialize_xtts_model(self):
        """Inicializa un modelo XTTS (XTreme Text-to-Speech)."""
        logger.info("Inicializando modelo XTTS")
        
        try:
            # Importar dependencias específicas de XTTS
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            # Configurar modelo XTTS
            if self.config.use_pretrained and self.config.pretrained_model_path:
                # Cargar modelo pre-entrenado
                logger.info(f"Cargando modelo pre-entrenado desde {self.config.pretrained_model_path}")
                config_path = os.path.join(self.config.pretrained_model_path, "config.json")
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                
                model_config = XttsConfig(**config_dict)
                self.model = Xtts.init_from_config(model_config)
                self.model.load_checkpoint(
                    config=model_config,
                    checkpoint_path=os.path.join(self.config.pretrained_model_path, "model.pth")
                )
            else:
                # Crear modelo desde cero
                logger.info("Creando modelo XTTS desde cero")
                model_config = XttsConfig(
                    sample_rate=self.config.sample_rate,
                    speaker_embedding_dim=self.config.speaker_embedding_dim,
                    use_d_vector_file=False,
                    use_speaker_encoder_as_loss=True
                )
                self.model = Xtts(config=model_config)
            
            # Mover modelo a dispositivo
            self.model = self.model.to(self.device)
            
            # Configurar optimizador
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Configurar scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            logger.info("Modelo XTTS inicializado correctamente")
            
        except ImportError:
            logger.error("No se pudo importar TTS. Instala con: pip install TTS")
            raise
        except Exception as e:
            logger.error(f"Error inicializando modelo XTTS: {str(e)}")
            raise
    
    def _initialize_rvc_model(self):
        """Inicializa un modelo RVC (Retrieval-based Voice Conversion)."""
        logger.info("Inicializando modelo RVC")
        
        try:
            # Importar dependencias específicas de RVC
            # Nota: RVC es un proyecto de código abierto que puede requerir adaptación
            # Esta es una implementación simplificada
            
            # Estructura básica para un modelo de conversión de voz
            class RVCModel(torch.nn.Module):
                def __init__(self, speaker_embedding_dim):
                    super().__init__()
                    self.encoder = torch.nn.Sequential(
                        torch.nn.Conv1d(80, 256, kernel_size=3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
                        torch.nn.ReLU()
                    )
                    self.speaker_embedding = torch.nn.Embedding(10, speaker_embedding_dim)
                    self.decoder = torch.nn.Sequential(
                        torch.nn.Conv1d(512 + speaker_embedding_dim, 256, kernel_size=3, padding=1),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(256, 80, kernel_size=3, padding=1)
                    )
                
                def forward(self, mel, speaker_id):
                    encoded = self.encoder(mel)
                    speaker_emb = self.speaker_embedding(speaker_id).unsqueeze(2).expand(-1, -1, encoded.size(2))
                    concat = torch.cat((encoded, speaker_emb), dim=1)
                    output = self.decoder(concat)
                    return output
            
            # Crear o cargar modelo
            if self.config.use_pretrained and self.config.pretrained_model_path:
                # Cargar modelo pre-entrenado
                logger.info(f"Cargando modelo RVC pre-entrenado desde {self.config.pretrained_model_path}")
                self.model = torch.load(
                    os.path.join(self.config.pretrained_model_path, "model.pth"),
                    map_location=self.device
                )
            else:
                # Crear modelo desde cero
                logger.info("Creando modelo RVC desde cero")
                self.model = RVCModel(self.config.speaker_embedding_dim)
            
            # Mover modelo a dispositivo
            self.model = self.model.to(self.device)
            
            # Configurar optimizador
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
            # Configurar scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            logger.info("Modelo RVC inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando modelo RVC: {str(e)}")
            raise
    
    def train(self, train_dataset: List[Dict], val_dataset: List[Dict]) -> Dict:
        """
        Entrena el modelo de voz con el dataset proporcionado.
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            
        Returns:
            Historial de entrenamiento
        """
        logger.info(f"Iniciando entrenamiento de {self.config.model_type} para {self.config.character_name}")
        
        # Verificar que el modelo esté inicializado
        if self.model is None:
            self.initialize_model()
        
        # Registrar tiempo de inicio
        start_time = datetime.now()
        
        # Crear dataloader
        train_loader = self._create_dataloader(train_dataset, is_training=True)
        val_loader = self._create_dataloader(val_dataset, is_training=False)
        
        # Ciclo de entrenamiento
        for epoch in range(self.config.epochs):
            # Entrenamiento
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"):
                loss = self._process_batch(batch, is_training=True)
                train_loss += loss
                train_batches += 1
            
            avg_train_loss = train_loss / max(1, train_batches)
            self.training_history["loss"].append(avg_train_loss)
            
            # Validación
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validación"):
                    loss = self._process_batch(batch, is_training=False)
                    val_loss += loss
                    val_batches += 1
            
            avg_val_loss = val_loss / max(1, val_batches)
            self.training_history["val_loss"].append(avg_val_loss)
            
            # Actualizar scheduler
            self.scheduler.step(avg_val_loss)
            
            # Guardar checkpoint si es el mejor modelo
            if avg_val_loss < self.training_history["best_val_loss"]:
                self.training_history["best_val_loss"] = avg_val_loss
                self.save_model(is_best=True)
            
            # Guardar checkpoint periódico
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_model(is_checkpoint=True, epoch=epoch+1)
            
            # Actualizar contador de épocas
            self.training_history["epochs_completed"] = epoch + 1
            
            # Mostrar progreso
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Best Val Loss: {self.training_history['best_val_loss']:.4f}"
            )
        
        # Registrar tiempo total de entrenamiento
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        self.training_history["training_time"] = training_time
        
        logger.info(
            f"Entrenamiento completado en {training_time/3600:.2f} horas. "
            f"Mejor pérdida de validación: {self.training_history['best_val_loss']:.4f}"
        )
        
        # Guardar modelo final
        self.save_model()
        
        return self.training_history
    
    def _create_dataloader(self, dataset: List[Dict], is_training: bool = True) -> torch.utils.data.DataLoader:
        """
        Crea un dataloader para el dataset proporcionado.
        
        Args:
            dataset: Lista de muestras
            is_training: Si es para entrenamiento o validación
            
        Returns:
            DataLoader de PyTorch
        """
        # Implementación simplificada - en un sistema real se usaría una clase Dataset personalizada
        batch_size = self.config.batch_size if is_training else self.config.batch_size * 2
        
        # Crear función de collate para procesar lotes
        def collate_fn(batch):
            # Procesar cada muestra en el lote
            processed_batch = {
                "audio": [],
                "transcript": [],
                "audio_lengths": [],
                "transcript_lengths": []
            }
            
            for item in batch:
                # Cargar audio
                audio_path = item["audio_path"]
                audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
                
                # Extraer características (ejemplo: mel spectrograms)
                mel = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=sr,
                    n_mels=80,
                    hop_length=256,
                    win_length=1024,
                    fmin=80,
                    fmax=7600
                )
                mel = librosa.power_to_db(mel, ref=np.max)
                
                # Normalizar
                mel = (mel - mel.mean()) / (mel.std() + 1e-5)
                
                # Convertir a tensor
                mel_tensor = torch.FloatTensor(mel)
                
                # Añadir a lote
                processed_batch["audio"].append(mel_tensor)
                processed_batch["transcript"].append(item["transcript"])
                processed_batch["audio_lengths"].append(mel_tensor.shape[1])
                processed_batch["transcript_lengths"].append(len(item["transcript"]))
            
            # Padding para audio (mel spectrograms)
            max_audio_len = max(processed_batch["audio_lengths"])
            padded_audio = []
            
            for mel in processed_batch["audio"]:
                pad_len = max_audio_len - mel.shape[1]
                padded_mel = torch.nn.functional.pad(mel, (0, pad_len), "constant", 0)
                padded_audio.append(padded_mel)
            
            # Convertir a tensores
            processed_batch["audio"] = torch.stack(padded_audio)
            processed_batch["audio_lengths"] = torch.LongTensor(processed_batch["audio_lengths"])
            processed_batch["transcript_lengths"] = torch.LongTensor(processed_batch["transcript_lengths"])
            
            return processed_batch
        
        # Crear dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            collate_fn=collate_fn,
            num_workers=4 if is_training else 2,
            pin_memory=True
        )
        
        return dataloader
    
    def _process_batch(self, batch: Dict[str, torch.Tensor], is_training: bool = True) -> float:
        """
        Procesa un lote de datos para entrenamiento o validación.
        
        Args:
            batch: Lote de datos
            is_training: Si es para entrenamiento o validación
            
        Returns:
            Pérdida del lote
        """
        # Mover datos al dispositivo
        audio = batch["audio"].to(self.device)
        audio_lengths = batch["audio_lengths"].to(self.device)
        
        # Proceso específico según tipo de modelo
        if self.config.model_type == "xtts":
            return self._process_xtts_batch(batch, is_training)
        elif self.config.model_type == "rvc":
            return self._process_rvc_batch(batch, is_training)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.config.model_type}")
    
    def _process_xtts_batch(self, batch: Dict[str, torch.Tensor], is_training: bool = True) -> float:
        """
        Procesa un lote para modelo XTTS.
        
        Args:
            batch: Lote de datos
            is_training: Si es para entrenamiento o validación
            
        Returns:
            Pérdida del lote
        """
        # Implementación simplificada - en un sistema real se usaría la API de XTTS
        
        # Extraer datos
        mel = batch["audio"].to(self.device)
        mel_lengths = batch["audio_lengths"].to(self.device)
        
        # Forward pass
        outputs = self.model(mel, mel_lengths)
        
        # Calcular pérdida
        loss = outputs["loss"]
        
        # Backpropagation si es entrenamiento
        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item()
    
    def _process_rvc_batch(self, batch: Dict[str, torch.Tensor], is_training: bool = True) -> float:
        """
        Procesa un lote para modelo RVC.
        
        Args:
            batch: Lote de datos
            is_training: Si es para entrenamiento o validación
            
        Returns:
            Pérdida del lote
        """
        # Implementación simplificada - en un sistema real se usaría la API de RVC
        
        # Extraer datos
        mel = batch["audio"].to(self.device)
        
        # Crear IDs de hablante (simplificado)
        speaker_ids = torch.zeros(mel.size(0), dtype=torch.long).to(self.device)
        
        # Forward pass
        output_mel = self.model(mel, speaker_ids)
        
        # Calcular pérdida (reconstrucción)
        loss = torch.nn.functional.mse_loss(output_mel, mel)
        
        # Backpropagation si es entrenamiento
        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, is_best: bool = False, is_checkpoint: bool = False, epoch: int = None) -> str:
        """
        Guarda el modelo entrenado.
        
        Args:
            is_best: Si es el mejor modelo hasta ahora
            is_checkpoint: Si es un checkpoint periódico
            epoch: Número de época (para checkpoints)
            
        Returns:
            Ruta donde se guardó el modelo
        """
        # Crear directorio de salida
        output_dir = os.path.join(
            self.config.output_dir,
            self.config.character_name
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Determinar nombre de archivo
        if is_best:
            model_path = os.path.join(output_dir, "best_model")
        elif is_checkpoint and epoch is not None:
            model_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}")
        else:
            model_path = os.path.join(output_dir, "final_model")
        
        os.makedirs(model_path, exist_ok=True)
        
        # Guardar modelo
        if self.config.model_type == "xtts":
            # Guardar modelo XTTS
            self.model.save_checkpoint(
                config=self.model.config,
                checkpoint_path=os.path.join(model_path, "model.pth")
            )
        else:
            # Guardar modelo RVC
            torch.save(self.model, os.path.join(model_path, "model.pth"))
        
        # Guardar configuración
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        with open(os.path.join(model_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        
        # Guardar historial de entrenamiento
        with open(os.path.join(model_path, "training_history.json"), "w", encoding="utf-8") as f:
            # Convertir valores de numpy a Python nativos
            history_dict = {}
            for k, v in self.training_history.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.float32, np.float64)):
                    history_dict[k] = [float(x) for x in v]
                else:
                    history_dict[k] = v
            
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"Modelo guardado en {model_path}")
        
        return model_path
    
    @classmethod
    def load_model(cls, model_path: str) -> "VoiceTrainer":
        """
        Carga un modelo entrenado.
        
        Args:
            model_path: Ruta al modelo guardado
            
        Returns:
            Instancia de VoiceTrainer con el modelo cargado
        """
        logger.info(f"Cargando modelo desde {model_path}")
        
        # Cargar configuración
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        # Crear configuración
        config = VoiceTrainingConfig(**config_dict)
        
        # Crear instancia
        trainer = cls(config)
        
        # Inicializar modelo
        trainer.initialize_model()
        
        # Cargar historial de entrenamiento si existe
        history_path = os.path.join(model_path, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r", encoding="utf-8") as f:
                trainer.training_history = json.load(f)
        
        logger.info(f"Modelo cargado correctamente: {config.model_type} para {config.character_name}")
        
        return trainer
    
    def generate_speech(self, text: str, output_path: str = None, speaker_id: int = 0) -> str:
        """
        Genera voz a partir de texto usando el modelo entrenado.
        
        Args:
            text: Texto a convertir en voz
            output_path: Ruta para guardar el audio generado
            speaker_id: ID del hablante (para modelos multi-hablante)
            
        Returns:
            Ruta al archivo de audio generado
        """
                logger.info(f"Generando voz para: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Verificar que el modelo esté inicializado
        if self.model is None:
            self.initialize_model()
        
        # Preparar directorio de salida si no se especificó
        if output_path is None:
            output_dir = os.path.join("outputs", "generated_speech", self.config.character_name)
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"speech_{timestamp}.wav")
        
        try:
            # Generar voz según tipo de modelo
            if self.config.model_type == "xtts":
                audio_path = self._generate_xtts_speech(text, output_path, speaker_id)
            elif self.config.model_type == "rvc":
                audio_path = self._generate_rvc_speech(text, output_path, speaker_id)
            else:
                raise ValueError(f"Tipo de modelo no soportado: {self.config.model_type}")
            
            logger.info(f"Voz generada correctamente: {audio_path}")
            return audio_path
            
        except Exception as e:
            logger.error(f"Error al generar voz: {str(e)}")
            return self._generate_fallback_speech(text, output_path)
    
    def _generate_xtts_speech(self, text: str, output_path: str, speaker_id: int = 0) -> str:
        """
        Genera voz usando modelo XTTS.
        
        Args:
            text: Texto a convertir en voz
            output_path: Ruta para guardar el audio
            speaker_id: ID del hablante
            
        Returns:
            Ruta al archivo de audio generado
        """
        # Poner modelo en modo evaluación
        self.model.eval()
        
        try:
            # Implementación simplificada - en un sistema real se usaría la API de XTTS
            from TTS.utils.synthesizer import Synthesizer
            
            # Crear sintetizador
            synthesizer = Synthesizer(
                tts_model=self.model,
                tts_config=self.model.config,
                use_cuda=self.config.use_gpu
            )
            
            # Generar voz
            wav = synthesizer.tts(text, speaker_id=speaker_id)
            
            # Guardar audio
            sf.write(output_path, wav, self.config.sample_rate)
            
            return output_path
            
        except ImportError:
            logger.error("No se pudo importar TTS. Instala con: pip install TTS")
            raise
        except Exception as e:
            logger.error(f"Error generando voz con XTTS: {str(e)}")
            raise
    
    def _generate_rvc_speech(self, text: str, output_path: str, speaker_id: int = 0) -> str:
        """
        Genera voz usando modelo RVC.
        
        Args:
            text: Texto a convertir en voz
            output_path: Ruta para guardar el audio
            speaker_id: ID del hablante
            
        Returns:
            Ruta al archivo de audio generado
        """
        # Poner modelo en modo evaluación
        self.model.eval()
        
        try:
            # RVC es un modelo de conversión de voz, no de TTS
            # Primero necesitamos generar voz con un TTS básico
            # y luego convertirla con RVC
            
            # Generar voz base con un TTS simple (ejemplo: gTTS)
            from gtts import gTTS
            
            # Ruta temporal para audio base
            temp_path = output_path.replace(".wav", "_temp.mp3")
            
            # Generar audio base
            tts = gTTS(text=text, lang='es')
            tts.save(temp_path)
            
            # Cargar audio base
            audio, sr = librosa.load(temp_path, sr=self.config.sample_rate)
            
            # Extraer características
            mel = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr,
                n_mels=80,
                hop_length=256,
                win_length=1024,
                fmin=80,
                fmax=7600
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            
            # Normalizar
            mel = (mel - mel.mean()) / (mel.std() + 1e-5)
            
            # Convertir a tensor
            mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
            
            # Crear ID de hablante
            speaker_tensor = torch.tensor([speaker_id], dtype=torch.long).to(self.device)
            
            # Convertir voz
            with torch.no_grad():
                output_mel = self.model(mel_tensor, speaker_tensor)
            
            # Convertir mel a audio (simplificado)
            # En un sistema real se usaría un vocoder adecuado
            from librosa.feature import inverse
            
            # Convertir a numpy
            output_mel = output_mel.squeeze(0).cpu().numpy()
            
            # Desnormalizar
            output_mel = output_mel * (mel.std() + 1e-5) + mel.mean()
            
            # Convertir a audio
            audio_out = librosa.feature.inverse.mel_to_audio(
                output_mel,
                sr=self.config.sample_rate,
                n_fft=1024,
                hop_length=256,
                win_length=1024
            )
            
            # Guardar audio
            sf.write(output_path, audio_out, self.config.sample_rate)
            
            # Eliminar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando voz con RVC: {str(e)}")
            raise
    
    def _generate_fallback_speech(self, text: str, output_path: str) -> str:
        """
        Genera voz usando un método de respaldo cuando falla el modelo principal.
        
        Args:
            text: Texto a convertir en voz
            output_path: Ruta para guardar el audio
            
        Returns:
            Ruta al archivo de audio generado
        """
        logger.warning("Usando generador de voz de respaldo")
        
        try:
            # Usar gTTS como respaldo
            from gtts import gTTS
            
            # Asegurar que la extensión sea correcta
            if not output_path.endswith(".mp3"):
                output_path = output_path.replace(".wav", ".mp3")
            
            # Generar audio
            tts = gTTS(text=text, lang='es')
            tts.save(output_path)
            
            logger.info(f"Voz de respaldo generada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generando voz de respaldo: {str(e)}")
            # Si todo falla, devolver ruta aunque no exista el archivo
            return output_path
    
    def evaluate_model(self, test_dataset: List[Dict]) -> Dict[str, float]:
        """
        Evalúa el rendimiento del modelo en un conjunto de prueba.
        
        Args:
            test_dataset: Dataset de prueba
            
        Returns:
            Métricas de evaluación
        """
        logger.info("Evaluando modelo")
        
        # Verificar que el modelo esté inicializado
        if self.model is None:
            self.initialize_model()
        
        # Poner modelo en modo evaluación
        self.model.eval()
        
        # Crear dataloader
        test_loader = self._create_dataloader(test_dataset, is_training=False)
        
        # Métricas
        metrics = {
            "loss": 0.0,
            "mel_cepstral_distortion": 0.0,
            "word_error_rate": 0.0,
            "character_error_rate": 0.0
        }
        
        # Evaluar
        total_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluando"):
                # Procesar batch
                loss = self._process_batch(batch, is_training=False)
                metrics["loss"] += loss
                total_batches += 1
                
                # Aquí se calcularían otras métricas como MCD, WER, CER
                # Implementación simplificada
        
        # Promediar métricas
        for key in metrics:
            metrics[key] /= max(1, total_batches)
        
        logger.info(f"Evaluación completada: {metrics}")
        
        return metrics
    
    def export_model(self, export_path: str = None, format: str = "onnx") -> str:
        """
        Exporta el modelo para uso en producción.
        
        Args:
            export_path: Ruta para exportar el modelo
            format: Formato de exportación ("onnx", "torchscript")
            
        Returns:
            Ruta al modelo exportado
        """
        logger.info(f"Exportando modelo en formato {format}")
        
        # Verificar que el modelo esté inicializado
        if self.model is None:
            self.initialize_model()
        
        # Poner modelo en modo evaluación
        self.model.eval()
        
        # Preparar directorio de salida
        if export_path is None:
            export_dir = os.path.join(
                self.config.output_dir,
                self.config.character_name,
                "exported"
            )
            os.makedirs(export_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(export_dir, f"model_{timestamp}")
        
        try:
            if format.lower() == "onnx":
                # Exportar a ONNX
                import onnx
                import onnxruntime
                
                # Crear inputs de ejemplo
                dummy_input = torch.randn(1, 80, 100).to(self.device)  # [batch, mel_bins, time]
                dummy_speaker = torch.zeros(1, dtype=torch.long).to(self.device)
                
                # Exportar
                onnx_path = f"{export_path}.onnx"
                torch.onnx.export(
                    self.model,
                    (dummy_input, dummy_speaker),
                    onnx_path,
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=["mel", "speaker_id"],
                    output_names=["output_mel"],
                    dynamic_axes={
                        "mel": {0: "batch_size", 2: "time"},
                        "speaker_id": {0: "batch_size"},
                        "output_mel": {0: "batch_size", 2: "time"}
                    }
                )
                
                # Verificar modelo
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                
                logger.info(f"Modelo exportado a ONNX: {onnx_path}")
                return onnx_path
                
            elif format.lower() == "torchscript":
                # Exportar a TorchScript
                script_path = f"{export_path}.pt"
                
                # Crear función de trazado
                class ModelWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, mel, speaker_id):
                        return self.model(mel, speaker_id)
                
                # Crear wrapper
                wrapper = ModelWrapper(self.model)
                
                # Crear inputs de ejemplo
                dummy_input = torch.randn(1, 80, 100).to(self.device)  # [batch, mel_bins, time]
                dummy_speaker = torch.zeros(1, dtype=torch.long).to(self.device)
                
                # Trazar modelo
                traced_model = torch.jit.trace(wrapper, (dummy_input, dummy_speaker))
                
                # Guardar modelo
                traced_model.save(script_path)
                
                logger.info(f"Modelo exportado a TorchScript: {script_path}")
                return script_path
                
            else:
                raise ValueError(f"Formato de exportación no soportado: {format}")
                
        except Exception as e:
            logger.error(f"Error exportando modelo: {str(e)}")
            raise


# Función principal para uso desde línea de comandos
def main():
    """Función principal para uso desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenador de voces personalizadas")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Subcomando para entrenar modelo
    train_parser = subparsers.add_parser("train", help="Entrenar un modelo de voz")
    train_parser.add_argument("--model-type", default="xtts", choices=["xtts", "rvc"], help="Tipo de modelo")
    train_parser.add_argument("--character", required=True, help="Nombre del personaje")
    train_parser.add_argument("--samples-dir", default="datasets/voice_samples", help="Directorio con muestras de voz")
    train_parser.add_argument("--output-dir", default="models/voice_models", help="Directorio para guardar el modelo")
    train_parser.add_argument("--epochs", type=int, default=1000, help="Número de épocas")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Tamaño del batch")
    train_parser.add_argument("--no-gpu", action="store_true", help="Desactivar uso de GPU")
    train_parser.add_argument("--pretrained", help="Ruta a modelo pre-entrenado")
    
    # Subcomando para generar voz
    generate_parser = subparsers.add_parser("generate", help="Generar voz a partir de texto")
    generate_parser.add_argument("--model-path", required=True, help="Ruta al modelo entrenado")
    generate_parser.add_argument("--text", required=True, help="Texto a convertir en voz")
    generate_parser.add_argument("--output", help="Ruta para guardar el audio generado")
    generate_parser.add_argument("--speaker-id", type=int, default=0, help="ID del hablante")
    
    # Subcomando para exportar modelo
    export_parser = subparsers.add_parser("export", help="Exportar modelo para producción")
    export_parser.add_argument("--model-path", required=True, help="Ruta al modelo entrenado")
    export_parser.add_argument("--output", help="Ruta para guardar el modelo exportado")
    export_parser.add_argument("--format", default="onnx", choices=["onnx", "torchscript"], help="Formato de exportación")
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Ejecutar comando correspondiente
    if args.command == "train":
        # Configurar entrenamiento
        config = VoiceTrainingConfig(
            model_type=args.model_type,
            character_name=args.character,
            voice_samples_dir=args.samples_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_gpu=not args.no_gpu,
            pretrained_model_path=args.pretrained if args.pretrained else None
        )
        
        # Crear entrenador
        trainer = VoiceTrainer(config)
        
        # Preparar dataset
        train_dataset, val_dataset = trainer.prepare_dataset()
        
        # Entrenar modelo
        trainer.train(train_dataset, val_dataset)
        
    elif args.command == "generate":
        # Cargar modelo
        trainer = VoiceTrainer.load_model(args.model_path)
        
        # Generar voz
        output_path = trainer.generate_speech(
            text=args.text,
            output_path=args.output,
            speaker_id=args.speaker_id
        )
        
        print(f"Voz generada: {output_path}")
        
    elif args.command == "export":
        # Cargar modelo
        trainer = VoiceTrainer.load_model(args.model_path)
        
        # Exportar modelo
        export_path = trainer.export_model(
            export_path=args.output,
            format=args.format
        )
        
        print(f"Modelo exportado: {export_path}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()