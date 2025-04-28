"""
Custom AI Model Trainer for Content Bot

Este módulo permite entrenar y ajustar modelos de lenguaje personalizados
para generar contenido viral y optimizado para nichos específicos.
Soporta fine-tuning de modelos como LLaMA, Grok y otros LLMs.
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/custom_trainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomModelTrainer:
    """
    Clase para entrenar y ajustar modelos de lenguaje personalizados
    para la generación de contenido optimizado para diferentes nichos.
    """
    
    def __init__(
        self, 
        base_model: str = "meta-llama/Llama-2-7b-hf",
        model_type: str = "llama",
        output_dir: str = "models/custom_llm",
        dataset_path: str = "datasets/viral_phrases.json",
        niche: Optional[str] = None,
        use_lora: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa el entrenador de modelos personalizados.
        
        Args:
            base_model: Modelo base a utilizar (LLaMA, Grok, etc.)
            model_type: Tipo de modelo (llama, grok, gpt)
            output_dir: Directorio para guardar el modelo entrenado
            dataset_path: Ruta al dataset de entrenamiento
            niche: Nicho específico para filtrar datos (finanzas, gaming, etc.)
            use_lora: Si se debe usar LoRA para fine-tuning eficiente
            device: Dispositivo para entrenamiento (cuda/cpu)
        """
        self.base_model = base_model
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.dataset_path = dataset_path
        self.niche = niche
        self.use_lora = use_lora
        self.device = device
        
        # Crear directorio de salida si no existe
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar tokenizer
        logger.info(f"Cargando tokenizer para {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            use_fast=True
        )
        
        # Asegurar que el tokenizer tenga token de padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuración específica según el tipo de modelo
        self._configure_model_settings()
        
        # Cargar modelo base
        self._load_base_model()
    
    def _configure_model_settings(self):
        """Configura ajustes específicos según el tipo de modelo."""
        if self.model_type == "llama":
            self.max_length = 2048
            self.lora_r = 16
            self.lora_alpha = 32
            self.lora_dropout = 0.05
            self.learning_rate = 2e-4
        elif self.model_type == "grok":
            self.max_length = 4096
            self.lora_r = 32
            self.lora_alpha = 64
            self.lora_dropout = 0.1
            self.learning_rate = 1e-4
        else:  # default/gpt
            self.max_length = 1024
            self.lora_r = 8
            self.lora_alpha = 16
            self.lora_dropout = 0.05
            self.learning_rate = 5e-5
    
    def _load_base_model(self):
        """Carga el modelo base y lo prepara para entrenamiento."""
        logger.info(f"Cargando modelo base {self.base_model} en {self.device}")
        
        # Configuración para carga eficiente en memoria
        load_in_8bit = self.device == "cuda" and torch.cuda.is_available()
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Preparar modelo para entrenamiento
            if load_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Aplicar LoRA si está habilitado
            if self.use_lora and self.device == "cuda":
                logger.info("Aplicando configuración LoRA para fine-tuning eficiente")
                lora_config = LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=self._get_target_modules()
                )
                self.model = get_peft_model(self.model, lora_config)
            
            logger.info(f"Modelo cargado exitosamente: {self.model.__class__.__name__}")
        
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            raise
    
    def _get_target_modules(self) -> List[str]:
        """Determina los módulos objetivo para LoRA según el tipo de modelo."""
        if self.model_type == "llama":
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif self.model_type == "grok":
            return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        else:  # default/gpt
            return ["c_attn", "c_proj", "c_fc"]
    
    def _load_and_prepare_dataset(self) -> Dataset:
        """
        Carga y prepara el dataset para entrenamiento.
        
        Returns:
            Dataset procesado y tokenizado
        """
        logger.info(f"Cargando dataset desde {self.dataset_path}")
        
        try:
            # Cargar datos según el formato
            if self.dataset_path.endswith('.json'):
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Filtrar por nicho si se especifica
                if self.niche:
                    data = [item for item in data if item.get('niche') == self.niche]
                
                # Crear dataset de HuggingFace
                dataset = Dataset.from_dict({
                    'text': [self._format_training_example(item) for item in data]
                })
            
            else:
                # Cargar dataset directamente con datasets
                dataset = load_dataset(self.dataset_path)
                if self.niche:
                    dataset = dataset.filter(lambda x: x.get('niche') == self.niche)
            
            logger.info(f"Dataset cargado con {len(dataset)} ejemplos")
            
            # Tokenizar dataset
            tokenized_dataset = dataset.map(
                lambda examples: self._tokenize_function(examples),
                batched=True,
                remove_columns=['text'] if 'text' in dataset.column_names else dataset.column_names
            )
            
            return tokenized_dataset
        
        except Exception as e:
            logger.error(f"Error al cargar el dataset: {str(e)}")
            raise
    
    def _format_training_example(self, item: Dict) -> str:
        """
        Formatea un ejemplo de entrenamiento según el tipo de contenido.
        
        Args:
            item: Diccionario con datos del ejemplo
            
        Returns:
            Texto formateado para entrenamiento
        """
        # Formato para guiones virales
        if 'script' in item:
            return f"### Nicho: {item.get('niche', 'general')}\n### Título: {item.get('title', '')}\n### Guión:\n{item['script']}\n### CTA:\n{item.get('cta', '')}"
        
        # Formato para frases virales
        elif 'phrase' in item:
            return f"### Nicho: {item.get('niche', 'general')}\n### Frase Viral:\n{item['phrase']}\n### Contexto:\n{item.get('context', '')}"
        
        # Formato para CTAs
        elif 'cta' in item:
            return f"### Nicho: {item.get('niche', 'general')}\n### CTA:\n{item['cta']}\n### Efectividad: {item.get('effectiveness', '0')}%"
        
        # Formato genérico
        else:
            return item.get('text', '')
    
    def _tokenize_function(self, examples):
        """Tokeniza los ejemplos para entrenamiento."""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def train(
        self,
        epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        save_steps: int = 100,
        eval_steps: int = 100,
        warmup_steps: int = 50,
        logging_steps: int = 10,
        eval_split: float = 0.1
    ) -> None:
        """
        Entrena el modelo con los parámetros especificados.
        
        Args:
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del batch
            gradient_accumulation_steps: Pasos de acumulación de gradiente
            save_steps: Cada cuántos pasos guardar checkpoint
            eval_steps: Cada cuántos pasos evaluar
            warmup_steps: Pasos de calentamiento para el scheduler
            logging_steps: Cada cuántos pasos registrar métricas
            eval_split: Proporción de datos para evaluación
        """
        logger.info(f"Iniciando entrenamiento del modelo {self.base_model}")
        
        try:
            # Cargar y preparar dataset
            dataset = self._load_and_prepare_dataset()
            
            # Dividir en train/eval
            dataset = dataset.train_test_split(test_size=eval_split)
            
            # Configurar argumentos de entrenamiento
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                evaluation_strategy="steps",
                eval_steps=eval_steps,
                save_strategy="steps",
                save_steps=save_steps,
                warmup_steps=warmup_steps,
                logging_steps=logging_steps,
                learning_rate=self.learning_rate,
                weight_decay=0.01,
                fp16=self.device == "cuda",
                logging_dir=f"{self.output_dir}/logs",
                report_to="tensorboard",
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                push_to_hub=False,
                remove_unused_columns=False
            )
            
            # Configurar data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Inicializar trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # Entrenar modelo
            logger.info("Comenzando entrenamiento...")
            trainer.train()
            
            # Guardar modelo entrenado
            self._save_model(trainer)
            
            logger.info(f"Entrenamiento completado. Modelo guardado en {self.output_dir}")
        
        except Exception as e:
            logger.error(f"Error durante el entrenamiento: {str(e)}")
            raise
    
    def _save_model(self, trainer: Trainer) -> None:
        """Guarda el modelo entrenado y su configuración."""
        # Guardar modelo
        trainer.save_model(str(self.output_dir))
        
        # Guardar tokenizer
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        # Guardar configuración
        config = {
            "base_model": self.base_model,
            "model_type": self.model_type,
            "niche": self.niche,
            "dataset": self.dataset_path,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "use_lora": self.use_lora
        }
        
        with open(f"{self.output_dir}/training_config.json", "w") as f:
            json.dump(config, f, indent=4)
    
    def generate_content(
        self, 
        prompt: str, 
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> List[str]:
        """
        Genera contenido con el modelo entrenado.
        
        Args:
            prompt: Texto inicial para la generación
            max_length: Longitud máxima de la generación
            temperature: Temperatura para sampling
            top_p: Valor de nucleus sampling
            top_k: Número de tokens más probables a considerar
            num_return_sequences: Número de secuencias a generar
            do_sample: Si se debe usar sampling o greedy decoding
            
        Returns:
            Lista de textos generados
        """
        logger.info(f"Generando contenido con prompt: {prompt[:50]}...")
        
        try:
            # Tokenizar prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generar texto
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Decodificar salidas
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            # Eliminar el prompt de las salidas
            prompt_length = len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
            generated_texts = [text[prompt_length:].strip() for text in generated_texts]
            
            return generated_texts
        
        except Exception as e:
            logger.error(f"Error al generar contenido: {str(e)}")
            return [f"Error: {str(e)}"]
    
    @classmethod
    def load_trained_model(cls, model_path: str) -> 'CustomModelTrainer':
        """
        Carga un modelo previamente entrenado.
        
        Args:
            model_path: Ruta al modelo guardado
            
        Returns:
            Instancia de CustomModelTrainer con el modelo cargado
        """
        logger.info(f"Cargando modelo entrenado desde {model_path}")
        
        try:
            # Cargar configuración
            with open(f"{model_path}/training_config.json", "r") as f:
                config = json.load(f)
            
            # Crear instancia
            trainer = cls(
                base_model=model_path,  # Usar el modelo guardado como base
                model_type=config.get("model_type", "llama"),
                output_dir=model_path,
                dataset_path=config.get("dataset", ""),
                niche=config.get("niche", None),
                use_lora=False  # No necesitamos LoRA para inferencia
            )
            
            logger.info(f"Modelo cargado exitosamente desde {model_path}")
            return trainer
        
        except Exception as e:
            logger.error(f"Error al cargar el modelo entrenado: {str(e)}")
            raise


class ContentOptimizer:
    """
    Clase para optimizar contenido utilizando modelos entrenados
    para diferentes nichos y formatos.
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        viral_phrases_path: str = "datasets/viral_phrases.json",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa el optimizador de contenido.
        
        Args:
            models_dir: Directorio con modelos entrenados
            viral_phrases_path: Ruta al dataset de frases virales
            device: Dispositivo para inferencia
        """
        self.models_dir = Path(models_dir)
        self.viral_phrases_path = viral_phrases_path
        self.device = device
        self.models = {}
        self.viral_phrases = self._load_viral_phrases()
        
        # Cargar modelos disponibles
        self._load_available_models()
    
    def _load_viral_phrases(self) -> Dict[str, List[str]]:
        """Carga frases virales por nicho desde el dataset."""
        try:
            with open(self.viral_phrases_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Organizar por nicho
            phrases_by_niche = {}
            for item in data:
                niche = item.get('niche', 'general')
                if 'phrase' in item:
                    if niche not in phrases_by_niche:
                        phrases_by_niche[niche] = []
                    phrases_by_niche[niche].append(item['phrase'])
            
            return phrases_by_niche
        
        except Exception as e:
            logger.error(f"Error al cargar frases virales: {str(e)}")
            return {}
    
    def _load_available_models(self) -> None:
        """Carga los modelos entrenados disponibles."""
        if not self.models_dir.exists():
            logger.warning(f"Directorio de modelos {self.models_dir} no existe")
            return
        
        # Buscar subdirectorios con modelos entrenados
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "training_config.json").exists():
                try:
                    # Cargar configuración
                    with open(model_dir / "training_config.json", "r") as f:
                        config = json.load(f)
                    
                    niche = config.get("niche", "general")
                    logger.info(f"Encontrado modelo para nicho: {niche}")
                    
                    # No cargar el modelo todavía, solo registrar su ubicación
                    self.models[niche] = {
                        "path": str(model_dir),
                        "config": config,
                        "model": None  # Se cargará bajo demanda
                    }
                
                except Exception as e:
                    logger.error(f"Error al cargar configuración del modelo {model_dir}: {str(e)}")
    
    def _get_model_for_niche(self, niche: str) -> Optional[CustomModelTrainer]:
        """
        Obtiene o carga el modelo para un nicho específico.
        
        Args:
            niche: Nicho para el que se requiere el modelo
            
        Returns:
            Instancia de CustomModelTrainer o None si no hay modelo
        """
        # Si no tenemos modelo para este nicho, usar el general
        if niche not in self.models and "general" in self.models:
            niche = "general"
        
        # Si aún no hay modelo, retornar None
        if niche not in self.models:
            return None
        
        # Cargar modelo si aún no está cargado
        if self.models[niche]["model"] is None:
            try:
                self.models[niche]["model"] = CustomModelTrainer.load_trained_model(
                    self.models[niche]["path"]
                )
            except Exception as e:
                logger.error(f"Error al cargar modelo para nicho {niche}: {str(e)}")
                return None
        
        return self.models[niche]["model"]
    
    def optimize_script(
        self,
        script: str,
        niche: str = "general",
        target_platform: str = "tiktok",
        include_cta: bool = True,
        cta_position: str = "middle",  # "start", "middle", "end"
        enhance_virality: bool = True
    ) -> Dict[str, str]:
        """
        Optimiza un guión para maximizar engagement.
        
        Args:
            script: Guión original
            niche: Nicho del contenido
            target_platform: Plataforma objetivo
            include_cta: Si se debe incluir CTA
            cta_position: Posición del CTA
            enhance_virality: Si se debe mejorar la viralidad
            
        Returns:
            Diccionario con guión optimizado y CTA
        """
        logger.info(f"Optimizando guión para nicho {niche} en {target_platform}")
        
        # Obtener modelo para el nicho
        model = self._get_model_for_niche(niche)
        
        # Si no hay modelo, aplicar optimizaciones básicas
        if model is None:
            return self._basic_script_optimization(
                script, niche, target_platform, include_cta, cta_position, enhance_virality
            )
        
        try:
            # Preparar prompt para el modelo
            prompt = f"""### Nicho: {niche}
### Plataforma: {target_platform}
### Tarea: Optimizar el siguiente guión para maximizar engagement y viralidad

### Guión Original:
{script}

### Guión Optimizado:"""
            
            # Generar guión optimizado
            optimized_script = model.generate_content(
                prompt,
                max_length=len(script) * 2,
                temperature=0.7
            )[0]
            
            # Generar CTA si se requiere
            cta = ""
            if include_cta:
                cta_prompt = f"""### Nicho: {niche}
### Plataforma: {target_platform}
### Tarea: Generar un CTA efectivo para el siguiente guión

### Guión:
{optimized_script}

### CTA:"""
                
                cta = model.generate_content(
                    cta_prompt,
                    max_length=100,
                    temperature=0.8
                )[0]
            
            # Insertar CTA en la posición adecuada
            final_script = self._insert_cta(optimized_script, cta, cta_position)
            
            return {
                "original_script": script,
                "optimized_script": final_script,
                "cta": cta,
                "niche": niche,
                "platform": target_platform
            }
        
        except Exception as e:
            logger.error(f"Error al optimizar guión: {str(e)}")
            return self._basic_script_optimization(
                script, niche, target_platform, include_cta, cta_position, enhance_virality
            )
    
    def _basic_script_optimization(
        self,
        script: str,
        niche: str,
        target_platform: str,
        include_cta: bool,
        cta_position: str,
        enhance_virality: bool
    ) -> Dict[str, str]:
        """Aplica optimizaciones básicas sin usar modelo entrenado."""
        logger.info("Aplicando optimizaciones básicas al guión")
        
        # Optimizaciones básicas según plataforma
        if target_platform in ["tiktok", "reels"]:
            # Para plataformas de video corto
            lines = script.split("\n")
            # Acortar si es muy largo
            if len(lines) > 15:
                lines = lines[:15]
            # Añadir elementos de enganche
            if lines and enhance_virality:
                lines[0] = f"¡{lines[0]}"
            optimized_script = "\n".join(lines)
        else:
            # Para plataformas de video largo
            optimized_script = script
        
        # Generar CTA básico según nicho
        cta = ""
        if include_cta:
            cta_templates = {
                "finanzas": "¡Sigue para más consejos financieros! 💰 Comenta '💸' si quieres la parte 2.",
                "gaming": "¡Dale like si te gustó! 🎮 Comenta qué juego quieres ver a continuación.",
                "salud": "¡Sigue para más rutinas! 💪 Comenta '🔥' si lo intentarás.",
                "tecnología": "¡Suscríbete para más reviews! 📱 ¿Qué dispositivo quieres que analice?",
                "humor": "¡Dale like si te reíste! 😂 Comenta '🤣' para más videos así.",
                "general": "¡Sigue para más contenido! 👉 Comenta si te gustó."
            }
            
            cta = cta_templates.get(niche, cta_templates["general"])
        
        # Insertar CTA
        final_script = self._insert_cta(optimized_script, cta, cta_position)
        
        return {
            "original_script": script,
            "optimized_script": final_script,
            "cta": cta,
            "niche": niche,
            "platform": target_platform
        }
    
    def _insert_cta(self, script: str, cta: str, position: str) -> str:
        """Inserta el CTA en la posición especificada del guión."""
        if not cta:
            return script
        
        lines = script.split("\n")
        
        if position == "start":
            return f"{cta}\n\n{script}"
        
        elif position == "end":
            return f"{script}\n\n{cta}"
        
        else:  # middle
            if len(lines) <= 2:
                return f"{script}\n\n{cta}"
            
            middle_idx = len(lines) // 2
            return "\n".join(lines[:middle_idx]) + f"\n\n{cta}\n\n" + "\n".join(lines[middle_idx:])
    
    def generate_viral_hooks(
        self,
        niche: str = "general",
        target_platform: str = "tiktok",
        num_hooks: int = 5,
        max_length: int = 50
    ) -> List[str]:
        """
        Genera ganchos virales para iniciar videos.
        
        Args:
            niche: Nicho del contenido
            target_platform: Plataforma objetivo
            num_hooks: Número de ganchos a generar
            max_length: Longitud máxima de cada gancho
            
        Returns:
            Lista de ganchos virales
        """
        logger.info(f"Generando {num_hooks} ganchos virales para {niche} en {target_platform}")
        
        # Obtener modelo para el nicho
        model = self._get_model_for_niche(niche)
        
        # Si tenemos frases virales para este nicho, usarlas como base
        viral_phrases = self.viral_phrases.get(niche, [])
        if not viral_phrases and "general" in self.viral_phrases:
            viral_phrases = self.viral_phrases.get("general", [])
        
        # Si no hay modelo, generar ganchos básicos
        if model is None:
            return self._basic_hook_generation(niche, target_platform, num_hooks, viral_phrases)
        
        try:
            # Preparar prompt para el modelo
            prompt = f"""### Nicho: {niche}
### Plataforma: {target_platform}
### Tarea: Generar {num_hooks} ganchos virales para iniciar videos

### Ejemplos de ganchos virales:
{"- " + "\n- ".join(viral_phrases[:5]) if viral_phrases else ""}

### Ganchos Virales Generados:
1."""
            
            # Generar ganchos
            generated_text = model.generate_content(
                prompt,
                max_length=max_length * num_hooks,
                temperature=0.9
            )[0]
            
            # Procesar resultado
            hooks = []
            for line in generated_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line[0] == "-"):
                    # Eliminar numeración o viñetas
                    clean_line = line.split(".", 1)[-1].strip() if "." in line else line[1:].strip()
                    if clean_line:
                        hooks.append(clean_line)
            
                        # Asegurar que tenemos suficientes ganchos
            while len(hooks) < num_hooks and viral_phrases:
                hooks.append(np.random.choice(viral_phrases))
            
            # Limitar a la cantidad solicitada
            return hooks[:num_hooks]
        
        except Exception as e:
            logger.error(f"Error al generar ganchos virales: {str(e)}")
            return self._basic_hook_generation(niche, target_platform, num_hooks, viral_phrases)
    
    def _basic_hook_generation(
        self,
        niche: str,
        target_platform: str,
        num_hooks: int,
        viral_phrases: List[str] = None
    ) -> List[str]:
        """Genera ganchos virales básicos sin usar modelo entrenado."""
        logger.info("Generando ganchos virales básicos")
        
        # Plantillas básicas por nicho
        templates = {
            "finanzas": [
                "¿Sabías que puedes duplicar tu dinero con este truco?",
                "El secreto que los bancos no quieren que sepas",
                "Así ahorré {X} euros en solo un mes",
                "3 formas de ganar dinero mientras duermes",
                "Lo que nadie te dice sobre invertir en {X}"
            ],
            "gaming": [
                "Este truco en {X} te hará ganar siempre",
                "Lo que los pros de {X} hacen diferente",
                "La configuración secreta que cambiará tu juego",
                "Nadie esperaba que esto funcionara en {X}",
                "El error que todos cometen en {X}"
            ],
            "salud": [
                "Haz esto por 7 días y notarás la diferencia",
                "El superalimento que cambió mi vida",
                "Lo que tu médico no te dice sobre {X}",
                "Transforma tu cuerpo con este simple hábito",
                "El secreto para {X} que descubrí por accidente"
            ],
            "tecnología": [
                "Este truco de {X} te ahorrará horas",
                "Lo que nadie te dice sobre tu {X}",
                "La función oculta de {X} que debes conocer",
                "Así puedes hacer que tu {X} dure el doble",
                "El ajuste que todos deberían hacer en su {X}"
            ],
            "humor": [
                "Cuando intentas {X} pero...",
                "Nadie me avisó que {X}",
                "POV: tu primer día haciendo {X}",
                "Lo que realmente pasa cuando {X}",
                "Mi reacción cuando {X}"
            ],
            "general": [
                "No creerás lo que pasó cuando {X}",
                "El secreto mejor guardado sobre {X}",
                "Esto cambiará tu forma de ver {X}",
                "Lo que nadie te dice sobre {X}",
                "Descubrí esto por accidente y cambió mi vida"
            ]
        }
        
        # Seleccionar plantillas según nicho
        niche_templates = templates.get(niche, templates["general"])
        
        # Generar ganchos
        hooks = []
        
        # Usar frases virales si están disponibles
        if viral_phrases and len(viral_phrases) >= num_hooks:
            return np.random.choice(viral_phrases, num_hooks, replace=False).tolist()
        
        # Combinar frases virales y plantillas
        if viral_phrases:
            hooks.extend(viral_phrases)
        
        # Completar con plantillas si es necesario
        while len(hooks) < num_hooks:
            template = np.random.choice(niche_templates)
            # Reemplazar {X} con términos relevantes según el nicho
            terms = {
                "finanzas": ["crypto", "bolsa", "inversiones", "ahorro", "presupuesto"],
                "gaming": ["Fortnite", "Minecraft", "Roblox", "FIFA", "Call of Duty"],
                "salud": ["perder peso", "ganar músculo", "dormir mejor", "reducir estrés", "energía"],
                "tecnología": ["smartphone", "laptop", "WiFi", "batería", "apps"],
                "humor": ["cocinar", "hacer ejercicio", "trabajar desde casa", "estudiar", "madrugar"],
                "general": ["esto", "hacerlo", "intentarlo", "aprenderlo", "descubrirlo"]
            }
            
            niche_terms = terms.get(niche, terms["general"])
            hook = template.replace("{X}", np.random.choice(niche_terms))
            
            if hook not in hooks:
                hooks.append(hook)
        
        return hooks[:num_hooks]
    
    def generate_trending_content(
        self,
        topic: str,
        niche: str = "general",
        target_platform: str = "tiktok",
        content_type: str = "script",  # "script", "outline", "hooks"
        length: str = "medium"  # "short", "medium", "long"
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Genera contenido basado en tendencias actuales.
        
        Args:
            topic: Tema o tendencia para generar contenido
            niche: Nicho del contenido
            target_platform: Plataforma objetivo
            content_type: Tipo de contenido a generar
            length: Longitud del contenido
            
        Returns:
            Diccionario con el contenido generado
        """
        logger.info(f"Generando contenido de tendencia sobre '{topic}' para {target_platform}")
        
        # Obtener modelo para el nicho
        model = self._get_model_for_niche(niche)
        
        # Determinar longitud en tokens según parámetro
        length_tokens = {
            "short": 150,
            "medium": 300,
            "long": 600
        }.get(length, 300)
        
        # Si no hay modelo, generar contenido básico
        if model is None:
            return self._basic_trending_content(topic, niche, target_platform, content_type, length)
        
        try:
            # Preparar prompt según tipo de contenido
            if content_type == "script":
                prompt = f"""### Nicho: {niche}
### Plataforma: {target_platform}
### Tendencia: {topic}
### Tarea: Generar un guión viral sobre esta tendencia

### Guión:"""
                
                # Generar guión
                content = model.generate_content(
                    prompt,
                    max_length=length_tokens,
                    temperature=0.8
                )[0]
                
                return {
                    "type": "script",
                    "content": content,
                    "topic": topic,
                    "niche": niche,
                    "platform": target_platform
                }
                
            elif content_type == "outline":
                prompt = f"""### Nicho: {niche}
### Plataforma: {target_platform}
### Tendencia: {topic}
### Tarea: Generar un esquema para video viral sobre esta tendencia

### Esquema:
1."""
                
                # Generar esquema
                content = model.generate_content(
                    prompt,
                    max_length=length_tokens,
                    temperature=0.7
                )[0]
                
                # Procesar resultado
                outline_points = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line and (line[0].isdigit() or line[0] == "-"):
                        # Eliminar numeración o viñetas
                        clean_line = line.split(".", 1)[-1].strip() if "." in line else line[1:].strip()
                        if clean_line:
                            outline_points.append(clean_line)
                
                return {
                    "type": "outline",
                    "content": outline_points,
                    "topic": topic,
                    "niche": niche,
                    "platform": target_platform
                }
                
            else:  # hooks
                # Usar método existente para generar ganchos
                hooks = self.generate_viral_hooks(
                    niche=niche,
                    target_platform=target_platform,
                    num_hooks=5,
                    max_length=50
                )
                
                return {
                    "type": "hooks",
                    "content": hooks,
                    "topic": topic,
                    "niche": niche,
                    "platform": target_platform
                }
        
        except Exception as e:
            logger.error(f"Error al generar contenido de tendencia: {str(e)}")
            return self._basic_trending_content(topic, niche, target_platform, content_type, length)
    
    def _basic_trending_content(
        self,
        topic: str,
        niche: str,
        target_platform: str,
        content_type: str,
        length: str
    ) -> Dict[str, Union[str, List[str]]]:
        """Genera contenido básico de tendencia sin usar modelo entrenado."""
        logger.info(f"Generando contenido básico de tendencia sobre '{topic}'")
        
        if content_type == "script":
            # Generar guión básico
            script_templates = {
                "tiktok": f"¡Hola a todos! Hoy vamos a hablar sobre {topic}.\n\n"
                          f"Esta tendencia está arrasando porque {topic} está cambiando la forma en que vemos {niche}.\n\n"
                          f"Tres cosas que debes saber:\n"
                          f"1. Es accesible para todos\n"
                          f"2. Está revolucionando {niche}\n"
                          f"3. Puedes empezar hoy mismo\n\n"
                          f"¡Comenta si quieres saber más sobre {topic}!",
                
                "youtube": f"¡Bienvenidos a un nuevo video! Hoy vamos a explorar a fondo {topic}.\n\n"
                           f"Esta tendencia está ganando popularidad por buenas razones. En este video, analizaremos:\n"
                           f"- Qué es exactamente {topic}\n"
                           f"- Por qué está revolucionando {niche}\n"
                           f"- Cómo puedes aprovecharla\n"
                           f"- Errores comunes a evitar\n\n"
                           f"Antes de comenzar, no olvides suscribirte y activar las notificaciones para más contenido sobre {niche}."
            }
            
            script = script_templates.get(target_platform, script_templates["tiktok"])
            
            return {
                "type": "script",
                "content": script,
                "topic": topic,
                "niche": niche,
                "platform": target_platform
            }
            
        elif content_type == "outline":
            # Generar esquema básico
            outline_templates = {
                "tiktok": [
                    f"Hook: ¿Has oído hablar de {topic}?",
                    f"Explicar qué es {topic} en 10 segundos",
                    f"Mostrar 3 ejemplos de {topic} en acción",
                    f"Explicar por qué {topic} está revolucionando {niche}",
                    f"CTA: Seguir para más contenido sobre {niche}"
                ],
                
                "youtube": [
                    f"Introducción a {topic} y su relevancia",
                    f"Historia y origen de {topic}",
                    f"Cómo {topic} está cambiando {niche}",
                    f"5 formas de aprovechar {topic}",
                    f"Casos de éxito con {topic}",
                    f"Errores comunes al implementar {topic}",
                    f"Conclusión y próximos pasos"
                ]
            }
            
            outline = outline_templates.get(target_platform, outline_templates["tiktok"])
            
            return {
                "type": "outline",
                "content": outline,
                "topic": topic,
                "niche": niche,
                "platform": target_platform
            }
            
        else:  # hooks
            # Generar ganchos básicos
            hook_templates = [
                f"¿Ya conoces la tendencia de {topic}? Está revolucionando {niche}",
                f"Lo que nadie te dice sobre {topic} (y deberías saber)",
                f"3 razones por las que {topic} está arrasando en {niche}",
                f"Probé {topic} durante una semana y esto es lo que pasó",
                f"El secreto detrás de {topic} que está cambiando {niche}"
            ]
            
            return {
                "type": "hooks",
                "content": hook_templates,
                "topic": topic,
                "niche": niche,
                "platform": target_platform
            }
    
    def adapt_content_for_platform(
        self,
        content: str,
        source_platform: str,
        target_platform: str,
        niche: str = "general"
    ) -> Dict[str, str]:
        """
        Adapta contenido de una plataforma a otra.
        
        Args:
            content: Contenido original
            source_platform: Plataforma de origen
            target_platform: Plataforma de destino
            niche: Nicho del contenido
            
        Returns:
            Diccionario con contenido adaptado
        """
        logger.info(f"Adaptando contenido de {source_platform} a {target_platform}")
        
        # Obtener modelo para el nicho
        model = self._get_model_for_niche(niche)
        
        # Si no hay modelo, aplicar adaptaciones básicas
        if model is None:
            return self._basic_platform_adaptation(content, source_platform, target_platform, niche)
        
        try:
            # Preparar prompt para el modelo
            prompt = f"""### Nicho: {niche}
### Plataforma Original: {source_platform}
### Plataforma Destino: {target_platform}
### Tarea: Adaptar el siguiente contenido de {source_platform} a {target_platform}

### Contenido Original ({source_platform}):
{content}

### Contenido Adaptado ({target_platform}):"""
            
            # Generar contenido adaptado
            adapted_content = model.generate_content(
                prompt,
                max_length=len(content) * 2,
                temperature=0.7
            )[0]
            
            return {
                "original_content": content,
                "adapted_content": adapted_content,
                "source_platform": source_platform,
                "target_platform": target_platform,
                "niche": niche
            }
        
        except Exception as e:
            logger.error(f"Error al adaptar contenido: {str(e)}")
            return self._basic_platform_adaptation(content, source_platform, target_platform, niche)
    
    def _basic_platform_adaptation(
        self,
        content: str,
        source_platform: str,
        target_platform: str,
        niche: str
    ) -> Dict[str, str]:
        """Aplica adaptaciones básicas de plataforma sin usar modelo entrenado."""
        logger.info("Aplicando adaptaciones básicas de plataforma")
        
        # Características por plataforma
        platform_characteristics = {
            "tiktok": {
                "max_length": 150,
                "style": "informal, directo, rápido",
                "intro": "¡Hola TikTok! ",
                "outro": " ¡Sigue para más! #viral #fyp"
            },
            "instagram": {
                "max_length": 200,
                "style": "visual, estético, tendencia",
                "intro": "✨ ",
                "outro": " 📸 #instagram #reels"
            },
            "youtube": {
                "max_length": 500,
                "style": "detallado, informativo, estructurado",
                "intro": "¡Hola YouTube! Bienvenidos a un nuevo video. ",
                "outro": " No olvides suscribirte y activar las notificaciones. ¡Hasta el próximo video!"
            },
            "twitter": {
                "max_length": 280,
                "style": "conciso, directo, opinión",
                "intro": "",
                "outro": " ¿Qué opinas? RT si estás de acuerdo."
            },
            "facebook": {
                "max_length": 400,
                "style": "conversacional, personal, detallado",
                "intro": "Hoy quiero compartir con ustedes ",
                "outro": " ¿Alguien más ha experimentado esto? ¡Comenta abajo!"
            }
        }
        
        # Obtener características de la plataforma destino
        target_chars = platform_characteristics.get(
            target_platform, 
            platform_characteristics["tiktok"]
        )
        
        # Adaptar contenido
        lines = content.split("\n")
        
        # Acortar si es necesario
        if len(content) > target_chars["max_length"]:
            # Mantener solo las líneas más importantes
            if len(lines) > 5:
                # Mantener primera, última y algunas del medio
                important_lines = [lines[0]]
                middle_lines = lines[1:-1]
                # Seleccionar algunas líneas del medio
                selected_middle = middle_lines[:min(3, len(middle_lines))]
                important_lines.extend(selected_middle)
                if len(lines) > 1:
                    important_lines.append(lines[-1])
                lines = important_lines
            
            # Unir y truncar si aún es muy largo
            adapted_content = "\n".join(lines)
            if len(adapted_content) > target_chars["max_length"]:
                adapted_content = adapted_content[:target_chars["max_length"] - 3] + "..."
        else:
            adapted_content = content
        
        # Añadir intro y outro específicos de la plataforma
        if not adapted_content.startswith(target_chars["intro"]) and target_chars["intro"]:
            adapted_content = target_chars["intro"] + adapted_content
        
        if not adapted_content.endswith(target_chars["outro"]) and target_chars["outro"]:
            adapted_content = adapted_content + target_chars["outro"]
        
        return {
            "original_content": content,
            "adapted_content": adapted_content,
            "source_platform": source_platform,
            "target_platform": target_platform,
            "niche": niche
        }
    
    def analyze_content_performance(
        self,
        content: str,
        niche: str = "general",
        target_platform: str = "tiktok"
    ) -> Dict[str, Union[str, float, List[str]]]:
        """
        Analiza el rendimiento potencial de un contenido.
        
        Args:
            content: Contenido a analizar
            niche: Nicho del contenido
            target_platform: Plataforma objetivo
            
        Returns:
            Diccionario con análisis de rendimiento
        """
        logger.info(f"Analizando rendimiento de contenido para {target_platform}")
        
        # Obtener modelo para el nicho
        model = self._get_model_for_niche(niche)
        
        # Si no hay modelo, realizar análisis básico
        if model is None:
            return self._basic_performance_analysis(content, niche, target_platform)
        
        try:
            # Preparar prompt para el modelo
            prompt = f"""### Nicho: {niche}
### Plataforma: {target_platform}
### Tarea: Analizar el potencial de engagement del siguiente contenido

### Contenido:
{content}

### Análisis:
- Puntuación (0-100): 
- Fortalezas:
1.
- Debilidades:
1.
- Sugerencias de mejora:
1."""
            
            # Generar análisis
            analysis_text = model.generate_content(
                prompt,
                max_length=500,
                temperature=0.5
            )[0]
            
            # Extraer puntuación
            score_match = re.search(r"Puntuación \(0-100\): (\d+)", analysis_text)
            score = int(score_match.group(1)) if score_match else 50
            
            # Extraer fortalezas
            strengths = []
            strengths_section = re.search(r"Fortalezas:(.*?)(?:Debilidades:|$)", analysis_text, re.DOTALL)
            if strengths_section:
                for line in strengths_section.group(1).strip().split("\n"):
                    line = line.strip()
                    if line and (line[0].isdigit() or line[0] == "-"):
                        clean_line = line.split(".", 1)[-1].strip() if "." in line else line[1:].strip()
                        if clean_line:
                            strengths.append(clean_line)
            
            # Extraer debilidades
            weaknesses = []
            weaknesses_section = re.search(r"Debilidades:(.*?)(?:Sugerencias|$)", analysis_text, re.DOTALL)
            if weaknesses_section:
                for line in weaknesses_section.group(1).strip().split("\n"):
                    line = line.strip()
                    if line and (line[0].isdigit() or line[0] == "-"):
                        clean_line = line.split(".", 1)[-1].strip() if "." in line else line[1:].strip()
                        if clean_line:
                            weaknesses.append(clean_line)
            
            # Extraer sugerencias
            suggestions = []
            suggestions_section = re.search(r"Sugerencias de mejora:(.*?)$", analysis_text, re.DOTALL)
            if suggestions_section:
                for line in suggestions_section.group(1).strip().split("\n"):
                    line = line.strip()
                    if line and (line[0].isdigit() or line[0] == "-"):
                        clean_line = line.split(".", 1)[-1].strip() if "." in line else line[1:].strip()
                        if clean_line:
                            suggestions.append(clean_line)
            
            return {
                "score": score,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "suggestions": suggestions,
                "niche": niche,
                "platform": target_platform
            }
        
        except Exception as e:
            logger.error(f"Error al analizar rendimiento: {str(e)}")
            return self._basic_performance_analysis(content, niche, target_platform)
    
    def _basic_performance_analysis(
        self,
        content: str,
        niche: str,
        target_platform: str
    ) -> Dict[str, Union[str, float, List[str]]]:
        """Realiza análisis básico de rendimiento sin usar modelo entrenado."""
        logger.info("Realizando análisis básico de rendimiento")
        
        # Análisis de longitud
        length_score = 0
        if target_platform == "tiktok" and len(content) < 300:
            length_score = 20
        elif target_platform == "youtube" and len(content) > 500:
            length_score = 20
        elif target_platform == "instagram" and len(content) < 400:
            length_score = 20
        else:
            length_score = 10
        
        # Análisis de engagement
        engagement_words = ["tú", "tu", "ustedes", "vosotros", "te", "os", "comenta", 
                           "like", "suscríbete", "comparte", "sígueme", "¿qué opinas?",
                           "¿qué piensas?", "¿estás de acuerdo?"]
        
        engagement_score = 0
        for word in engagement_words:
            if word.lower() in content.lower():
                engagement_score += 5
        
        engagement_score = min(engagement_score, 30)
        
        # Análisis de estructura
        structure_score = 0
        if "\n" in content:
            structure_score += 10
        
        if ":" in content or "-" in content:
            structure_score += 10
        
        if "?" in content or "¿" in content:
            structure_score += 10
        
        structure_score = min(structure_score, 30)
        
        # Análisis de llamada a la acción
        cta_score = 0
        cta_phrases = ["suscríbete", "dale like", "comenta", "comparte", "sígueme", 
                      "no olvides", "activa", "guarda", "visita"]
        
        for phrase in cta_phrases:
            if phrase.lower() in content.lower():
                cta_score += 5
        
        cta_score = min(cta_score, 20)
        
        # Puntuación total
        total_score = length_score + engagement_score + structure_score + cta_score
        
        # Fortalezas y debilidades
        strengths = []
        weaknesses = []
        
        if length_score >= 15:
            strengths.append(f"Longitud adecuada para {target_platform}")
        else:
            weaknesses.append(f"Longitud no óptima para {target_platform}")
        
        if engagement_score >= 20:
            strengths.append("Buen nivel de engagement con la audiencia")
        else:
            weaknesses.append("Podría mejorar la interacción con la audiencia")
        
        if structure_score >= 20:
            strengths.append("Buena estructura y organización")
        else:
            weaknesses.append("Estructura mejorable para facilitar la lectura")
        
        if cta_score >= 15:
            strengths.append("Incluye llamadas a la acción efectivas")
        else:
            weaknesses.append("Falta o débil llamada a la acción")
        
        # Sugerencias
        suggestions = []
        
        if length_score < 15:
            if target_platform in ["tiktok", "instagram"]:
                suggestions.append("Acortar el contenido para mantener la atención")
            else:
                suggestions.append("Expandir el contenido para más profundidad")
        
        if engagement_score < 20:
            suggestions.append("Añadir preguntas o llamadas a la participación")
        
        if structure_score < 20:
            suggestions.append("Mejorar la estructura con párrafos y puntos clave")
        
        if cta_score < 15:
            suggestions.append("Incluir una llamada a la acción clara al final")
        
        return {
            "score": total_score,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "suggestions": suggestions,
            "niche": niche,
            "platform": target_platform
        }


# Función de utilidad para usar desde línea de comandos
def main():
    """Función principal para uso desde línea de comandos."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenador y optimizador de modelos de IA para contenido viral")
    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")
    
    # Subcomando para entrenar modelo
    train_parser = subparsers.add_parser("train", help="Entrenar un modelo personalizado")
    train_parser.add_argument("--base-model", default="meta-llama/Llama-2-7b-hf", help="Modelo base a utilizar")
    train_parser.add_argument("--model-type", default="llama", choices=["llama", "grok", "gpt"], help="Tipo de modelo")
    train_parser.add_argument("--output-dir", default="models/custom_llm", help="Directorio para guardar el modelo")
    train_parser.add_argument("--dataset", default="datasets/viral_phrases.json", help="Dataset de entrenamiento")
    train_parser.add_argument("--niche", help="Nicho específico para filtrar datos")
    train_parser.add_argument("--epochs", type=int, default=3, help="Número de épocas de entrenamiento")
    train_parser.add_argument("--batch-size", type=int, default=4, help="Tamaño del batch")
    train_parser.add_argument("--no-lora", action="store_true", help="Desactivar LoRA para fine-tuning")
    
    # Subcomando para generar contenido
    generate_parser = subparsers.add_parser("generate", help="Generar contenido optimizado")
    generate_parser.add_argument("--model-path", default="models/custom_llm", help="Ruta al modelo entrenado")
    generate_parser.add_argument("--prompt", required=True, help="Prompt inicial para generación")
    generate_parser.add_argument("--max-length", type=int, default=512, help="Longitud máxima de generación")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura para sampling")
    generate_parser.add_argument("--output-file", help="Archivo para guardar la salida")
    
    # Subcomando para optimizar guión
    optimize_parser = subparsers.add_parser("optimize", help="Optimizar un guión existente")
    optimize_parser.add_argument("--input-file", required=True, help="Archivo con guión a optimizar")
    optimize_parser.add_argument("--niche", default="general", help="Nicho del contenido")
    optimize_parser.add_argument("--platform", default="tiktok", help="Plataforma objetivo")
    optimize_parser.add_argument("--include-cta", action="store_true", help="Incluir CTA")
    optimize_parser.add_argument("--output-file", help="Archivo para guardar la salida")
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Ejecutar comando correspondiente
    if args.command == "train":
        trainer = CustomModelTrainer(
            base_model=args.base_model,
            model_type=args.model_type,
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            niche=args.niche,
            use_lora=not args.no_lora
        )
        
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
    elif args.command == "generate":
        try:
            trainer = CustomModelTrainer.load_trained_model(args.model_path)
            generated_texts = trainer.generate_content(
                prompt=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
            
                        for i, text in enumerate(generated_texts):
                print(f"Generación {i+1}:\n{text}\n")
                
            # Guardar en archivo si se especificó
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    for i, text in enumerate(generated_texts):
                        f.write(f"Generación {i+1}:\n{text}\n\n")
                print(f"Resultados guardados en {args.output_file}")
                
        except Exception as e:
            print(f"Error al generar contenido: {str(e)}")
            
    elif args.command == "optimize":
        try:
            # Leer guión de entrada
            with open(args.input_file, 'r', encoding='utf-8') as f:
                script = f.read()
            
            # Crear optimizador
            optimizer = ContentOptimizer()
            
            # Optimizar guión
            result = optimizer.optimize_script(
                script=script,
                niche=args.niche,
                target_platform=args.platform,
                include_cta=args.include_cta
            )
            
            # Mostrar resultado
            print(f"Guión original ({len(result['original_script'])} caracteres):")
            print("-" * 50)
            print(result['original_script'])
            print("-" * 50)
            print(f"\nGuión optimizado ({len(result['optimized_script'])} caracteres):")
            print("-" * 50)
            print(result['optimized_script'])
            print("-" * 50)
            
            # Guardar en archivo si se especificó
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Guión original ({len(result['original_script'])} caracteres):\n")
                    f.write("-" * 50 + "\n")
                    f.write(result['original_script'] + "\n")
                    f.write("-" * 50 + "\n\n")
                    f.write(f"Guión optimizado ({len(result['optimized_script'])} caracteres):\n")
                    f.write("-" * 50 + "\n")
                    f.write(result['optimized_script'] + "\n")
                    f.write("-" * 50 + "\n")
                print(f"Resultados guardados en {args.output_file}")
                
        except Exception as e:
            print(f"Error al optimizar guión: {str(e)}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()