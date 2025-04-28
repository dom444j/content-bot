import os
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import re
import whisper
import torch
from transformers import pipeline, MarianMTModel, MarianTokenizer
from googletrans import Translator
import webvtt
from webvtt import WebVTT, Caption
import srt

from data.knowledge_base import KnowledgeBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SubtitleGenerator:
    """
    Clase para generar subtítulos para videos con soporte para múltiples idiomas
    y con capacidad para integrar CTAs estratégicamente.
    """
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.config = self._load_config()
        self.whisper_model = None
        self.translator = None
        self.translation_models = {}
        self.cache = {}
        logger.info("SubtitleGenerator initialized")
        
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración para la generación de subtítulos"""
        try:
            config = self.knowledge_base.get_subtitle_config()
            if not config:
                # Configuración por defecto
                config = {
                    "output_directory": "uploads/subtitles",
                    "cache_enabled": True,
                    "cache_max_size": 100,
                    "whisper": {
                        "model_size": "base",  # tiny, base, small, medium, large
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "compute_type": "float16" if torch.cuda.is_available() else "float32"
                    },
                    "translation": {
                        "method": "googletrans",  # googletrans, huggingface
                        "huggingface_models": {
                            "es-en": "Helsinki-NLP/opus-mt-es-en",
                            "en-es": "Helsinki-NLP/opus-mt-en-es",
                            "en-fr": "Helsinki-NLP/opus-mt-en-fr",
                            "en-de": "Helsinki-NLP/opus-mt-en-de"
                        }
                    },
                    "subtitle_formats": ["srt", "vtt"],
                    "max_chars_per_line": 42,
                    "max_lines_per_caption": 2,
                    "min_duration": 0.7,  # segundos
                    "max_duration": 7.0,  # segundos
                    "cta_integration": {
                        "enabled": True,
                        "highlight_color": "#FF0000",
                        "bold_enabled": True,
                        "timing_ranges": [
                            [4, 8],  # segundos desde el inicio
                            [-5, -2]  # segundos desde el final (valores negativos)
                        ]
                    },
                    "languages": {
                        "default": "es",
                        "auto_translate": ["en", "fr", "de"],
                        "language_codes": {
                            "es": "Spanish",
                            "en": "English",
                            "fr": "French",
                            "de": "German",
                            "pt": "Portuguese",
                            "it": "Italian"
                        }
                    }
                }
                self.knowledge_base.save_subtitle_config(config)
            return config
        except Exception as e:
            logger.error(f"Error loading subtitle config: {str(e)}")
            # Configuración mínima en caso de error
            return {
                "output_directory": "uploads/subtitles",
                "whisper": {
                    "model_size": "base",
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                },
                "languages": {
                    "default": "es",
                    "auto_translate": ["en"]
                }
            }
    
    def _load_whisper_model(self):
        """Carga el modelo de Whisper para transcripción"""
        if self.whisper_model is None:
            try:
                whisper_config = self.config.get("whisper", {})
                model_size = whisper_config.get("model_size", "base")
                device = whisper_config.get("device", "cpu")
                compute_type = whisper_config.get("compute_type", "float32")
                
                logger.info(f"Loading Whisper model: {model_size} on {device} with {compute_type}")
                self.whisper_model = whisper.load_model(
                    model_size, 
                    device=device, 
                    compute_type=compute_type
                )
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {str(e)}")
                raise
    
    def _load_translator(self):
        """Carga el traductor para subtítulos multilingües"""
        if self.translator is None:
            try:
                translation_config = self.config.get("translation", {})
                method = translation_config.get("method", "googletrans")
                
                if method == "googletrans":
                    self.translator = Translator()
                    logger.info("Google Translator loaded successfully")
                elif method == "huggingface":
                    # Los modelos de HuggingFace se cargarán bajo demanda
                    logger.info("HuggingFace translation will be loaded on demand")
                else:
                    logger.warning(f"Unknown translation method: {method}, using googletrans")
                    self.translator = Translator()
            except Exception as e:
                logger.error(f"Error loading translator: {str(e)}")
                # Fallback a googletrans
                self.translator = Translator()
    
    def _load_translation_model(self, src_lang: str, tgt_lang: str):
        """Carga un modelo de traducción específico de HuggingFace"""
        lang_pair = f"{src_lang}-{tgt_lang}"
        if lang_pair not in self.translation_models:
            try:
                translation_config = self.config.get("translation", {})
                huggingface_models = translation_config.get("huggingface_models", {})
                
                if lang_pair in huggingface_models:
                    model_name = huggingface_models[lang_pair]
                    logger.info(f"Loading translation model: {model_name}")
                    
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    
                    self.translation_models[lang_pair] = {
                        "tokenizer": tokenizer,
                        "model": model
                    }
                    logger.info(f"Translation model for {lang_pair} loaded successfully")
                else:
                    logger.warning(f"No HuggingFace model found for {lang_pair}, will use googletrans")
            except Exception as e:
                logger.error(f"Error loading translation model for {lang_pair}: {str(e)}")
    
    def generate_subtitles(self, 
                          audio_path: str, 
                          output_path: str = None,
                          language: str = None,
                          translate_to: List[str] = None,
                          cta_texts: List[Dict[str, Any]] = None,
                          formats: List[str] = None) -> Dict[str, Dict[str, str]]:
        """
        Genera subtítulos para un archivo de audio o video
        
        Args:
            audio_path: Ruta al archivo de audio o video
            output_path: Directorio de salida (opcional)
            language: Idioma del audio (opcional, se detecta automáticamente)
            translate_to: Lista de idiomas a los que traducir (opcional)
            cta_texts: Lista de textos CTA con tiempos para integrar (opcional)
            formats: Formatos de salida (srt, vtt)
            
        Returns:
            Diccionario con rutas a los archivos de subtítulos generados por idioma y formato
        """
        try:
            # Verificar que el archivo existe
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Cargar modelos si no están cargados
            self._load_whisper_model()
            self._load_translator()
            
            # Configurar idioma y traducciones
            if language is None:
                language = self.config.get("languages", {}).get("default", "es")
            
            if translate_to is None:
                translate_to = self.config.get("languages", {}).get("auto_translate", [])
            
            # Configurar formatos de salida
            if formats is None:
                formats = self.config.get("subtitle_formats", ["srt", "vtt"])
            
            # Configurar directorio de salida
            if output_path is None:
                output_dir = self.config.get("output_directory", "uploads/subtitles")
                timestamp = int(time.time())
                output_path = os.path.join(output_dir, f"subtitles_{timestamp}")
            
            os.makedirs(output_path, exist_ok=True)
            
            # Transcribir audio
            logger.info(f"Transcribing audio: {audio_path}")
            result = self.whisper_model.transcribe(audio_path, language=language)
            
            # Procesar segmentos
            segments = result["segments"]
            
            # Integrar CTAs si se proporcionan
            if cta_texts and self.config.get("cta_integration", {}).get("enabled", True):
                segments = self._integrate_ctas(segments, cta_texts)
            
            # Generar subtítulos en el idioma original
            subtitle_paths = {}
            subtitle_paths[language] = {}
            
            for format in formats:
                if format == "srt":
                    srt_path = os.path.join(output_path, f"subtitles_{language}.srt")
                    self._generate_srt(segments, srt_path, language)
                    subtitle_paths[language]["srt"] = srt_path
                
                elif format == "vtt":
                    vtt_path = os.path.join(output_path, f"subtitles_{language}.vtt")
                    self._generate_vtt(segments, vtt_path, language)
                    subtitle_paths[language]["vtt"] = vtt_path
            
            # Traducir subtítulos si se solicita
            for target_lang in translate_to:
                if target_lang != language:
                    translated_segments = self._translate_segments(segments, language, target_lang)
                    subtitle_paths[target_lang] = {}
                    
                    for format in formats:
                        if format == "srt":
                            srt_path = os.path.join(output_path, f"subtitles_{target_lang}.srt")
                            self._generate_srt(translated_segments, srt_path, target_lang)
                            subtitle_paths[target_lang]["srt"] = srt_path
                        
                        elif format == "vtt":
                            vtt_path = os.path.join(output_path, f"subtitles_{target_lang}.vtt")
                            self._generate_vtt(translated_segments, vtt_path, target_lang)
                            subtitle_paths[target_lang]["vtt"] = vtt_path
            
            logger.info(f"Generated subtitles in {len(subtitle_paths)} languages")
            return subtitle_paths
            
        except Exception as e:
            logger.error(f"Error generating subtitles: {str(e)}")
            raise
    
    def _integrate_ctas(self, segments: List[Dict[str, Any]], cta_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Integra textos de CTA en los segmentos de subtítulos
        
        Args:
            segments: Lista de segmentos de subtítulos
            cta_texts: Lista de textos CTA con tiempos
                [{"text": "¡Sigue ahora!", "start": 5.0, "end": 7.0}]
                o [{"text": "¡Suscríbete!", "position": "start+5"}]
                o [{"text": "¡Comenta!", "position": "end-3"}]
        
        Returns:
            Lista de segmentos con CTAs integrados
        """
        try:
            # Copiar segmentos para no modificar el original
            modified_segments = segments.copy()
            
            # Obtener configuración de CTAs
            cta_config = self.config.get("cta_integration", {})
            highlight_color = cta_config.get("highlight_color", "#FF0000")
            bold_enabled = cta_config.get("bold_enabled", True)
            
            # Procesar cada CTA
            for cta in cta_texts:
                cta_text = cta.get("text", "")
                
                # Determinar tiempo de inicio y fin
                if "start" in cta and "end" in cta:
                    # Tiempo explícito
                    start_time = cta["start"]
                    end_time = cta["end"]
                elif "position" in cta:
                    # Posición relativa
                    position = cta["position"]
                    
                    if position.startswith("start+"):
                        # Segundos desde el inicio
                        seconds = float(position.replace("start+", ""))
                        start_time = seconds
                        end_time = start_time + 3.0  # Duración por defecto
                    elif position.startswith("end-"):
                        # Segundos antes del final
                        seconds = float(position.replace("end-", ""))
                        # Obtener duración total del video
                        if segments:
                            total_duration = segments[-1]["end"]
                            start_time = total_duration - seconds
                            end_time = start_time + 3.0
                        else:
                            logger.warning("Cannot determine video duration, skipping CTA")
                            continue
                    else:
                        logger.warning(f"Unknown CTA position: {position}, skipping")
                        continue
                else:
                    # Usar rangos de tiempo predeterminados
                    timing_ranges = cta_config.get("timing_ranges", [[4, 8], [-5, -2]])
                    
                    # Seleccionar un rango aleatorio
                    import random
                    selected_range = random.choice(timing_ranges)
                    
                    if selected_range[0] >= 0:
                        # Segundos desde el inicio
                        start_time = selected_range[0]
                        end_time = selected_range[1]
                    else:
                        # Segundos antes del final
                        if segments:
                            total_duration = segments[-1]["end"]
                            start_time = total_duration + selected_range[0]  # Valor negativo
                            end_time = total_duration + selected_range[1]    # Valor negativo
                        else:
                            logger.warning("Cannot determine video duration, using default timing")
                            start_time = 5.0
                            end_time = 8.0
                
                # Formatear texto CTA
                formatted_cta = cta_text
                if bold_enabled:
                    # Formato SRT/VTT para negrita
                    formatted_cta = f"<b>{formatted_cta}</b>"
                
                # Formato SRT/VTT para color
                formatted_cta = f'<font color="{highlight_color}">{formatted_cta}</font>'
                
                # Buscar segmentos que se superpongan con el tiempo del CTA
                for i, segment in enumerate(modified_segments):
                    seg_start = segment["start"]
                    seg_end = segment["end"]
                    
                    # Verificar si hay superposición
                    if (seg_start <= end_time and seg_end >= start_time):
                        # Modificar el texto del segmento para incluir el CTA
                        original_text = segment["text"]
                        
                        # Añadir CTA al final del texto
                        modified_segments[i]["text"] = f"{original_text} {formatted_cta}"
            
            return modified_segments
            
        except Exception as e:
            logger.error(f"Error integrating CTAs: {str(e)}")
            return segments
    
    def _translate_segments(self, 
                          segments: List[Dict[str, Any]], 
                          src_lang: str, 
                          tgt_lang: str) -> List[Dict[str, Any]]:
        """
        Traduce los segmentos de subtítulos a otro idioma
        
        Args:
            segments: Lista de segmentos de subtítulos
            src_lang: Idioma de origen
            tgt_lang: Idioma de destino
            
        Returns:
            Lista de segmentos traducidos
        """
        try:
            # Copiar segmentos para no modificar el original
            translated_segments = []
            
            # Obtener método de traducción
            translation_config = self.config.get("translation", {})
            method = translation_config.get("method", "googletrans")
            
            # Traducir cada segmento
            for segment in segments:
                # Copiar segmento
                translated_segment = segment.copy()
                
                # Extraer texto a traducir
                text = segment["text"]
                
                # Preservar formato (negrita, color) durante la traducción
                # Extraer etiquetas HTML
                html_tags = []
                pattern = r'<[^>]+>'
                for match in re.finditer(pattern, text):
                    html_tags.append((match.start(), match.end(), match.group()))
                
                # Texto sin etiquetas HTML para traducción
                clean_text = re.sub(pattern, '', text)
                
                # Traducir texto
                translated_text = ""
                
                if method == "huggingface":
                    # Usar modelo de HuggingFace si está disponible
                    lang_pair = f"{src_lang}-{tgt_lang}"
                    
                    if lang_pair not in self.translation_models:
                        self._load_translation_model(src_lang, tgt_lang)
                    
                    if lang_pair in self.translation_models:
                        model_data = self.translation_models[lang_pair]
                        tokenizer = model_data["tokenizer"]
                        model = model_data["model"]
                        
                        inputs = tokenizer(clean_text, return_tensors="pt", padding=True)
                        outputs = model.generate(**inputs)
                        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    else:
                        # Fallback a googletrans
                        translated_text = self.translator.translate(
                            clean_text, src=src_lang, dest=tgt_lang
                        ).text
                
                else:  # googletrans
                    translated_text = self.translator.translate(
                        clean_text, src=src_lang, dest=tgt_lang
                    ).text
                
                # Reinsert HTML tags
                if html_tags:
                    # Convertir el texto traducido a una lista de caracteres para inserción
                    chars = list(translated_text)
                    
                    # Insertar etiquetas en orden inverso para mantener índices válidos
                    for start, end, tag in sorted(html_tags, reverse=True):
                        # Calcular posición proporcional en el texto traducido
                        if len(clean_text) > 0:
                            ratio = start / len(clean_text)
                            pos = min(int(ratio * len(translated_text)), len(translated_text))
                        else:
                            pos = 0
                        
                        chars.insert(pos, tag)
                    
                    translated_text = ''.join(chars)
                
                # Actualizar segmento traducido
                translated_segment["text"] = translated_text
                translated_segments.append(translated_segment)
            
            return translated_segments
            
        except Exception as e:
            logger.error(f"Error translating segments: {str(e)}")
            # Devolver segmentos originales en caso de error
            return segments
    
    def _generate_srt(self, segments: List[Dict[str, Any]], output_path: str, language: str):
        """
        Genera un archivo de subtítulos en formato SRT
        
        Args:
            segments: Lista de segmentos de subtítulos
            output_path: Ruta de salida para el archivo SRT
            language: Código de idioma
        """
        try:
            # Configuración de formato
            max_chars = self.config.get("max_chars_per_line", 42)
            max_lines = self.config.get("max_lines_per_caption", 2)
            
            # Crear subtítulos SRT
            srt_subtitles = []
            
            for i, segment in enumerate(segments):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                
                # Formatear tiempos
                start_time_str = self._format_time_srt(start_time)
                end_time_str = self._format_time_srt(end_time)
                
                # Dividir texto largo en múltiples líneas
                lines = self._split_text(text, max_chars, max_lines)
                
                                # Crear entrada SRT
                srt_entry = srt.Subtitle(
                    index=i+1,
                    start=srt.srt_timestamp_to_timedelta(start_time_str),
                    end=srt.srt_timestamp_to_timedelta(end_time_str),
                    content="\n".join(lines)
                )
                
                srt_subtitles.append(srt_entry)
            
            # Guardar archivo SRT
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(srt.compose(srt_subtitles))
            
            logger.info(f"Generated SRT subtitles: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating SRT subtitles: {str(e)}")
            raise
    
    def _generate_vtt(self, segments: List[Dict[str, Any]], output_path: str, language: str):
        """
        Genera un archivo de subtítulos en formato VTT
        
        Args:
            segments: Lista de segmentos de subtítulos
            output_path: Ruta de salida para el archivo VTT
            language: Código de idioma
        """
        try:
            # Configuración de formato
            max_chars = self.config.get("max_chars_per_line", 42)
            max_lines = self.config.get("max_lines_per_caption", 2)
            
            # Crear subtítulos VTT
            vtt = WebVTT()
            
            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                
                # Formatear tiempos
                start_time_str = self._format_time_vtt(start_time)
                end_time_str = self._format_time_vtt(end_time)
                
                # Dividir texto largo en múltiples líneas
                lines = self._split_text(text, max_chars, max_lines)
                
                # Crear entrada VTT
                caption = Caption(
                    start=start_time_str,
                    end=end_time_str,
                    text="\n".join(lines)
                )
                
                vtt.captions.append(caption)
            
            # Guardar archivo VTT
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            vtt.save(output_path)
            
            logger.info(f"Generated VTT subtitles: {output_path}")
            
        except Exception as e:
            logger.error(f"Error generating VTT subtitles: {str(e)}")
            raise
    
    def _format_time_srt(self, seconds: float) -> str:
        """
        Formatea un tiempo en segundos al formato SRT (HH:MM:SS,mmm)
        
        Args:
            seconds: Tiempo en segundos
            
        Returns:
            Tiempo formateado para SRT
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def _format_time_vtt(self, seconds: float) -> str:
        """
        Formatea un tiempo en segundos al formato VTT (HH:MM:SS.mmm)
        
        Args:
            seconds: Tiempo en segundos
            
        Returns:
            Tiempo formateado para VTT
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int((seconds - int(seconds)) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"
    
    def _split_text(self, text: str, max_chars: int, max_lines: int) -> List[str]:
        """
        Divide un texto en múltiples líneas respetando límites de caracteres
        
        Args:
            text: Texto a dividir
            max_chars: Máximo de caracteres por línea
            max_lines: Máximo de líneas
            
        Returns:
            Lista de líneas
        """
        # Si el texto ya tiene saltos de línea, respetarlos
        if "\n" in text:
            predefined_lines = text.split("\n")
            result_lines = []
            
            for line in predefined_lines:
                if len(line) <= max_chars:
                    result_lines.append(line)
                else:
                    # Dividir líneas largas
                    words = line.split()
                    current_line = ""
                    
                    for word in words:
                        test_line = current_line + " " + word if current_line else word
                        
                        if len(test_line) <= max_chars:
                            current_line = test_line
                        else:
                            result_lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        result_lines.append(current_line)
            
            # Limitar número de líneas
            if len(result_lines) > max_lines:
                result_lines = result_lines[:max_lines]
                # Añadir elipsis a la última línea si se truncó
                if len(result_lines) > 0:
                    result_lines[-1] = result_lines[-1].rstrip() + "..."
            
            return result_lines
        
        # Dividir texto por palabras
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
                
                # Limitar número de líneas
                if len(lines) >= max_lines - 1:
                    break
        
        if current_line and len(lines) < max_lines:
            lines.append(current_line)
        
        # Si se truncó el texto, añadir elipsis
        if len(words) > 0 and len(" ".join(words)) > len(" ".join(lines)):
            lines[-1] = lines[-1].rstrip() + "..."
        
        return lines
    
    def extract_subtitles(self, video_path: str, output_path: str = None) -> str:
        """
        Extrae subtítulos incrustados de un archivo de video
        
        Args:
            video_path: Ruta al archivo de video
            output_path: Ruta de salida para el archivo de subtítulos (opcional)
            
        Returns:
            Ruta al archivo de subtítulos extraído o None si no se encontraron
        """
        try:
            # Verificar que el archivo existe
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Generar nombre de archivo si no se proporciona
            if output_path is None:
                output_dir = self.config.get("output_directory", "uploads/subtitles")
                os.makedirs(output_dir, exist_ok=True)
                
                basename = os.path.basename(video_path)
                name, _ = os.path.splitext(basename)
                output_path = os.path.join(output_dir, f"{name}_extracted.srt")
            
            # Usar ffmpeg para extraer subtítulos
            import subprocess
            
            # Primero, verificar si hay subtítulos
            cmd_check = [
                "ffprobe", 
                "-v", "error",
                "-select_streams", "s",
                "-show_entries", "stream=index:stream_tags=language",
                "-of", "csv=p=0",
                video_path
            ]
            
            result = subprocess.run(cmd_check, capture_output=True, text=True)
            
            if not result.stdout.strip():
                logger.warning(f"No subtitles found in {video_path}")
                return None
            
            # Extraer subtítulos
            cmd_extract = [
                "ffmpeg",
                "-i", video_path,
                "-map", "0:s:0",  # Primera pista de subtítulos
                "-c", "copy",
                output_path
            ]
            
            subprocess.run(cmd_extract, capture_output=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Extracted subtitles to {output_path}")
                return output_path
            else:
                logger.warning(f"Failed to extract subtitles from {video_path}")
                return None
            
        except Exception as e:
            logger.error(f"Error extracting subtitles: {str(e)}")
            return None
    
    def merge_subtitles(self, video_path: str, subtitle_path: str, output_path: str = None) -> str:
        """
        Fusiona subtítulos con un archivo de video
        
        Args:
            video_path: Ruta al archivo de video
            subtitle_path: Ruta al archivo de subtítulos
            output_path: Ruta de salida para el video con subtítulos (opcional)
            
        Returns:
            Ruta al video con subtítulos incrustados
        """
        try:
            # Verificar que los archivos existen
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if not os.path.exists(subtitle_path):
                raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
            
            # Generar nombre de archivo si no se proporciona
            if output_path is None:
                dirname = os.path.dirname(video_path)
                basename = os.path.basename(video_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(dirname, f"{name}_subtitled{ext}")
            
            # Usar ffmpeg para fusionar video y subtítulos
            import subprocess
            
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-i", subtitle_path,
                "-c:v", "copy",
                "-c:a", "copy",
                "-c:s", "mov_text",  # Formato de subtítulos compatible
                "-map", "0:v",
                "-map", "0:a",
                "-map", "1",
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True)
            
            if os.path.exists(output_path):
                logger.info(f"Merged subtitles into video: {output_path}")
                return output_path
            else:
                logger.warning(f"Failed to merge subtitles with video")
                return None
            
        except Exception as e:
            logger.error(f"Error merging subtitles: {str(e)}")
            return None
    
    def burn_subtitles(self, video_path: str, subtitle_path: str, output_path: str = None,
                     font_size: int = 24, font_color: str = "white",
                     position: str = "bottom") -> str:
        """
        Quema (incrusta permanentemente) subtítulos en un archivo de video
        
        Args:
            video_path: Ruta al archivo de video
            subtitle_path: Ruta al archivo de subtítulos
            output_path: Ruta de salida para el video con subtítulos (opcional)
            font_size: Tamaño de fuente
            font_color: Color de fuente
            position: Posición (bottom, top, middle)
            
        Returns:
            Ruta al video con subtítulos quemados
        """
        try:
            # Verificar que los archivos existen
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            if not os.path.exists(subtitle_path):
                raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
            
            # Generar nombre de archivo si no se proporciona
            if output_path is None:
                dirname = os.path.dirname(video_path)
                basename = os.path.basename(video_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(dirname, f"{name}_hardsubbed{ext}")
            
            # Determinar posición vertical
            y_position = "h-th-50" # middle
            if position == "bottom":
                y_position = "h-th-30"
            elif position == "top":
                y_position = "th+30"
            
            # Usar ffmpeg para quemar subtítulos
            import subprocess
            
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"subtitles={subtitle_path}:force_style='FontSize={font_size},PrimaryColour=&H{self._color_to_hex(font_color)},Alignment=2,MarginV=30'",
                "-c:a", "copy",
                output_path
            ]
            
            subprocess.run(cmd, capture_output=True)
            
            if os.path.exists(output_path):
                logger.info(f"Burned subtitles into video: {output_path}")
                return output_path
            else:
                logger.warning(f"Failed to burn subtitles into video")
                return None
            
        except Exception as e:
            logger.error(f"Error burning subtitles: {str(e)}")
            return None
    
    def _color_to_hex(self, color_name: str) -> str:
        """
        Convierte un nombre de color a formato hexadecimal para ffmpeg
        
        Args:
            color_name: Nombre del color
            
        Returns:
            Color en formato hexadecimal
        """
        color_map = {
            "white": "FFFFFF",
            "black": "000000",
            "red": "FF0000",
            "green": "00FF00",
            "blue": "0000FF",
            "yellow": "FFFF00",
            "cyan": "00FFFF",
            "magenta": "FF00FF"
        }
        
        return color_map.get(color_name.lower(), "FFFFFF")
    
    def detect_language(self, audio_path: str) -> str:
        """
        Detecta el idioma de un archivo de audio
        
        Args:
            audio_path: Ruta al archivo de audio
            
        Returns:
            Código de idioma detectado
        """
        try:
            # Cargar modelo si no está cargado
            self._load_whisper_model()
            
            # Detectar idioma con Whisper
            logger.info(f"Detecting language for: {audio_path}")
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Obtener características de audio
            mel = whisper.log_mel_spectrogram(audio).to(self.whisper_model.device)
            
            # Detectar idioma
            _, probs = self.whisper_model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            
            logger.info(f"Detected language: {detected_lang}")
            return detected_lang
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            # Devolver idioma por defecto en caso de error
            return self.config.get("languages", {}).get("default", "es")
    
    def add_cta_to_subtitles(self, subtitle_path: str, cta_texts: List[Dict[str, Any]], 
                            output_path: str = None) -> str:
        """
        Añade textos de CTA a un archivo de subtítulos existente
        
        Args:
            subtitle_path: Ruta al archivo de subtítulos
            cta_texts: Lista de textos CTA con tiempos
            output_path: Ruta de salida para el archivo modificado (opcional)
            
        Returns:
            Ruta al archivo de subtítulos modificado
        """
        try:
            # Verificar que el archivo existe
            if not os.path.exists(subtitle_path):
                raise FileNotFoundError(f"Subtitle file not found: {subtitle_path}")
            
            # Generar nombre de archivo si no se proporciona
            if output_path is None:
                dirname = os.path.dirname(subtitle_path)
                basename = os.path.basename(subtitle_path)
                name, ext = os.path.splitext(basename)
                output_path = os.path.join(dirname, f"{name}_with_cta{ext}")
            
            # Determinar formato de subtítulos
            _, ext = os.path.splitext(subtitle_path)
            
            if ext.lower() == ".srt":
                # Cargar subtítulos SRT
                with open(subtitle_path, "r", encoding="utf-8") as f:
                    srt_content = f.read()
                
                subtitles = list(srt.parse(srt_content))
                
                # Obtener configuración de CTAs
                cta_config = self.config.get("cta_integration", {})
                highlight_color = cta_config.get("highlight_color", "#FF0000")
                bold_enabled = cta_config.get("bold_enabled", True)
                
                # Procesar cada CTA
                for cta in cta_texts:
                    cta_text = cta.get("text", "")
                    
                    # Determinar tiempo de inicio y fin
                    if "start" in cta and "end" in cta:
                        start_time = cta["start"]
                        end_time = cta["end"]
                    elif "position" in cta:
                        position = cta["position"]
                        
                        if position.startswith("start+"):
                            seconds = float(position.replace("start+", ""))
                            start_time = seconds
                            end_time = start_time + 3.0
                        elif position.startswith("end-"):
                            seconds = float(position.replace("end-", ""))
                            # Obtener duración total
                            if subtitles:
                                total_duration = subtitles[-1].end.total_seconds()
                                start_time = total_duration - seconds
                                end_time = start_time + 3.0
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                    
                    # Formatear texto CTA
                    formatted_cta = cta_text
                    if bold_enabled:
                        formatted_cta = f"<b>{formatted_cta}</b>"
                    
                    formatted_cta = f'<font color="{highlight_color}">{formatted_cta}</font>'
                    
                    # Buscar subtítulos que se superpongan con el tiempo del CTA
                    for subtitle in subtitles:
                        sub_start = subtitle.start.total_seconds()
                        sub_end = subtitle.end.total_seconds()
                        
                        if (sub_start <= end_time and sub_end >= start_time):
                            # Modificar el texto del subtítulo
                            subtitle.content = f"{subtitle.content} {formatted_cta}"
                
                # Guardar subtítulos modificados
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(srt.compose(subtitles))
                
            elif ext.lower() == ".vtt":
                # Cargar subtítulos VTT
                vtt = WebVTT.read(subtitle_path)
                
                # Obtener configuración de CTAs
                cta_config = self.config.get("cta_integration", {})
                highlight_color = cta_config.get("highlight_color", "#FF0000")
                bold_enabled = cta_config.get("bold_enabled", True)
                
                # Procesar cada CTA
                for cta in cta_texts:
                    cta_text = cta.get("text", "")
                    
                    # Determinar tiempo de inicio y fin
                    if "start" in cta and "end" in cta:
                        start_time = cta["start"]
                        end_time = cta["end"]
                    elif "position" in cta:
                        position = cta["position"]
                        
                        if position.startswith("start+"):
                            seconds = float(position.replace("start+", ""))
                            start_time = seconds
                            end_time = start_time + 3.0
                        elif position.startswith("end-"):
                            seconds = float(position.replace("end-", ""))
                            # Obtener duración total
                            if vtt.captions:
                                last_caption = vtt.captions[-1]
                                h, m, s = map(float, last_caption.end.split(':'))
                                total_duration = h * 3600 + m * 60 + s
                                start_time = total_duration - seconds
                                end_time = start_time + 3.0
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                    
                    # Formatear texto CTA
                    formatted_cta = cta_text
                    if bold_enabled:
                        formatted_cta = f"<b>{formatted_cta}</b>"
                    
                    formatted_cta = f'<c.{highlight_color.replace("#", "")}>{formatted_cta}</c>'
                    
                    # Buscar subtítulos que se superpongan con el tiempo del CTA
                    for caption in vtt.captions:
                        # Convertir tiempos a segundos
                        h_start, m_start, s_start = map(float, caption.start.split(':'))
                        h_end, m_end, s_end = map(float, caption.end.split(':'))
                        
                        sub_start = h_start * 3600 + m_start * 60 + s_start
                        sub_end = h_end * 3600 + m_end * 60 + s_end
                        
                        if (sub_start <= end_time and sub_end >= start_time):
                            # Modificar el texto del subtítulo
                            caption.text = f"{caption.text} {formatted_cta}"
                
                # Guardar subtítulos modificados
                vtt.save(output_path)
            
            else:
                logger.warning(f"Unsupported subtitle format: {ext}")
                return subtitle_path
            
            logger.info(f"Added CTAs to subtitles: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding CTAs to subtitles: {str(e)}")
            return subtitle_path