import os
import json
import logging
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/engagement.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SentimentAnalyzer")

class SentimentAnalyzer:
    """
    Sistema para analizar el sentimiento en comentarios y mensajes.
    Identifica tendencias de sentimiento, temas emergentes y proporciona
    insights sobre la percepción del contenido y la marca.
    """
    
    def __init__(self, data_path: str = "data/engagement/sentiment"):
        """
        Inicializa el analizador de sentimiento.
        
        Args:
            data_path: Ruta al directorio de datos de sentimiento
        """
        self.data_path = data_path
        self.sentiment_data = []
        self.content_sentiment = {}
        self.user_sentiment = {}
        self.keyword_sentiment = {}
        self.sentiment_lexicon = {}
        
        # Crear directorios necesarios
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Cargar datos existentes
        self._load_data()
        
        # Cargar léxico de sentimiento
        self._load_sentiment_lexicon()
        
        logger.info("Analizador de sentimiento inicializado")
    
    def _load_data(self) -> None:
        """Carga los datos de sentimiento desde archivos JSON."""
        try:
            # Cargar datos de sentimiento
            sentiment_path = os.path.join(self.data_path, "sentiment_data.json")
            if os.path.exists(sentiment_path):
                with open(sentiment_path, "r", encoding="utf-8") as f:
                    self.sentiment_data = json.load(f)
                logger.info(f"Datos de sentimiento cargados: {len(self.sentiment_data)} registros")
            
            # Cargar sentimiento por contenido
            content_path = os.path.join(self.data_path, "content_sentiment.json")
            if os.path.exists(content_path):
                with open(content_path, "r", encoding="utf-8") as f:
                    self.content_sentiment = json.load(f)
                logger.info(f"Sentimiento por contenido cargado: {len(self.content_sentiment)} elementos")
            
            # Cargar sentimiento por usuario
            user_path = os.path.join(self.data_path, "user_sentiment.json")
            if os.path.exists(user_path):
                with open(user_path, "r", encoding="utf-8") as f:
                    self.user_sentiment = json.load(f)
                logger.info(f"Sentimiento por usuario cargado: {len(self.user_sentiment)} usuarios")
            
            # Cargar sentimiento por palabra clave
            keyword_path = os.path.join(self.data_path, "keyword_sentiment.json")
            if os.path.exists(keyword_path):
                with open(keyword_path, "r", encoding="utf-8") as f:
                    self.keyword_sentiment = json.load(f)
                logger.info(f"Sentimiento por palabra clave cargado: {len(self.keyword_sentiment)} palabras")
        
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            self.sentiment_data = []
            self.content_sentiment = {}
            self.user_sentiment = {}
            self.keyword_sentiment = {}
    
    def _save_data(self) -> None:
        """Guarda los datos de sentimiento en archivos JSON."""
        try:
            # Guardar datos de sentimiento
            sentiment_path = os.path.join(self.data_path, "sentiment_data.json")
            with open(sentiment_path, "w", encoding="utf-8") as f:
                json.dump(self.sentiment_data, f, indent=4)
            
            # Guardar sentimiento por contenido
            content_path = os.path.join(self.data_path, "content_sentiment.json")
            with open(content_path, "w", encoding="utf-8") as f:
                json.dump(self.content_sentiment, f, indent=4)
            
            # Guardar sentimiento por usuario
            user_path = os.path.join(self.data_path, "user_sentiment.json")
            with open(user_path, "w", encoding="utf-8") as f:
                json.dump(self.user_sentiment, f, indent=4)
            
            # Guardar sentimiento por palabra clave
            keyword_path = os.path.join(self.data_path, "keyword_sentiment.json")
            with open(keyword_path, "w", encoding="utf-8") as f:
                json.dump(self.keyword_sentiment, f, indent=4)
            
            logger.info("Datos de sentimiento guardados correctamente")
        
        except Exception as e:
            logger.error(f"Error al guardar datos: {str(e)}")
    
    def _load_sentiment_lexicon(self) -> None:
        """Carga el léxico de sentimiento desde un archivo JSON."""
        lexicon_path = os.path.join(self.data_path, "sentiment_lexicon.json")
        
        try:
            if os.path.exists(lexicon_path):
                with open(lexicon_path, "r", encoding="utf-8") as f:
                    self.sentiment_lexicon = json.load(f)
                logger.info(f"Léxico de sentimiento cargado: {len(self.sentiment_lexicon)} palabras")
            else:
                # Crear léxico básico por defecto
                self.sentiment_lexicon = {
                    # Palabras positivas
                    "excelente": 1.0,
                    "bueno": 0.8,
                    "genial": 0.9,
                    "increíble": 1.0,
                    "maravilloso": 0.9,
                    "fantástico": 0.9,
                    "encanta": 0.8,
                    "gusta": 0.7,
                    "útil": 0.6,
                    "recomiendo": 0.8,
                    "gracias": 0.6,
                    "feliz": 0.7,
                    "perfecto": 0.9,
                    "mejor": 0.7,
                    "fácil": 0.6,
                    
                    # Palabras negativas
                    "malo": -0.8,
                    "terrible": -1.0,
                    "horrible": -1.0,
                    "pésimo": -0.9,
                    "odio": -0.9,
                    "decepción": -0.8,
                    "decepcionado": -0.8,
                    "difícil": -0.6,
                    "problema": -0.7,
                    "error": -0.7,
                    "falla": -0.7,
                    "peor": -0.8,
                    "inútil": -0.8,
                    "aburrido": -0.6,
                    "costoso": -0.5,
                    
                    # Emojis
                    "😊": 0.8,
                    "😃": 0.9,
                    "😄": 0.9,
                    "😁": 0.8,
                    "😍": 1.0,
                    "👍": 0.7,
                    "❤️": 0.9,
                    "😢": -0.7,
                    "😭": -0.8,
                    "😡": -0.9,
                    "👎": -0.7,
                    "😠": -0.8,
                    "🤮": -1.0,
                    "😕": -0.5,
                    "😒": -0.6
                                }
                
                # Guardar léxico por defecto
                with open(lexicon_path, "w", encoding="utf-8") as f:
                    json.dump(self.sentiment_lexicon, f, indent=4)
                
                logger.info(f"Léxico de sentimiento por defecto creado: {len(self.sentiment_lexicon)} palabras")
        
        except Exception as e:
            logger.error(f"Error al cargar léxico de sentimiento: {str(e)}")
            # Crear un léxico mínimo en caso de error
            self.sentiment_lexicon = {"bueno": 0.8, "malo": -0.8}
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analiza el sentimiento de un texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con resultados del análisis
        """
        if not text or not isinstance(text, str):
            return {
                "sentiment_score": 0,
                "sentiment_label": "neutral",
                "confidence": 0,
                "keywords": []
            }
        
        # Normalizar texto
        normalized_text = self._normalize_text(text)
        
        # Extraer palabras clave
        keywords = self._extract_keywords(normalized_text)
        
        # Calcular puntuación de sentimiento
        sentiment_score, confidence = self._calculate_sentiment_score(normalized_text, keywords)
        
        # Determinar etiqueta de sentimiento
        sentiment_label = self._get_sentiment_label(sentiment_score)
        
        return {
            "sentiment_score": round(sentiment_score, 2),
            "sentiment_label": sentiment_label,
            "confidence": round(confidence, 2),
            "keywords": keywords
        }
    
    def _normalize_text(self, text: str) -> str:
        """
        Normaliza el texto para análisis.
        
        Args:
            text: Texto a normalizar
            
        Returns:
            Texto normalizado
        """
        # Convertir a minúsculas
        text = text.lower()
        
        # Preservar emojis y caracteres especiales importantes para el sentimiento
        
        # Eliminar URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Eliminar menciones
        text = re.sub(r'@\w+', '', text)
        
        # Eliminar hashtags pero mantener el texto
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Eliminar caracteres especiales pero mantener emojis
        text = re.sub(r'[^\w\s\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]', ' ', text)
        
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrae palabras clave del texto con su sentimiento asociado.
        
        Args:
            text: Texto normalizado
            
        Returns:
            Lista de palabras clave con su sentimiento
        """
        keywords = []
        
        # Dividir texto en palabras
        words = text.split()
        
        # Extraer n-gramas (1, 2 y 3 palabras)
        ngrams = []
        
        # Unigrams
        ngrams.extend(words)
        
        # Bigrams
        for i in range(len(words) - 1):
            ngrams.append(words[i] + " " + words[i + 1])
        
        # Trigrams
        for i in range(len(words) - 2):
            ngrams.append(words[i] + " " + words[i + 1] + " " + words[i + 2])
        
        # Filtrar n-gramas que están en el léxico
        for ngram in ngrams:
            if ngram in self.sentiment_lexicon:
                keywords.append({
                    "text": ngram,
                    "sentiment": self.sentiment_lexicon[ngram]
                })
        
        # Ordenar por valor absoluto de sentimiento (más significativos primero)
        keywords.sort(key=lambda x: abs(x["sentiment"]), reverse=True)
        
        # Limitar a las 10 palabras clave más significativas
        return keywords[:10]
    
    def _calculate_sentiment_score(self, text: str, keywords: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Calcula la puntuación de sentimiento del texto.
        
        Args:
            text: Texto normalizado
            keywords: Palabras clave extraídas
            
        Returns:
            Tupla con (puntuación de sentimiento, confianza)
        """
        if not keywords:
            return 0.0, 0.0
        
        # Calcular puntuación basada en palabras clave encontradas
        total_score = sum(keyword["sentiment"] for keyword in keywords)
        
        # Normalizar puntuación entre -1 y 1
        if len(keywords) > 0:
            normalized_score = total_score / len(keywords)
        else:
            normalized_score = 0
        
        # Ajustar por modificadores (negaciones, intensificadores)
        normalized_score = self._adjust_for_modifiers(text, normalized_score)
        
        # Calcular confianza basada en la cantidad de palabras clave encontradas
        # y la consistencia de sus puntuaciones
        word_count = len(text.split())
        keyword_coverage = min(1.0, len(keywords) / max(1, word_count / 2))
        
        # Calcular desviación estándar de puntuaciones (consistencia)
        if len(keywords) > 1:
            sentiment_values = [k["sentiment"] for k in keywords]
            consistency = 1.0 - min(1.0, np.std(sentiment_values))
        else:
            consistency = 0.5
        
        # Calcular confianza final
        confidence = (keyword_coverage * 0.7) + (consistency * 0.3)
        
        return normalized_score, confidence
    
    def _adjust_for_modifiers(self, text: str, score: float) -> float:
        """
        Ajusta la puntuación de sentimiento basado en modificadores.
        
        Args:
            text: Texto normalizado
            score: Puntuación inicial
            
        Returns:
            Puntuación ajustada
        """
        # Detectar negaciones
        negations = ["no", "nunca", "ni", "tampoco", "sin", "nada"]
        has_negation = any(neg in text.split() for neg in negations)
        
        # Detectar intensificadores
        intensifiers = ["muy", "mucho", "demasiado", "extremadamente", "totalmente", "completamente"]
        intensifier_count = sum(1 for intensifier in intensifiers if intensifier in text.split())
        
        # Aplicar ajustes
        adjusted_score = score
        
        # Negación invierte el sentimiento
        if has_negation:
            adjusted_score = -adjusted_score
        
        # Intensificadores aumentan la magnitud
        if intensifier_count > 0:
            # Aumentar hasta un 50% con múltiples intensificadores
            intensity_factor = min(1.5, 1 + (0.1 * intensifier_count))
            adjusted_score = adjusted_score * intensity_factor
        
        # Asegurar que el resultado esté entre -1 y 1
        return max(-1.0, min(1.0, adjusted_score))
    
    def _get_sentiment_label(self, score: float) -> str:
        """
        Convierte una puntuación numérica en una etiqueta de sentimiento.
        
        Args:
            score: Puntuación de sentimiento (-1 a 1)
            
        Returns:
            Etiqueta de sentimiento
        """
        if score >= 0.6:
            return "very_positive"
        elif score >= 0.2:
            return "positive"
        elif score > -0.2:
            return "neutral"
        elif score > -0.6:
            return "negative"
        else:
            return "very_negative"
    
    def analyze_comment(self, comment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza un comentario y actualiza las métricas de sentimiento.
        
        Args:
            comment_data: Datos del comentario
            
        Returns:
            Resultados del análisis
        """
        # Verificar campos requeridos
        required_fields = ["text", "user_id", "content_id", "platform"]
        for field in required_fields:
            if field not in comment_data:
                logger.error(f"Falta campo requerido: {field}")
                return {"error": f"Falta campo requerido: {field}"}
        
        # Generar ID único para el análisis
        analysis_id = f"sa_{len(self.sentiment_data) + 1}_{int(datetime.now().timestamp())}"
        
        # Añadir timestamp si no existe
        if "timestamp" not in comment_data:
            comment_data["timestamp"] = datetime.now().isoformat()
        
        # Analizar texto
        text = comment_data["text"]
        analysis_result = self.analyze_text(text)
        
        # Crear registro de análisis
        sentiment_record = {
            "analysis_id": analysis_id,
            "timestamp": comment_data["timestamp"],
            "user_id": comment_data["user_id"],
            "content_id": comment_data["content_id"],
            "platform": comment_data["platform"],
            "text": text,
            "sentiment_score": analysis_result["sentiment_score"],
            "sentiment_label": analysis_result["sentiment_label"],
            "confidence": analysis_result["confidence"],
            "keywords": analysis_result["keywords"]
        }
        
        # Añadir campos adicionales si existen
        for key, value in comment_data.items():
            if key not in sentiment_record and key != "text":
                sentiment_record[key] = value
        
        # Guardar registro
        self.sentiment_data.append(sentiment_record)
        
        # Actualizar métricas
        self._update_metrics(sentiment_record)
        
        # Guardar datos
        self._save_data()
        
        logger.info(f"Comentario analizado: {analysis_id} - {analysis_result['sentiment_label']}")
        
        return {
            "analysis_id": analysis_id,
            **analysis_result
        }
    
    def _update_metrics(self, sentiment_record: Dict[str, Any]) -> None:
        """
        Actualiza las métricas de sentimiento basadas en un nuevo análisis.
        
        Args:
            sentiment_record: Registro de análisis de sentimiento
        """
        content_id = sentiment_record["content_id"]
        user_id = sentiment_record["user_id"]
        sentiment_score = sentiment_record["sentiment_score"]
        sentiment_label = sentiment_record["sentiment_label"]
        keywords = sentiment_record["keywords"]
        timestamp = sentiment_record["timestamp"]
        
        # Actualizar métricas de contenido
        if content_id not in self.content_sentiment:
            self.content_sentiment[content_id] = {
                "total_comments": 0,
                "sentiment_distribution": {
                    "very_positive": 0,
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "very_negative": 0
                },
                "average_sentiment": 0,
                "sentiment_timeline": {},
                "top_keywords": {},
                "first_comment": timestamp,
                "last_comment": timestamp
            }
        
        content_metric = self.content_sentiment[content_id]
        content_metric["total_comments"] += 1
        content_metric["sentiment_distribution"][sentiment_label] += 1
        content_metric["last_comment"] = timestamp
        
        # Actualizar promedio de sentimiento
        total_score = (content_metric["average_sentiment"] * (content_metric["total_comments"] - 1) + 
                      sentiment_score)
        content_metric["average_sentiment"] = total_score / content_metric["total_comments"]
        
        # Actualizar timeline de sentimiento (por día)
        comment_date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
        if comment_date not in content_metric["sentiment_timeline"]:
            content_metric["sentiment_timeline"][comment_date] = {
                "count": 0,
                "average": 0
            }
        
        timeline_entry = content_metric["sentiment_timeline"][comment_date]
        total_score = (timeline_entry["average"] * timeline_entry["count"] + sentiment_score)
        timeline_entry["count"] += 1
        timeline_entry["average"] = total_score / timeline_entry["count"]
        
        # Actualizar palabras clave
        for keyword in keywords:
            keyword_text = keyword["text"]
            if keyword_text not in content_metric["top_keywords"]:
                content_metric["top_keywords"][keyword_text] = {
                    "count": 0,
                    "sentiment": 0
                }
            
            keyword_entry = content_metric["top_keywords"][keyword_text]
            keyword_entry["count"] += 1
            # Actualizar sentimiento promedio de la palabra clave
            keyword_entry["sentiment"] = ((keyword_entry["sentiment"] * (keyword_entry["count"] - 1) + 
                                         keyword["sentiment"]) / keyword_entry["count"])
        
        # Actualizar métricas de usuario
        if user_id not in self.user_sentiment:
            self.user_sentiment[user_id] = {
                "total_comments": 0,
                "sentiment_distribution": {
                    "very_positive": 0,
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "very_negative": 0
                },
                "average_sentiment": 0,
                "sentiment_timeline": {},
                "content_sentiment": {},
                "first_comment": timestamp,
                "last_comment": timestamp
            }
        
        user_metric = self.user_sentiment[user_id]
        user_metric["total_comments"] += 1
        user_metric["sentiment_distribution"][sentiment_label] += 1
        user_metric["last_comment"] = timestamp
        
        # Actualizar promedio de sentimiento
        total_score = (user_metric["average_sentiment"] * (user_metric["total_comments"] - 1) + 
                      sentiment_score)
        user_metric["average_sentiment"] = total_score / user_metric["total_comments"]
        
        # Actualizar timeline de sentimiento (por día)
        if comment_date not in user_metric["sentiment_timeline"]:
            user_metric["sentiment_timeline"][comment_date] = {
                "count": 0,
                "average": 0
            }
        
        timeline_entry = user_metric["sentiment_timeline"][comment_date]
        total_score = (timeline_entry["average"] * timeline_entry["count"] + sentiment_score)
        timeline_entry["count"] += 1
        timeline_entry["average"] = total_score / timeline_entry["count"]
        
        # Actualizar sentimiento por contenido
        if content_id not in user_metric["content_sentiment"]:
            user_metric["content_sentiment"][content_id] = {
                "count": 0,
                "average": 0
            }
        
        content_entry = user_metric["content_sentiment"][content_id]
        total_score = (content_entry["average"] * content_entry["count"] + sentiment_score)
        content_entry["count"] += 1
        content_entry["average"] = total_score / content_entry["count"]
        
        # Actualizar métricas de palabras clave
        for keyword in keywords:
            keyword_text = keyword["text"]
            if keyword_text not in self.keyword_sentiment:
                self.keyword_sentiment[keyword_text] = {
                    "count": 0,
                    "average_sentiment": 0,
                    "content_mentions": {},
                    "first_mention": timestamp,
                    "last_mention": timestamp
                }
            
            keyword_metric = self.keyword_sentiment[keyword_text]
            keyword_metric["count"] += 1
            keyword_metric["last_mention"] = timestamp
            
            # Actualizar promedio de sentimiento
            total_score = (keyword_metric["average_sentiment"] * (keyword_metric["count"] - 1) + 
                          keyword["sentiment"])
            keyword_metric["average_sentiment"] = total_score / keyword_metric["count"]
            
            # Actualizar menciones por contenido
            if content_id not in keyword_metric["content_mentions"]:
                keyword_metric["content_mentions"][content_id] = 0
            keyword_metric["content_mentions"][content_id] += 1
    
    def get_content_sentiment(self, content_id: str) -> Dict[str, Any]:
        """
        Obtiene métricas de sentimiento para un contenido específico.
        
        Args:
            content_id: ID del contenido
            
        Returns:
            Métricas de sentimiento del contenido
        """
        if content_id not in self.content_sentiment:
            logger.warning(f"No hay datos de sentimiento para el contenido: {content_id}")
            return {
                "content_id": content_id,
                "sentiment_score": 0,
                "total_comments": 0,
                "sentiment_distribution": {
                    "very_positive": 0,
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "very_negative": 0
                },
                "top_keywords": []
            }
        
        metrics = self.content_sentiment[content_id]
        
        # Calcular porcentaje de sentimiento positivo
        total = metrics["total_comments"]
        positive_count = metrics["sentiment_distribution"]["very_positive"] + metrics["sentiment_distribution"]["positive"]
        positive_percentage = (positive_count / total) if total > 0 else 0
        
        # Ordenar palabras clave por frecuencia
        sorted_keywords = sorted(
            [{"text": k, "count": v["count"], "sentiment": v["sentiment"]} 
             for k, v in metrics["top_keywords"].items()],
            key=lambda x: x["count"],
            reverse=True
        )
        
        return {
            "content_id": content_id,
            "sentiment_score": round(metrics["average_sentiment"], 2),
            "positive_percentage": round(positive_percentage * 100, 1),
            "total_comments": metrics["total_comments"],
            "sentiment_distribution": metrics["sentiment_distribution"],
            "top_keywords": sorted_keywords[:10],
            "sentiment_timeline": metrics["sentiment_timeline"],
            "first_comment": metrics["first_comment"],
            "last_comment": metrics["last_comment"]
        }
    
    def get_user_sentiment(self, user_id: str) -> Dict[str, Any]:
        """
        Obtiene métricas de sentimiento para un usuario específico.
        
        Args:
            user_id: ID del usuario
            
        Returns:
            Métricas de sentimiento del usuario
        """
        if user_id not in self.user_sentiment:
            logger.warning(f"No hay datos de sentimiento para el usuario: {user_id}")
            return {
                "user_id": user_id,
                "sentiment_score": 0,
                "total_comments": 0,
                "sentiment_distribution": {
                    "very_positive": 0,
                    "positive": 0,
                    "neutral": 0,
                    "negative": 0,
                    "very_negative": 0
                }
            }
        
        metrics = self.user_sentiment[user_id]
        
        # Ordenar contenido por número de comentarios
        sorted_content = sorted(
            [{"content_id": k, "count": v["count"], "sentiment": v["average"]} 
             for k, v in metrics["content_sentiment"].items()],
            key=lambda x: x["count"],
            reverse=True
        )
        
        return {
            "user_id": user_id,
            "sentiment_score": round(metrics["average_sentiment"], 2),
            "total_comments": metrics["total_comments"],
            "sentiment_distribution": metrics["sentiment_distribution"],
            "content_sentiment": sorted_content[:10],
            "sentiment_timeline": metrics["sentiment_timeline"],
            "first_comment": metrics["first_comment"],
            "last_comment": metrics["last_comment"]
        }
    
    def get_keyword_sentiment(self, keyword: str) -> Dict[str, Any]:
        """
        Obtiene métricas de sentimiento para una palabra clave específica.
        
        Args:
            keyword: Palabra clave
            
        Returns:
            Métricas de sentimiento de la palabra clave
        """
        if keyword not in self.keyword_sentiment:
            logger.warning(f"No hay datos de sentimiento para la palabra clave: {keyword}")
            return {
                "keyword": keyword,
                "sentiment_score": 0,
                "mention_count": 0,
                "content_mentions": []
            }
        
        metrics = self.keyword_sentiment[keyword]
        
        # Ordenar contenido por número de menciones
        sorted_content = sorted(
            [{"content_id": k, "count": v} 
             for k, v in metrics["content_mentions"].items()],
            key=lambda x: x["count"],
            reverse=True
        )
        
        return {
            "keyword": keyword,
            "sentiment_score": round(metrics["average_sentiment"], 2),
            "mention_count": metrics["count"],
            "content_mentions": sorted_content[:10],
            "first_mention": metrics["first_mention"],
            "last_mention": metrics["last_mention"]
        }
    
    def get_sentiment_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Obtiene tendencias de sentimiento durante un período de tiempo.
        
        Args:
            days: Número de días a analizar
            
        Returns:
            Tendencias de sentimiento
        """
        # Calcular fecha de inicio
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Inicializar datos de tendencias
        trends = {
            "daily_sentiment": {},
            "sentiment_distribution": {
                "very_positive": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "very_negative": 0
            },
            "total_comments": 0,
            "average_sentiment": 0,
            "trending_keywords": {}
        }
        
        # Generar fechas para el período
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            trends["daily_sentiment"][date_str] = {
                "count": 0,
                "average": 0
            }
            current_date += timedelta(days=1)
        
        # Filtrar comentarios por fecha
        filtered_comments = []
        total_sentiment = 0
        
        for comment in self.sentiment_data:
            try:
                comment_date = datetime.fromisoformat(comment["timestamp"])
                if start_date <= comment_date <= end_date:
                    filtered_comments.append(comment)
                    
                    # Actualizar contadores
                    date_str = comment_date.strftime("%Y-%m-%d")
                    
                    # Actualizar sentimiento diario
                    daily = trends["daily_sentiment"][date_str]
                    total_score = (daily["average"] * daily["count"] + comment["sentiment_score"])
                    daily["count"] += 1
                    daily["average"] = total_score / daily["count"] if daily["count"] > 0 else 0
                    
                    # Actualizar distribución de sentimiento
                    trends["sentiment_distribution"][comment["sentiment_label"]] += 1
                    
                    # Actualizar total y promedio
                    trends["total_comments"] += 1
                    total_sentiment += comment["sentiment_score"]
                    
                    # Actualizar palabras clave
                    for keyword in comment["keywords"]:
                        keyword_text = keyword["text"]
                        if keyword_text not in trends["trending_keywords"]:
                            trends["trending_keywords"][keyword_text] = {
                                "count": 0,
                                "sentiment": 0
                            }
                        
                        keyword_entry = trends["trending_keywords"][keyword_text]
                        keyword_entry["count"] += 1
                        # Actualizar sentimiento promedio
                        keyword_entry["sentiment"] = ((keyword_entry["sentiment"] * (keyword_entry["count"] - 1) + 
                                                     keyword["sentiment"]) / keyword_entry["count"])
            
            except (ValueError, KeyError):
                continue
        
        # Calcular promedio general
        if trends["total_comments"] > 0:
            trends["average_sentiment"] = total_sentiment / trends["total_comments"]
        
        # Ordenar palabras clave por frecuencia
        sorted_keywords = sorted(
            [{"text": k, "count": v["count"], "sentiment": v["sentiment"]} 
             for k, v in trends["trending_keywords"].items()],
            key=lambda x: x["count"],
            reverse=True
        )
        
        trends["trending_keywords"] = sorted_keywords[:20]
        
        return trends
    
    def update_sentiment_lexicon(self, updates: Dict[str, float]) -> bool:
        """
        Actualiza el léxico de sentimiento con nuevas palabras o valores.
        
        Args:
            updates: Diccionario de actualizaciones {palabra: valor}
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        try:
            # Validar valores
            for word, value in updates.items():
                if not isinstance(value, (int, float)):
                    logger.error(f"Valor no válido para '{word}': {value}")
                    return False
                
                # Asegurar que el valor esté entre -1 y 1
                updates[word] = max(-1.0, min(1.0, float(value)))
            
            # Actualizar léxico
            self.sentiment_lexicon.update(updates)
            
            # Guardar léxico actualizado
            lexicon_path = os.path.join(self.data_path, "sentiment_lexicon.json")
            with open(lexicon_path, "w", encoding="utf-8") as f:
                json.dump(self.sentiment_lexicon, f, indent=4)
            
            logger.info(f"Léxico de sentimiento actualizado: {len(updates)} palabras")
            return True
        
        except Exception as e:
            logger.error(f"Error al actualizar léxico de sentimiento: {str(e)}")
            return False
    
    def export_sentiment_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None, format_type: str = "json") -> Dict[str, Any]:
        """
        Exporta datos de sentimiento para análisis externo.
        
        Args:
            start_date: Fecha de inicio (opcional)
            end_date: Fecha de fin (opcional)
            format_type: Formato de exportación (json, csv)
            
        Returns:
            Datos de exportación y ruta del archivo
        """
        # Validar fechas
        if start_date:
            try:
                start_date_obj = datetime.fromisoformat(start_date)
            except ValueError:
                logger.error(f"Formato de fecha de inicio no válido: {start_date}")
                return {
                    "status": "error",
                    "message": "Formato de fecha de inicio no válido"
                }
        else:
            # Por defecto, último mes
            start_date_obj = datetime.now() - timedelta(days=30)
            start_date = start_date_obj.isoformat()
        
        if end_date:
            try:
                end_date_obj = datetime.fromisoformat(end_date)
            except ValueError:
                logger.error(f"Formato de fecha de fin no válido: {end_date}")
                return {
                    "status": "error",
                    "message": "Formato de fecha de fin no válido"
                }
        else:
            end_date_obj = datetime.now()
            end_date = end_date_obj.isoformat()
        
        # Validar formato
        valid_formats = ["json", "csv"]
        if format_type not in valid_formats:
            logger.error(f"Formato no válido: {format_type}")
            return {
                "status": "error",
                "message": f"Formato no válido. Debe ser uno de: {', '.join(valid_formats)}"
            }
        
        # Filtrar comentarios por fecha
        filtered_comments = []
        for comment in self.sentiment_data:
            try:
                comment_date = datetime.fromisoformat(comment["timestamp"])
                if start_date_obj <= comment_date <= end_date_obj:
                    filtered_comments.append(comment)
            except ValueError:
                continue
        
        # Crear directorio de exportación
        export_dir = "data/engagement/exports"
        os.makedirs(export_dir, exist_ok=True)
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sentiment_export_{timestamp}.{format_type}"
        filepath = os.path.join(export_dir, filename)
        
        # Exportar datos
        if format_type == "json":
            export_data = {
                "metadata": {
                    "export_date": datetime.now().isoformat(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "comment_count": len(filtered_comments)
                },
                "comments": filtered_comments,
                "content_sentiment": self.content_sentiment,
                "keyword_sentiment": self.keyword_sentiment
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4)
        
        elif format_type == "csv":
            import csv
            
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                # Definir campos
                fieldnames = [
                    "analysis_id", "timestamp", "user_id", "content_id", "platform", 
                    "sentiment_score", "sentiment_label", "confidence", "text"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Escribir comentarios
                for comment in filtered_comments:
                    # Crear fila con solo los campos definidos
                    row = {field: comment.get(field, "") for field in fieldnames}
                    writer.writerow(row)
        
        logger.info(f"Datos de sentimiento exportados: {filepath}")
        
        return {
                        "status": "success",
            "message": "Datos exportados correctamente",
            "filepath": filepath,
            "format": format_type,
            "record_count": len(filtered_comments)
        }
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen general de las métricas de sentimiento.
        
        Returns:
            Resumen de métricas de sentimiento
        """
        # Inicializar resumen
        summary = {
            "total_comments": len(self.sentiment_data),
            "total_content": len(self.content_sentiment),
            "total_users": len(self.user_sentiment),
            "total_keywords": len(self.keyword_sentiment),
            "sentiment_distribution": {
                "very_positive": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "very_negative": 0
            },
            "average_sentiment": 0,
            "top_content": [],
            "top_users": [],
            "top_keywords": []
        }
        
        # Si no hay datos, devolver resumen vacío
        if not self.sentiment_data:
            return summary
        
        # Calcular distribución de sentimiento y promedio
        total_sentiment = 0
        for comment in self.sentiment_data:
            summary["sentiment_distribution"][comment["sentiment_label"]] += 1
            total_sentiment += comment["sentiment_score"]
        
        # Calcular promedio general
        summary["average_sentiment"] = round(total_sentiment / len(self.sentiment_data), 2)
        
        # Obtener contenido más comentado
        sorted_content = sorted(
            [(content_id, data["total_comments"], data["average_sentiment"]) 
             for content_id, data in self.content_sentiment.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        summary["top_content"] = [
            {
                "content_id": content_id,
                "total_comments": comment_count,
                "sentiment_score": round(sentiment, 2)
            }
            for content_id, comment_count, sentiment in sorted_content[:10]
        ]
        
        # Obtener usuarios más activos
        sorted_users = sorted(
            [(user_id, data["total_comments"], data["average_sentiment"]) 
             for user_id, data in self.user_sentiment.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        summary["top_users"] = [
            {
                "user_id": user_id,
                "total_comments": comment_count,
                "sentiment_score": round(sentiment, 2)
            }
            for user_id, comment_count, sentiment in sorted_users[:10]
        ]
        
        # Obtener palabras clave más mencionadas
        sorted_keywords = sorted(
            [(keyword, data["count"], data["average_sentiment"]) 
             for keyword, data in self.keyword_sentiment.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        summary["top_keywords"] = [
            {
                "keyword": keyword,
                "mention_count": mention_count,
                "sentiment_score": round(sentiment, 2)
            }
            for keyword, mention_count, sentiment in sorted_keywords[:20]
        ]
        
        # Calcular porcentaje de sentimiento positivo
        positive_count = (summary["sentiment_distribution"]["very_positive"] + 
                         summary["sentiment_distribution"]["positive"])
        total_count = len(self.sentiment_data)
        summary["positive_percentage"] = round((positive_count / total_count) * 100, 1) if total_count > 0 else 0
        
        return summary
    
    def compare_content_sentiment(self, content_ids: List[str]) -> Dict[str, Any]:
        """
        Compara el sentimiento entre diferentes contenidos.
        
        Args:
            content_ids: Lista de IDs de contenido a comparar
            
        Returns:
            Comparación de sentimiento entre contenidos
        """
        if not content_ids or not isinstance(content_ids, list):
            logger.error("Lista de IDs de contenido no válida")
            return {
                "status": "error",
                "message": "Lista de IDs de contenido no válida"
            }
        
        # Filtrar contenidos existentes
        valid_content_ids = [cid for cid in content_ids if cid in self.content_sentiment]
        
        if not valid_content_ids:
            logger.warning("Ninguno de los IDs de contenido proporcionados tiene datos de sentimiento")
            return {
                "status": "warning",
                "message": "Ninguno de los IDs de contenido proporcionados tiene datos de sentimiento",
                "content_comparison": []
            }
        
        # Preparar datos de comparación
        comparison = []
        
        for content_id in valid_content_ids:
            metrics = self.content_sentiment[content_id]
            
            # Calcular porcentaje de sentimiento positivo
            total = metrics["total_comments"]
            positive_count = (metrics["sentiment_distribution"]["very_positive"] + 
                             metrics["sentiment_distribution"]["positive"])
            positive_percentage = (positive_count / total) * 100 if total > 0 else 0
            
            # Obtener palabras clave principales
            sorted_keywords = sorted(
                [(k, v["count"], v["sentiment"]) 
                 for k, v in metrics["top_keywords"].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            comparison.append({
                "content_id": content_id,
                "total_comments": metrics["total_comments"],
                "sentiment_score": round(metrics["average_sentiment"], 2),
                "positive_percentage": round(positive_percentage, 1),
                "sentiment_distribution": metrics["sentiment_distribution"],
                "top_keywords": [
                    {"text": k, "count": c, "sentiment": round(s, 2)} 
                    for k, c, s in sorted_keywords
                ],
                "first_comment": metrics["first_comment"],
                "last_comment": metrics["last_comment"]
            })
        
        # Ordenar por puntuación de sentimiento (de mayor a menor)
        comparison.sort(key=lambda x: x["sentiment_score"], reverse=True)
        
        return {
            "status": "success",
            "message": f"Comparación de {len(valid_content_ids)} contenidos",
            "content_comparison": comparison
        }
    
    def detect_sentiment_shifts(self, content_id: str, window_days: int = 7) -> Dict[str, Any]:
        """
        Detecta cambios significativos en el sentimiento a lo largo del tiempo.
        
        Args:
            content_id: ID del contenido a analizar
            window_days: Tamaño de la ventana de tiempo en días
            
        Returns:
            Análisis de cambios de sentimiento
        """
        if content_id not in self.content_sentiment:
            logger.warning(f"No hay datos de sentimiento para el contenido: {content_id}")
            return {
                "status": "warning",
                "message": f"No hay datos de sentimiento para el contenido: {content_id}",
                "shifts": []
            }
        
        metrics = self.content_sentiment[content_id]
        timeline = metrics["sentiment_timeline"]
        
        if not timeline:
            return {
                "status": "warning",
                "message": "No hay suficientes datos para detectar cambios",
                "shifts": []
            }
        
        # Ordenar fechas
        dates = sorted(timeline.keys())
        
        if len(dates) < window_days:
            return {
                "status": "warning",
                "message": f"No hay suficientes datos para una ventana de {window_days} días",
                "shifts": []
            }
        
        # Calcular cambios de sentimiento
        shifts = []
        
        for i in range(len(dates) - window_days):
            start_window = dates[i:i+window_days]
            end_window = dates[i+1:i+window_days+1]
            
            # Calcular promedio de sentimiento para cada ventana
            start_avg = sum(timeline[d]["average"] * timeline[d]["count"] for d in start_window) / sum(timeline[d]["count"] for d in start_window)
            end_avg = sum(timeline[d]["average"] * timeline[d]["count"] for d in end_window) / sum(timeline[d]["count"] for d in end_window)
            
            # Calcular cambio
            change = end_avg - start_avg
            
            # Detectar cambios significativos (más de 0.2 en escala de -1 a 1)
            if abs(change) >= 0.2:
                shifts.append({
                    "start_date": start_window[0],
                    "end_date": end_window[-1],
                    "start_sentiment": round(start_avg, 2),
                    "end_sentiment": round(end_avg, 2),
                    "change": round(change, 2),
                    "direction": "positive" if change > 0 else "negative"
                })
        
        return {
            "status": "success",
            "content_id": content_id,
            "total_days": len(dates),
            "window_size": window_days,
            "shifts": shifts
        }
    
    def batch_analyze_comments(self, comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza un lote de comentarios en una sola operación.
        
        Args:
            comments: Lista de comentarios a analizar
            
        Returns:
            Resultados del análisis por lotes
        """
        if not comments or not isinstance(comments, list):
            logger.error("Lista de comentarios no válida")
            return {
                "status": "error",
                "message": "Lista de comentarios no válida"
            }
        
        results = []
        success_count = 0
        error_count = 0
        
        # Procesar cada comentario
        for comment in comments:
            try:
                # Verificar campos requeridos
                required_fields = ["text", "user_id", "content_id", "platform"]
                missing_fields = [field for field in required_fields if field not in comment]
                
                if missing_fields:
                    error_result = {
                        "status": "error",
                        "message": f"Faltan campos requeridos: {', '.join(missing_fields)}",
                        "original_comment": comment
                    }
                    results.append(error_result)
                    error_count += 1
                    continue
                
                # Analizar comentario
                analysis_result = self.analyze_comment(comment)
                
                # Añadir resultado
                results.append({
                    "status": "success",
                    "analysis_id": analysis_result["analysis_id"],
                    "sentiment_score": analysis_result["sentiment_score"],
                    "sentiment_label": analysis_result["sentiment_label"],
                    "confidence": analysis_result["confidence"]
                })
                
                success_count += 1
            
            except Exception as e:
                logger.error(f"Error al analizar comentario: {str(e)}")
                results.append({
                    "status": "error",
                    "message": f"Error al analizar comentario: {str(e)}",
                    "original_comment": comment
                })
                error_count += 1
        
        return {
            "status": "completed",
            "total_processed": len(comments),
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        }
    
    def reset_sentiment_data(self, confirm: bool = False) -> Dict[str, Any]:
        """
        Reinicia todos los datos de sentimiento (¡PRECAUCIÓN!).
        
        Args:
            confirm: Confirmación para reiniciar datos
            
        Returns:
            Estado de la operación
        """
        if not confirm:
            return {
                "status": "warning",
                "message": "Operación cancelada. Se requiere confirmación explícita para reiniciar datos."
            }
        
        try:
            # Hacer copia de seguridad antes de reiniciar
            backup_dir = os.path.join(self.data_path, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Guardar copia de seguridad
            backup_files = {
                "sentiment_data.json": self.sentiment_data,
                "content_sentiment.json": self.content_sentiment,
                "user_sentiment.json": self.user_sentiment,
                "keyword_sentiment.json": self.keyword_sentiment
            }
            
            for filename, data in backup_files.items():
                backup_path = os.path.join(backup_dir, f"{filename}.{timestamp}.bak")
                with open(backup_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
            
            # Reiniciar datos
            self.sentiment_data = []
            self.content_sentiment = {}
            self.user_sentiment = {}
            self.keyword_sentiment = {}
            
            # Guardar datos vacíos
            self._save_data()
            
            logger.warning("Todos los datos de sentimiento han sido reiniciados")
            
            return {
                "status": "success",
                "message": "Todos los datos de sentimiento han sido reiniciados",
                "backup_location": backup_dir
            }
        
        except Exception as e:
            logger.error(f"Error al reiniciar datos: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al reiniciar datos: {str(e)}"
            }