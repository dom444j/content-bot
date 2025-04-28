"""
Herramientas para análisis de sentimiento de comentarios y feedback de usuarios.
Proporciona funciones para evaluar la polaridad y subjetividad del texto.
"""

import re
from typing import Dict, List, Tuple, Union, Optional
import numpy as np

# Intenta importar bibliotecas populares de análisis de sentimiento
# con manejo de errores para permitir instalación bajo demanda
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

class SentimentAnalyzer:
    """Clase para analizar el sentimiento de textos."""
    
    def __init__(self, lang: str = "es"):
        """
        Inicializa el analizador de sentimiento.
        
        Args:
            lang: Código de idioma (es, en, etc.)
        """
        self.lang = lang
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # Diccionarios de palabras positivas/negativas en español
        # Esto es un conjunto básico que deberías expandir
        self.positive_words_es = {
            "bueno", "excelente", "increíble", "genial", "fantástico", 
            "maravilloso", "útil", "recomendado", "me gusta", "encanta",
            "gracias", "perfecto", "top", "crack", "brutal", "espectacular"
        }
        
        self.negative_words_es = {
            "malo", "terrible", "horrible", "pésimo", "inútil", 
            "decepcionante", "aburrido", "odio", "no recomendado", "estafa",
            "basura", "timo", "pérdida", "tiempo", "mentira", "falso"
        }
        
        # Emojis comunes y su polaridad
        self.emoji_sentiment = {
            "😀": 0.8, "😃": 0.8, "😄": 0.8, "😁": 0.8, "😆": 0.7, 
            "😅": 0.6, "🤣": 0.7, "😂": 0.7, "🙂": 0.5, "😊": 0.8,
            "😍": 0.9, "🥰": 0.9, "❤️": 0.9, "👍": 0.7, "🔥": 0.8,
            "😢": -0.6, "😭": -0.7, "😡": -0.8, "👎": -0.7, "🤮": -0.9,
            "😠": -0.8, "😤": -0.7, "😒": -0.6, "😔": -0.5, "😟": -0.6
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analiza el sentimiento del texto proporcionado.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con puntuaciones de sentimiento
        """
        if not text or text.strip() == "":
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        # Normalizar texto
        text = text.lower()
        
        # Análisis basado en el idioma
        if self.lang == "es":
            return self._analyze_spanish(text)
        else:
            return self._analyze_english(text)
    
    def _analyze_spanish(self, text: str) -> Dict[str, float]:
        """Análisis específico para español."""
        # Contar palabras positivas y negativas
        words = re.findall(r'\b\w+\b', text)
        positive_count = sum(1 for word in words if word in self.positive_words_es)
        negative_count = sum(1 for word in words if word in self.negative_words_es)
        
        # Analizar emojis
        emoji_score = 0
        emoji_count = 0
        for emoji, score in self.emoji_sentiment.items():
            count = text.count(emoji)
            if count > 0:
                emoji_score += score * count
                emoji_count += count
        
        # Calcular puntuaciones
        total_words = len(words) if words else 1
        positive = positive_count / total_words if total_words > 0 else 0
        negative = negative_count / total_words if total_words > 0 else 0
        
        # Incorporar puntuación de emojis
        if emoji_count > 0:
            emoji_weight = min(0.5, emoji_count / 10)  # Limitar influencia de emojis
            text_weight = 1 - emoji_weight
            
            # Ajustar puntuaciones con emojis
            if emoji_score > 0:
                positive = (positive * text_weight) + (emoji_score * emoji_weight)
            elif emoji_score < 0:
                negative = (negative * text_weight) + (abs(emoji_score) * emoji_weight)
        
        # Calcular puntuación compuesta
        compound = positive - negative
        neutral = 1.0 - (positive + negative)
        
        return {
            "compound": compound,
            "positive": positive,
            "negative": negative,
            "neutral": neutral
        }
    
    def _analyze_english(self, text: str) -> Dict[str, float]:
        """Análisis para inglés usando VADER o TextBlob."""
        # Usar VADER si está disponible
        if VADER_AVAILABLE:
            scores = self.vader.polarity_scores(text)
            return scores
        
        # Usar TextBlob como respaldo
        elif TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convertir al formato de VADER para consistencia
            if polarity > 0:
                pos = polarity
                neg = 0
            else:
                pos = 0
                neg = abs(polarity)
                
            return {
                "compound": polarity,
                "positive": pos,
                "negative": neg,
                "neutral": 1.0 - subjectivity
            }
        
        # Método básico si no hay bibliotecas disponibles
        else:
            return self._analyze_spanish(text)  # Usar el método básico
    
    def get_sentiment_label(self, text: str) -> str:
        """
        Obtiene una etiqueta de sentimiento (positivo, negativo, neutral).
        
        Args:
            text: Texto a analizar
            
        Returns:
            Etiqueta de sentimiento
        """
        scores = self.analyze(text)
        compound = scores["compound"]
        
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        else:
            return "neutral"
    
    def get_emoji_for_sentiment(self, text: str) -> str:
        """
        Devuelve un emoji representativo del sentimiento del texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Emoji representativo
        """
        label = self.get_sentiment_label(text)
        
        if label == "positive":
            return "😊"
        elif label == "negative":
            return "😔"
        else:
            return "🙂"
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analiza el sentimiento de múltiples textos.
        
        Args:
            texts: Lista de textos a analizar
            
        Returns:
            Lista de resultados de sentimiento
        """
        return [self.analyze(text) for text in texts]

def get_comment_sentiment(comment: str, lang: str = "es") -> Dict[str, float]:
    """
    Función de conveniencia para analizar el sentimiento de un comentario.
    
    Args:
        comment: Texto del comentario
        lang: Código de idioma
        
    Returns:
        Diccionario con puntuaciones de sentimiento
    """
    analyzer = SentimentAnalyzer(lang=lang)
    return analyzer.analyze(comment)

def get_average_sentiment(comments: List[str], lang: str = "es") -> Dict[str, float]:
    """
    Calcula el sentimiento promedio de una lista de comentarios.
    
    Args:
        comments: Lista de comentarios
        lang: Código de idioma
        
    Returns:
        Diccionario con puntuaciones promedio
    """
    if not comments:
        return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
    
    analyzer = SentimentAnalyzer(lang=lang)
    results = analyzer.batch_analyze(comments)
    
    # Calcular promedios
    avg_compound = sum(r["compound"] for r in results) / len(results)
    avg_positive = sum(r["positive"] for r in results) / len(results)
    avg_negative = sum(r["negative"] for r in results) / len(results)
    avg_neutral = sum(r["neutral"] for r in results) / len(results)
    
    return {
        "compound": avg_compound,
        "positive": avg_positive,
        "negative": avg_negative,
        "neutral": avg_neutral
    }

def should_adjust_content(comments: List[str], threshold: float = 0.1) -> bool:
    """
    Determina si el contenido debe ajustarse basado en el sentimiento de los comentarios.
    
    Args:
        comments: Lista de comentarios
        threshold: Umbral para considerar ajuste
        
    Returns:
        True si se debe ajustar el contenido
    """
    sentiment = get_average_sentiment(comments)
    return sentiment["compound"] < -threshold

# Ejemplo de uso
if __name__ == "__main__":
    # Probar con algunos comentarios
    comments = [
        "Me encantó este video, muy útil! 😊",
        "No me gustó nada, pérdida de tiempo 👎",
        "Interesante pero podría ser mejor"
    ]
    
    analyzer = SentimentAnalyzer()
    
    for comment in comments:
        sentiment = analyzer.analyze(comment)
        label = analyzer.get_sentiment_label(comment)
        emoji = analyzer.get_emoji_for_sentiment(comment)
        
        print(f"Comentario: {comment}")
        print(f"Sentimiento: {label} {emoji}")
        print(f"Puntuaciones: {sentiment}")
        print("-" * 50)
    
    # Analizar sentimiento promedio
    avg_sentiment = get_average_sentiment(comments)
    print(f"Sentimiento promedio: {avg_sentiment}")