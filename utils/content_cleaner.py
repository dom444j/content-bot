"""
Herramientas para limpiar y verificar contenido generado.
Incluye funciones para detectar y eliminar contenido con copyright,
lenguaje inapropiado, y asegurar cumplimiento con políticas de plataformas.
"""

import re
import os
import hashlib
from typing import Dict, List, Tuple, Union, Optional, Set
import json
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'errors', 'system_errors', 'content_cleaner.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('content_cleaner')

class ContentCleaner:
    """Clase para limpiar y verificar contenido."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el limpiador de contenido.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path or os.path.join('config', 'compliance.json')
        self.config = self._load_config()
        
        # Cargar listas de palabras prohibidas
        self.prohibited_words = set(self.config.get('prohibited_words', []))
        self.sensitive_topics = set(self.config.get('sensitive_topics', []))
        self.trademark_terms = set(self.config.get('trademark_terms', []))
        
        # Expresiones regulares para detectar problemas comunes
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b')
        
        # Cargar hashes de contenido conocido con copyright
        self.copyright_hashes = set(self.config.get('copyright_hashes', []))
    
    def _load_config(self) -> Dict:
        """
        Carga la configuración desde el archivo.
        
        Returns:
            Diccionario con la configuración
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return self._create_default_config()
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """
        Crea una configuración predeterminada.
        
        Returns:
            Diccionario con configuración predeterminada
        """
        default_config = {
            'prohibited_words': [
                # Palabras prohibidas en español
                "palabrota1", "palabrota2", 
                # Añadir más según sea necesario
            ],
            'sensitive_topics': [
                "política", "religión", "drogas", "violencia", "discriminación"
            ],
            'trademark_terms': [
                "coca-cola", "adidas", "nike", "apple", "microsoft"
            ],
            'copyright_hashes': [],
            'max_similarity_threshold': 0.85,
            'min_originality_score': 0.7
        }
        
        # Guardar configuración predeterminada
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Configuración predeterminada creada en: {self.config_path}")
        except Exception as e:
            logger.error(f"Error al crear configuración predeterminada: {str(e)}")
        
        return default_config
    
    def clean_text(self, text: str) -> str:
        """
        Limpia el texto eliminando contenido problemático.
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        if not text:
            return ""
        
        # Eliminar URLs
        text = self.url_pattern.sub('[URL]', text)
        
        # Eliminar emails
        text = self.email_pattern.sub('[EMAIL]', text)
        
        # Eliminar números de teléfono
        text = self.phone_pattern.sub('[TELÉFONO]', text)
        
        # Censurar palabras prohibidas
        for word in self.prohibited_words:
            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
            text = pattern.sub('[CENSURADO]', text)
        
        return text
    
    def check_copyright(self, text: str) -> Tuple[bool, float]:
        """
        Verifica si el texto puede tener problemas de copyright.
        
        Args:
            text: Texto a verificar
            
        Returns:
            Tupla (tiene_problemas, puntuación_similitud)
        """
        # Calcular hash del texto
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Verificar si el hash está en la lista de contenido con copyright
        if text_hash in self.copyright_hashes:
            return True, 1.0
        
        # Implementar verificación de similitud básica
        # En una implementación real, usarías un servicio como Copyscape o similar
        max_similarity = 0.0
        
        # Aquí simularemos una verificación básica
        # En producción, conectarías con una API externa
        similarity_score = self._simulate_copyright_check(text)
        
        threshold = self.config.get('max_similarity_threshold', 0.85)
        return similarity_score > threshold, similarity_score
    
    def _simulate_copyright_check(self, text: str) -> float:
        """
        Simula una verificación de copyright.
        En producción, esto se conectaría a un servicio externo.
        
        Args:
            text: Texto a verificar
            
        Returns:
            Puntuación de similitud (0-1)
        """
        # Esta es una simulación muy básica
        # En producción, usarías un servicio real
        
        # Detectar frases comunes que podrían indicar contenido copiado
        common_phrases = [
            "como todos sabemos", "es un hecho bien conocido",
            "según estudios recientes", "los expertos coinciden"
        ]
        
        phrase_count = sum(1 for phrase in common_phrases if phrase.lower() in text.lower())
        
        # Calcular una puntuación básica basada en frases comunes
        base_score = min(0.5, phrase_count * 0.1)
        
        # Añadir aleatoriedad para simular
        import random
        random_factor = random.uniform(0, 0.3)
        
        return base_score + random_factor
    
    def check_trademarks(self, text: str) -> List[str]:
        """
        Identifica términos de marca registrada en el texto.
        
        Args:
            text: Texto a verificar
            
        Returns:
            Lista de términos de marca encontrados
        """
        found_terms = []
        
        for term in self.trademark_terms:
            pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                found_terms.append(term)
        
        return found_terms
    
    def check_sensitive_topics(self, text: str) -> List[str]:
        """
        Identifica temas sensibles en el texto.
        
        Args:
            text: Texto a verificar
            
        Returns:
            Lista de temas sensibles encontrados
        """
        found_topics = []
        
        for topic in self.sensitive_topics:
            pattern = re.compile(r'\b' + re.escape(topic) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                found_topics.append(topic)
        
        return found_topics
    
    def calculate_originality_score(self, text: str) -> float:
        """
        Calcula una puntuación de originalidad para el texto.
        
        Args:
            text: Texto a evaluar
            
        Returns:
            Puntuación de originalidad (0-1)
        """
        # Verificar copyright
        has_copyright, similarity = self.check_copyright(text)
        
        # Verificar marcas
        trademark_terms = self.check_trademarks(text)
        
        # Verificar temas sensibles
        sensitive_topics = self.check_sensitive_topics(text)
        
        # Calcular puntuación base
        base_score = 1.0
        
        # Penalizar por similitud de copyright
        base_score -= similarity * 0.5
        
        # Penalizar por términos de marca
        base_score -= len(trademark_terms) * 0.05
        
        # Penalizar por temas sensibles
        base_score -= len(sensitive_topics) * 0.03
        
        # Asegurar que la puntuación esté en el rango 0-1
        return max(0.0, min(1.0, base_score))
    
    def is_content_safe(self, text: str) -> Tuple[bool, Dict]:
        """
        Determina si el contenido es seguro para publicar.
        
        Args:
            text: Texto a verificar
            
        Returns:
            Tupla (es_seguro, detalles)
        """
        # Verificar copyright
        has_copyright, similarity = self.check_copyright(text)
        
        # Verificar marcas
        trademark_terms = self.check_trademarks(text)
        
        # Verificar temas sensibles
        sensitive_topics = self.check_sensitive_topics(text)
        
        # Calcular puntuación de originalidad
        originality_score = self.calculate_originality_score(text)
        
        # Determinar si es seguro
        min_score = self.config.get('min_originality_score', 0.7)
        is_safe = (
            not has_copyright and
            len(trademark_terms) == 0 and
            len(sensitive_topics) == 0 and
            originality_score >= min_score
        )
        
        # Preparar detalles
        details = {
            "has_copyright": has_copyright,
            "copyright_similarity": similarity,
            "trademark_terms": trademark_terms,
            "sensitive_topics": sensitive_topics,
            "originality_score": originality_score,
            "is_safe": is_safe
        }
        
        return is_safe, details
    
    def clean_and_verify(self, text: str) -> Tuple[str, bool, Dict]:
        """
        Limpia y verifica el texto en un solo paso.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Tupla (texto_limpio, es_seguro, detalles)
        """
        # Limpiar el texto
        cleaned_text = self.clean_text(text)
        
        # Verificar si es seguro
        is_safe, details = self.is_content_safe(cleaned_text)
        
        return cleaned_text, is_safe, details

def clean_text(text: str) -> str:
    """
    Función de conveniencia para limpiar texto.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    cleaner = ContentCleaner()
    return cleaner.clean_text(text)

def is_content_safe(text: str) -> bool:
    """
    Función de conveniencia para verificar si el contenido es seguro.
    
    Args:
        text: Texto a verificar
        
    Returns:
        True si el contenido es seguro
    """
    cleaner = ContentCleaner()
    is_safe, _ = cleaner.is_content_safe(text)
    return is_safe

# Ejemplo de uso
if __name__ == "__main__":
    # Probar con algunos textos
    texts = [
        "Este es un texto normal sin problemas.",
        "Visita mi sitio web en www.ejemplo.com para más información.",
        "Contacta conmigo en ejemplo@correo.com o al teléfono 123-456-7890.",
        "Me encanta tomar Coca-Cola mientras uso mi iPhone de Apple.",
        "Este contenido habla sobre política y religión de manera controversial."
    ]
    
    cleaner = ContentCleaner()
    
    for text in texts:
        cleaned_text = cleaner.clean_text(text)
        is_safe, details = cleaner.is_content_safe(text)
        
        print(f"Texto original: {text}")
        print(f"Texto limpio: {cleaned_text}")
        print(f"¿Es seguro?: {'Sí' if is_safe else 'No'}")
        print(f"Detalles: {details}")
        print("-" * 50)