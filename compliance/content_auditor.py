"""
Content Auditor - Auditoría de Contenido y CTAs

Este módulo se encarga de verificar que el contenido y los CTAs (Calls to Action)
cumplan con las políticas de las plataformas, detectando posibles problemas
antes de la publicación.
"""

import os
import json
import logging
import re
import time
import hashlib
import requests
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import numpy as np
from collections import Counter

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/content_auditor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ContentAuditor")

class ContentAuditor:
    """
    Audita el contenido y los CTAs para asegurar el cumplimiento de las políticas
    de las plataformas y evitar problemas como shadowbans o demonetización.
    """
    
    def __init__(self, config_path: str = "config/platforms.json", 
                 rules_path: str = "compliance/rules/content_rules.json",
                 cta_database_path: str = "data/cta_database.json"):
        """
        Inicializa el auditor de contenido.
        
        Args:
            config_path: Ruta al archivo de configuración de plataformas
            rules_path: Ruta al archivo de reglas de contenido
            cta_database_path: Ruta a la base de datos de CTAs
        """
        self.config_path = config_path
        self.rules_path = rules_path
        self.cta_database_path = cta_database_path
        
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)
        os.makedirs(os.path.dirname(cta_database_path), exist_ok=True)
        
        # Cargar configuración
        self.platforms = self._load_config()
        
        # Cargar reglas
        self.rules = self._load_rules()
        
        # Cargar base de datos de CTAs
        self.cta_database = self._load_cta_database()
        
        # Inicializar contadores de uso de CTAs
        self.cta_usage_counter = Counter()
        for cta_id, cta_data in self.cta_database.items():
            self.cta_usage_counter[cta_id] = cta_data.get("usage_count", 0)
        
        logger.info(f"ContentAuditor inicializado con {len(self.rules)} reglas y {len(self.cta_database)} CTAs")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración de plataformas"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def _load_rules(self) -> Dict[str, Any]:
        """Carga las reglas de contenido"""
        try:
            if os.path.exists(self.rules_path):
                with open(self.rules_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.info(f"Creando nuevas reglas de contenido")
                # Reglas predeterminadas
                default_rules = {
                    "global": {
                        "prohibited_words": [
                            "spam", "scam", "estafa", "hack", "crackear", "piratear",
                            "ilegal", "illegal", "pornografía", "pornography", "xxx",
                            "violencia extrema", "extreme violence", "drogas ilegales",
                            "illegal drugs", "hate speech", "discurso de odio"
                        ],
                        "suspicious_patterns": [
                            r"\b(?:ganar|earn)\s+\d+[k]?\s+(?:dólares|euros|USD|EUR)\b",
                            r"\b(?:garantizado|guaranteed)\b",
                            r"\b(?:secreto|secret)\s+(?:revelado|revealed)\b",
                            r"\b(?:médicos|doctors)\s+(?:odian|hate)\b",
                            r"\b(?:100%|cien por ciento)\s+(?:efectivo|effective)\b"
                        ],
                        "max_cta_frequency": 3,  # Máximo número de CTAs por minuto
                        "min_cta_interval": 15,  # Segundos mínimos entre CTAs
                        "max_hashtags": 30,      # Máximo número de hashtags
                        "max_mentions": 20,      # Máximo número de menciones
                        "max_emojis_title": 10,  # Máximo número de emojis en título
                        "max_caps_percentage": 30  # Porcentaje máximo de mayúsculas
                    },
                    "youtube": {
                        "prohibited_words": [
                            "clickbait", "sub4sub", "like4like", "subscribe", "suscríbete"
                        ],
                        "max_hashtags": 15,
                        "title_max_length": 100,
                        "description_max_length": 5000,
                        "max_tags": 500,
                        "suspicious_patterns": [
                            r"\b(?:sub|suscríbete)\s+(?:y|and)\s+(?:gana|win)\b"
                        ]
                    },
                    "tiktok": {
                        "prohibited_words": [
                            "instagram", "youtube", "facebook", "twitter", "follow4follow"
                        ],
                        "max_hashtags": 30,
                        "caption_max_length": 150,
                        "suspicious_patterns": [
                            r"\blink\s+(?:in|en)\s+(?:bio|biografía)\b"
                        ]
                    },
                    "instagram": {
                        "prohibited_words": [
                            "follow4follow", "like4like", "tiktok", "youtube"
                        ],
                        "max_hashtags": 30,
                        "caption_max_length": 2200,
                        "suspicious_patterns": [
                            r"\blink\s+(?:in|en)\s+(?:bio|biografía)\b"
                        ]
                    }
                }
                
                # Guardar reglas predeterminadas
                os.makedirs(os.path.dirname(self.rules_path), exist_ok=True)
                with open(self.rules_path, 'w', encoding='utf-8') as f:
                    json.dump(default_rules, f, indent=2, ensure_ascii=False)
                
                return default_rules
        except Exception as e:
            logger.error(f"Error al cargar reglas: {str(e)}")
            return {
                "global": {
                    "prohibited_words": [],
                    "suspicious_patterns": [],
                    "max_cta_frequency": 3,
                    "min_cta_interval": 15,
                    "max_hashtags": 30,
                    "max_mentions": 20,
                    "max_emojis_title": 10,
                    "max_caps_percentage": 30
                }
            }
    
    def _load_cta_database(self) -> Dict[str, Any]:
        """Carga la base de datos de CTAs"""
        try:
            if os.path.exists(self.cta_database_path):
                with open(self.cta_database_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.info(f"Creando nueva base de datos de CTAs")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar base de datos de CTAs: {str(e)}")
            return {}
    
    def _save_cta_database(self) -> bool:
        """Guarda la base de datos de CTAs"""
        try:
            with open(self.cta_database_path, 'w', encoding='utf-8') as f:
                json.dump(self.cta_database, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error al guardar base de datos de CTAs: {str(e)}")
            return False
    
    def register_cta(self, cta_text: str, cta_type: str, platform: str, 
                    timing: float, performance_data: Dict[str, Any] = None) -> str:
        """
        Registra un nuevo CTA en la base de datos
        
        Args:
            cta_text: Texto del CTA
            cta_type: Tipo de CTA (e.g., "follow", "subscribe", "comment", "like")
            platform: Plataforma objetivo
            timing: Tiempo en segundos donde aparece el CTA
            performance_data: Datos de rendimiento iniciales
            
        Returns:
            ID del CTA registrado
        """
        # Generar ID único
        cta_id = hashlib.md5(f"{cta_text}_{platform}_{int(time.time())}".encode()).hexdigest()
        
        # Datos de rendimiento por defecto
        if performance_data is None:
            performance_data = {
                "conversion_rate": 0.0,
                "engagement_rate": 0.0,
                "retention_rate": 0.0,
                "samples": 0
            }
        
        # Registrar CTA
        self.cta_database[cta_id] = {
            "text": cta_text,
            "type": cta_type,
            "platform": platform,
            "timing": timing,
            "creation_date": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "usage_count": 0,
            "performance": performance_data,
            "compliance_score": 0.0,
            "flags": []
        }
        
        # Actualizar base de datos
        self._save_cta_database()
        
        logger.info(f"CTA registrado: {cta_id} - '{cta_text}' para {platform}")
        return cta_id
    
    def update_cta_performance(self, cta_id: str, performance_data: Dict[str, Any]) -> bool:
        """
        Actualiza los datos de rendimiento de un CTA
        
        Args:
            cta_id: ID del CTA
            performance_data: Nuevos datos de rendimiento
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        if cta_id not in self.cta_database:
            logger.warning(f"CTA no encontrado: {cta_id}")
            return False
        
        cta = self.cta_database[cta_id]
        
        # Actualizar datos de rendimiento
        current_performance = cta.get("performance", {})
        current_samples = current_performance.get("samples", 0)
        new_samples = performance_data.get("samples", 1)
        total_samples = current_samples + new_samples
        
        # Calcular promedios ponderados
        if total_samples > 0:
            for key in ["conversion_rate", "engagement_rate", "retention_rate"]:
                if key in performance_data:
                    current_value = current_performance.get(key, 0.0)
                    new_value = performance_data.get(key, 0.0)
                    
                    # Promedio ponderado
                    weighted_avg = (current_value * current_samples + new_value * new_samples) / total_samples
                    current_performance[key] = weighted_avg
        
        # Actualizar muestras
        current_performance["samples"] = total_samples
        
        # Actualizar CTA
        cta["performance"] = current_performance
        cta["last_used"] = datetime.now().isoformat()
        cta["usage_count"] = cta.get("usage_count", 0) + 1
        
        # Actualizar contador
        self.cta_usage_counter[cta_id] = cta["usage_count"]
        
        # Guardar cambios
        self._save_cta_database()
        
        logger.info(f"Rendimiento de CTA actualizado: {cta_id} - {current_performance}")
        return True
    
    def get_cta_recommendations(self, platform: str, content_type: str, 
                               duration: float = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Obtiene recomendaciones de CTAs basadas en rendimiento y cumplimiento
        
        Args:
            platform: Plataforma objetivo
            content_type: Tipo de contenido
            duration: Duración del contenido en segundos
            limit: Número máximo de recomendaciones
            
        Returns:
            Lista de CTAs recomendados con puntuaciones
        """
        recommendations = []
        
        # Filtrar CTAs por plataforma
        platform_ctas = [
            (cta_id, cta_data) 
            for cta_id, cta_data in self.cta_database.items() 
            if cta_data.get("platform") == platform
        ]
        
        # Si no hay CTAs para esta plataforma, devolver lista vacía
        if not platform_ctas:
            logger.warning(f"No hay CTAs registrados para {platform}")
            return []
        
        # Calcular puntuaciones
        for cta_id, cta_data in platform_ctas:
            # Factores de puntuación
            performance = cta_data.get("performance", {})
            conversion_rate = performance.get("conversion_rate", 0.0)
            engagement_rate = performance.get("engagement_rate", 0.0)
                        retention_rate = performance.get("retention_rate", 0.0)
            compliance_score = cta_data.get("compliance_score", 0.0)
            usage_count = cta_data.get("usage_count", 0)
            
            # Calcular puntuación compuesta
            performance_score = (conversion_rate * 0.5) + (engagement_rate * 0.3) + (retention_rate * 0.2)
            
            # Ajustar por uso (favorecer CTAs menos usados para variedad)
            usage_factor = 1.0 / (1.0 + np.log1p(usage_count))
            
            # Puntuación final
            final_score = (performance_score * 0.7) + (compliance_score * 0.2) + (usage_factor * 0.1)
            
            # Añadir a recomendaciones
            recommendations.append({
                "id": cta_id,
                "text": cta_data.get("text", ""),
                "type": cta_data.get("type", ""),
                "score": final_score,
                "performance_score": performance_score,
                "compliance_score": compliance_score,
                "usage_count": usage_count,
                "timing": cta_data.get("timing", 0.0)
            })
        
        # Ordenar por puntuación (mayor primero)
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        # Limitar resultados
        recommendations = recommendations[:limit]
        
        # Si hay duración, ajustar timings
        if duration and recommendations:
            # Distribuir CTAs a lo largo del contenido
            if len(recommendations) > 1:
                # Evitar principio y final
                start_buffer = duration * 0.1  # 10% del inicio
                end_buffer = duration * 0.1    # 10% del final
                available_duration = duration - start_buffer - end_buffer
                
                # Calcular intervalos
                interval = available_duration / (len(recommendations) - 1) if len(recommendations) > 1 else available_duration
                
                # Asignar timings
                for i, rec in enumerate(recommendations):
                    rec["suggested_timing"] = start_buffer + (i * interval)
            else:
                # Si solo hay un CTA, ponerlo a la mitad
                recommendations[0]["suggested_timing"] = duration / 2
        
        return recommendations
    
    def audit_content(self, content: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """
        Audita un contenido para verificar cumplimiento con políticas
        
        Args:
            content: Diccionario con el contenido a auditar
                - title: Título del contenido
                - description: Descripción o caption
                - tags: Lista de etiquetas
                - ctas: Lista de CTAs con timing
                - duration: Duración en segundos (para videos)
            platform: Plataforma objetivo
            
        Returns:
            Resultado de la auditoría con puntuación y problemas detectados
        """
        # Inicializar resultado
        result = {
            "platform": platform,
            "timestamp": datetime.now().isoformat(),
            "compliance_score": 0.0,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Obtener reglas
        global_rules = self.rules.get("global", {})
        platform_rules = self.rules.get(platform, {})
        
        # Combinar reglas
        rules = {**global_rules}
        for key, value in platform_rules.items():
            if isinstance(value, list) and key in rules and isinstance(rules[key], list):
                # Combinar listas
                rules[key] = list(set(rules[key] + value))
            else:
                # Sobrescribir o añadir
                rules[key] = value
        
        # Verificar título
        if "title" in content:
            title_issues = self._check_text(content["title"], rules, "title")
            result["issues"].extend(title_issues)
            
            # Verificar longitud del título
            if "title_max_length" in rules and len(content["title"]) > rules["title_max_length"]:
                result["issues"].append({
                    "type": "title_too_long",
                    "severity": "medium",
                    "message": f"El título excede la longitud máxima ({len(content['title'])}/{rules['title_max_length']})"
                })
        
        # Verificar descripción
        if "description" in content:
            desc_issues = self._check_text(content["description"], rules, "description")
            result["issues"].extend(desc_issues)
            
            # Verificar longitud de la descripción
            if "description_max_length" in rules and len(content["description"]) > rules["description_max_length"]:
                result["issues"].append({
                    "type": "description_too_long",
                    "severity": "medium",
                    "message": f"La descripción excede la longitud máxima ({len(content['description'])}/{rules['description_max_length']})"
                })
        
        # Verificar etiquetas
        if "tags" in content and isinstance(content["tags"], list):
            # Verificar número máximo de etiquetas
            if "max_tags" in rules and len(content["tags"]) > rules["max_tags"]:
                result["issues"].append({
                    "type": "too_many_tags",
                    "severity": "low",
                    "message": f"Demasiadas etiquetas ({len(content['tags'])}/{rules['max_tags']})"
                })
            
            # Verificar cada etiqueta
            for tag in content["tags"]:
                tag_issues = self._check_text(tag, rules, "tag")
                result["issues"].extend(tag_issues)
        
        # Verificar hashtags
        if "description" in content:
            hashtags = re.findall(r'#\w+', content["description"])
            if "max_hashtags" in rules and len(hashtags) > rules["max_hashtags"]:
                result["issues"].append({
                    "type": "too_many_hashtags",
                    "severity": "medium",
                    "message": f"Demasiados hashtags ({len(hashtags)}/{rules['max_hashtags']})"
                })
        
        # Verificar menciones
        if "description" in content:
            mentions = re.findall(r'@\w+', content["description"])
            if "max_mentions" in rules and len(mentions) > rules["max_mentions"]:
                result["issues"].append({
                    "type": "too_many_mentions",
                    "severity": "low",
                    "message": f"Demasiadas menciones ({len(mentions)}/{rules['max_mentions']})"
                })
        
        # Verificar CTAs
        if "ctas" in content and isinstance(content["ctas"], list) and "duration" in content:
            cta_issues = self._check_ctas(content["ctas"], content["duration"], rules)
            result["issues"].extend(cta_issues)
        
        # Calcular puntuación de cumplimiento
        total_issues = len(result["issues"])
        severe_issues = sum(1 for issue in result["issues"] if issue.get("severity") == "high")
        medium_issues = sum(1 for issue in result["issues"] if issue.get("severity") == "medium")
        
        # Fórmula de puntuación: 100 - (severos*15 + medios*5 + bajos*1)
        compliance_score = 100 - (severe_issues * 15 + medium_issues * 5 + (total_issues - severe_issues - medium_issues))
        compliance_score = max(0, min(100, compliance_score))  # Limitar entre 0 y 100
        
        result["compliance_score"] = compliance_score
        
        # Generar recomendaciones
        if compliance_score < 70:
            result["recommendations"].append({
                "type": "general",
                "message": "La puntuación de cumplimiento es baja. Revisa los problemas detectados."
            })
        
        # Recomendar CTAs si no hay suficientes
        if "ctas" in content and len(content["ctas"]) < 2 and "duration" in content and content["duration"] > 60:
            result["recommendations"].append({
                "type": "cta",
                "message": "Considera añadir más CTAs para mejorar la conversión."
            })
        
        # Clasificar resultado
        if compliance_score >= 80:
            result["status"] = "approved"
        elif compliance_score >= 60:
            result["status"] = "warning"
        else:
            result["status"] = "rejected"
        
        return result
    
    def _check_text(self, text: str, rules: Dict[str, Any], content_type: str) -> List[Dict[str, Any]]:
        """
        Verifica un texto contra las reglas
        
        Args:
            text: Texto a verificar
            rules: Reglas a aplicar
            content_type: Tipo de contenido (title, description, tag)
            
        Returns:
            Lista de problemas detectados
        """
        issues = []
        
        # Verificar palabras prohibidas
        if "prohibited_words" in rules:
            for word in rules["prohibited_words"]:
                if re.search(r'\b' + re.escape(word) + r'\b', text.lower()):
                    issues.append({
                        "type": "prohibited_word",
                        "severity": "high",
                        "message": f"Palabra prohibida detectada: '{word}'",
                        "content_type": content_type
                    })
        
        # Verificar patrones sospechosos
        if "suspicious_patterns" in rules:
            for pattern in rules["suspicious_patterns"]:
                if re.search(pattern, text.lower()):
                    issues.append({
                        "type": "suspicious_pattern",
                        "severity": "medium",
                        "message": f"Patrón sospechoso detectado: '{pattern}'",
                        "content_type": content_type
                    })
        
        # Verificar porcentaje de mayúsculas (para títulos y descripciones)
        if content_type in ["title", "description"] and "max_caps_percentage" in rules:
            caps_count = sum(1 for c in text if c.isupper() and c.isalpha())
            total_chars = sum(1 for c in text if c.isalpha())
            
            if total_chars > 0:
                caps_percentage = (caps_count / total_chars) * 100
                if caps_percentage > rules["max_caps_percentage"]:
                    issues.append({
                        "type": "too_many_caps",
                        "severity": "medium",
                        "message": f"Demasiadas mayúsculas ({caps_percentage:.1f}% > {rules['max_caps_percentage']}%)",
                        "content_type": content_type
                    })
        
        # Verificar emojis en título
        if content_type == "title" and "max_emojis_title" in rules:
            # Patrón simple para detectar emojis
            emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # símbolos y pictogramas
                                       u"\U0001F680-\U0001F6FF"  # transporte y símbolos de mapas
                                       u"\U0001F700-\U0001F77F"  # alchemical symbols
                                       u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                                       u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                       u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                       u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                       u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                       u"\U00002702-\U000027B0"  # Dingbats
                                       u"\U000024C2-\U0001F251" 
                                       "]+", flags=re.UNICODE)
            
            emojis = emoji_pattern.findall(text)
            if len(emojis) > rules["max_emojis_title"]:
                issues.append({
                    "type": "too_many_emojis",
                    "severity": "low",
                    "message": f"Demasiados emojis en el título ({len(emojis)} > {rules['max_emojis_title']})",
                    "content_type": content_type
                })
        
        return issues
    
    def _check_ctas(self, ctas: List[Dict[str, Any]], duration: float, rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verifica los CTAs contra las reglas
        
        Args:
            ctas: Lista de CTAs con timing
            duration: Duración del contenido en segundos
            rules: Reglas a aplicar
            
        Returns:
            Lista de problemas detectados
        """
        issues = []
        
        # Verificar frecuencia de CTAs
        if "max_cta_frequency" in rules and duration > 0:
            cta_count = len(ctas)
            max_ctas = int(duration / 60 * rules["max_cta_frequency"])
            
            if cta_count > max_ctas:
                issues.append({
                    "type": "too_many_ctas",
                    "severity": "medium",
                    "message": f"Demasiados CTAs ({cta_count} > {max_ctas} recomendados)"
                })
        
        # Verificar intervalo mínimo entre CTAs
        if "min_cta_interval" in rules and len(ctas) > 1:
            # Ordenar CTAs por timing
            sorted_ctas = sorted(ctas, key=lambda x: x.get("timing", 0))
            
            for i in range(1, len(sorted_ctas)):
                prev_timing = sorted_ctas[i-1].get("timing", 0)
                curr_timing = sorted_ctas[i].get("timing", 0)
                
                interval = curr_timing - prev_timing
                if interval < rules["min_cta_interval"]:
                    issues.append({
                        "type": "cta_interval_too_short",
                        "severity": "low",
                        "message": f"Intervalo entre CTAs demasiado corto ({interval:.1f}s < {rules['min_cta_interval']}s)",
                        "cta_positions": [prev_timing, curr_timing]
                    })
        
        # Verificar texto de cada CTA
        for cta in ctas:
            if "text" in cta:
                cta_text_issues = self._check_text(cta["text"], rules, "cta")
                issues.extend(cta_text_issues)
        
        return issues
    
    def update_compliance_score(self, cta_id: str, compliance_score: float, flags: List[str] = None) -> bool:
        """
        Actualiza la puntuación de cumplimiento de un CTA
        
        Args:
            cta_id: ID del CTA
            compliance_score: Puntuación de cumplimiento (0-100)
            flags: Lista de flags o problemas detectados
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        if cta_id not in self.cta_database:
            logger.warning(f"CTA no encontrado: {cta_id}")
            return False
        
        cta = self.cta_database[cta_id]
        
        # Actualizar puntuación
        cta["compliance_score"] = max(0, min(100, compliance_score))
        
        # Actualizar flags
        if flags is not None:
            cta["flags"] = flags
        
        # Guardar cambios
        self._save_cta_database()
        
        logger.info(f"Puntuación de cumplimiento actualizada: {cta_id} - {compliance_score}")
        return True
    
    def get_cta_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas sobre los CTAs registrados
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            "total_ctas": len(self.cta_database),
            "by_platform": {},
            "by_type": {},
            "top_performing": [],
            "compliance_issues": []
        }
        
        # Contadores
        platforms = Counter()
        types = Counter()
        
        # Procesar cada CTA
        for cta_id, cta_data in self.cta_database.items():
            platform = cta_data.get("platform", "unknown")
            cta_type = cta_data.get("type", "unknown")
            
            # Contar por plataforma y tipo
            platforms[platform] += 1
            types[cta_type] += 1
            
            # Verificar rendimiento
            performance = cta_data.get("performance", {})
            conversion_rate = performance.get("conversion_rate", 0.0)
            
            # Añadir a top performing si tiene buen rendimiento
            if conversion_rate > 0.1 and performance.get("samples", 0) > 5:
                stats["top_performing"].append({
                    "id": cta_id,
                    "text": cta_data.get("text", ""),
                    "platform": platform,
                    "type": cta_type,
                    "conversion_rate": conversion_rate,
                    "samples": performance.get("samples", 0)
                })
            
            # Verificar problemas de cumplimiento
            compliance_score = cta_data.get("compliance_score", 0.0)
            if compliance_score < 60 and cta_data.get("flags"):
                stats["compliance_issues"].append({
                    "id": cta_id,
                    "text": cta_data.get("text", ""),
                    "platform": platform,
                    "compliance_score": compliance_score,
                    "flags": cta_data.get("flags", [])
                })
        
        # Ordenar top performing por tasa de conversión
        stats["top_performing"].sort(key=lambda x: x["conversion_rate"], reverse=True)
        stats["top_performing"] = stats["top_performing"][:10]  # Top 10
        
        # Ordenar problemas de cumplimiento por puntuación (peor primero)
        stats["compliance_issues"].sort(key=lambda x: x["compliance_score"])
        stats["compliance_issues"] = stats["compliance_issues"][:10]  # Top 10 problemas
        
        # Añadir contadores
        stats["by_platform"] = dict(platforms)
        stats["by_type"] = dict(types)
        
        return stats
    
    def export_cta_database(self, output_path: str = None) -> str:
        """
        Exporta la base de datos de CTAs a un archivo JSON
        
        Args:
            output_path: Ruta de salida (opcional)
            
        Returns:
            Ruta al archivo exportado
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/exports/cta_database_{timestamp}.json"
        
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Exportar base de datos
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.cta_database, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Base de datos de CTAs exportada a: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error al exportar base de datos: {str(e)}")
            return None
    
    def import_cta_database(self, input_path: str, merge: bool = False) -> bool:
        """
        Importa una base de datos de CTAs desde un archivo JSON
        
        Args:
            input_path: Ruta al archivo a importar
            merge: Si es True, combina con la base de datos existente
            
        Returns:
            True si se importó correctamente, False en caso contrario
        """
        try:
            # Verificar que existe el archivo
            if not os.path.exists(input_path):
                logger.error(f"Archivo no encontrado: {input_path}")
                return False
            
            # Cargar base de datos
            with open(input_path, 'r', encoding='utf-8') as f:
                imported_db = json.load(f)
            
            # Validar formato
            if not isinstance(imported_db, dict):
                logger.error(f"Formato inválido: {input_path}")
                return False
            
            # Combinar o reemplazar
            if merge:
                # Combinar bases de datos
                self.cta_database.update(imported_db)
            else:
                # Reemplazar base de datos
                self.cta_database = imported_db
            
            # Actualizar contadores
            self.cta_usage_counter = Counter()
            for cta_id, cta_data in self.cta_database.items():
                self.cta_usage_counter[cta_id] = cta_data.get("usage_count", 0)
            
            # Guardar cambios
            self._save_cta_database()
            
            logger.info(f"Base de datos de CTAs importada desde: {input_path}")
            return True
        except Exception as e:
            logger.error(f"Error al importar base de datos: {str(e)}")
            return False

# Ejemplo de uso
if __name__ == "__main__":
    auditor = ContentAuditor()
    
    # Registrar un CTA de ejemplo
    cta_id = auditor.register_cta(
        cta_text="¡Dale like y suscríbete para más contenido!",
        cta_type="subscribe",
        platform="youtube",
        timing=120.0
    )
    
    # Auditar contenido de ejemplo
    content = {
        "title": "10 TRUCOS INCREÍBLES que DEBES CONOCER!!",
        "description": "En este video te muestro los mejores trucos para mejorar tu productividad. #productividad #trucos #consejos",
        "tags": ["productividad", "trucos", "consejos", "tutorial"],
        "ctas": [
            {"text": "¡Dale like y suscríbete para más contenido!", "timing": 30.0},
            {"text": "Comenta qué truco te gustó más", "timing": 120.0},
            {"text": "Visita mi web para más recursos", "timing": 240.0}
        ],
        "duration": 300.0
    }
    
    result = auditor.audit_content(content, "youtube")
    print(json.dumps(result, indent=2))
    
    # Obtener recomendaciones de CTAs
    recommendations = auditor.get_cta_recommendations("youtube", "tutorial", 300.0, 3)
    print(json.dumps(recommendations, indent=2))