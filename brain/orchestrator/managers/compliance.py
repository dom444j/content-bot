"""
ComplianceManager - Gestor de cumplimiento normativo para el sistema de monetización

Este módulo se encarga de verificar que todo el contenido generado cumpla con:
- Términos de servicio de las plataformas
- Políticas de monetización
- Directrices de contenido
- Regulaciones legales aplicables
- Detección de contenido problemático

Características principales:
- Verificación automática de contenido antes de publicación
- Monitoreo continuo de cambios en políticas de plataformas
- Filtrado de CTAs potencialmente problemáticos
- Detección de infracciones de copyright
- Evaluación de riesgo de shadowban
"""

import os
import sys
import logging
import json
import re
import time
import datetime
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum, auto

# Configuración del logger
logger = logging.getLogger("orchestrator.compliance")

class ContentRiskLevel(Enum):
    """Niveles de riesgo para el contenido."""
    SAFE = auto()
    LOW_RISK = auto()
    MEDIUM_RISK = auto()
    HIGH_RISK = auto()
    CRITICAL = auto()

class ComplianceIssueType(Enum):
    """Tipos de problemas de cumplimiento."""
    COPYRIGHT = auto()
    TERMS_OF_SERVICE = auto()
    COMMUNITY_GUIDELINES = auto()
    MONETIZATION_POLICY = auto()
    LEGAL_REGULATION = auto()
    MISLEADING_CONTENT = auto()
    INAPPROPRIATE_CONTENT = auto()
    SPAM_POLICY = auto()
    FALSE_CLAIMS = auto()
    PRIVACY_VIOLATION = auto()

class ComplianceManager:
    """
    Gestor de cumplimiento normativo para verificar que el contenido cumpla con
    las políticas de las plataformas y regulaciones legales.
    """
    
    def __init__(self, config: Dict[str, Any] = None, shadowban_manager = None):
        """
        Inicializa el gestor de cumplimiento.
        
        Args:
            config: Configuración del gestor de cumplimiento
            shadowban_manager: Referencia al gestor de shadowbans
        """
        self.config = config or {}
        self.shadowban_manager = shadowban_manager
        
        # Cargar reglas de cumplimiento
        self.platform_policies = self._load_platform_policies()
        self.content_guidelines = self._load_content_guidelines()
        self.cta_patterns = self._load_cta_patterns()
        self.copyright_rules = self._load_copyright_rules()
        
        # Palabras y frases prohibidas o de alto riesgo por plataforma
        self.prohibited_terms = self._load_prohibited_terms()
        
        # Historial de verificaciones
        self.compliance_history = {}
        
        # Caché de políticas para evitar consultas repetidas
        self.policy_cache = {}
        self.policy_cache_expiry = {}
        self.policy_cache_ttl = 86400  # 24 horas en segundos
        
        logger.info("ComplianceManager inicializado correctamente")
    
    def _load_platform_policies(self) -> Dict[str, Any]:
        """
        Carga las políticas de las plataformas desde archivos de configuración.
        
        Returns:
            Dict[str, Any]: Políticas por plataforma
        """
        try:
            policy_path = os.path.join("config", "platform_policies.json")
            if os.path.exists(policy_path):
                with open(policy_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de políticas no encontrado: {policy_path}")
                return self._get_default_platform_policies()
        except Exception as e:
            logger.error(f"Error al cargar políticas de plataformas: {str(e)}")
            return self._get_default_platform_policies()
    
    def _get_default_platform_policies(self) -> Dict[str, Any]:
        """
        Proporciona políticas predeterminadas para las plataformas principales.
        
        Returns:
            Dict[str, Any]: Políticas predeterminadas
        """
        return {
            "youtube": {
                "tos_url": "https://www.youtube.com/t/terms",
                "community_guidelines_url": "https://www.youtube.com/howyoutubeworks/policies/community-guidelines/",
                "monetization_policy_url": "https://support.google.com/youtube/answer/1311392",
                "max_hashtags": 15,
                "max_description_length": 5000,
                "max_title_length": 100,
                "restricted_content": ["gambling", "adult", "violence", "hate_speech", "dangerous_activities"]
            },
            "tiktok": {
                "tos_url": "https://www.tiktok.com/legal/terms-of-service",
                "community_guidelines_url": "https://www.tiktok.com/community-guidelines",
                "monetization_policy_url": "https://www.tiktok.com/creators/creator-portal/",
                "max_hashtags": 30,
                "max_description_length": 2200,
                "restricted_content": ["gambling", "adult", "violence", "hate_speech", "dangerous_activities"]
            },
            "instagram": {
                "tos_url": "https://help.instagram.com/581066165581870",
                "community_guidelines_url": "https://help.instagram.com/477434105621119",
                "monetization_policy_url": "https://creators.instagram.com/",
                "max_hashtags": 30,
                "max_description_length": 2200,
                "restricted_content": ["gambling", "adult", "violence", "hate_speech", "dangerous_activities"]
            }
        }
    
    def _load_content_guidelines(self) -> Dict[str, Any]:
        """
        Carga las directrices de contenido desde archivos de configuración.
        
        Returns:
            Dict[str, Any]: Directrices de contenido
        """
        try:
            guidelines_path = os.path.join("config", "content_guidelines.json")
            if os.path.exists(guidelines_path):
                with open(guidelines_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de directrices no encontrado: {guidelines_path}")
                return self._get_default_content_guidelines()
        except Exception as e:
            logger.error(f"Error al cargar directrices de contenido: {str(e)}")
            return self._get_default_content_guidelines()
    
    def _get_default_content_guidelines(self) -> Dict[str, Any]:
        """
        Proporciona directrices de contenido predeterminadas.
        
        Returns:
            Dict[str, Any]: Directrices predeterminadas
        """
        return {
            "general": {
                "prohibited_content": [
                    "contenido para adultos explícito",
                    "violencia gráfica",
                    "discurso de odio",
                    "acoso",
                    "información personal identificable",
                    "actividades ilegales",
                    "contenido engañoso",
                    "spam"
                ],
                "restricted_content": [
                    "contenido político controvertido",
                    "temas sensibles",
                    "lenguaje fuerte",
                    "referencias a sustancias controladas",
                    "contenido médico no verificado",
                    "teorías conspirativas"
                ]
            },
            "cta_guidelines": {
                "prohibited": [
                    "promesas falsas",
                    "garantías de resultados",
                    "lenguaje engañoso",
                    "urgencia artificial extrema",
                    "solicitudes agresivas"
                ],
                "recommended": [
                    "llamados a la acción claros",
                    "beneficios realistas",
                    "lenguaje honesto",
                    "propuestas de valor claras"
                ]
            }
        }
    
    def _load_cta_patterns(self) -> Dict[str, List[str]]:
        """
        Carga patrones de CTAs problemáticos.
        
        Returns:
            Dict[str, List[str]]: Patrones de CTAs por categoría
        """
        try:
            patterns_path = os.path.join("config", "cta_patterns.json")
            if os.path.exists(patterns_path):
                with open(patterns_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de patrones CTA no encontrado: {patterns_path}")
                return self._get_default_cta_patterns()
        except Exception as e:
            logger.error(f"Error al cargar patrones CTA: {str(e)}")
            return self._get_default_cta_patterns()
    
    def _get_default_cta_patterns(self) -> Dict[str, List[str]]:
        """
        Proporciona patrones CTA predeterminados.
        
        Returns:
            Dict[str, List[str]]: Patrones CTA predeterminados
        """
        return {
            "false_promises": [
                r"garant[ií]za(do|mos)",
                r"100\s*%\s*seguro",
                r"nunca\s*fallar?[áa]",
                r"resultados\s*instant[áa]neos",
                r"ganar?\s*dinero\s*r[áa]pido",
                r"hacerte\s*rico",
                r"millonario\s*en\s*\d+\s*(d[íi]as|semanas|meses)"
            ],
            "urgency": [
                r"[úu]ltima\s*oportunidad",
                r"oferta\s*por\s*tiempo\s*limitado",
                r"solo\s*hoy",
                r"antes\s*que\s*sea\s*tarde",
                r"no\s*esperes\s*m[áa]s"
            ],
            "aggressive": [
                r"tienes\s*que",
                r"debes",
                r"obligatorio",
                r"no\s*puedes\s*perderte",
                r"no\s*hay\s*excusas"
            ],
            "misleading": [
                r"secreto\s*que\s*no\s*quieren\s*que\s*sepas",
                r"lo\s*que\s*no\s*te\s*cuentan",
                r"truco\s*oculto",
                r"m[ée]todo\s*prohibido",
                r"hackea\s*el\s*sistema"
            ]
        }
    
    def _load_copyright_rules(self) -> Dict[str, Any]:
        """
        Carga reglas de copyright.
        
        Returns:
            Dict[str, Any]: Reglas de copyright
        """
        try:
            copyright_path = os.path.join("config", "copyright_rules.json")
            if os.path.exists(copyright_path):
                with open(copyright_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de reglas de copyright no encontrado: {copyright_path}")
                return self._get_default_copyright_rules()
        except Exception as e:
            logger.error(f"Error al cargar reglas de copyright: {str(e)}")
            return self._get_default_copyright_rules()
    
    def _get_default_copyright_rules(self) -> Dict[str, Any]:
        """
        Proporciona reglas de copyright predeterminadas.
        
        Returns:
            Dict[str, Any]: Reglas de copyright predeterminadas
        """
        return {
            "fair_use": {
                "max_clip_length": 15,  # segundos
                "max_percentage": 10,   # porcentaje del contenido original
                "required_attribution": True,
                "transformative_use": True
            },
            "music": {
                "allowed_sources": ["royalty_free", "licensed", "creative_commons"],
                "max_sample_length": 10  # segundos
            },
            "images": {
                "allowed_sources": ["own_creation", "licensed", "creative_commons", "public_domain"],
                "required_attribution": True
            },
            "text": {
                "max_quote_length": 250,  # caracteres
                "required_citation": True
            }
        }
    
    def _load_prohibited_terms(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Carga términos prohibidos por plataforma.
        
        Returns:
            Dict[str, Dict[str, List[str]]]: Términos prohibidos por plataforma y nivel de riesgo
        """
        try:
            terms_path = os.path.join("config", "prohibited_terms.json")
            if os.path.exists(terms_path):
                with open(terms_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de términos prohibidos no encontrado: {terms_path}")
                return self._get_default_prohibited_terms()
        except Exception as e:
            logger.error(f"Error al cargar términos prohibidos: {str(e)}")
            return self._get_default_prohibited_terms()
    
    def _get_default_prohibited_terms(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Proporciona términos prohibidos predeterminados.
        
        Returns:
            Dict[str, Dict[str, List[str]]]: Términos prohibidos predeterminados
        """
        # Nota: Estos son términos genéricos de ejemplo
        # En un sistema real, se necesitaría una lista más completa y específica
        return {
            "all": {
                "high_risk": [
                    "pornografía", "drogas ilegales", "armas", "terrorismo",
                    "suicidio", "abuso", "violencia extrema", "odio racial"
                ],
                "medium_risk": [
                    "alcohol", "tabaco", "apuestas", "desnudez", "violencia",
                    "política controvertida", "religión controvertida"
                ],
                "low_risk": [
                    "maldiciones", "lenguaje fuerte", "temas para adultos",
                    "controversia", "crítica"
                ]
            },
            "youtube": {
                "high_risk": [
                    "contenido no apto para anunciantes", "clickbait extremo"
                ],
                "medium_risk": [
                    "temas sensibles", "noticias controvertidas"
                ]
            },
            "tiktok": {
                "high_risk": [
                    "desafíos peligrosos", "desinformación médica"
                ],
                "medium_risk": [
                    "bailes provocativos", "bromas pesadas"
                ]
            },
            "instagram": {
                "high_risk": [
                    "promoción de cirugías estéticas", "productos de dieta no verificados"
                ],
                "medium_risk": [
                    "retoque corporal extremo", "estereotipos negativos"
                ]
            }
        }
    
    def verify_content(self, content: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """
        Verifica que el contenido cumpla con las políticas de la plataforma.
        
        Args:
            content: Contenido a verificar (incluye texto, metadatos, etc.)
            platform: Plataforma donde se publicará
            
        Returns:
            Dict[str, Any]: Resultado de la verificación con detalles
        """
        start_time = time.time()
        content_id = content.get('id', str(uuid.uuid4()))
        
        logger.info(f"Iniciando verificación de cumplimiento para contenido {content_id} en {platform}")
        
        # Inicializar resultado
        result = {
            'content_id': content_id,
            'platform': platform,
            'timestamp': datetime.datetime.now().isoformat(),
            'passed': True,
            'risk_level': ContentRiskLevel.SAFE.name,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Verificar texto (título, descripción, etc.)
        if 'text' in content:
            text_result = self._verify_text_content(content['text'], platform)
            self._update_verification_result(result, text_result)
        
        # Verificar CTAs
        if 'cta' in content:
            cta_result = self._verify_cta(content['cta'], platform)
            self._update_verification_result(result, cta_result)
        
        # Verificar hashtags
        if 'hashtags' in content:
            hashtag_result = self._verify_hashtags(content['hashtags'], platform)
            self._update_verification_result(result, hashtag_result)
        
        # Verificar metadatos
        if 'metadata' in content:
            metadata_result = self._verify_metadata(content['metadata'], platform)
            self._update_verification_result(result, metadata_result)
        
        # Verificar contenido multimedia (si hay URLs o paths)
        if 'media' in content:
            media_result = self._verify_media_content(content['media'], platform)
            self._update_verification_result(result, media_result)
        
        # Verificar riesgo de shadowban
        if self.shadowban_manager:
            shadowban_risk = self.shadowban_manager.assess_shadowban_risk(content, platform)
            if shadowban_risk['risk_level'] > 0.5:
                result['warnings'].append({
                    'type': 'SHADOWBAN_RISK',
                    'message': f"Alto riesgo de shadowban: {shadowban_risk['reason']}",
                    'risk_score': shadowban_risk['risk_level']
                })
                result['recommendations'].append(
                    f"Considerar modificar: {shadowban_risk['recommendation']}"
                )
        
        # Determinar nivel de riesgo final
        if len(result['issues']) > 0:
            result['passed'] = False
            # El nivel de riesgo es el más alto de todos los problemas
            risk_levels = [issue.get('risk_level', ContentRiskLevel.LOW_RISK.name) for issue in result['issues']]
            result['risk_level'] = max(risk_levels, key=lambda x: ContentRiskLevel[x].value)
        
        # Guardar en historial
        self.compliance_history[content_id] = {
            'timestamp': datetime.datetime.now().isoformat(),
            'platform': platform,
            'result': result,
            'content_summary': {
                'title': content.get('text', {}).get('title', ''),
                'description_snippet': content.get('text', {}).get('description', '')[:100] + '...' if content.get('text', {}).get('description', '') else ''
            }
        }
        
        execution_time = time.time() - start_time
        logger.info(f"Verificación de cumplimiento completada para {content_id} en {execution_time:.2f}s. Resultado: {result['passed']}")
        
        return result
    
    def _update_verification_result(self, result: Dict[str, Any], component_result: Dict[str, Any]) -> None:
        """
        Actualiza el resultado de verificación con los resultados de un componente.
        
        Args:
            result: Resultado de verificación a actualizar
            component_result: Resultado del componente a integrar
        """
        if not component_result['passed']:
            result['passed'] = False
            result['issues'].extend(component_result['issues'])
        
        result['warnings'].extend(component_result.get('warnings', []))
        result['recommendations'].extend(component_result.get('recommendations', []))
        
        # Actualizar nivel de riesgo si el componente tiene un nivel más alto
        if ContentRiskLevel[component_result['risk_level']].value > ContentRiskLevel[result['risk_level']].value:
            result['risk_level'] = component_result['risk_level']
    
    def _verify_text_content(self, text_content: Dict[str, str], platform: str) -> Dict[str, Any]:
        """
        Verifica el contenido textual.
        
        Args:
            text_content: Contenido textual (título, descripción, etc.)
            platform: Plataforma donde se publicará
            
        Returns:
            Dict[str, Any]: Resultado de la verificación
        """
        result = {
            'passed': True,
            'risk_level': ContentRiskLevel.SAFE.name,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Verificar cada campo de texto
        for field_name, text in text_content.items():
            if not text:
                continue
                
            # Verificar longitud según plataforma
            max_length = self.platform_policies.get(platform, {}).get(f"max_{field_name}_length")
            if max_length and len(text) > max_length:
                result['issues'].append({
                    'type': ComplianceIssueType.TERMS_OF_SERVICE.name,
                    'field': field_name,
                    'message': f"El {field_name} excede la longitud máxima permitida en {platform}: {len(text)}/{max_length} caracteres",
                    'risk_level': ContentRiskLevel.MEDIUM_RISK.name
                })
                result['recommendations'].append(
                    f"Acortar el {field_name} a menos de {max_length} caracteres"
                )
            
            # Verificar términos prohibidos
            prohibited_terms = self._check_prohibited_terms(text, platform)
            if prohibited_terms:
                for term_info in prohibited_terms:
                    if term_info['risk_level'] == 'high_risk':
                        risk_level = ContentRiskLevel.HIGH_RISK.name
                        result['issues'].append({
                            'type': ComplianceIssueType.COMMUNITY_GUIDELINES.name,
                            'field': field_name,
                            'message': f"Término de alto riesgo detectado en {field_name}: '{term_info['term']}'",
                            'risk_level': risk_level
                        })
                    elif term_info['risk_level'] == 'medium_risk':
                        risk_level = ContentRiskLevel.MEDIUM_RISK.name
                        result['warnings'].append({
                            'type': 'RESTRICTED_TERM',
                            'field': field_name,
                            'message': f"Término de riesgo medio detectado en {field_name}: '{term_info['term']}'",
                            'risk_level': risk_level
                        })
                    else:
                        risk_level = ContentRiskLevel.LOW_RISK.name
                        result['warnings'].append({
                            'type': 'SENSITIVE_TERM',
                            'field': field_name,
                            'message': f"Término sensible detectado en {field_name}: '{term_info['term']}'",
                            'risk_level': risk_level
                        })
                    
                    result['recommendations'].append(
                        f"Considerar reemplazar o reformular el término '{term_info['term']}'"
                    )
                    
                    # Actualizar nivel de riesgo si es necesario
                    if ContentRiskLevel[risk_level].value > ContentRiskLevel[result['risk_level']].value:
                        result['risk_level'] = risk_level
            
            # Verificar contenido engañoso
            misleading_patterns = [
                (r"cura\s+para", "afirmaciones médicas no verificadas"),
                (r"garant[ií]a\s+de\s+resultados", "garantías de resultados"),
                (r"secreto\s+que\s+no\s+quieren\s+que\s+sepas", "teorías conspirativas"),
                (r"m[ée]todo\s+revolucionario", "exageraciones de marketing"),
                (r"100\s*%\s*efectivo", "afirmaciones absolutas")
            ]
            
            for pattern, description in misleading_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    result['warnings'].append({
                        'type': 'POTENTIALLY_MISLEADING',
                        'field': field_name,
                        'message': f"Posible contenido engañoso en {field_name}: {description}",
                        'risk_level': ContentRiskLevel.MEDIUM_RISK.name
                    })
                    result['recommendations'].append(
                        f"Reformular para evitar {description}"
                    )
                    
                    if ContentRiskLevel.MEDIUM_RISK.value > ContentRiskLevel[result['risk_level']].value:
                        result['risk_level'] = ContentRiskLevel.MEDIUM_RISK.name
        
        # Actualizar estado general
        if len(result['issues']) > 0:
            result['passed'] = False
        
        return result
    
    def _verify_cta(self, cta: str, platform: str) -> Dict[str, Any]:
        """
        Verifica el llamado a la acción (CTA).
        
        Args:
            cta: Texto del CTA
            platform: Plataforma donde se publicará
            
        Returns:
            Dict[str, Any]: Resultado de la verificación
        """
        result = {
            'passed': True,
            'risk_level': ContentRiskLevel.SAFE.name,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Verificar patrones problemáticos en CTAs
        for category, patterns in self.cta_patterns.items():
            for pattern in patterns:
                if re.search(pattern, cta, re.IGNORECASE):
                    if category in ["false_promises", "misleading"]:
                        result['issues'].append({
                            'type': ComplianceIssueType.MISLEADING_CONTENT.name,
                            'message': f"CTA potencialmente engañoso detectado: '{category}'",
                            'pattern': pattern,
                            'risk_level': ContentRiskLevel.HIGH_RISK.name
                        })
                        result['passed'] = False
                        result['risk_level'] = ContentRiskLevel.HIGH_RISK.name
                    else:
                        result['warnings'].append({
                            'type': 'CTA_CONCERN',
                            'message': f"CTA potencialmente problemático: '{category}'",
                            'pattern': pattern,
                            'risk_level': ContentRiskLevel.MEDIUM_RISK.name
                        })
                        if ContentRiskLevel.MEDIUM_RISK.value > ContentRiskLevel[result['risk_level']].value:
                            result['risk_level'] = ContentRiskLevel.MEDIUM_RISK.name
                    
                    result['recommendations'].append(
                        f"Reformular CTA para evitar patrón '{pattern}'"
                    )
        
        # Verificar longitud del CTA
        if len(cta) > 100:
            result['warnings'].append({
                'type': 'CTA_LENGTH',
                'message': f"CTA demasiado largo ({len(cta)} caracteres). Los CTAs efectivos suelen ser concisos.",
                'risk_level': ContentRiskLevel.LOW_RISK.name
            })
            result['recommendations'].append(
                "Acortar CTA a menos de 100 caracteres para mayor efectividad"
            )
        
        # Verificar si el CTA es demasiado genérico
        generic_ctas = [
            "suscríbete", "like", "comenta", "comparte", "sígueme", 
            "dale like", "no olvides suscribirte"
        ]
        
        if cta.lower() in generic_ctas:
            result['warnings'].append({
                'type': 'GENERIC_CTA',
                'message': "CTA demasiado genérico. Podría tener menor efectividad.",
                'risk_level': ContentRiskLevel.LOW_RISK.name
            })
            result['recommendations'].append(
                "Personalizar el CTA para hacerlo más específico y relevante al contenido"
            )
        
        return result
    
    def _verify_hashtags(self, hashtags: List[str], platform: str) -> Dict[str, Any]:
        """
        Verifica los hashtags.
        
        Args:
            hashtags: Lista de hashtags
            platform: Plataforma donde se publicará
            
        Returns:
            Dict[str, Any]: Resultado de la verificación
        """
        result = {
            'passed': True,
            'risk_level': ContentRiskLevel.SAFE.name,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Verificar número máximo de hashtags
        max_hashtags = self.platform_policies.get(platform, {}).get("max_hashtags", 30)
        if len(hashtags) > max_hashtags:
            result['issues'].append({
                'type': ComplianceIssueType.TERMS_OF_SERVICE.name,
                'message': f"Demasiados hashtags: {len(hashtags)}/{max_hashtags} permitidos en {platform}",
                'risk_level': ContentRiskLevel.MEDIUM_RISK.name
            })
            result['passed'] = False
            result['risk_level'] = ContentRiskLevel.MEDIUM_RISK.name
            result['recommendations'].append(
                f"Reducir el número de hashtags a {max_hashtags} o menos"
            )
        
        # Verificar hashtags prohibidos o sensibles
        prohibited_hashtags = [
            "shadowban", "banme", "cprght", "cpyright", 
            "nsfw", "porn", "sex", "xxx", "adult"
        ]
        
        sensitive_hashtags = [
            "diet", "weightloss", "thinspiration", "surgery",
            "gambling", "betting", "casino", "alcohol", "drugs"
        ]
        
        for hashtag in hashtags:
            clean_hashtag = hashtag.replace('#', '').lower()
            
            if clean_hashtag in prohibited_hashtags:
                result['issues'].append({
                    'type': ComplianceIssueType.COMMUNITY_GUIDELINES.name,
                    'message': f"Hashtag prohibido detectado: #{clean_hashtag}",
                    'risk_level': ContentRiskLevel.HIGH_RISK.name
                })
                                result['passed'] = False
                result['risk_level'] = ContentRiskLevel.HIGH_RISK.name
            
            if clean_hashtag in sensitive_hashtags:
                result['warnings'].append({
                    'type': 'SENSITIVE_HASHTAG',
                    'message': f"Hashtag sensible detectado: #{clean_hashtag}",
                    'risk_level': ContentRiskLevel.MEDIUM_RISK.name
                })
                result['recommendations'].append(
                    f"Considerar reemplazar el hashtag #{clean_hashtag} por una alternativa menos sensible"
                )
                if ContentRiskLevel.MEDIUM_RISK.value > ContentRiskLevel[result['risk_level']].value:
                    result['risk_level'] = ContentRiskLevel.MEDIUM_RISK.name
        
        return result
    
    def _verify_metadata(self, metadata: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """
        Verifica los metadatos del contenido.
        
        Args:
            metadata: Metadatos del contenido
            platform: Plataforma donde se publicará
            
        Returns:
            Dict[str, Any]: Resultado de la verificación
        """
        result = {
            'passed': True,
            'risk_level': ContentRiskLevel.SAFE.name,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Verificar categoría/tema
        if 'category' in metadata:
            restricted_categories = {
                "youtube": ["adult", "gambling", "controversial_politics", "violence"],
                "tiktok": ["adult", "gambling", "controversial_politics", "medical_advice"],
                "instagram": ["adult", "gambling", "self_harm", "extreme_weight_loss"]
            }
            
            platform_restricted = restricted_categories.get(platform, [])
            if metadata['category'].lower() in platform_restricted:
                result['issues'].append({
                    'type': ComplianceIssueType.MONETIZATION_POLICY.name,
                    'message': f"Categoría restringida para monetización en {platform}: {metadata['category']}",
                    'risk_level': ContentRiskLevel.HIGH_RISK.name
                })
                result['passed'] = False
                result['risk_level'] = ContentRiskLevel.HIGH_RISK.name
                result['recommendations'].append(
                    f"Cambiar la categoría o adaptar el contenido para cumplir con las políticas de {platform}"
                )
        
        # Verificar etiquetas de edad/audiencia
        if 'audience' in metadata:
            if metadata['audience'].lower() == 'mature' or metadata['audience'].lower() == 'adult':
                result['issues'].append({
                    'type': ComplianceIssueType.MONETIZATION_POLICY.name,
                    'message': f"Audiencia marcada como '{metadata['audience']}' puede limitar la monetización",
                    'risk_level': ContentRiskLevel.HIGH_RISK.name
                })
                result['passed'] = False
                result['risk_level'] = ContentRiskLevel.HIGH_RISK.name
                result['recommendations'].append(
                    "Adaptar el contenido para una audiencia general o revisar la clasificación"
                )
        
        # Verificar etiquetas de contenido patrocinado
        if metadata.get('sponsored', False) and not metadata.get('sponsored_disclosure', False):
            result['issues'].append({
                'type': ComplianceIssueType.LEGAL_REGULATION.name,
                'message': "Contenido patrocinado sin divulgación adecuada",
                'risk_level': ContentRiskLevel.HIGH_RISK.name
            })
            result['passed'] = False
            result['risk_level'] = ContentRiskLevel.HIGH_RISK.name
            result['recommendations'].append(
                "Añadir divulgación clara de contenido patrocinado según regulaciones FTC/ASA/locales"
            )
        
        # Verificar etiquetas de ubicación (geobloqueo)
        if 'geo_restrictions' in metadata and metadata['geo_restrictions']:
            result['warnings'].append({
                'type': 'GEO_RESTRICTION',
                'message': f"Contenido con restricciones geográficas: {', '.join(metadata['geo_restrictions'])}",
                'risk_level': ContentRiskLevel.MEDIUM_RISK.name
            })
            if ContentRiskLevel.MEDIUM_RISK.value > ContentRiskLevel[result['risk_level']].value:
                result['risk_level'] = ContentRiskLevel.MEDIUM_RISK.name
            result['recommendations'].append(
                "Considerar eliminar restricciones geográficas para maximizar alcance"
            )
        
        return result
    
    def _verify_media_content(self, media: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """
        Verifica el contenido multimedia.
        
        Args:
            media: Información del contenido multimedia
            platform: Plataforma donde se publicará
            
        Returns:
            Dict[str, Any]: Resultado de la verificación
        """
        result = {
            'passed': True,
            'risk_level': ContentRiskLevel.SAFE.name,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Verificar duración del video
        if 'duration' in media:
            duration = media['duration']
            platform_limits = {
                "youtube": {"min": 0, "max": 43200, "optimal": (8*60, 15*60)},  # 12 horas max, 8-15 min óptimo
                "tiktok": {"min": 0, "max": 600, "optimal": (15, 60)},  # 10 min max, 15-60s óptimo
                "instagram": {"min": 0, "max": 900, "optimal": (30, 90)}  # 15 min max, 30-90s óptimo
            }
            
            limits = platform_limits.get(platform, {"min": 0, "max": float('inf'), "optimal": (0, float('inf'))})
            
            if duration > limits["max"]:
                result['issues'].append({
                    'type': ComplianceIssueType.TERMS_OF_SERVICE.name,
                    'message': f"Duración del video ({duration}s) excede el límite de {platform} ({limits['max']}s)",
                    'risk_level': ContentRiskLevel.HIGH_RISK.name
                })
                result['passed'] = False
                result['risk_level'] = ContentRiskLevel.HIGH_RISK.name
                result['recommendations'].append(
                    f"Acortar el video a menos de {limits['max']} segundos"
                )
            elif duration < limits["min"]:
                result['issues'].append({
                    'type': ComplianceIssueType.TERMS_OF_SERVICE.name,
                    'message': f"Duración del video ({duration}s) por debajo del mínimo de {platform} ({limits['min']}s)",
                    'risk_level': ContentRiskLevel.HIGH_RISK.name
                })
                result['passed'] = False
                result['risk_level'] = ContentRiskLevel.HIGH_RISK.name
                result['recommendations'].append(
                    f"Alargar el video a al menos {limits['min']} segundos"
                )
            elif duration < limits["optimal"][0] or duration > limits["optimal"][1]:
                result['warnings'].append({
                    'type': 'SUBOPTIMAL_DURATION',
                    'message': f"Duración del video ({duration}s) fuera del rango óptimo para {platform} ({limits['optimal'][0]}-{limits['optimal'][1]}s)",
                    'risk_level': ContentRiskLevel.LOW_RISK.name
                })
                result['recommendations'].append(
                    f"Considerar ajustar la duración a {limits['optimal'][0]}-{limits['optimal'][1]} segundos para optimizar engagement"
                )
        
        # Verificar resolución del video
        if 'resolution' in media:
            width, height = media['resolution']
            min_resolutions = {
                "youtube": (426, 240),  # 240p mínimo
                "tiktok": (540, 960),   # 540x960 mínimo
                "instagram": (600, 600)  # 600x600 mínimo
            }
            
            min_width, min_height = min_resolutions.get(platform, (0, 0))
            
            if width < min_width or height < min_height:
                result['issues'].append({
                    'type': ComplianceIssueType.TERMS_OF_SERVICE.name,
                    'message': f"Resolución del video ({width}x{height}) por debajo del mínimo recomendado para {platform} ({min_width}x{min_height})",
                    'risk_level': ContentRiskLevel.MEDIUM_RISK.name
                })
                result['passed'] = False
                result['risk_level'] = ContentRiskLevel.MEDIUM_RISK.name
                result['recommendations'].append(
                    f"Aumentar la resolución a al menos {min_width}x{min_height}"
                )
        
        # Verificar relación de aspecto
        if 'resolution' in media:
            width, height = media['resolution']
            aspect_ratio = width / height if height > 0 else 0
            
            platform_ratios = {
                "youtube": {"preferred": 16/9, "acceptable": [16/9, 4/3]},
                "tiktok": {"preferred": 9/16, "acceptable": [9/16]},
                "instagram": {"preferred": 9/16, "acceptable": [9/16, 1/1, 4/5]}
            }
            
            ratios = platform_ratios.get(platform, {"preferred": 16/9, "acceptable": [16/9, 4/3, 1/1, 9/16]})
            
            # Verificar si la relación de aspecto es aceptable (con margen de error)
            is_acceptable = False
            for ratio in ratios["acceptable"]:
                if abs(aspect_ratio - ratio) < 0.1:  # Margen de error del 10%
                    is_acceptable = True
                    break
            
            if not is_acceptable:
                result['warnings'].append({
                    'type': 'ASPECT_RATIO',
                    'message': f"Relación de aspecto ({aspect_ratio:.2f}) no óptima para {platform}",
                    'risk_level': ContentRiskLevel.MEDIUM_RISK.name
                })
                if ContentRiskLevel.MEDIUM_RISK.value > ContentRiskLevel[result['risk_level']].value:
                    result['risk_level'] = ContentRiskLevel.MEDIUM_RISK.name
                
                preferred_ratio = ratios["preferred"]
                if preferred_ratio > 1:
                    preferred_desc = "horizontal"
                elif preferred_ratio < 1:
                    preferred_desc = "vertical"
                else:
                    preferred_desc = "cuadrado"
                
                result['recommendations'].append(
                    f"Ajustar a formato {preferred_desc} ({preferred_ratio:.2f}) para mejor rendimiento en {platform}"
                )
        
        # Verificar copyright de música/audio
        if 'audio_source' in media:
            audio_source = media['audio_source'].lower()
            safe_sources = ["original", "licensed", "royalty_free", "creative_commons", "public_domain"]
            
            if audio_source not in safe_sources:
                result['issues'].append({
                    'type': ComplianceIssueType.COPYRIGHT.name,
                    'message': f"Fuente de audio '{audio_source}' puede tener problemas de copyright",
                    'risk_level': ContentRiskLevel.HIGH_RISK.name
                })
                result['passed'] = False
                result['risk_level'] = ContentRiskLevel.HIGH_RISK.name
                result['recommendations'].append(
                    "Utilizar música de fuentes seguras: original, con licencia, libre de regalías o dominio público"
                )
        
        return result
    
    def _check_prohibited_terms(self, text: str, platform: str) -> List[Dict[str, str]]:
        """
        Verifica si el texto contiene términos prohibidos.
        
        Args:
            text: Texto a verificar
            platform: Plataforma donde se publicará
            
        Returns:
            List[Dict[str, str]]: Lista de términos prohibidos encontrados con su nivel de riesgo
        """
        found_terms = []
        
        # Verificar términos prohibidos generales (para todas las plataformas)
        for risk_level, terms in self.prohibited_terms.get("all", {}).items():
            for term in terms:
                if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                    found_terms.append({
                        'term': term,
                        'risk_level': risk_level
                    })
        
        # Verificar términos prohibidos específicos de la plataforma
        if platform in self.prohibited_terms:
            for risk_level, terms in self.prohibited_terms[platform].items():
                for term in terms:
                    if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
                        found_terms.append({
                            'term': term,
                            'risk_level': risk_level
                        })
        
        return found_terms
    
    def check_platform_policy_updates(self, platform: str) -> Dict[str, Any]:
        """
        Verifica si hay actualizaciones en las políticas de la plataforma.
        
        Args:
            platform: Plataforma a verificar
            
        Returns:
            Dict[str, Any]: Información sobre actualizaciones de políticas
        """
        # Verificar si hay una versión en caché y si no ha expirado
        cache_key = f"policy_update_{platform}"
        if cache_key in self.policy_cache and time.time() < self.policy_cache_expiry.get(cache_key, 0):
            return self.policy_cache[cache_key]
        
        try:
            # En un sistema real, esto podría consultar una API o hacer web scraping
            # Para este ejemplo, simulamos una verificación
            policy_urls = {
                "youtube": {
                    "tos": self.platform_policies.get("youtube", {}).get("tos_url", ""),
                    "community": self.platform_policies.get("youtube", {}).get("community_guidelines_url", ""),
                    "monetization": self.platform_policies.get("youtube", {}).get("monetization_policy_url", "")
                },
                "tiktok": {
                    "tos": self.platform_policies.get("tiktok", {}).get("tos_url", ""),
                    "community": self.platform_policies.get("tiktok", {}).get("community_guidelines_url", ""),
                    "monetization": self.platform_policies.get("tiktok", {}).get("monetization_policy_url", "")
                },
                "instagram": {
                    "tos": self.platform_policies.get("instagram", {}).get("tos_url", ""),
                    "community": self.platform_policies.get("instagram", {}).get("community_guidelines_url", ""),
                    "monetization": self.platform_policies.get("instagram", {}).get("monetization_policy_url", "")
                }
            }
            
            # Simular resultado (en un sistema real, esto vendría de una API o análisis)
            result = {
                "platform": platform,
                "last_checked": datetime.datetime.now().isoformat(),
                "has_updates": False,
                "update_date": None,
                "policy_urls": policy_urls.get(platform, {}),
                "changes": []
            }
            
            # Guardar en caché
            self.policy_cache[cache_key] = result
            self.policy_cache_expiry[cache_key] = time.time() + self.policy_cache_ttl
            
            return result
            
        except Exception as e:
            logger.error(f"Error al verificar actualizaciones de políticas para {platform}: {str(e)}")
            return {
                "platform": platform,
                "last_checked": datetime.datetime.now().isoformat(),
                "has_updates": False,
                "error": str(e)
            }
    
    def get_compliance_history(self, content_id: str = None, platform: str = None, 
                              start_date: str = None, end_date: str = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de verificaciones de cumplimiento.
        
        Args:
            content_id: ID del contenido (opcional)
            platform: Plataforma (opcional)
            start_date: Fecha de inicio en formato ISO (opcional)
            end_date: Fecha de fin en formato ISO (opcional)
            limit: Límite de resultados (por defecto 100)
            
        Returns:
            List[Dict[str, Any]]: Historial de verificaciones
        """
        results = []
        
        # Convertir fechas si se proporcionan
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.datetime.fromisoformat(start_date)
            except ValueError:
                logger.warning(f"Formato de fecha de inicio inválido: {start_date}")
        
        if end_date:
            try:
                end_datetime = datetime.datetime.fromisoformat(end_date)
            except ValueError:
                logger.warning(f"Formato de fecha de fin inválido: {end_date}")
        
        # Filtrar resultados
        for history_id, history_data in self.compliance_history.items():
            # Filtrar por content_id si se proporciona
            if content_id and history_id != content_id:
                continue
            
            # Filtrar por plataforma si se proporciona
            if platform and history_data.get('platform') != platform:
                continue
            
            # Filtrar por fecha de inicio si se proporciona
            if start_datetime:
                history_datetime = datetime.datetime.fromisoformat(history_data.get('timestamp'))
                if history_datetime < start_datetime:
                    continue
            
            # Filtrar por fecha de fin si se proporciona
            if end_datetime:
                history_datetime = datetime.datetime.fromisoformat(history_data.get('timestamp'))
                if history_datetime > end_datetime:
                    continue
            
            # Añadir a resultados
            results.append(history_data)
            
            # Limitar resultados
            if len(results) >= limit:
                break
        
        return results
    
    def get_compliance_stats(self, platform: str = None, 
                            start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de cumplimiento.
        
        Args:
            platform: Plataforma (opcional)
            start_date: Fecha de inicio en formato ISO (opcional)
            end_date: Fecha de fin en formato ISO (opcional)
            
        Returns:
            Dict[str, Any]: Estadísticas de cumplimiento
        """
        # Obtener historial filtrado
        history = self.get_compliance_history(
            platform=platform,
            start_date=start_date,
            end_date=end_date,
            limit=10000  # Límite alto para incluir todos los registros relevantes
        )
        
        # Inicializar estadísticas
        stats = {
            "total_checks": len(history),
            "passed": 0,
            "failed": 0,
            "risk_levels": {
                ContentRiskLevel.SAFE.name: 0,
                ContentRiskLevel.LOW_RISK.name: 0,
                ContentRiskLevel.MEDIUM_RISK.name: 0,
                ContentRiskLevel.HIGH_RISK.name: 0,
                ContentRiskLevel.CRITICAL.name: 0
            },
            "issue_types": {},
            "platforms": {},
            "recommendations": {}
        }
        
        # Calcular estadísticas
        for entry in history:
            result = entry.get('result', {})
            
            # Contar verificaciones pasadas/fallidas
            if result.get('passed', True):
                stats["passed"] += 1
            else:
                stats["failed"] += 1
            
            # Contar por nivel de riesgo
            risk_level = result.get('risk_level', ContentRiskLevel.SAFE.name)
            stats["risk_levels"][risk_level] = stats["risk_levels"].get(risk_level, 0) + 1
            
            # Contar por plataforma
            platform = entry.get('platform', 'unknown')
            stats["platforms"][platform] = stats["platforms"].get(platform, 0) + 1
            
            # Contar por tipo de problema
            for issue in result.get('issues', []):
                issue_type = issue.get('type', 'UNKNOWN')
                stats["issue_types"][issue_type] = stats["issue_types"].get(issue_type, 0) + 1
            
            # Contar recomendaciones comunes
            for recommendation in result.get('recommendations', []):
                # Usar solo las primeras palabras como clave para agrupar recomendaciones similares
                rec_key = ' '.join(recommendation.split()[:3]) + '...'
                stats["recommendations"][rec_key] = stats["recommendations"].get(rec_key, 0) + 1
        
        # Calcular porcentajes
        if stats["total_checks"] > 0:
            stats["pass_rate"] = (stats["passed"] / stats["total_checks"]) * 100
            stats["fail_rate"] = (stats["failed"] / stats["total_checks"]) * 100
        else:
            stats["pass_rate"] = 0
            stats["fail_rate"] = 0
        
        # Ordenar recomendaciones por frecuencia
        stats["recommendations"] = dict(
            sorted(stats["recommendations"].items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Ordenar tipos de problemas por frecuencia
        stats["issue_types"] = dict(
            sorted(stats["issue_types"].items(), key=lambda x: x[1], reverse=True)
        )
        
        return stats