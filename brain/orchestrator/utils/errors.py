"""
Excepciones específicas para el Orchestrator
"""
import logging

logger = logging.getLogger(__name__)

class OrchestratorError(Exception):
    """Excepción base para errores del Orchestrator"""
    def __init__(self, message, code=None):
        self.message = message
        self.code = code
        logger.error(f"OrchestratorError: {message} (code: {code})")
        super().__init__(self.message)

class ContentGenerationError(OrchestratorError):
    """Error en la generación de contenido"""
    pass

class PublishingError(OrchestratorError):
    """Error en la publicación de contenido"""
    pass

class AnalysisError(OrchestratorError):
    """Error en el análisis de datos"""
    pass

class MonetizationError(OrchestratorError):
    """Error en la monetización"""
    pass

class ComplianceError(OrchestratorError):
    """Error en la verificación de cumplimiento"""
    pass

class ShadowbanError(OrchestratorError):
    """Error relacionado con shadowbans"""
    pass

class PlatformError(OrchestratorError):
    """Error específico de plataforma"""
    pass

class RateLimitError(PlatformError):
    """Error de límite de tasa alcanzado"""
    pass

class ConfigurationError(OrchestratorError):
    """Error en la configuración"""
    pass

class TaskError(OrchestratorError):
    """Error en la gestión de tareas"""
    pass

class PersistenceError(OrchestratorError):
    """Error en la persistencia de datos"""
    pass