"""
Policy Monitor - Monitoreo de Términos de Servicio (ToS)

Este módulo se encarga de rastrear y analizar los cambios en las políticas
y términos de servicio de las diferentes plataformas, alertando sobre
cambios relevantes que puedan afectar la estrategia de contenido.
"""

import os
import json
import logging
import requests
import time
import hashlib
import difflib
import re
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Optional, Any, Union

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/policy_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PolicyMonitor")

class PolicyMonitor:
    """
    Monitorea los cambios en las políticas y términos de servicio de las plataformas.
    Detecta actualizaciones y analiza su impacto en la estrategia de contenido.
    """
    
    def __init__(self, config_path: str = "config/platforms.json", 
                 history_path: str = "data/policy_history.json"):
        """
        Inicializa el monitor de políticas.
        
        Args:
            config_path: Ruta al archivo de configuración de plataformas
            history_path: Ruta al archivo de historial de políticas
        """
        self.config_path = config_path
        self.history_path = history_path
        
        # Crear directorios si no existen
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        # Cargar configuración
        self.platforms = self._load_config()
        
        # Cargar historial
        self.policy_history = self._load_history()
        
        # Palabras clave críticas para monitorear
        self.critical_keywords = [
            "prohibido", "prohibited", "banned", "ban", "suspensión", "suspension",
            "terminación", "termination", "copyright", "derechos de autor", 
            "monetización", "monetization", "algoritmo", "algorithm",
            "penalización", "penalty", "restricción", "restriction",
            "contenido", "content", "promoción", "promotion", "afiliados", "affiliate",
            "enlaces", "links", "CTA", "call to action", "llamada a la acción"
        ]
        
        logger.info(f"PolicyMonitor inicializado con {len(self.platforms)} plataformas")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración de plataformas"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_path}")
                return {
                    "youtube": {
                        "name": "YouTube",
                        "tos_url": "https://www.youtube.com/t/terms",
                        "community_guidelines_url": "https://www.youtube.com/howyoutubeworks/policies/community-guidelines/",
                        "monetization_policy_url": "https://www.youtube.com/howyoutubeworks/policies/monetization-policies/"
                    },
                    "tiktok": {
                        "name": "TikTok",
                        "tos_url": "https://www.tiktok.com/legal/terms-of-service",
                        "community_guidelines_url": "https://www.tiktok.com/community-guidelines"
                    },
                    "instagram": {
                        "name": "Instagram",
                        "tos_url": "https://help.instagram.com/581066165581870",
                        "community_guidelines_url": "https://help.instagram.com/477434105621119"
                    }
                }
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def _load_history(self) -> Dict[str, Dict[str, Any]]:
        """Carga el historial de políticas"""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.info(f"Creando nuevo historial de políticas")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar historial: {str(e)}")
            return {}
    
    def _save_history(self) -> bool:
        """Guarda el historial de políticas"""
        try:
            with open(self.history_path, 'w', encoding='utf-8') as f:
                json.dump(self.policy_history, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error al guardar historial: {str(e)}")
            return False
    
    def fetch_policy_content(self, url: str) -> Optional[str]:
        """
        Obtiene el contenido de una política desde una URL
        
        Args:
            url: URL de la política
            
        Returns:
            Contenido de texto de la política o None si hay error
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parsear HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Eliminar scripts y estilos
            for script in soup(["script", "style"]):
                script.extract()
            
            # Obtener texto
            text = soup.get_text(separator=' ', strip=True)
            
            # Limpiar espacios múltiples
            text = re.sub(r'\s+', ' ', text)
            
            return text
        except Exception as e:
            logger.error(f"Error al obtener política desde {url}: {str(e)}")
            return None
    
    def calculate_hash(self, content: str) -> str:
        """Calcula el hash SHA-256 del contenido"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def check_platform_policies(self, platform_id: str) -> Dict[str, Any]:
        """
        Verifica las políticas de una plataforma específica
        
        Args:
            platform_id: Identificador de la plataforma
            
        Returns:
            Diccionario con resultados del análisis
        """
        if platform_id not in self.platforms:
            logger.warning(f"Plataforma no encontrada: {platform_id}")
            return {"status": "error", "message": "Plataforma no encontrada"}
        
        platform = self.platforms[platform_id]
        platform_name = platform.get("name", platform_id)
        results = {
            "platform": platform_name,
            "timestamp": datetime.now().isoformat(),
            "changes_detected": False,
            "policies": {}
        }
        
        # Verificar cada política
        for policy_type, url_key in [
            ("tos", "tos_url"),
            ("community_guidelines", "community_guidelines_url"),
            ("monetization_policy", "monetization_policy_url")
        ]:
            if url_key in platform:
                url = platform[url_key]
                logger.info(f"Verificando {policy_type} de {platform_name}: {url}")
                
                # Obtener contenido actual
                content = self.fetch_policy_content(url)
                if not content:
                    results["policies"][policy_type] = {
                        "status": "error",
                        "message": "No se pudo obtener el contenido"
                    }
                    continue
                
                # Calcular hash
                current_hash = self.calculate_hash(content)
                
                # Verificar si existe en historial
                history_key = f"{platform_id}_{policy_type}"
                if history_key in self.policy_history:
                    previous = self.policy_history[history_key]
                    previous_hash = previous.get("hash")
                    
                    if current_hash != previous_hash:
                        # Detectar cambios
                        changes = self._analyze_changes(
                            previous.get("content", ""), 
                            content
                        )
                        
                        results["changes_detected"] = True
                        results["policies"][policy_type] = {
                            "status": "changed",
                            "last_update": previous.get("timestamp"),
                            "changes": changes
                        }
                        
                        # Actualizar historial
                        self.policy_history[history_key] = {
                            "hash": current_hash,
                            "timestamp": results["timestamp"],
                            "content": content
                        }
                    else:
                        results["policies"][policy_type] = {
                            "status": "unchanged",
                            "last_update": previous.get("timestamp")
                        }
                else:
                    # Primera vez que se verifica
                    results["policies"][policy_type] = {
                        "status": "new",
                        "message": "Primera verificación"
                    }
                    
                    # Guardar en historial
                    self.policy_history[history_key] = {
                        "hash": current_hash,
                        "timestamp": results["timestamp"],
                        "content": content
                    }
        
        # Guardar historial actualizado
        self._save_history()
        
        return results
    
    def _analyze_changes(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """
        Analiza los cambios entre dos versiones de una política
        
        Args:
            old_content: Contenido anterior
            new_content: Contenido actual
            
        Returns:
            Diccionario con análisis de cambios
        """
        # Dividir en líneas
        old_lines = old_content.split('. ')
        new_lines = new_content.split('. ')
        
        # Calcular diferencias
        diff = list(difflib.ndiff(old_lines, new_lines))
        
        # Analizar cambios
        added = []
        removed = []
        critical_changes = []
        
        for line in diff:
            if line.startswith('+ '):
                added.append(line[2:])
                # Verificar palabras clave críticas
                for keyword in self.critical_keywords:
                    if keyword.lower() in line.lower():
                        critical_changes.append({
                            "type": "added",
                            "content": line[2:],
                            "keyword": keyword
                        })
            elif line.startswith('- '):
                removed.append(line[2:])
                # Verificar palabras clave críticas
                for keyword in self.critical_keywords:
                    if keyword.lower() in line.lower():
                        critical_changes.append({
                            "type": "removed",
                            "content": line[2:],
                            "keyword": keyword
                        })
        
        return {
            "added_count": len(added),
            "removed_count": len(removed),
            "critical_changes": critical_changes,
            "critical_count": len(critical_changes),
            "added_sample": added[:5],  # Muestra de hasta 5 adiciones
            "removed_sample": removed[:5]  # Muestra de hasta 5 eliminaciones
        }
    
    def check_all_platforms(self) -> Dict[str, Any]:
        """
        Verifica las políticas de todas las plataformas configuradas
        
        Returns:
            Diccionario con resultados del análisis para todas las plataformas
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "platforms": {}
        }
        
        for platform_id in self.platforms:
            results["platforms"][platform_id] = self.check_platform_policies(platform_id)
        
        return results
    
    def get_critical_changes(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Obtiene los cambios críticos detectados en un período de tiempo
        
        Args:
            days: Número de días hacia atrás para buscar cambios
            
        Returns:
            Lista de cambios críticos con detalles
        """
        critical_changes = []
        cutoff_date = datetime.now().timestamp() - (days * 86400)
        
        for history_key, data in self.policy_history.items():
            timestamp = data.get("timestamp")
            if timestamp:
                dt = datetime.fromisoformat(timestamp)
                if dt.timestamp() >= cutoff_date:
                    platform_id, policy_type = history_key.split('_', 1)
                    platform_name = self.platforms.get(platform_id, {}).get("name", platform_id)
                    
                    critical_changes.append({
                        "platform": platform_name,
                        "policy_type": policy_type,
                        "timestamp": timestamp,
                        "url": self.platforms.get(platform_id, {}).get(f"{policy_type}_url", ""),
                        "changes": data.get("changes", {})
                    })
        
        # Ordenar por fecha (más reciente primero)
        critical_changes.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return critical_changes
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Genera un informe de cumplimiento basado en las políticas actuales
        
        Returns:
            Diccionario con informe de cumplimiento
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "platforms": {},
            "critical_changes": self.get_critical_changes(30),
            "recommendations": []
        }
        
        # Analizar cada plataforma
        for platform_id, platform in self.platforms.items():
            platform_name = platform.get("name", platform_id)
            platform_report = {
                "name": platform_name,
                "policies": {}
            }
            
            # Verificar cada política
            for policy_type in ["tos", "community_guidelines", "monetization_policy"]:
                history_key = f"{platform_id}_{policy_type}"
                if history_key in self.policy_history:
                    data = self.policy_history[history_key]
                    platform_report["policies"][policy_type] = {
                        "last_update": data.get("timestamp"),
                        "critical_keywords": []
                    }
                    
                    # Buscar palabras clave críticas en el contenido
                    content = data.get("content", "")
                    for keyword in self.critical_keywords:
                        if keyword.lower() in content.lower():
                            # Encontrar contexto (50 caracteres antes y después)
                            index = content.lower().find(keyword.lower())
                            start = max(0, index - 50)
                            end = min(len(content), index + len(keyword) + 50)
                            context = content[start:end]
                            
                            platform_report["policies"][policy_type]["critical_keywords"].append({
                                "keyword": keyword,
                                "context": context
                            })
            
            report["platforms"][platform_id] = platform_report
            
            # Generar recomendaciones
            if platform_report["policies"]:
                for policy_type, policy_data in platform_report["policies"].items():
                    if policy_data.get("critical_keywords"):
                        keywords = policy_data["critical_keywords"]
                        report["recommendations"].append({
                            "platform": platform_name,
                            "policy_type": policy_type,
                            "message": f"Revisar {len(keywords)} palabras clave críticas en {policy_type}",
                            "keywords": [k["keyword"] for k in keywords[:5]]  # Mostrar hasta 5 palabras clave
                        })
        
        return report

# Ejemplo de uso
if __name__ == "__main__":
    monitor = PolicyMonitor()
    results = monitor.check_all_platforms()
    print(json.dumps(results, indent=2))
    
    report = monitor.generate_compliance_report()
    print(json.dumps(report, indent=2))