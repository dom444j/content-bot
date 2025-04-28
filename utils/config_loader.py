"""
Módulo para cargar configuraciones y credenciales de manera segura.
Prioriza variables de entorno sobre archivos de configuración.
"""

import os
import json
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Cargar variables de entorno desde .env
load_dotenv()

def load_platform_config():
    """
    Carga la configuración de plataformas, priorizando variables de entorno
    sobre el archivo platforms.json.
    
    Returns:
        dict: Configuración de plataformas con credenciales.
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'platforms.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Archivo de configuración no encontrado: {config_path}")
        config = {}
    except json.JSONDecodeError:
        logger.error(f"Error al decodificar JSON en: {config_path}")
        config = {}
    
    # Sobreescribir con variables de entorno si están disponibles
    # YouTube
    if 'youtube' in config:
        config['youtube']['api_key'] = os.getenv('YOUTUBE_API_KEY', config['youtube'].get('api_key', ''))
        config['youtube']['client_id'] = os.getenv('YOUTUBE_CLIENT_ID', config['youtube'].get('client_id', ''))
        config['youtube']['client_secret'] = os.getenv('YOUTUBE_CLIENT_SECRET', config['youtube'].get('client_secret', ''))
        config['youtube']['refresh_token'] = os.getenv('YOUTUBE_REFRESH_TOKEN', config['youtube'].get('refresh_token', ''))
        config['youtube']['channel_id'] = os.getenv('YOUTUBE_CHANNEL_ID', config['youtube'].get('channel_id', ''))
    
    # TikTok
    if 'tiktok' in config:
        config['tiktok']['api_key'] = os.getenv('TIKTOK_API_KEY', config['tiktok'].get('api_key', ''))
        config['tiktok']['client_key'] = os.getenv('TIKTOK_CLIENT_KEY', config['tiktok'].get('client_key', ''))
        config['tiktok']['client_secret'] = os.getenv('TIKTOK_CLIENT_SECRET', config['tiktok'].get('client_secret', ''))
        config['tiktok']['access_token'] = os.getenv('TIKTOK_ACCESS_TOKEN', config['tiktok'].get('access_token', ''))
        config['tiktok']['open_id'] = os.getenv('TIKTOK_OPEN_ID', config['tiktok'].get('open_id', ''))
    
    # Instagram
    if 'instagram' in config:
        config['instagram']['api_key'] = os.getenv('INSTAGRAM_API_KEY', config['instagram'].get('api_key', ''))
        config['instagram']['client_id'] = os.getenv('INSTAGRAM_CLIENT_ID', config['instagram'].get('client_id', ''))
        config['instagram']['client_secret'] = os.getenv('INSTAGRAM_CLIENT_SECRET', config['instagram'].get('client_secret', ''))
        config['instagram']['access_token'] = os.getenv('INSTAGRAM_ACCESS_TOKEN', config['instagram'].get('access_token', ''))
        config['instagram']['user_id'] = os.getenv('INSTAGRAM_USER_ID', config['instagram'].get('user_id', ''))
    
    # Añadir más plataformas según sea necesario...
    
    return config

def get_platform_credentials(platform_name):
    """
    Obtiene las credenciales para una plataforma específica.
    
    Args:
        platform_name (str): Nombre de la plataforma (youtube, tiktok, etc.)
        
    Returns:
        dict: Credenciales de la plataforma.
    """
    config = load_platform_config()
    return config.get(platform_name, {})