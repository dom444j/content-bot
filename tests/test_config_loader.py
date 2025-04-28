"""
Pruebas para el módulo de carga de configuraciones.
"""

import os
import sys
import unittest
from dotenv import load_dotenv

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_platform_config, get_platform_credentials

class TestConfigLoader(unittest.TestCase):
    """Pruebas para el cargador de configuraciones."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Asegurarse de que las variables de entorno estén cargadas
        load_dotenv()
        
    def test_load_platform_config(self):
        """Prueba la carga de la configuración completa."""
        config = load_platform_config()
        self.assertIsInstance(config, dict)
        
    def test_get_platform_credentials(self):
        """Prueba la obtención de credenciales para plataformas específicas."""
        youtube_config = get_platform_credentials('youtube')
        self.assertIsInstance(youtube_config, dict)
        
        # Verificar que las credenciales de YouTube se carguen correctamente
        # (esto asume que has configurado las variables de entorno)
        if 'YOUTUBE_API_KEY' in os.environ:
            self.assertEqual(youtube_config.get('api_key'), os.environ.get('YOUTUBE_API_KEY'))

if __name__ == '__main__':
    unittest.main()