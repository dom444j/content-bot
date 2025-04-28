"""
Herramientas para extraer tendencias de diversas plataformas.
Proporciona funciones para obtener hashtags populares, temas tendencia,
y contenido viral de YouTube, TikTok, Twitter/X, Google Trends, etc.
"""

import os
import json
import time
import random
import logging
import requests
from typing import Dict, List, Tuple, Union, Optional, Any
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import csv

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'errors', 'api_errors', 'trend_scraper.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('trend_scraper')

class TrendScraper:
    """Clase para extraer tendencias de diversas plataformas."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el scraper de tendencias.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config_path = config_path or os.path.join('config', 'platforms.json')
        self.config = self._load_config()
        
        # Configurar headers para requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Inicializar caché
        self.cache = {}
        self.cache_duration = 3600  # 1 hora en segundos
    
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
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def _get_api_key(self, platform: str) -> Optional[str]:
        """
        Obtiene la clave API para una plataforma.
        
        Args:
            platform: Nombre de la plataforma
            
        Returns:
            Clave API o None si no está disponible
        """
        try:
            return self.config.get(platform, {}).get('api_key')
        except Exception as e:
            logger.error(f"Error al obtener clave API para {platform}: {str(e)}")
            return None
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Obtiene datos de la caché si están disponibles y no han expirado.
        
        Args:
            key: Clave de caché
            
        Returns:
            Datos en caché o None
        """
        if key in self.cache:
            timestamp, data = self.cache[key]
            if time.time() - timestamp < self.cache_duration:
                return data
        return None
    
    def _save_to_cache(self, key: str, data: Any) -> None:
        """
        Guarda datos en la caché.
        
        Args:
            key: Clave de caché
            data: Datos a guardar
        """
        self.cache[key] = (time.time(), data)
    
    def _make_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[Dict]:
        """
        Realiza una solicitud HTTP con manejo de errores.
        
        Args:
            url: URL para la solicitud
            params: Parámetros de consulta
            headers: Cabeceras HTTP
            
        Returns:
            Respuesta JSON o None en caso de error
        """
        try:
            # Usar headers personalizados o predeterminados
            request_headers = headers or self.headers
            
            # Realizar solicitud
            response = requests.get(url, params=params, headers=request_headers, timeout=10)
            
            # Verificar estado
            response.raise_for_status()
            
            # Intentar devolver JSON
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en solicitud a {url}: {str(e)}")
            return None
        except ValueError as e:
            logger.error(f"Error al procesar respuesta JSON de {url}: {str(e)}")
            return None
    
    def _make_html_request(self, url: str, params: Optional[Dict] = None) -> Optional[str]:
        """
        Realiza una solicitud HTTP y devuelve HTML.
        
        Args:
            url: URL para la solicitud
            params: Parámetros de consulta
            
        Returns:
            Contenido HTML o None en caso de error
        """
        try:
                        response = requests.get(url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en solicitud HTML a {url}: {str(e)}")
            return None
    
    def get_youtube_trends(self, region_code: str = "ES", category_id: str = "0", max_results: int = 10) -> List[Dict]:
        """
        Obtiene videos tendencia en YouTube.
        
        Args:
            region_code: Código de región (ES, US, MX, etc.)
            category_id: ID de categoría (0 para todas)
            max_results: Número máximo de resultados
            
        Returns:
            Lista de videos tendencia
        """
        # Verificar caché
        cache_key = f"youtube_trends_{region_code}_{category_id}_{max_results}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Obtener clave API
        api_key = self._get_api_key('youtube')
        if not api_key:
            logger.error("No se encontró clave API para YouTube")
            return []
        
        # Construir URL
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            'part': 'snippet,statistics',
            'chart': 'mostPopular',
            'regionCode': region_code,
            'maxResults': max_results,
            'key': api_key
        }
        
        # Añadir categoría si no es 0
        if category_id != "0":
            params['videoCategoryId'] = category_id
        
        # Realizar solicitud
        response = self._make_request(url, params)
        if not response or 'items' not in response:
            return []
        
        # Procesar resultados
        trends = []
        for item in response.get('items', []):
            snippet = item.get('snippet', {})
            statistics = item.get('statistics', {})
            
            trend = {
                'id': item.get('id'),
                'title': snippet.get('title'),
                'description': snippet.get('description'),
                'channel': snippet.get('channelTitle'),
                'published_at': snippet.get('publishedAt'),
                'tags': snippet.get('tags', []),
                'views': int(statistics.get('viewCount', 0)),
                'likes': int(statistics.get('likeCount', 0)),
                'comments': int(statistics.get('commentCount', 0)),
                'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url')
            }
            trends.append(trend)
        
        # Guardar en caché
        self._save_to_cache(cache_key, trends)
        
        return trends
    
    def get_tiktok_trends(self, region: str = "es", max_results: int = 10) -> List[Dict]:
        """
        Obtiene tendencias de TikTok.
        Nota: TikTok no tiene una API pública oficial, esto usa web scraping.
        
        Args:
            region: Código de región
            max_results: Número máximo de resultados
            
        Returns:
            Lista de tendencias
        """
        # Verificar caché
        cache_key = f"tiktok_trends_{region}_{max_results}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Simular tendencias (en producción, implementar web scraping)
        # Nota: TikTok bloquea scraping, en producción usar servicios como Proxycrawl
        trends = self._simulate_tiktok_trends(max_results)
        
        # Guardar en caché
        self._save_to_cache(cache_key, trends)
        
        return trends
    
    def _simulate_tiktok_trends(self, count: int) -> List[Dict]:
        """
        Simula tendencias de TikTok para desarrollo.
        En producción, reemplazar con scraping real.
        
        Args:
            count: Número de tendencias a simular
            
        Returns:
            Lista de tendencias simuladas
        """
        # Hashtags populares simulados
        popular_hashtags = [
            "fyp", "parati", "viral", "humor", "comedia", "challenge",
            "baile", "música", "receta", "fitness", "moda", "belleza",
            "consejos", "aprendizaje", "finanzas", "tecnología"
        ]
        
        # Generar tendencias simuladas
        trends = []
        for i in range(count):
            # Seleccionar hashtags aleatorios
            hashtags = random.sample(popular_hashtags, k=random.randint(2, 5))
            
            trend = {
                'id': f"trend_{i}_{int(time.time())}",
                'title': f"Tendencia #{i+1}",
                'description': f"Esta es una tendencia simulada #{i+1}",
                'hashtags': hashtags,
                'views': random.randint(100000, 10000000),
                'likes': random.randint(10000, 1000000),
                'shares': random.randint(1000, 100000),
                'comments': random.randint(500, 50000),
                'created_at': (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat()
            }
            trends.append(trend)
        
        return trends
    
    def get_twitter_trends(self, woeid: int = 23424950, max_results: int = 10) -> List[Dict]:
        """
        Obtiene tendencias de Twitter/X.
        
        Args:
            woeid: ID de ubicación (23424950 es España)
            max_results: Número máximo de resultados
            
        Returns:
            Lista de tendencias
        """
        # Verificar caché
        cache_key = f"twitter_trends_{woeid}_{max_results}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Obtener claves API
        api_key = self._get_api_key('x')
        if not api_key:
            logger.warning("No se encontró clave API para Twitter/X, usando simulación")
            trends = self._simulate_twitter_trends(max_results)
            self._save_to_cache(cache_key, trends)
            return trends
        
        # En producción, implementar llamada a la API de Twitter/X
        # Aquí usamos simulación por simplicidad
        trends = self._simulate_twitter_trends(max_results)
        
        # Guardar en caché
        self._save_to_cache(cache_key, trends)
        
        return trends
    
    def _simulate_twitter_trends(self, count: int) -> List[Dict]:
        """
        Simula tendencias de Twitter/X para desarrollo.
        
        Args:
            count: Número de tendencias a simular
            
        Returns:
            Lista de tendencias simuladas
        """
        # Temas populares simulados
        popular_topics = [
            "Actualidad", "Política", "Deportes", "Entretenimiento",
            "Tecnología", "Ciencia", "Salud", "Economía", "Cultura",
            "Música", "Cine", "Series", "Videojuegos", "Moda"
        ]
        
        # Generar tendencias simuladas
        trends = []
        for i in range(count):
            topic = random.choice(popular_topics)
            
            trend = {
                'id': i + 1,
                'name': f"#{topic.lower().replace(' ', '')}{random.randint(1, 100)}",
                'topic': topic,
                'tweet_volume': random.randint(5000, 500000),
                'url': f"https://twitter.com/search?q=%23{topic.lower().replace(' ', '')}"
            }
            trends.append(trend)
        
        return trends
    
    def get_google_trends(self, country: str = "ES", max_results: int = 10) -> List[Dict]:
        """
        Obtiene tendencias de búsqueda de Google.
        
        Args:
            country: Código de país
            max_results: Número máximo de resultados
            
        Returns:
            Lista de tendencias
        """
        # Verificar caché
        cache_key = f"google_trends_{country}_{max_results}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # En producción, usar la API de Google Trends o pytrends
        # Aquí usamos simulación por simplicidad
        trends = self._simulate_google_trends(max_results)
        
        # Guardar en caché
        self._save_to_cache(cache_key, trends)
        
        return trends
    
    def _simulate_google_trends(self, count: int) -> List[Dict]:
        """
        Simula tendencias de Google para desarrollo.
        
        Args:
            count: Número de tendencias a simular
            
        Returns:
            Lista de tendencias simuladas
        """
        # Temas populares simulados
        popular_topics = [
            "Noticias", "Deportes", "Celebridades", "Películas", "Series",
            "Tecnología", "Videojuegos", "Música", "Política", "Economía",
            "Salud", "Recetas", "Tutoriales", "Compras", "Viajes"
        ]
        
        # Generar tendencias simuladas
        trends = []
        for i in range(count):
            topic = random.choice(popular_topics)
            
            trend = {
                'id': i + 1,
                'query': f"{topic} {random.choice(['nuevo', 'mejor', 'cómo', 'qué', 'dónde'])}",
                'topic': topic,
                'search_volume': random.randint(10000, 1000000),
                'related_queries': [
                    f"{topic} {random.choice(['2023', 'gratis', 'online', 'español', 'precio'])}" 
                    for _ in range(3)
                ]
            }
            trends.append(trend)
        
        return trends
    
    def get_instagram_trends(self, max_results: int = 10) -> List[Dict]:
        """
        Obtiene tendencias de Instagram.
        
        Args:
            max_results: Número máximo de resultados
            
        Returns:
            Lista de tendencias
        """
        # Verificar caché
        cache_key = f"instagram_trends_{max_results}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Instagram no tiene API pública para tendencias
        # Simulamos tendencias basadas en hashtags populares
        trends = self._simulate_instagram_trends(max_results)
        
        # Guardar en caché
        self._save_to_cache(cache_key, trends)
        
        return trends
    
    def _simulate_instagram_trends(self, count: int) -> List[Dict]:
        """
        Simula tendencias de Instagram para desarrollo.
        
        Args:
            count: Número de tendencias a simular
            
        Returns:
            Lista de tendencias simuladas
        """
        # Hashtags populares simulados
        popular_hashtags = [
            "love", "instagood", "fashion", "photooftheday", "beautiful",
            "art", "photography", "happy", "picoftheday", "cute", "follow",
            "travel", "style", "instadaily", "nature", "reels", "viral"
        ]
        
        # Categorías simuladas
        categories = [
            "Moda", "Viajes", "Comida", "Fitness", "Belleza", "Arte",
            "Fotografía", "Lifestyle", "Música", "Deportes", "Tecnología"
        ]
        
        # Generar tendencias simuladas
        trends = []
        for i in range(count):
            hashtag = random.choice(popular_hashtags)
            category = random.choice(categories)
            
            trend = {
                'id': i + 1,
                'hashtag': f"#{hashtag}",
                'category': category,
                'posts': random.randint(100000, 10000000),
                'engagement_rate': round(random.uniform(1.5, 8.5), 2),
                'related_hashtags': random.sample(popular_hashtags, k=3)
            }
            trends.append(trend)
        
        return trends
    
    def get_all_trends(self, max_per_platform: int = 5) -> Dict[str, List[Dict]]:
        """
        Obtiene tendencias de todas las plataformas soportadas.
        
        Args:
            max_per_platform: Número máximo de tendencias por plataforma
            
        Returns:
            Diccionario con tendencias por plataforma
        """
        # Verificar caché
        cache_key = f"all_trends_{max_per_platform}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Obtener tendencias de cada plataforma
        youtube_trends = self.get_youtube_trends(max_results=max_per_platform)
        tiktok_trends = self.get_tiktok_trends(max_results=max_per_platform)
        twitter_trends = self.get_twitter_trends(max_results=max_per_platform)
        google_trends = self.get_google_trends(max_results=max_per_platform)
        instagram_trends = self.get_instagram_trends(max_results=max_per_platform)
        
        # Combinar resultados
        all_trends = {
            'youtube': youtube_trends,
            'tiktok': tiktok_trends,
            'twitter': twitter_trends,
            'google': google_trends,
            'instagram': instagram_trends
        }
        
        # Guardar en caché
        self._save_to_cache(cache_key, all_trends)
        
        return all_trends
    
    def save_trends_to_file(self, trends: Dict[str, List[Dict]], output_dir: str = None) -> Dict[str, str]:
        """
        Guarda las tendencias en archivos JSON.
        
        Args:
            trends: Diccionario con tendencias por plataforma
            output_dir: Directorio de salida (por defecto: datasets)
            
        Returns:
            Diccionario con rutas de archivos guardados
        """
        # Configurar directorio de salida
        if output_dir is None:
            output_dir = os.path.join('datasets')
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar cada plataforma en un archivo
        saved_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for platform, platform_trends in trends.items():
            filename = f"{platform}_trends_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(platform_trends, f, indent=2, ensure_ascii=False)
                saved_files[platform] = filepath
                logger.info(f"Tendencias de {platform} guardadas en {filepath}")
            except Exception as e:
                logger.error(f"Error al guardar tendencias de {platform}: {str(e)}")
        
        return saved_files
    
    def extract_hashtags(self, trends: Dict[str, List[Dict]]) -> List[str]:
        """
        Extrae hashtags únicos de todas las tendencias.
        
        Args:
            trends: Diccionario con tendencias por plataforma
            
        Returns:
            Lista de hashtags únicos
        """
        hashtags = set()
        
        # Procesar YouTube
        for item in trends.get('youtube', []):
            tags = item.get('tags', [])
            for tag in tags:
                if tag.startswith('#'):
                    hashtags.add(tag.lower())
                else:
                    hashtags.add(f"#{tag.lower().replace(' ', '')}")
        
        # Procesar TikTok
        for item in trends.get('tiktok', []):
            for tag in item.get('hashtags', []):
                hashtags.add(f"#{tag.lower().replace(' ', '')}")
        
        # Procesar Twitter
        for item in trends.get('twitter', []):
            name = item.get('name', '')
            if name.startswith('#'):
                hashtags.add(name.lower())
        
        # Procesar Instagram
        for item in trends.get('instagram', []):
            hashtag = item.get('hashtag', '')
            if hashtag.startswith('#'):
                hashtags.add(hashtag.lower())
            
            for related in item.get('related_hashtags', []):
                hashtags.add(f"#{related.lower().replace(' ', '')}")
        
        return sorted(list(hashtags))
    
    def save_hashtags_to_csv(self, hashtags: List[str], output_path: str = None) -> str:
        """
        Guarda hashtags en un archivo CSV.
        
        Args:
            hashtags: Lista de hashtags
            output_path: Ruta de salida (por defecto: datasets/hashtags.csv)
            
        Returns:
            Ruta del archivo guardado
        """
        # Configurar ruta de salida
        if output_path is None:
            output_path = os.path.join('datasets', 'hashtags.csv')
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['hashtag', 'fecha'])
                
                today = datetime.now().strftime('%Y-%m-%d')
                for hashtag in hashtags:
                    writer.writerow([hashtag, today])
            
            logger.info(f"Hashtags guardados en {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error al guardar hashtags: {str(e)}")
            return ""

# Funciones de conveniencia para uso directo

def get_trending_hashtags(max_results: int = 20) -> List[str]:
    """
    Función de conveniencia para obtener hashtags tendencia.
    
    Args:
        max_results: Número máximo de hashtags
        
    Returns:
        Lista de hashtags tendencia
    """
    scraper = TrendScraper()
    trends = scraper.get_all_trends(max_per_platform=10)
    hashtags = scraper.extract_hashtags(trends)
    return hashtags[:max_results]

def update_trend_datasets() -> Dict[str, str]:
    """
    Actualiza todos los datasets de tendencias.
    
    Returns:
        Diccionario con rutas de archivos actualizados
    """
    scraper = TrendScraper()
    
    # Obtener tendencias
    trends = scraper.get_all_trends(max_per_platform=20)
    
    # Guardar tendencias
    saved_files = scraper.save_trends_to_file(trends)
    
    # Extraer y guardar hashtags
    hashtags = scraper.extract_hashtags(trends)
    hashtags_path = scraper.save_hashtags_to_csv(hashtags)
    
    if hashtags_path:
        saved_files['hashtags'] = hashtags_path
    
    return saved_files

# Ejemplo de uso
if __name__ == "__main__":
    # Crear scraper
    scraper = TrendScraper()
    
    # Obtener tendencias de todas las plataformas
    print("Obteniendo tendencias...")
    trends = scraper.get_all_trends(max_per_platform=5)
    
    # Mostrar resultados
    for platform, platform_trends in trends.items():
        print(f"\n=== Tendencias en {platform.upper()} ===")
        for i, trend in enumerate(platform_trends, 1):
            if platform == 'youtube':
                print(f"{i}. {trend['title']} - {trend['views']} vistas")
            elif platform == 'tiktok':
                print(f"{i}. {trend['title']} - {', '.join(trend['hashtags'])}")
            elif platform == 'twitter':
                print(f"{i}. {trend['name']} - {trend['tweet_volume']} tweets")
            elif platform == 'google':
                print(f"{i}. {trend['query']} - {trend['search_volume']} búsquedas")
            elif platform == 'instagram':
                print(f"{i}. {trend['hashtag']} - {trend['posts']} posts")
    
    # Extraer hashtags
    hashtags = scraper.extract_hashtags(trends)
    print(f"\n=== Hashtags Populares ({len(hashtags)}) ===")
    for i, hashtag in enumerate(hashtags[:10], 1):
        print(f"{i}. {hashtag}")
    
    # Actualizar datasets
    print("\nActualizando datasets...")
    saved_files = scraper.save_trends_to_file(trends)
    for platform, filepath in saved_files.items():
        print(f"- {platform}: {filepath}")
    
    # Guardar hashtags
    hashtags_path = scraper.save_hashtags_to_csv(hashtags)
    print(f"- hashtags: {hashtags_path}")