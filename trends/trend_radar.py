"""
Módulo de detección de tendencias.

Este módulo se encarga de monitorear diversas fuentes para identificar tendencias emergentes
que puedan ser aprovechadas para la creación de contenido. Utiliza APIs y técnicas de scraping
para recopilar datos de plataformas como Twitter/X, Google Trends, YouTube, TikTok, etc.
"""

import os
import json
import time
import random
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'trends.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('trend_radar')

class TrendRadar:
    """
    Clase principal para la detección de tendencias en múltiples plataformas.
    """
    
    def __init__(self, config_path: str = 'config/platforms.json'):
        """
        Inicializa el detector de tendencias.
        
        Args:
            config_path: Ruta al archivo de configuración con las claves API
        """
        self.config_path = config_path
        self.load_config()
        self.trends_cache = {}
        self.last_update = {}
        self.update_interval = {
            'twitter': 3600,  # 1 hora
            'google_trends': 7200,  # 2 horas
            'youtube': 10800,  # 3 horas
            'tiktok': 3600,  # 1 hora
            'instagram': 7200,  # 2 horas
            'reddit': 10800,  # 3 horas
        }
        
    def load_config(self) -> None:
        """Carga la configuración desde el archivo JSON."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info("Configuración cargada correctamente")
        except Exception as e:
            logger.error(f"Error al cargar la configuración: {e}")
            self.config = {
                "twitter": {"api_key": "", "api_secret": "", "access_token": "", "access_secret": ""},
                "google_trends": {"api_key": ""},
                "youtube": {"api_key": ""},
                "tiktok": {"api_key": ""},
                "instagram": {"api_key": "", "api_secret": ""},
                "reddit": {"client_id": "", "client_secret": ""}
            }
    
    def get_twitter_trends(self, location_id: int = 1) -> List[Dict[str, Any]]:
        """
        Obtiene las tendencias actuales de Twitter/X.
        
        Args:
            location_id: ID de ubicación (1 = mundial)
            
        Returns:
            Lista de tendencias con nombre, volumen y URL
        """
        if not self._should_update('twitter'):
            return self.trends_cache.get('twitter', [])
            
        logger.info("Obteniendo tendencias de Twitter/X")
        
        # Si no hay API key configurada, usar modo simulación
        if not self.config['twitter']['api_key']:
            trends = self._simulate_twitter_trends()
            self.trends_cache['twitter'] = trends
            self.last_update['twitter'] = time.time()
            return trends
            
        try:
            # Aquí iría la implementación real con la API de Twitter
            # Usando tweepy u otra biblioteca
            # Por ahora, simulamos los resultados
            trends = self._simulate_twitter_trends()
            self.trends_cache['twitter'] = trends
            self.last_update['twitter'] = time.time()
            return trends
        except Exception as e:
            logger.error(f"Error al obtener tendencias de Twitter: {e}")
            return self.trends_cache.get('twitter', [])
    
    def get_google_trends(self, region: str = 'ES') -> List[Dict[str, Any]]:
        """
        Obtiene las tendencias actuales de Google Trends.
        
        Args:
            region: Código de país (ES = España)
            
        Returns:
            Lista de tendencias con nombre, volumen y URL
        """
        if not self._should_update('google_trends'):
            return self.trends_cache.get('google_trends', [])
            
        logger.info(f"Obteniendo tendencias de Google para región {region}")
        
        try:
            # Aquí iría la implementación real con la API de Google Trends
            # Usando pytrends u otra biblioteca
            # Por ahora, simulamos los resultados
            trends = self._simulate_google_trends(region)
            self.trends_cache['google_trends'] = trends
            self.last_update['google_trends'] = time.time()
            return trends
        except Exception as e:
            logger.error(f"Error al obtener tendencias de Google: {e}")
            return self.trends_cache.get('google_trends', [])
    
    def get_youtube_trends(self, region_code: str = 'ES') -> List[Dict[str, Any]]:
        """
        Obtiene los videos en tendencia de YouTube.
        
        Args:
            region_code: Código de país (ES = España)
            
        Returns:
            Lista de videos en tendencia con título, canal, vistas y URL
        """
        if not self._should_update('youtube'):
            return self.trends_cache.get('youtube', [])
            
        logger.info(f"Obteniendo tendencias de YouTube para región {region_code}")
        
        try:
            # Implementación con YouTube Data API
            if self.config['youtube']['api_key']:
                url = f"https://www.googleapis.com/youtube/v3/videos"
                params = {
                    'part': 'snippet,statistics',
                    'chart': 'mostPopular',
                    'regionCode': region_code,
                    'maxResults': 20,
                    'key': self.config['youtube']['api_key']
                }
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    trends = []
                    for item in data.get('items', []):
                        trends.append({
                            'title': item['snippet']['title'],
                            'channel': item['snippet']['channelTitle'],
                            'views': int(item['statistics'].get('viewCount', 0)),
                            'likes': int(item['statistics'].get('likeCount', 0)),
                            'comments': int(item['statistics'].get('commentCount', 0)),
                            'url': f"https://www.youtube.com/watch?v={item['id']}",
                            'thumbnail': item['snippet']['thumbnails']['high']['url'],
                            'published_at': item['snippet']['publishedAt'],
                            'category_id': item['snippet']['categoryId'],
                            'platform': 'youtube'
                        })
                    self.trends_cache['youtube'] = trends
                    self.last_update['youtube'] = time.time()
                    return trends
                else:
                    logger.warning(f"Error en la API de YouTube: {response.status_code}")
                    
            # Si no hay API key o falló la petición, usar simulación
            trends = self._simulate_youtube_trends()
            self.trends_cache['youtube'] = trends
            self.last_update['youtube'] = time.time()
            return trends
        except Exception as e:
            logger.error(f"Error al obtener tendencias de YouTube: {e}")
            return self.trends_cache.get('youtube', [])
    
    def get_tiktok_trends(self) -> List[Dict[str, Any]]:
        """
        Obtiene las tendencias actuales de TikTok.
        
        Returns:
            Lista de tendencias con hashtag, videos y vistas
        """
        if not self._should_update('tiktok'):
            return self.trends_cache.get('tiktok', [])
            
        logger.info("Obteniendo tendencias de TikTok")
        
        try:
            # Aquí iría la implementación real con la API de TikTok
            # Por ahora, simulamos los resultados
            trends = self._simulate_tiktok_trends()
            self.trends_cache['tiktok'] = trends
            self.last_update['tiktok'] = time.time()
            return trends
        except Exception as e:
            logger.error(f"Error al obtener tendencias de TikTok: {e}")
            return self.trends_cache.get('tiktok', [])
    
    def get_all_trends(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Obtiene tendencias de todas las plataformas soportadas.
        
        Returns:
            Diccionario con tendencias por plataforma
        """
        logger.info("Obteniendo tendencias de todas las plataformas")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Ejecutar las consultas en paralelo
            twitter_future = executor.submit(self.get_twitter_trends)
            google_future = executor.submit(self.get_google_trends)
            youtube_future = executor.submit(self.get_youtube_trends)
            tiktok_future = executor.submit(self.get_tiktok_trends)
            
            # Recopilar resultados
            all_trends = {
                'twitter': twitter_future.result(),
                'google_trends': google_future.result(),
                'youtube': youtube_future.result(),
                'tiktok': tiktok_future.result()
            }
        
        return all_trends
    
    def get_cross_platform_trends(self) -> List[Dict[str, Any]]:
        """
        Identifica tendencias que aparecen en múltiples plataformas.
        
        Returns:
            Lista de tendencias con puntuación de relevancia cruzada
        """
        logger.info("Analizando tendencias multiplataforma")
        
        all_trends = self.get_all_trends()
        trend_keywords = {}
        
        # Extraer palabras clave de todas las tendencias
        for platform, trends in all_trends.items():
            for trend in trends:
                # Extraer palabras clave según la plataforma
                if platform == 'twitter':
                    keywords = self._extract_keywords(trend['name'])
                elif platform == 'google_trends':
                    keywords = self._extract_keywords(trend['query'])
                elif platform == 'youtube':
                    keywords = self._extract_keywords(trend['title'])
                elif platform == 'tiktok':
                    keywords = self._extract_keywords(trend['hashtag'])
                else:
                    continue
                
                # Actualizar contador de apariciones
                for keyword in keywords:
                    if keyword not in trend_keywords:
                        trend_keywords[keyword] = {
                            'count': 0,
                            'platforms': set(),
                            'examples': {}
                        }
                    
                    trend_keywords[keyword]['count'] += 1
                    trend_keywords[keyword]['platforms'].add(platform)
                    
                    if platform not in trend_keywords[keyword]['examples']:
                        trend_keywords[keyword]['examples'][platform] = []
                    
                    if len(trend_keywords[keyword]['examples'][platform]) < 3:
                        trend_keywords[keyword]['examples'][platform].append(trend)
        
        # Filtrar y ordenar tendencias multiplataforma
        cross_platform = []
        for keyword, data in trend_keywords.items():
            if len(data['platforms']) >= 2:  # Aparece en al menos 2 plataformas
                cross_platform.append({
                    'keyword': keyword,
                    'relevance_score': data['count'] * len(data['platforms']),
                    'platform_count': len(data['platforms']),
                    'platforms': list(data['platforms']),
                    'examples': data['examples']
                })
        
        # Ordenar por puntuación de relevancia
        cross_platform.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return cross_platform[:20]  # Devolver las 20 más relevantes
    
    def get_trending_categories(self) -> Dict[str, int]:
        """
        Identifica las categorías temáticas más populares en las tendencias.
        
        Returns:
            Diccionario con categorías y su puntuación
        """
        all_trends = self.get_all_trends()
        categories = {
            'tecnología': 0,
            'entretenimiento': 0,
            'deportes': 0,
            'política': 0,
            'finanzas': 0,
            'salud': 0,
            'educación': 0,
            'gaming': 0,
            'moda': 0,
            'comida': 0,
            'viajes': 0,
            'música': 0,
            'cine': 0,
            'celebridades': 0,
            'humor': 0
        }
        
        # Palabras clave por categoría
        category_keywords = {
            'tecnología': ['tech', 'tecnología', 'ai', 'ia', 'inteligencia artificial', 'app', 'iphone', 'android', 'gadget', 'software', 'hardware'],
            'entretenimiento': ['entretenimiento', 'show', 'película', 'serie', 'tv', 'televisión', 'streaming', 'netflix', 'disney'],
            'deportes': ['deporte', 'fútbol', 'baloncesto', 'tenis', 'f1', 'olimpiadas', 'liga', 'champions', 'mundial'],
            'política': ['política', 'gobierno', 'elecciones', 'presidente', 'congreso', 'ley', 'ministro', 'partido'],
            'finanzas': ['finanzas', 'economía', 'bolsa', 'bitcoin', 'crypto', 'cripto', 'inversión', 'dinero', 'banco', 'ahorro'],
            'salud': ['salud', 'fitness', 'ejercicio', 'dieta', 'nutrición', 'bienestar', 'yoga', 'médico', 'hospital'],
            'educación': ['educación', 'universidad', 'escuela', 'aprendizaje', 'curso', 'estudio', 'profesor', 'estudiante'],
            'gaming': ['gaming', 'videojuego', 'ps5', 'xbox', 'nintendo', 'fortnite', 'minecraft', 'gamer', 'twitch', 'steam'],
            'moda': ['moda', 'ropa', 'estilo', 'fashion', 'outfit', 'zapatos', 'marca', 'diseñador', 'modelo'],
            'comida': ['comida', 'receta', 'cocina', 'chef', 'restaurante', 'foodie', 'vegano', 'dieta', 'bebida'],
            'viajes': ['viaje', 'turismo', 'hotel', 'vacaciones', 'playa', 'montaña', 'aventura', 'destino', 'aerolínea'],
            'música': ['música', 'canción', 'álbum', 'concierto', 'artista', 'cantante', 'spotify', 'banda', 'festival'],
            'cine': ['cine', 'película', 'actor', 'actriz', 'director', 'hollywood', 'tráiler', 'estreno', 'oscar'],
            'celebridades': ['celebridad', 'famoso', 'influencer', 'estrella', 'celebrity', 'instagram', 'gossip', 'escándalo'],
            'humor': ['humor', 'meme', 'risa', 'comedia', 'gracioso', 'broma', 'viral', 'divertido', 'parodia']
        }
        
        # Analizar todas las tendencias
        for platform, trends in all_trends.items():
            for trend in trends:
                text = ''
                if platform == 'twitter':
                    text = trend.get('name', '')
                elif platform == 'google_trends':
                    text = trend.get('query', '')
                elif platform == 'youtube':
                    text = trend.get('title', '') + ' ' + trend.get('channel', '')
                elif platform == 'tiktok':
                    text = trend.get('hashtag', '') + ' ' + trend.get('description', '')
                
                text = text.lower()
                
                # Incrementar puntuación de categorías que coinciden
                for category, keywords in category_keywords.items():
                    for keyword in keywords:
                        if keyword.lower() in text:
                            categories[category] += 1
                            break
        
        # Ordenar categorías por puntuación
        sorted_categories = {k: v for k, v in sorted(categories.items(), key=lambda item: item[1], reverse=True)}
        
        return sorted_categories
    
    def _should_update(self, platform: str) -> bool:
        """Determina si se debe actualizar la caché para una plataforma."""
        last_time = self.last_update.get(platform, 0)
        interval = self.update_interval.get(platform, 3600)
        return (time.time() - last_time) > interval
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrae palabras clave de un texto."""
        if not text:
            return []
            
        # Eliminar caracteres especiales y convertir a minúsculas
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text.lower())
        
        # Dividir en palabras
        words = text.split()
        
        # Filtrar palabras comunes y cortas
        stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'a', 'de', 'en', 'que', 'por', 'con', 'para', 'al', 'del', 'se', 'su', 'sus', 'mi', 'mis', 'tu', 'tus', 'es', 'son', 'fue', 'fueron', 'ser', 'estar', 'como', 'más', 'menos', 'pero', 'si', 'no', 'sin', 'sobre', 'entre', 'cada', 'todo', 'toda', 'todos', 'todas'}
        keywords = [word for word in words if len(word) > 3 and word not in stopwords]
        
        return keywords
    
    # Métodos de simulación para desarrollo y pruebas
    
    def _simulate_twitter_trends(self) -> List[Dict[str, Any]]:
        """Simula tendencias de Twitter para desarrollo."""
        trends = [
            {"name": "#IA2025", "volume": 125000, "url": "https://twitter.com/search?q=%23IA2025"},
            {"name": "Nuevas Criptomonedas", "volume": 98000, "url": "https://twitter.com/search?q=Nuevas%20Criptomonedas"},
            {"name": "#FinanzasPersonales", "volume": 87000, "url": "https://twitter.com/search?q=%23FinanzasPersonales"},
            {"name": "Mindfulness Diario", "volume": 76000, "url": "https://twitter.com/search?q=Mindfulness%20Diario"},
            {"name": "#GamingPro", "volume": 65000, "url": "https://twitter.com/search?q=%23GamingPro"},
            {"name": "Recetas Saludables", "volume": 54000, "url": "https://twitter.com/search?q=Recetas%20Saludables"},
            {"name": "#TechNews", "volume": 43000, "url": "https://twitter.com/search?q=%23TechNews"},
            {"name": "Ahorro Inteligente", "volume": 32000, "url": "https://twitter.com/search?q=Ahorro%20Inteligente"},
            {"name": "#MemesDelDía", "volume": 21000, "url": "https://twitter.com/search?q=%23MemesDelDía"},
            {"name": "Fitness en Casa", "volume": 19000, "url": "https://twitter.com/search?q=Fitness%20en%20Casa"},
            {"name": "#CriptomonedasHoy", "volume": 18000, "url": "https://twitter.com/search?q=%23CriptomonedasHoy"},
            {"name": "Gadgets 2025", "volume": 17000, "url": "https://twitter.com/search?q=Gadgets%202025"},
            {"name": "#SaludMental", "volume": 16000, "url": "https://twitter.com/search?q=%23SaludMental"},
            {"name": "Inversiones Seguras", "volume": 15000, "url": "https://twitter.com/search?q=Inversiones%20Seguras"},
            {"name": "#TutorialesRápidos", "volume": 14000, "url": "https://twitter.com/search?q=%23TutorialesRápidos"}
        ]
        
        # Añadir plataforma y timestamp
        for trend in trends:
            trend['platform'] = 'twitter'
            trend['timestamp'] = datetime.now().isoformat()
            
        return trends
    
    def _simulate_google_trends(self, region: str) -> List[Dict[str, Any]]:
        """Simula tendencias de Google para desarrollo."""
        trends = [
            {"query": "Cómo invertir en criptomonedas 2025", "volume": 250000, "url": "https://www.google.com/search?q=Cómo+invertir+en+criptomonedas+2025"},
            {"query": "Mejores ejercicios mindfulness", "volume": 180000, "url": "https://www.google.com/search?q=Mejores+ejercicios+mindfulness"},
            {"query": "Inteligencia artificial para principiantes", "volume": 170000, "url": "https://www.google.com/search?q=Inteligencia+artificial+para+principiantes"},
            {"query": "Estrategias gaming profesional", "volume": 160000, "url": "https://www.google.com/search?q=Estrategias+gaming+profesional"},
            {"query": "Recetas fitness rápidas", "volume": 150000, "url": "https://www.google.com/search?q=Recetas+fitness+rápidas"},
            {"query": "Cómo ahorrar dinero rápido", "volume": 140000, "url": "https://www.google.com/search?q=Cómo+ahorrar+dinero+rápido"},
            {"query": "Nuevos gadgets tecnológicos", "volume": 130000, "url": "https://www.google.com/search?q=Nuevos+gadgets+tecnológicos"},
            {"query": "Memes virales 2025", "volume": 120000, "url": "https://www.google.com/search?q=Memes+virales+2025"},
            {"query": "Técnicas de meditación para ansiedad", "volume": 110000, "url": "https://www.google.com/search?q=Técnicas+de+meditación+para+ansiedad"},
            {"query": "Mejores criptomonedas para invertir", "volume": 100000, "url": "https://www.google.com/search?q=Mejores+criptomonedas+para+invertir"}
        ]
        
        # Añadir plataforma, región y timestamp
        for trend in trends:
            trend['platform'] = 'google_trends'
            trend['region'] = region
            trend['timestamp'] = datetime.now().isoformat()
            
        return trends
    
    def _simulate_youtube_trends(self) -> List[Dict[str, Any]]:
        """Simula tendencias de YouTube para desarrollo."""
        trends = [
            {"title": "10 CRIPTOMONEDAS que EXPLOTARÁN en 2025", "channel": "CryptoMaster", "views": 1500000, "likes": 120000, "comments": 15000, "url": "https://www.youtube.com/watch?v=abc123", "thumbnail": "https://i.ytimg.com/vi/abc123/hqdefault.jpg", "published_at": "2023-05-15T14:30:00Z", "category_id": "22"},
            {"title": "Rutina de MINDFULNESS en 10 MINUTOS | Cambia tu vida", "channel": "MeditaciónYoga", "views": 980000, "likes": 89000, "comments": 7500, "url": "https://www.youtube.com/watch?v=def456", "thumbnail": "https://i.ytimg.com/vi/def456/hqdefault.jpg", "published_at": "2023-05-16T10:15:00Z", "category_id": "26"},
            {"title": "INTELIGENCIA ARTIFICIAL explicada en 5 MINUTOS", "channel": "TechSimple", "views": 870000, "likes": 76000, "comments": 8200, "url": "https://www.youtube.com/watch?v=ghi789", "thumbnail": "https://i.ytimg.com/vi/ghi789/hqdefault.jpg", "published_at": "2023-05-14T18:45:00Z", "category_id": "28"},
            {"title": "TRUCOS PRO de GAMING que NADIE te CUENTA", "channel": "GamerProfesional", "views": 760000, "likes": 65000, "comments": 9100, "url": "https://www.youtube.com/watch?v=jkl012", "thumbnail": "https://i.ytimg.com/vi/jkl012/hqdefault.jpg", "published_at": "2023-05-17T12:00:00Z", "category_id": "20"},
            {"title": "AHORRA 1000€ en 30 DÍAS con este MÉTODO", "channel": "FinanzasPersonales", "views": 650000, "likes": 54000, "comments": 6300, "url": "https://www.youtube.com/watch?v=mno345", "thumbnail": "https://i.ytimg.com/vi/mno345/hqdefault.jpg", "published_at": "2023-05-13T09:30:00Z", "category_id": "22"},
            {"title": "Los MEMES más VIRALES de 2025", "channel": "HumorDigital", "views": 540000, "likes": 43000, "comments": 5200, "url": "https://www.youtube.com/watch?v=pqr678", "thumbnail": "https://i.ytimg.com/vi/pqr678/hqdefault.jpg", "published_at": "2023-05-18T15:20:00Z", "category_id": "23"},
            {"title": "5 GADGETS que CAMBIARÁN tu VIDA en 2025", "channel": "TechReviews", "views": 430000, "likes": 32000, "comments": 4100, "url": "https://www.youtube.com/watch?v=stu901", "thumbnail": "https://i.ytimg.com/vi/stu901/hqdefault.jpg", "published_at": "2023-05-12T11:45:00Z", "category_id": "28"},
            {"title": "RECETAS FITNESS en 15 MINUTOS | Sin excusas", "channel": "NutriciónFit", "views": 320000, "likes": 28000, "comments": 3200, "url": "https://www.youtube.com/watch?v=vwx234", "thumbnail": "https://i.ytimg.com/vi/vwx234/hqdefault.jpg", "published_at": "2023-05-19T08:10:00Z", "category_id": "26"},
            {"title": "TUTORIAL: Cómo CREAR tu PRIMER NFT", "channel": "CryptoArtista", "views": 290000, "likes": 24000, "comments": 2800, "url": "https://www.youtube.com/watch?v=yz5678", "thumbnail": "https://i.ytimg.com/vi/yz5678/hqdefault.jpg", "published_at": "2023-05-11T16:30:00Z", "category_id": "24"},
            {"title": "MEDITACIÓN GUIADA para DORMIR en 10 MINUTOS", "channel": "BienestarMental", "views": 280000, "likes": 23000, "comments": 1900, "url": "https://www.youtube.com/watch?v=abc890", "thumbnail": "https://i.ytimg.com/vi/abc890/hqdefault.jpg", "published_at": "2023-05-20T20:00:00Z", "category_id": "26"}
        ]
        
        # Añadir plataforma y timestamp
        for trend in trends:
            trend['platform'] = 'youtube'
            trend['timestamp'] = datetime.now().isoformat()
            
        return trends
    
    def _simulate_tiktok_trends(self) -> List[Dict[str, Any]]:
        """Simula tendencias de TikTok para desarrollo."""
        trends = [
            {"hashtag": "#CryptoTips", "videos": 25000, "views": 1200000000, "description": "Consejos rápidos sobre criptomonedas", "url": "https://www.tiktok.com/tag/cryptotips"},
            {"hashtag": "#MindfulnessChallenge", "videos": 18000, "views": 980000000, "description": "Reto de mindfulness de 7 días", "url": "https://www.tiktok.com/tag/mindfulnesschallenge"},
            {"hashtag": "#IAexplicada", "videos": 15000, "views": 870000000, "description": "Explicaciones sencillas sobre IA", "url": "https://www.tiktok.com/tag/iaexplicada"},
            {"hashtag": "#GamingHacks", "videos": 12000, "views": 760000000, "description": "Trucos para gamers", "url": "https://www.tiktok.com/tag/gaminghacks"},
            {"hashtag": "#AhorroChallenge", "videos": 10000, "views": 650000000, "description": "Reto de ahorro de 30 días", "url": "https://www.tiktok.com/tag/ahorroChallenge"},
                        {"hashtag": "#MemesVirales", "videos": 9000, "views": 540000000, "description": "Los memes más divertidos", "url": "https://www.tiktok.com/tag/memesvirales"},
            {"hashtag": "#TechGadgets", "videos": 8000, "views": 430000000, "description": "Los gadgets más innovadores", "url": "https://www.tiktok.com/tag/techgadgets"},
            {"hashtag": "#RecetasFit", "videos": 7000, "views": 320000000, "description": "Recetas fitness rápidas", "url": "https://www.tiktok.com/tag/recetasfit"},
            {"hashtag": "#NFTCreators", "videos": 6000, "views": 290000000, "description": "Creadores de NFTs", "url": "https://www.tiktok.com/tag/nftcreators"},
            {"hashtag": "#MeditaciónGuiada", "videos": 5000, "views": 280000000, "description": "Meditaciones cortas guiadas", "url": "https://www.tiktok.com/tag/meditacionguiada"}
        ]
        
        # Añadir plataforma y timestamp
        for trend in trends:
            trend['platform'] = 'tiktok'
            trend['timestamp'] = datetime.now().isoformat()
            
        return trends

if __name__ == "__main__":
    # Ejemplo de uso
    radar = TrendRadar()
    
    # Obtener tendencias de una plataforma específica
    twitter_trends = radar.get_twitter_trends()
    print(f"Tendencias de Twitter: {len(twitter_trends)}")
    
    # Obtener tendencias de todas las plataformas
    all_trends = radar.get_all_trends()
    for platform, trends in all_trends.items():
        print(f"Tendencias de {platform}: {len(trends)}")
    
    # Obtener tendencias multiplataforma
    cross_platform = radar.get_cross_platform_trends()
    print(f"Tendencias multiplataforma: {len(cross_platform)}")
    for trend in cross_platform[:5]:
        print(f"- {trend['keyword']} (Score: {trend['relevance_score']}, Plataformas: {', '.join(trend['platforms'])})")
    
    # Obtener categorías populares
    categories = radar.get_trending_categories()
    print("\nCategorías populares:")
    for category, score in list(categories.items())[:5]:
        print(f"- {category}: {score}")