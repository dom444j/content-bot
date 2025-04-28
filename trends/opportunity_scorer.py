"""
Módulo de evaluación de oportunidades de tendencias.

Este módulo analiza las tendencias detectadas y las evalúa para determinar
cuáles representan las mejores oportunidades para crear contenido viral.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Importar el detector de tendencias
from trends.trend_radar import TrendRadar

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'trends.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('opportunity_scorer')

class OpportunityScorer:
    """
    Clase para evaluar y puntuar oportunidades de tendencias.
    """
    
    def __init__(self, config_path: str = 'config/niches.json'):
        """
        Inicializa el evaluador de oportunidades.
        
        Args:
            config_path: Ruta al archivo de configuración con los nichos objetivo
        """
        self.config_path = config_path
        self.trend_radar = TrendRadar()
        self.load_config()
        
        # Factores de ponderación para diferentes métricas
        self.weights = {
            'relevance': 0.25,       # Relevancia para nuestros nichos
            'volume': 0.20,          # Volumen de búsqueda/menciones
            'growth': 0.20,          # Tasa de crecimiento
            'competition': 0.15,     # Nivel de competencia (inverso)
            'monetization': 0.10,    # Potencial de monetización
            'longevity': 0.10        # Potencial de longevidad
        }
        
        # Caché de oportunidades evaluadas
        self.opportunities_cache = {}
        self.last_evaluation = datetime.now() - timedelta(hours=24)  # Forzar primera evaluación
        
    def load_config(self) -> None:
        """Carga la configuración desde el archivo JSON."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            logger.info("Configuración de nichos cargada correctamente")
            
            # Extraer palabras clave de nichos
            self.niche_keywords = {}
            for niche, data in self.config.get('niches', {}).items():
                self.niche_keywords[niche] = data.get('keywords', [])
                
        except Exception as e:
            logger.error(f"Error al cargar la configuración de nichos: {e}")
            # Configuración por defecto
            self.config = {
                "niches": {
                    "finanzas": {
                        "keywords": ["finanzas", "inversión", "ahorro", "dinero", "crypto", "bitcoin", "economía"],
                        "priority": 5
                    },
                    "tecnología": {
                        "keywords": ["tecnología", "ia", "inteligencia artificial", "gadgets", "tech", "software", "hardware"],
                        "priority": 5
                    },
                    "salud": {
                        "keywords": ["salud", "fitness", "bienestar", "ejercicio", "nutrición", "dieta", "mindfulness"],
                        "priority": 4
                    },
                    "gaming": {
                        "keywords": ["gaming", "videojuegos", "juegos", "ps5", "xbox", "nintendo", "fortnite"],
                        "priority": 4
                    },
                    "humor": {
                        "keywords": ["humor", "memes", "comedia", "risa", "divertido", "gracioso", "viral"],
                        "priority": 3
                    }
                },
                "blacklist": ["política", "religión", "guerra", "desastre", "tragedia", "muerte", "accidente"]
            }
            self.niche_keywords = {}
            for niche, data in self.config.get('niches', {}).items():
                self.niche_keywords[niche] = data.get('keywords', [])
    
    def evaluate_opportunities(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Evalúa todas las tendencias y las puntúa como oportunidades.
        
        Args:
            force_refresh: Si es True, fuerza una actualización de las tendencias
            
        Returns:
            Lista de oportunidades ordenadas por puntuación
        """
        # Verificar si necesitamos actualizar
        time_since_last = datetime.now() - self.last_evaluation
        if not force_refresh and time_since_last < timedelta(hours=3) and self.opportunities_cache:
            logger.info("Usando caché de oportunidades (última evaluación hace menos de 3 horas)")
            return self.opportunities_cache
            
        logger.info("Evaluando nuevas oportunidades de tendencias")
        
        # Obtener tendencias de todas las plataformas
        all_trends = self.trend_radar.get_all_trends()
        
        # Obtener tendencias multiplataforma (tienen mayor peso)
        cross_platform = self.trend_radar.get_cross_platform_trends()
        
        # Lista para almacenar todas las oportunidades evaluadas
        opportunities = []
        
        # Procesar tendencias multiplataforma primero (tienen prioridad)
        for trend in cross_platform:
            # Verificar si está en la lista negra
            if self._is_blacklisted(trend['keyword']):
                continue
                
            # Calcular puntuaciones para cada métrica
            scores = {
                'relevance': self._calculate_relevance(trend),
                'volume': self._calculate_volume(trend),
                'growth': self._calculate_growth(trend),
                'competition': self._calculate_competition(trend),
                'monetization': self._calculate_monetization(trend),
                'longevity': self._calculate_longevity(trend)
            }
            
            # Calcular puntuación ponderada total
            total_score = sum(scores[metric] * self.weights[metric] for metric in scores)
            
            # Crear objeto de oportunidad
            opportunity = {
                'keyword': trend['keyword'],
                'total_score': round(total_score, 2),
                'scores': {k: round(v, 2) for k, v in scores.items()},
                'platforms': trend['platforms'],
                'platform_count': trend['platform_count'],
                'examples': trend['examples'],
                'niches': self._get_relevant_niches(trend['keyword']),
                'type': 'cross_platform',
                'timestamp': datetime.now().isoformat()
            }
            
            opportunities.append(opportunity)
        
        # Procesar tendencias individuales de cada plataforma
        for platform, trends in all_trends.items():
            for trend in trends:
                # Extraer palabra clave principal según la plataforma
                if platform == 'twitter':
                    keyword = trend['name']
                elif platform == 'google_trends':
                    keyword = trend['query']
                elif platform == 'youtube':
                    keyword = trend['title']
                elif platform == 'tiktok':
                    keyword = trend['hashtag']
                else:
                    continue
                
                # Verificar si está en la lista negra
                if self._is_blacklisted(keyword):
                    continue
                    
                # Verificar si ya está incluida como tendencia multiplataforma
                if any(opp['keyword'].lower() in keyword.lower() or keyword.lower() in opp['keyword'].lower() 
                       for opp in opportunities if opp['type'] == 'cross_platform'):
                    continue
                
                # Calcular puntuaciones para cada métrica
                scores = {
                    'relevance': self._calculate_relevance({'keyword': keyword}),
                    'volume': self._calculate_volume_single(trend, platform),
                    'growth': 0.5,  # Valor por defecto para tendencias de plataforma única
                    'competition': self._calculate_competition_single(trend, platform),
                    'monetization': self._calculate_monetization_single(trend, platform),
                    'longevity': self._calculate_longevity_single(trend, platform)
                }
                
                # Calcular puntuación ponderada total (con penalización por ser de una sola plataforma)
                total_score = sum(scores[metric] * self.weights[metric] for metric in scores) * 0.8
                
                # Crear objeto de oportunidad
                opportunity = {
                    'keyword': keyword,
                    'total_score': round(total_score, 2),
                    'scores': {k: round(v, 2) for k, v in scores.items()},
                    'platforms': [platform],
                    'platform_count': 1,
                    'examples': {platform: [trend]},
                    'niches': self._get_relevant_niches(keyword),
                    'type': 'single_platform',
                    'platform': platform,
                    'timestamp': datetime.now().isoformat()
                }
                
                opportunities.append(opportunity)
        
        # Ordenar oportunidades por puntuación total
        opportunities.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Actualizar caché y timestamp
        self.opportunities_cache = opportunities
        self.last_evaluation = datetime.now()
        
        return opportunities
    
    def get_top_opportunities(self, limit: int = 10, niche: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtiene las mejores oportunidades, opcionalmente filtradas por nicho.
        
        Args:
            limit: Número máximo de oportunidades a devolver
            niche: Nicho específico para filtrar (opcional)
            
        Returns:
            Lista de las mejores oportunidades
        """
        opportunities = self.evaluate_opportunities()
        
        if niche:
            # Filtrar por nicho específico
            filtered = [opp for opp in opportunities if niche in opp['niches']]
            return filtered[:limit]
        else:
            # Devolver las mejores sin filtrar
            return opportunities[:limit]
    
    def get_opportunities_by_platform(self, platform: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene oportunidades filtradas por plataforma.
        
        Args:
            platform: Plataforma para filtrar (twitter, youtube, etc.)
            limit: Número máximo de oportunidades a devolver
            
        Returns:
            Lista de oportunidades para la plataforma especificada
        """
        opportunities = self.evaluate_opportunities()
        
        # Filtrar por plataforma
        filtered = [opp for opp in opportunities if platform in opp['platforms']]
        return filtered[:limit]
    
    def get_niche_distribution(self) -> Dict[str, int]:
        """
        Obtiene la distribución de oportunidades por nicho.
        
        Returns:
            Diccionario con recuento de oportunidades por nicho
        """
        opportunities = self.evaluate_opportunities()
        
        distribution = {}
        for niche in self.config.get('niches', {}).keys():
            distribution[niche] = 0
            
        for opp in opportunities:
            for niche in opp['niches']:
                if niche in distribution:
                    distribution[niche] += 1
        
        # Ordenar por cantidad
        sorted_dist = {k: v for k, v in sorted(distribution.items(), key=lambda item: item[1], reverse=True)}
        
        return sorted_dist
    
    def _is_blacklisted(self, text: str) -> bool:
        """Verifica si un texto contiene palabras de la lista negra."""
        text = text.lower()
        blacklist = self.config.get('blacklist', [])
        
        for word in blacklist:
            if word.lower() in text:
                return True
                
        return False
    
    def _get_relevant_niches(self, text: str) -> List[str]:
        """Identifica los nichos relevantes para un texto."""
        text = text.lower()
        relevant_niches = []
        
        for niche, keywords in self.niche_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    relevant_niches.append(niche)
                    break
        
        return relevant_niches
    
    def _calculate_relevance(self, trend: Dict[str, Any]) -> float:
        """Calcula la relevancia de una tendencia para nuestros nichos."""
        keyword = trend['keyword'].lower()
        
        # Verificar relevancia para cada nicho
        niche_scores = []
        for niche, data in self.config.get('niches', {}).items():
            priority = data.get('priority', 3)
            keywords = data.get('keywords', [])
            
            # Verificar si alguna palabra clave del nicho está en la tendencia
            for kw in keywords:
                if kw.lower() in keyword:
                    niche_scores.append(priority / 5.0)  # Normalizar a escala 0-1
                    break
        
        if not niche_scores:
            return 0.1  # Relevancia mínima si no coincide con ningún nicho
            
        # Devolver la puntuación máxima entre todos los nichos
        return max(niche_scores)
    
    def _calculate_volume(self, trend: Dict[str, Any]) -> float:
        """Calcula la puntuación de volumen para tendencias multiplataforma."""
        # Para tendencias multiplataforma, usamos platform_count y ejemplos
        platform_count = trend['platform_count']
        
        # Calcular volumen promedio en cada plataforma
        volumes = []
        for platform, examples in trend['examples'].items():
            for example in examples:
                if platform == 'twitter' and 'volume' in example:
                    volumes.append(min(1.0, example['volume'] / 100000))
                elif platform == 'google_trends' and 'volume' in example:
                    volumes.append(min(1.0, example['volume'] / 200000))
                elif platform == 'youtube' and 'views' in example:
                    volumes.append(min(1.0, example['views'] / 1000000))
                elif platform == 'tiktok' and 'views' in example:
                    volumes.append(min(1.0, example['views'] / 1000000000))
        
        if not volumes:
            base_volume = 0.5  # Valor por defecto
        else:
            base_volume = sum(volumes) / len(volumes)
        
        # Ajustar por número de plataformas (más plataformas = más volumen)
        platform_factor = min(1.0, platform_count / 4)  # Normalizar a máximo 1.0
        
        return base_volume * (1 + platform_factor)
    
    def _calculate_volume_single(self, trend: Dict[str, Any], platform: str) -> float:
        """Calcula la puntuación de volumen para tendencias de plataforma única."""
        if platform == 'twitter' and 'volume' in trend:
            return min(1.0, trend['volume'] / 100000)
        elif platform == 'google_trends' and 'volume' in trend:
            return min(1.0, trend['volume'] / 200000)
        elif platform == 'youtube' and 'views' in trend:
            return min(1.0, trend['views'] / 1000000)
        elif platform == 'tiktok' and 'views' in trend:
            return min(1.0, trend['views'] / 1000000000)
        else:
            return 0.5  # Valor por defecto
    
    def _calculate_growth(self, trend: Dict[str, Any]) -> float:
        """Calcula la tasa de crecimiento estimada de una tendencia."""
        # Para este prototipo, asignamos un valor aleatorio ponderado
        # En una implementación real, se analizaría el crecimiento histórico
        
        # Tendencias en múltiples plataformas suelen tener mayor crecimiento
        platform_count = trend.get('platform_count', 1)
        base_growth = 0.5  # Valor base
        
        # Ajustar por número de plataformas
        platform_factor = min(0.5, (platform_count - 1) * 0.15)  # Máximo +0.5 por plataformas
        
        # Simular variabilidad
        random_factor = np.random.normal(0, 0.1)  # Distribución normal con desviación 0.1
        
        growth = base_growth + platform_factor + random_factor
        
        # Limitar a rango 0-1
        return max(0.0, min(1.0, growth))
    
    def _calculate_competition(self, trend: Dict[str, Any]) -> float:
        """Calcula el nivel de competencia (inverso) para una tendencia."""
        # Estimar competencia basada en plataformas y ejemplos
        # Menos competencia = puntuación más alta
        
        keyword = trend['keyword'].lower()
        
        # Palabras clave genéricas suelen tener más competencia
        if len(keyword) < 8:
            base_competition = 0.3  # Alta competencia (baja puntuación)
        elif len(keyword) < 15:
            base_competition = 0.5  # Competencia media
        else:
            base_competition = 0.7  # Baja competencia (alta puntuación)
        
        # Ajustar por número de plataformas (más plataformas = más competencia)
        platform_count = trend.get('platform_count', 1)
        platform_factor = max(0.0, 0.1 - (platform_count - 1) * 0.05)  # Máximo -0.15 por plataformas
        
        # Simular variabilidad
        random_factor = np.random.normal(0, 0.1)
        
        competition = base_competition + platform_factor + random_factor
        
        # Limitar a rango 0-1
        return max(0.0, min(1.0, competition))
    
    def _calculate_competition_single(self, trend: Dict[str, Any], platform: str) -> float:
        """Calcula el nivel de competencia para tendencias de plataforma única."""
        # Estimar competencia basada en la plataforma y métricas específicas
        
        if platform == 'twitter':
            text = trend.get('name', '')
            volume = trend.get('volume', 50000)
            # Más volumen = más competencia
            competition = 1.0 - min(1.0, volume / 150000)
        elif platform == 'google_trends':
            text = trend.get('query', '')
            volume = trend.get('volume', 100000)
            competition = 1.0 - min(1.0, volume / 300000)
        elif platform == 'youtube':
            text = trend.get('title', '')
            views = trend.get('views', 500000)
            competition = 1.0 - min(1.0, views / 2000000)
        elif platform == 'tiktok':
            text = trend.get('hashtag', '')
            videos = trend.get('videos', 10000)
            competition = 1.0 - min(1.0, videos / 30000)
        else:
            text = ''
            competition = 0.5
        
        # Ajustar por longitud del texto (textos más específicos suelen tener menos competencia)
        if text:
            length_factor = min(0.2, len(text) / 100)
            competition += length_factor
        
        # Limitar a rango 0-1
        return max(0.0, min(1.0, competition))
    
    def _calculate_monetization(self, trend: Dict[str, Any]) -> float:
        """Calcula el potencial de monetización de una tendencia."""
        keyword = trend['keyword'].lower()
        
        # Palabras clave relacionadas con monetización
        monetization_keywords = [
            'comprar', 'vender', 'precio', 'oferta', 'descuento', 'barato', 'mejor', 
            'review', 'tutorial', 'cómo', 'guía', 'aprende', 'curso', 'masterclass',
            'inversión', 'ganar', 'dinero', 'crypto', 'bitcoin', 'finanzas', 'ahorro',
            'producto', 'servicio', 'premium', 'profesional', 'experto'
        ]
        
        # Verificar si contiene palabras clave de monetización
        for word in monetization_keywords:
            if word in keyword:
                return min(1.0, 0.7 + np.random.normal(0, 0.1))
        
        # Verificar si está en nichos de alta monetización
        high_monetization_niches = ['finanzas', 'tecnología', 'gaming']
        for niche in self._get_relevant_niches(keyword):
            if niche in high_monetization_niches:
                return min(1.0, 0.6 + np.random.normal(0, 0.1))
        
        # Valor por defecto con variabilidad
        return max(0.0, min(1.0, 0.4 + np.random.normal(0, 0.1)))
    
    def _calculate_monetization_single(self, trend: Dict[str, Any], platform: str) -> float:
        """Calcula el potencial de monetización para tendencias de plataforma única."""
        # Extraer texto según plataforma
        if platform == 'twitter':
            text = trend.get('name', '')
        elif platform == 'google_trends':
            text = trend.get('query', '')
        elif platform == 'youtube':
            text = trend.get('title', '') + ' ' + trend.get('channel', '')
        elif platform == 'tiktok':
            text = trend.get('hashtag', '') + ' ' + trend.get('description', '')
        else:
            text = ''
        
        text = text.lower()
        
        # Palabras clave relacionadas con monetización
        monetization_keywords = [
            'comprar', 'vender', 'precio', 'oferta', 'descuento', 'barato', 'mejor', 
            'review', 'tutorial', 'cómo', 'guía', 'aprende', 'curso', 'masterclass',
            'inversión', 'ganar', 'dinero', 'crypto', 'bitcoin', 'finanzas', 'ahorro',
            'producto', 'servicio', 'premium', 'profesional', 'experto'
        ]
        
        # Verificar si contiene palabras clave de monetización
        for word in monetization_keywords:
            if word in text:
                return min(1.0, 0.7 + np.random.normal(0, 0.1))
        
        # Verificar si está en nichos de alta monetización
        high_monetization_niches = ['finanzas', 'tecnología', 'gaming']
        for niche in self._get_relevant_niches(text):
            if niche in high_monetization_niches:
                return min(1.0, 0.6 + np.random.normal(0, 0.1))
        
        # Valor por defecto con variabilidad
        return max(0.0, min(1.0, 0.4 + np.random.normal(0, 0.1)))
    
    def _calculate_longevity(self, trend: Dict[str, Any]) -> float:
        """Calcula el potencial de longevidad de una tendencia."""
        keyword = trend['keyword'].lower()
        
        # Palabras clave efímeras (baja longevidad)
        ephemeral_keywords = [
            'meme', 'viral', 'challenge', 'reto', 'trend', 'tendencia', 'hoy', 'ahora',
            'noticia', 'última', 'reciente', 'actualidad', 'breaking'
        ]
        
        # Palabras clave duraderas (alta longevidad)
        evergreen_keywords = [
            'guía', 'tutorial', 'cómo', 'aprende', 'curso', 'masterclass', 'completo',
            'definitivo', 'esencial', 'básico', 'fundamental', 'principiantes', 'experto',
            'profesional', 'mejor', 'top', 'review', 'análisis', 'comparativa'
        ]
        
        # Verificar si contiene palabras clave efímeras
        for word in ephemeral_keywords:
            if word in keyword:
                return max(0.0, min(1.0, 0.3 + np.random.normal(0, 0.1)))
        
        # Verificar si contiene palabras clave duraderas
        for word in evergreen_keywords:
            if word in keyword:
                return max(0.0, min(1.0, 0.8 + np.random.normal(0, 0.1)))
        
        # Verificar si está en nichos de alta longevidad
        high_longevity_niches = ['educación', 'finanzas', 'salud']
        for niche in self._get_relevant_niches(keyword):
            if niche in high_longevity_niches:
                return max(0.0, min(1.0, 0.7 + np.random.normal(0, 0.1)))
        
        # Valor por defecto con variabilidad
        return max(0.0, min(1.0, 0.5 + np.random.normal(0, 0.1)))
    
    def _calculate_longevity_single(self, trend: Dict[str, Any], platform: str) -> float:
        """Calcula el potencial de longevidad para tendencias de plataforma única."""
        # Extraer texto según plataforma
        if platform == 'twitter':
            text = trend.get('name', '')
        elif platform == 'google_trends':
            text = trend.get('query', '')
        elif platform == 'youtube':
            text = trend.get('title', '')
        elif platform == 'tiktok':
            text = trend.get('hashtag', '') + ' ' + trend.get('description', '')
        else:
            text = ''
        
        text = text.lower()
        
        # Palabras clave efímeras (baja longevidad)
        ephemeral_keywords = [
            'meme', 'viral', 'challenge', 'reto', 'trend', 'tendencia', 'hoy', 'ahora',
            'noticia', 'última', 'reciente', 'actualidad', 'breaking'
        ]
        
        # Palabras clave duraderas (alta longevidad)
        evergreen_keywords = [
            'guía', 'tutorial', 'cómo', 'aprende', 'curso', 'masterclass', 'completo',
            'definitivo', 'esencial', 'básico', 'fundamental', 'principiantes', 'experto',
            'profesional', 'mejor', 'top', 'review', 'análisis', 'comparativa'
        ]
        
        # Verificar si contiene palabras clave efímeras
        for word in ephemeral_keywords:
            if word in text:
                return max(0.0, min(1.0, 0.3 + np.random.normal(0, 0.1)))
        
        # Verificar si contiene palabras clave duraderas
        for word in evergreen_keywords:
            if word in text:
                return max(0.0, min(1.0, 0.8 + np.random.normal(0, 0.1)))
        
                # Verificar si está en nichos de alta longevidad
        high_longevity_niches = ['educación', 'finanzas', 'salud']
        for niche in self._get_relevant_niches(text):
            if niche in high_longevity_niches:
                return max(0.0, min(1.0, 0.7 + np.random.normal(0, 0.1)))
        
        # Valor por defecto con variabilidad
        return max(0.0, min(1.0, 0.5 + np.random.normal(0, 0.1)))


if __name__ == "__main__":
    # Ejemplo de uso
    scorer = OpportunityScorer()
    
    # Obtener las mejores oportunidades
    top_opportunities = scorer.get_top_opportunities(limit=5)
    print(f"Top 5 oportunidades:")
    for i, opp in enumerate(top_opportunities, 1):
        print(f"{i}. {opp['keyword']} (Score: {opp['total_score']})")
        print(f"   Plataformas: {', '.join(opp['platforms'])}")
        print(f"   Nichos: {', '.join(opp['niches']) if opp['niches'] else 'Ninguno'}")
        print(f"   Métricas: {opp['scores']}")
        print()
    
    # Obtener oportunidades por nicho
    tech_opportunities = scorer.get_top_opportunities(limit=3, niche='tecnología')
    print(f"\nTop 3 oportunidades de tecnología:")
    for i, opp in enumerate(tech_opportunities, 1):
        print(f"{i}. {opp['keyword']} (Score: {opp['total_score']})")
    
    # Obtener oportunidades por plataforma
    youtube_opportunities = scorer.get_opportunities_by_platform('youtube', limit=3)
    print(f"\nTop 3 oportunidades de YouTube:")
    for i, opp in enumerate(youtube_opportunities, 1):
        print(f"{i}. {opp['keyword']} (Score: {opp['total_score']})")
    
    # Obtener distribución de nichos
    niche_distribution = scorer.get_niche_distribution()
    print(f"\nDistribución de oportunidades por nicho:")
    for niche, count in niche_distribution.items():
        print(f"- {niche}: {count} oportunidades")