import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class AudienceSegmenter:
    """
    Sistema de segmentación de audiencia para personalizar CTAs y contenido.
    Utiliza análisis de comportamiento, demografía y engagement para crear
    segmentos de audiencia optimizados.
    """
    
    def __init__(self, data_path: str = None):
        """
        Inicializa el segmentador de audiencia.
        
        Args:
            data_path: Ruta al directorio de datos
        """
        # Configurar logging
        self.logger = logging.getLogger('AudienceSegmenter')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.FileHandler('logs/audience_segmenter.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Configurar rutas
        if data_path is None:
            self.data_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data'
            )
        else:
            self.data_path = data_path
            
        # Crear directorio si no existe
        os.makedirs(self.data_path, exist_ok=True)
        
        # Cargar datos de segmentos
        self.segments_file = os.path.join(self.data_path, 'audience_segments.json')
        self.audience_data_file = os.path.join(self.data_path, 'audience_data.json')
        
        self.segments = self._load_segments()
        self.audience_data = self._load_audience_data()
        
        # Configuración de segmentación
        self.min_cluster_size = 50  # Mínimo de usuarios por segmento
        self.max_segments = 8       # Máximo de segmentos a crear
        self.refresh_interval = 7   # Días entre actualizaciones de segmentos
        
        # Métricas de segmentación
        self.segment_metrics = {
            'last_update': None,
            'total_users': 0,
            'segment_sizes': {},
            'segment_engagement': {},
            'segment_conversion': {}
        }
        
        self.logger.info("Segmentador de audiencia inicializado")
    
    def _load_segments(self) -> Dict[str, Any]:
        """Carga los segmentos de audiencia desde el archivo JSON"""
        try:
            if os.path.exists(self.segments_file):
                with open(self.segments_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {
                    'segments': {},
                    'segment_metrics': {},
                    'last_update': None
                }
        except Exception as e:
            self.logger.error(f"Error al cargar segmentos: {str(e)}")
            return {
                'segments': {},
                'segment_metrics': {},
                'last_update': None
            }
    
    def _load_audience_data(self) -> Dict[str, Any]:
        """Carga los datos de audiencia desde el archivo JSON"""
        try:
            if os.path.exists(self.audience_data_file):
                with open(self.audience_data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {
                    'users': {},
                    'interactions': {},
                    'demographics': {},
                    'last_update': None
                }
        except Exception as e:
            self.logger.error(f"Error al cargar datos de audiencia: {str(e)}")
            return {
                'users': {},
                'interactions': {},
                'demographics': {},
                'last_update': None
            }
    
    def _save_segments(self) -> bool:
        """Guarda los segmentos de audiencia en el archivo JSON"""
        try:
            with open(self.segments_file, 'w', encoding='utf-8') as f:
                json.dump(self.segments, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar segmentos: {str(e)}")
            return False
    
    def _save_audience_data(self) -> bool:
        """Guarda los datos de audiencia en el archivo JSON"""
        try:
            with open(self.audience_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.audience_data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar datos de audiencia: {str(e)}")
            return False
    
    def update_user_data(self, user_id: str, platform: str, data: Dict[str, Any]) -> bool:
        """
        Actualiza los datos de un usuario específico.
        
        Args:
            user_id: ID del usuario
            platform: Plataforma (youtube, tiktok, instagram, etc.)
            data: Datos del usuario a actualizar
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        try:
            # Crear ID único para el usuario en la plataforma
            platform_user_id = f"{platform}_{user_id}"
            
            # Inicializar datos del usuario si no existen
            if platform_user_id not in self.audience_data['users']:
                self.audience_data['users'][platform_user_id] = {
                    'user_id': user_id,
                    'platform': platform,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'interactions': [],
                    'demographics': {},
                    'segment': None
                }
            
            # Actualizar última vez visto
            self.audience_data['users'][platform_user_id]['last_seen'] = datetime.now().isoformat()
            
            # Actualizar datos demográficos si se proporcionan
            if 'demographics' in data:
                for key, value in data['demographics'].items():
                    self.audience_data['users'][platform_user_id]['demographics'][key] = value
            
            # Actualizar interacciones si se proporcionan
            if 'interaction' in data:
                interaction = data['interaction']
                interaction['timestamp'] = datetime.now().isoformat()
                self.audience_data['users'][platform_user_id]['interactions'].append(interaction)
                
                # Limitar el número de interacciones almacenadas (últimas 50)
                if len(self.audience_data['users'][platform_user_id]['interactions']) > 50:
                    self.audience_data['users'][platform_user_id]['interactions'] = \
                        self.audience_data['users'][platform_user_id]['interactions'][-50:]
            
            # Guardar datos
            self._save_audience_data()
            
            # Verificar si es necesario actualizar segmentos
            self._check_segment_update()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al actualizar datos de usuario: {str(e)}")
            return False
    
    def _check_segment_update(self) -> None:
        """Verifica si es necesario actualizar los segmentos de audiencia"""
        try:
            # Verificar si nunca se han actualizado los segmentos
            if self.segments['last_update'] is None:
                self._update_segments()
                return
            
            # Verificar si ha pasado el intervalo de actualización
            last_update = datetime.fromisoformat(self.segments['last_update'])
            days_since_update = (datetime.now() - last_update).days
            
            if days_since_update >= self.refresh_interval:
                self._update_segments()
                
        except Exception as e:
            self.logger.error(f"Error al verificar actualización de segmentos: {str(e)}")
    
    def _update_segments(self) -> None:
        """Actualiza los segmentos de audiencia utilizando clustering"""
        try:
            self.logger.info("Iniciando actualización de segmentos de audiencia")
            
            # Verificar si hay suficientes usuarios para segmentar
            if len(self.audience_data['users']) < self.min_cluster_size:
                self.logger.info(f"No hay suficientes usuarios para segmentar ({len(self.audience_data['users'])} < {self.min_cluster_size})")
                return
            
            # Preparar datos para clustering
            user_features = self._prepare_user_features()
            
            if user_features.empty:
                self.logger.warning("No se pudieron preparar características de usuarios para segmentación")
                return
            
            # Normalizar datos
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(user_features.drop('user_id', axis=1))
            
            # Reducir dimensionalidad si hay muchas características
            if scaled_features.shape[1] > 10:
                pca = PCA(n_components=min(10, scaled_features.shape[1]))
                scaled_features = pca.fit_transform(scaled_features)
            
            # Determinar número óptimo de clusters (entre 2 y max_segments)
            optimal_clusters = self._find_optimal_clusters(scaled_features)
            
            # Aplicar KMeans
            kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_features)
            
            # Asignar segmentos a usuarios
            user_ids = user_features['user_id'].values
            
            # Crear nuevos segmentos
            new_segments = {}
            segment_sizes = {}
            
            for i in range(optimal_clusters):
                segment_id = f"segment_{i+1}"
                segment_users = user_ids[clusters == i]
                segment_size = len(segment_users)
                
                # Guardar tamaño del segmento
                segment_sizes[segment_id] = segment_size
                
                # Crear perfil del segmento
                segment_profile = self._create_segment_profile(user_features, user_ids, clusters, i)
                
                # Guardar segmento
                new_segments[segment_id] = {
                    'id': segment_id,
                    'name': f"Segmento {i+1}",
                    'size': segment_size,
                    'profile': segment_profile,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                
                # Asignar segmento a usuarios
                for user_id in segment_users:
                    if user_id in self.audience_data['users']:
                        self.audience_data['users'][user_id]['segment'] = segment_id
            
            # Actualizar segmentos
            self.segments['segments'] = new_segments
            self.segments['segment_metrics'] = {
                'total_users': len(user_ids),
                'segment_sizes': segment_sizes,
                'optimal_clusters': optimal_clusters
            }
            self.segments['last_update'] = datetime.now().isoformat()
            
            # Guardar cambios
            self._save_segments()
            self._save_audience_data()
            
            self.logger.info(f"Segmentación completada: {optimal_clusters} segmentos creados")
            
        except Exception as e:
            self.logger.error(f"Error al actualizar segmentos: {str(e)}")
    
    def _prepare_user_features(self) -> pd.DataFrame:
        """Prepara las características de los usuarios para clustering"""
        try:
            user_data = []
            
            for user_id, user in self.audience_data['users'].items():
                # Características básicas
                user_features = {
                    'user_id': user_id,
                    'days_active': (datetime.fromisoformat(user['last_seen']) - 
                                   datetime.fromisoformat(user['first_seen'])).days + 1,
                    'interaction_count': len(user['interactions']),
                }
                
                # Características demográficas
                if 'demographics' in user:
                    for key, value in user['demographics'].items():
                        if isinstance(value, (int, float)):
                            user_features[f"demo_{key}"] = value
                
                # Características de interacción
                if len(user['interactions']) > 0:
                    # Calcular tasas de interacción
                    likes = sum(1 for i in user['interactions'] if i.get('type') == 'like')
                    comments = sum(1 for i in user['interactions'] if i.get('type') == 'comment')
                    shares = sum(1 for i in user['interactions'] if i.get('type') == 'share')
                    clicks = sum(1 for i in user['interactions'] if i.get('type') == 'click')
                    
                    total = len(user['interactions'])
                    
                    user_features['like_rate'] = likes / total if total > 0 else 0
                    user_features['comment_rate'] = comments / total if total > 0 else 0
                    user_features['share_rate'] = shares / total if total > 0 else 0
                    user_features['click_rate'] = clicks / total if total > 0 else 0
                    
                    # Calcular tiempo promedio de visualización
                    view_times = [i.get('view_time', 0) for i in user['interactions'] 
                                 if 'view_time' in i and isinstance(i['view_time'], (int, float))]
                    
                    if view_times:
                        user_features['avg_view_time'] = sum(view_times) / len(view_times)
                    else:
                        user_features['avg_view_time'] = 0
                    
                    # Calcular tasa de conversión de CTA
                    cta_interactions = [i for i in user['interactions'] if 'cta_response' in i]
                    cta_conversions = [i for i in cta_interactions if i['cta_response'] == True]
                    
                    if cta_interactions:
                        user_features['cta_conversion_rate'] = len(cta_conversions) / len(cta_interactions)
                    else:
                        user_features['cta_conversion_rate'] = 0
                
                user_data.append(user_features)
            
            # Crear DataFrame
            df = pd.DataFrame(user_data)
            
            # Eliminar filas con valores NaN
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error al preparar características de usuarios: {str(e)}")
            return pd.DataFrame()
    
    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """
        Encuentra el número óptimo de clusters utilizando el método del codo
        
        Args:
            data: Datos normalizados para clustering
            
        Returns:
            Número óptimo de clusters
        """
        try:
            # Limitar el número máximo de clusters según la cantidad de datos
            max_possible_clusters = min(self.max_segments, len(data) // self.min_cluster_size)
            
            if max_possible_clusters <= 2:
                return max(2, max_possible_clusters)
            
            # Calcular inercia para diferentes números de clusters
            inertias = []
            cluster_range = range(2, max_possible_clusters + 1)
            
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            # Encontrar el punto de inflexión (método del codo)
            deltas = np.diff(inertias)
            delta_deltas = np.diff(deltas)
            
            # El punto donde la segunda derivada es máxima es el codo
            if len(delta_deltas) > 0:
                elbow_point = np.argmax(delta_deltas) + 2
                optimal_clusters = cluster_range[elbow_point]
            else:
                optimal_clusters = 3  # Valor predeterminado si no hay suficientes datos
            
            return optimal_clusters
            
        except Exception as e:
            self.logger.error(f"Error al encontrar número óptimo de clusters: {str(e)}")
            return 3  # Valor predeterminado en caso de error
    
    def _create_segment_profile(self, user_features: pd.DataFrame, 
                               user_ids: np.ndarray, clusters: np.ndarray, 
                               cluster_idx: int) -> Dict[str, Any]:
        """
        Crea un perfil para un segmento específico
        
        Args:
            user_features: DataFrame con características de usuarios
            user_ids: Array con IDs de usuarios
            clusters: Array con asignaciones de clusters
            cluster_idx: Índice del cluster a perfilar
            
        Returns:
            Diccionario con perfil del segmento
        """
        try:
            # Filtrar usuarios del segmento
            segment_mask = clusters == cluster_idx
            segment_user_ids = user_ids[segment_mask]
            
            # Crear máscara para el DataFrame
            df_mask = user_features['user_id'].isin(segment_user_ids)
            segment_df = user_features[df_mask].drop('user_id', axis=1)
            
            # Calcular estadísticas del segmento
            profile = {
                'avg_days_active': segment_df['days_active'].mean(),
                'avg_interaction_count': segment_df['interaction_count'].mean(),
            }
            
            # Añadir tasas de interacción si están disponibles
            interaction_metrics = ['like_rate', 'comment_rate', 'share_rate', 
                                  'click_rate', 'avg_view_time', 'cta_conversion_rate']
            
            for metric in interaction_metrics:
                if metric in segment_df.columns:
                    profile[metric] = segment_df[metric].mean()
            
            # Añadir características demográficas si están disponibles
            demo_columns = [col for col in segment_df.columns if col.startswith('demo_')]
            
            for col in demo_columns:
                profile[col] = segment_df[col].mean()
            
            # Determinar características distintivas del segmento
            if len(user_features) > 1:  # Necesitamos más de un segmento para comparar
                # Calcular medias globales
                global_means = user_features.drop('user_id', axis=1).mean()
                
                # Calcular diferencias relativas
                distinctive_features = {}
                
                for col in segment_df.columns:
                    if col in global_means and global_means[col] > 0:
                        segment_mean = segment_df[col].mean()
                        relative_diff = (segment_mean - global_means[col]) / global_means[col]
                        
                        if abs(relative_diff) > 0.2:  # 20% de diferencia
                            distinctive_features[col] = {
                                'segment_value': segment_mean,
                                'global_value': global_means[col],
                                'relative_diff': relative_diff
                            }
                
                profile['distinctive_features'] = distinctive_features
            
            # Generar nombre descriptivo para el segmento
            segment_name = self._generate_segment_name(profile)
            profile['suggested_name'] = segment_name
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Error al crear perfil de segmento: {str(e)}")
            return {'error': str(e)}
    
    def _generate_segment_name(self, profile: Dict[str, Any]) -> str:
        """
        Genera un nombre descriptivo para un segmento basado en su perfil
        
        Args:
            profile: Perfil del segmento
            
        Returns:
            Nombre descriptivo del segmento
        """
        try:
            # Características para nombrar segmentos
            engagement_level = None
            activity_level = None
            conversion_level = None
            
            # Determinar nivel de engagement
            if 'like_rate' in profile and 'comment_rate' in profile and 'share_rate' in profile:
                engagement_score = (profile['like_rate'] + profile['comment_rate'] * 2 + profile['share_rate'] * 3)
                
                if engagement_score > 0.8:
                    engagement_level = "Super Fans"
                elif engagement_score > 0.5:
                    engagement_level = "Activos"
                elif engagement_score > 0.2:
                    engagement_level = "Casuales"
                else:
                    engagement_level = "Observadores"
            
            # Determinar nivel de actividad
            if 'avg_days_active' in profile:
                if profile['avg_days_active'] > 30:
                    activity_level = "Leales"
                elif profile['avg_days_active'] > 14:
                    activity_level = "Recurrentes"
                elif profile['avg_days_active'] > 7:
                    activity_level = "Regulares"
                else:
                    activity_level = "Nuevos"
            
            # Determinar nivel de conversión
            if 'cta_conversion_rate' in profile:
                if profile['cta_conversion_rate'] > 0.3:
                    conversion_level = "Convertidores"
                elif profile['cta_conversion_rate'] > 0.1:
                    conversion_level = "Interesados"
                else:
                    conversion_level = "Exploradores"
            
            # Generar nombre combinando características
            name_parts = []
            
            if engagement_level:
                name_parts.append(engagement_level)
            
            if activity_level:
                name_parts.append(activity_level)
            
            if conversion_level:
                name_parts.append(conversion_level)
            
            if name_parts:
                return " ".join(name_parts)
            else:
                return "Segmento Genérico"
                
        except Exception as e:
            self.logger.error(f"Error al generar nombre de segmento: {str(e)}")
            return "Segmento Sin Nombre"
    
    def get_segment_for_user(self, user_id: str, platform: str) -> Dict[str, Any]:
        """
        Obtiene el segmento asignado a un usuario específico
        
        Args:
            user_id: ID del usuario
            platform: Plataforma (youtube, tiktok, instagram, etc.)
            
        Returns:
            Información del segmento del usuario
        """
        try:
            # Crear ID único para el usuario en la plataforma
            platform_user_id = f"{platform}_{user_id}"
            
            # Verificar si el usuario existe
            if platform_user_id not in self.audience_data['users']:
                return {'error': 'Usuario no encontrado'}
            
            # Obtener segmento del usuario
            user_data = self.audience_data['users'][platform_user_id]
            segment_id = user_data.get('segment')
            
            if not segment_id or segment_id not in self.segments['segments']:
                return {'segment': None, 'message': 'Usuario sin segmento asignado'}
            
            # Obtener información del segmento
            segment_info = self.segments['segments'][segment_id]
            
            return {
                'user_id': user_id,
                'platform': platform,
                'segment_id': segment_id,
                'segment_name': segment_info.get('name', f"Segmento {segment_id}"),
                'segment_profile': segment_info.get('profile', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error al obtener segmento de usuario: {str(e)}")
            return {'error': str(e)}
    
    def get_all_segments(self) -> Dict[str, Any]:
        """
        Obtiene información de todos los segmentos
        
        Returns:
            Diccionario con información de todos los segmentos
        """
        try:
            return {
                'segments': self.segments['segments'],
                'metrics': self.segments.get('segment_metrics', {}),
                'last_update': self.segments.get('last_update'),
                'total_users': len(self.audience_data['users'])
            }
        except Exception as e:
            self.logger.error(f"Error al obtener todos los segmentos: {str(e)}")
            return {'error': str(e)}
    
    def get_segment_recommendations(self, segment_id: str) -> Dict[str, Any]:
        """
        Obtiene recomendaciones de CTAs y contenido para un segmento específico
        
        Args:
            segment_id: ID del segmento
            
        Returns:
            Recomendaciones para el segmento
        """
        try:
            # Verificar si el segmento existe
            if segment_id not in self.segments['segments']:
                return {'error': 'Segmento no encontrado'}
            
            segment_info = self.segments['segments'][segment_id]
            segment_profile = segment_info.get('profile', {})
            
            # Generar recomendaciones basadas en el perfil del segmento
            recommendations = {
                'segment_id': segment_id,
                'segment_name': segment_info.get('name', f"Segmento {segment_id}"),
                'cta_recommendations': self._generate_cta_recommendations(segment_profile),
                'content_recommendations': self._generate_content_recommendations(segment_profile),
                'timing_recommendations': self._generate_timing_recommendations(segment_profile)
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al obtener recomendaciones de segmento: {str(e)}")
            return {'error': str(e)}
    
    def _generate_cta_recommendations(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones de CTAs basadas en el perfil del segmento
        
        Args:
            profile: Perfil del segmento
            
        Returns:
            Lista de recomendaciones de CTAs
        """
        recommendations = []
        
        try:
            # Determinar tipo de CTA basado en tasas de interacción
            like_rate = profile.get('like_rate', 0)
            comment_rate = profile.get('comment_rate', 0)
            share_rate = profile.get('share_rate', 0)
            cta_conversion_rate = profile.get('cta_conversion_rate', 0)
            
            # CTAs para segmentos con alta tasa de comentarios
            if comment_rate > 0.2:
                recommendations.append({
                    'type': 'comment',
                    'style': 'pregunta',
                    'example': '¿Qué opinas sobre esto? Comenta abajo 👇',
                    'timing': '6-8s',
                    'confidence': 0.85
                })
                
                recommendations.append({
                    'type': 'comment',
                    'style': 'gamificación',
                    'example': 'Comenta "🔥" si quieres ver la parte 2',
                    'timing': '5-7s',
                    'confidence': 0.8
                })
            
            # CTAs para segmentos con alta tasa de compartidos
            if share_rate > 0.1:
                recommendations.append({
                    'type': 'share',
                    'style': 'valor',
                    'example': 'Comparte este video con alguien que necesite esta información',
                    'timing': '7-9s',
                    'confidence': 0.75
                })
            
            # CTAs para segmentos con alta tasa de likes
            if like_rate > 0.3:
                recommendations.append({
                    'type': 'like',
                    'style': 'directo',
                    'example': 'Dale like si te ha servido esta información',
                    'timing': '4-6s',
                    'confidence': 0.7
                })
            
            # CTAs para segmentos con alta tasa de conversión
            if cta_conversion_rate > 0.2:
                recommendations.append({
                    'type': 'conversion',
                    'style': 'oferta',
                    'example': 'Enlace en la bio para acceder al recurso gratuito',
                    'timing': '8-10s',
                    'confidence': 0.9
                })
            
            # CTAs genéricos si no hay suficientes datos
            if not recommendations:
                recommendations = [
                    {
                        'type': 'follow',
                        'style': 'valor',
                        'example': 'Sigue para más contenido como este',
                        'timing': '5-7s',
                        'confidence': 0.6
                    },
                    {
                        'type': 'engagement',
                        'style': 'pregunta',
                        'example': '¿Te ha gustado? Deja tu opinión abajo',
                        'timing': '6-8s',
                        'confidence': 0.5
                    }
                ]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones de CTAs: {str(e)}")
            return [
                {
                    'type': 'generic',
                    'style': 'simple',
                    'example': 'Sigue para más contenido',
                    'timing': '5-7s',
                    'confidence': 0.5
                }
            ]
    
    def _generate_content_recommendations(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones de contenido basadas en el perfil del segmento
        
        Args:
            profile: Perfil del segmento
            
        Returns:
            Lista de recomendaciones de contenido
        """
        recommendations = []
        
        try:
            # Determinar preferencias de contenido basadas en el perfil
            avg_view_time = profile.get('avg_view_time', 0)
            
                        # Recomendaciones basadas en tiempo de visualización
            if avg_view_time > 60:  # Más de 1 minuto
                recommendations.append({
                    'content_type': 'largo',
                    'duration': '3-5 minutos',
                    'format': 'tutorial detallado',
                    'style': 'educativo',
                    'example': 'Guía paso a paso sobre [tema]',
                    'confidence': 0.85
                })
            elif avg_view_time > 30:  # Entre 30s y 1 minuto
                recommendations.append({
                    'content_type': 'medio',
                    'duration': '1-3 minutos',
                    'format': 'explicación concisa',
                    'style': 'informativo',
                    'example': 'Los 3 puntos clave sobre [tema]',
                    'confidence': 0.8
                })
            else:  # Menos de 30s
                recommendations.append({
                    'content_type': 'corto',
                    'duration': '15-30 segundos',
                    'format': 'clip de impacto',
                    'style': 'entretenido',
                    'example': 'El dato que nadie te cuenta sobre [tema]',
                    'confidence': 0.75
                })
            
            # Recomendaciones basadas en tasas de interacción
            like_rate = profile.get('like_rate', 0)
            comment_rate = profile.get('comment_rate', 0)
            share_rate = profile.get('share_rate', 0)
            
            # Contenido para audiencias que comentan mucho
            if comment_rate > 0.2:
                recommendations.append({
                    'content_type': 'participativo',
                    'format': 'preguntas y respuestas',
                    'style': 'conversacional',
                    'example': 'Respondiendo a sus preguntas sobre [tema]',
                    'confidence': 0.8
                })
            
            # Contenido para audiencias que comparten mucho
            if share_rate > 0.15:
                recommendations.append({
                    'content_type': 'viral',
                    'format': 'dato sorprendente',
                    'style': 'impactante',
                    'example': '5 datos que te sorprenderán sobre [tema]',
                    'confidence': 0.75
                })
            
            # Contenido para audiencias con alta tasa de likes
            if like_rate > 0.3:
                recommendations.append({
                    'content_type': 'entretenimiento',
                    'format': 'storytelling',
                    'style': 'narrativo',
                    'example': 'Mi experiencia con [tema]',
                    'confidence': 0.7
                })
            
            # Recomendaciones genéricas si no hay suficientes datos
            if not recommendations:
                recommendations = [
                    {
                        'content_type': 'informativo',
                        'format': 'lista',
                        'style': 'directo',
                        'example': '3 consejos sobre [tema]',
                        'confidence': 0.6
                    },
                    {
                        'content_type': 'educativo',
                        'format': 'tutorial',
                        'style': 'paso a paso',
                        'example': 'Cómo hacer [tema] en 60 segundos',
                        'confidence': 0.5
                    }
                ]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones de contenido: {str(e)}")
            return [
                {
                    'content_type': 'genérico',
                    'format': 'informativo',
                    'style': 'simple',
                    'example': 'Información sobre [tema]',
                    'confidence': 0.5
                }
            ]
    
    def _generate_timing_recommendations(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera recomendaciones de timing para publicación basadas en el perfil del segmento
        
        Args:
            profile: Perfil del segmento
            
        Returns:
            Diccionario con recomendaciones de timing
        """
        try:
            # Valores predeterminados
            recommendations = {
                'best_days': ['lunes', 'miércoles', 'viernes'],
                'best_times': ['12:00', '18:00', '21:00'],
                'posting_frequency': '3 veces por semana',
                'confidence': 0.6
            }
            
            # Personalizar según características del segmento
            avg_days_active = profile.get('avg_days_active', 0)
            
            # Ajustar frecuencia según actividad
            if avg_days_active > 20:  # Usuarios muy activos
                recommendations['posting_frequency'] = '5-7 veces por semana'
                recommendations['confidence'] = 0.8
            elif avg_days_active > 10:  # Usuarios moderadamente activos
                recommendations['posting_frequency'] = '3-4 veces por semana'
                recommendations['confidence'] = 0.7
            else:  # Usuarios poco activos
                recommendations['posting_frequency'] = '1-2 veces por semana'
                recommendations['confidence'] = 0.6
            
            # Añadir recomendaciones de consistencia
            recommendations['consistency_tips'] = [
                'Mantén un horario regular de publicación',
                'Establece un día específico para contenido premium',
                'Responde a comentarios dentro de las primeras 2 horas'
            ]
            
            # Añadir recomendaciones de experimentación
            recommendations['experimentation'] = {
                'suggestion': 'Prueba diferentes horarios cada 2 semanas',
                'metrics_to_track': ['engagement rate', 'retention', 'conversion'],
                'evaluation_period': '14 días'
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error al generar recomendaciones de timing: {str(e)}")
            return {
                'best_days': ['lunes', 'miércoles', 'viernes'],
                'best_times': ['12:00', '18:00', '21:00'],
                'posting_frequency': '3 veces por semana',
                'confidence': 0.5
            }