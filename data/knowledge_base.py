"""
Knowledge Base - Base de conocimiento del sistema de monetización

Este módulo gestiona el almacenamiento y recuperación de datos críticos:
- Canales y sus configuraciones
- CTAs efectivos y su historial de rendimiento
- Assets visuales y de audio
- Métricas de rendimiento
- Reputación de contenido
- Muestras de voces y personajes
"""

import os
import json
import logging
import datetime
import time
from typing import Dict, List, Any, Optional, Union
import pymongo
from bson.objectid import ObjectId

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'knowledge_base.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('KnowledgeBase')

class KnowledgeBase:
    """
    Clase principal para la gestión de la base de conocimiento.
    Implementa el patrón Singleton para asegurar una única instancia.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeBase, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa la base de conocimiento si aún no está inicializada"""
        if self._initialized:
            return
            
        logger.info("Inicializando Knowledge Base...")
        
        # Configuración de almacenamiento
        self.storage_type = os.environ.get('STORAGE_TYPE', 'file')  # 'file', 'mongodb', 's3'
        self.mongodb_uri = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/')
        self.db_name = os.environ.get('DB_NAME', 'content_bot')
        
        # Directorios de almacenamiento local
        self.data_dir = os.path.join('data', 'storage')
        self.channels_dir = os.path.join(self.data_dir, 'channels')
        self.ctas_dir = os.path.join(self.data_dir, 'ctas')
        self.assets_dir = os.path.join(self.data_dir, 'assets')
        self.metrics_dir = os.path.join(self.data_dir, 'metrics')
        self.voices_dir = os.path.join(self.data_dir, 'voices')
        
        # Crear directorios si no existen
        os.makedirs(self.channels_dir, exist_ok=True)
        os.makedirs(self.ctas_dir, exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.voices_dir, exist_ok=True)
        
        # Conexión a MongoDB (si se usa)
        self.mongo_client = None
        self.db = None
        if self.storage_type == 'mongodb':
            try:
                self.mongo_client = pymongo.MongoClient(self.mongodb_uri)
                self.db = self.mongo_client[self.db_name]
                logger.info(f"Conexión a MongoDB establecida: {self.db_name}")
            except Exception as e:
                logger.error(f"Error al conectar a MongoDB: {str(e)}")
                logger.info("Usando almacenamiento de archivos como respaldo")
                self.storage_type = 'file'
        
        self._initialized = True
        logger.info(f"Knowledge Base inicializada correctamente usando almacenamiento: {self.storage_type}")
    
    def _save_to_file(self, directory: str, filename: str, data: Dict) -> bool:
        """Guarda datos en un archivo JSON"""
        try:
            filepath = os.path.join(directory, f"{filename}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error al guardar en archivo {filepath}: {str(e)}")
            return False
    
    def _load_from_file(self, directory: str, filename: str) -> Optional[Dict]:
        """Carga datos desde un archivo JSON"""
        try:
            filepath = os.path.join(directory, f"{filename}.json")
            if not os.path.exists(filepath):
                return None
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error al cargar desde archivo {filepath}: {str(e)}")
            return None
    
    def _list_files(self, directory: str) -> List[str]:
        """Lista los archivos en un directorio"""
        try:
            if not os.path.exists(directory):
                return []
            return [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.json')]
        except Exception as e:
            logger.error(f"Error al listar archivos en {directory}: {str(e)}")
            return []
    
    def _save_to_mongodb(self, collection: str, data: Dict, query: Dict = None) -> bool:
        """Guarda datos en MongoDB"""
        if self.db is None:
            return self._save_to_file(self.data_dir, f"{collection}_{data.get('id', 'unknown')}", data)
        
        try:
            if query and self.db[collection].find_one(query):
                self.db[collection].update_one(query, {"$set": data})
            else:
                if '_id' not in data and 'id' in data:
                    data['_id'] = data['id']
                self.db[collection].insert_one(data)
            return True
        except Exception as e:
            logger.error(f"Error al guardar en MongoDB {collection}: {str(e)}")
            # Fallback a almacenamiento de archivos
            return self._save_to_file(self.data_dir, f"{collection}_{data.get('id', 'unknown')}", data)
    
    def _load_from_mongodb(self, collection: str, query: Dict) -> Optional[Dict]:
        """Carga datos desde MongoDB"""
        if self.db is None:
            return None
        
        try:
            result = self.db[collection].find_one(query)
            if result:
                # Convertir ObjectId a string para serialización JSON
                if '_id' in result and isinstance(result['_id'], ObjectId):
                    result['_id'] = str(result['_id'])
            return result
        except Exception as e:
            logger.error(f"Error al cargar desde MongoDB {collection}: {str(e)}")
            return None
    
    def _list_from_mongodb(self, collection: str, query: Dict = None) -> List[Dict]:
        """Lista documentos desde MongoDB"""
        if self.db is None:
            return []
        
        try:
            if query is None:
                query = {}
            results = list(self.db[collection].find(query))
            # Convertir ObjectId a string para serialización JSON
            for result in results:
                if '_id' in result and isinstance(result['_id'], ObjectId):
                    result['_id'] = str(result['_id'])
            return results
        except Exception as e:
            logger.error(f"Error al listar desde MongoDB {collection}: {str(e)}")
            return []
    
    # === CANALES ===
    
    def save_channel(self, channel_data: Dict) -> bool:
        """
        Guarda la información de un canal
        
        Args:
            channel_data: Diccionario con la información del canal
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        channel_id = channel_data.get('id')
        if not channel_id:
            logger.error("No se puede guardar canal sin ID")
            return False
        
        # Añadir timestamp de actualización
        channel_data['updated_at'] = datetime.datetime.now().isoformat()
        
        if self.storage_type == 'mongodb':
            return self._save_to_mongodb('channels', channel_data, {'id': channel_id})
        else:
            return self._save_to_file(self.channels_dir, channel_id, channel_data)
    
    def get_channel(self, channel_id: str) -> Optional[Dict]:
        """
        Obtiene la información de un canal
        
        Args:
            channel_id: ID del canal
            
        Returns:
            Dict: Información del canal o None si no existe
        """
        if self.storage_type == 'mongodb':
            return self._load_from_mongodb('channels', {'id': channel_id})
        else:
            return self._load_from_file(self.channels_dir, channel_id)
    
    def get_channels(self) -> Dict[str, Dict]:
        """
        Obtiene todos los canales
        
        Returns:
            Dict: Diccionario con todos los canales (id -> datos)
        """
        channels = {}
        
        if self.storage_type == 'mongodb':
            channel_list = self._list_from_mongodb('channels')
            for channel in channel_list:
                channel_id = channel.get('id')
                if channel_id:
                    channels[channel_id] = channel
        else:
            channel_ids = self._list_files(self.channels_dir)
            for channel_id in channel_ids:
                channel = self._load_from_file(self.channels_dir, channel_id)
                if channel:
                    channels[channel_id] = channel
        
        return channels
    
    def delete_channel(self, channel_id: str) -> bool:
        """
        Elimina un canal
        
        Args:
            channel_id: ID del canal
            
        Returns:
            bool: True si se eliminó correctamente, False en caso contrario
        """
        if self.storage_type == 'mongodb' and self.db:
            try:
                self.db['channels'].delete_one({'id': channel_id})
                return True
            except Exception as e:
                logger.error(f"Error al eliminar canal de MongoDB: {str(e)}")
                return False
        else:
            try:
                filepath = os.path.join(self.channels_dir, f"{channel_id}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    return True
                return False
            except Exception as e:
                logger.error(f"Error al eliminar archivo de canal: {str(e)}")
                return False
    
    # === CTAs ===
    
    def save_cta(self, cta_data: Dict) -> bool:
        """
        Guarda un CTA
        
        Args:
            cta_data: Diccionario con la información del CTA
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        cta_id = cta_data.get('id')
        if not cta_id:
            # Generar ID si no existe
            cta_id = f"cta_{int(time.time())}_{cta_data.get('type', 'unknown')}"
            cta_data['id'] = cta_id
        
        # Añadir timestamp de creación/actualización
        if 'created_at' not in cta_data:
            cta_data['created_at'] = datetime.datetime.now().isoformat()
        cta_data['updated_at'] = datetime.datetime.now().isoformat()
        
        if self.storage_type == 'mongodb':
            return self._save_to_mongodb('ctas', cta_data, {'id': cta_id})
        else:
            return self._save_to_file(self.ctas_dir, cta_id, cta_data)
    
    def get_cta(self, cta_id: str) -> Optional[Dict]:
        """
        Obtiene un CTA
        
        Args:
            cta_id: ID del CTA
            
        Returns:
            Dict: Información del CTA o None si no existe
        """
        if self.storage_type == 'mongodb':
            return self._load_from_mongodb('ctas', {'id': cta_id})
        else:
            return self._load_from_file(self.ctas_dir, cta_id)
    
    def get_ctas(self, filters: Dict = None) -> List[Dict]:
        """
        Obtiene CTAs según filtros
        
        Args:
            filters: Filtros para la búsqueda (opcional)
            
        Returns:
            List: Lista de CTAs que cumplen los filtros
        """
        ctas = []
        
        if self.storage_type == 'mongodb':
            query = filters if filters else {}
            ctas = self._list_from_mongodb('ctas', query)
        else:
            cta_ids = self._list_files(self.ctas_dir)
            for cta_id in cta_ids:
                cta = self._load_from_file(self.ctas_dir, cta_id)
                if cta:
                    # Aplicar filtros si existen
                    if filters:
                        match = True
                        for key, value in filters.items():
                            if key not in cta or cta[key] != value:
                                match = False
                                break
                        if match:
                            ctas.append(cta)
                    else:
                        ctas.append(cta)
        
        return ctas
    
    def update_cta_performance(self, cta_id: str, metrics: Dict) -> bool:
        """
        Actualiza las métricas de rendimiento de un CTA
        
        Args:
            cta_id: ID del CTA
            metrics: Métricas de rendimiento
            
        Returns:
            bool: True si se actualizó correctamente, False en caso contrario
        """
        cta = self.get_cta(cta_id)
        if not cta:
            logger.error(f"No se encontró el CTA {cta_id}")
            return False
        
        # Actualizar métricas
        if 'performance' not in cta:
            cta['performance'] = []
        
        # Añadir nueva entrada de rendimiento
        performance_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': metrics
        }
        cta['performance'].append(performance_entry)
        
        # Actualizar reputación si existe
        if 'reputation_score' in metrics:
            cta['reputation_score'] = metrics['reputation_score']
        
        # Guardar CTA actualizado
        return self.save_cta(cta)
    
    # === ASSETS ===
    
    def save_asset(self, asset_data: Dict, asset_content: bytes = None) -> bool:
        """
        Guarda un asset (imagen, audio, etc.)
        
        Args:
            asset_data: Metadatos del asset
            asset_content: Contenido binario del asset (opcional)
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        asset_id = asset_data.get('id')
        if not asset_id:
            # Generar ID si no existe
            asset_id = f"asset_{int(time.time())}_{asset_data.get('type', 'unknown')}"
            asset_data['id'] = asset_id
        
        # Añadir timestamp de creación/actualización
        if 'created_at' not in asset_data:
            asset_data['created_at'] = datetime.datetime.now().isoformat()
        asset_data['updated_at'] = datetime.datetime.now().isoformat()
        
        # Guardar metadatos
        if self.storage_type == 'mongodb':
            metadata_saved = self._save_to_mongodb('assets', asset_data, {'id': asset_id})
        else:
            metadata_saved = self._save_to_file(self.assets_dir, asset_id, asset_data)
        
        # Guardar contenido binario si existe
        if asset_content and metadata_saved:
            try:
                # Crear subdirectorio para el tipo de asset
                asset_type = asset_data.get('type', 'misc')
                asset_type_dir = os.path.join(self.assets_dir, asset_type)
                os.makedirs(asset_type_dir, exist_ok=True)
                
                # Guardar contenido
                filepath = os.path.join(asset_type_dir, f"{asset_id}.bin")
                with open(filepath, 'wb') as f:
                    f.write(asset_content)
                return True
            except Exception as e:
                logger.error(f"Error al guardar contenido de asset: {str(e)}")
                return False
        
        return metadata_saved
    
    def get_asset(self, asset_id: str, include_content: bool = False) -> Optional[Dict]:
        """
        Obtiene un asset
        
        Args:
            asset_id: ID del asset
            include_content: Si se debe incluir el contenido binario
            
        Returns:
            Dict: Información del asset o None si no existe
        """
        # Obtener metadatos
        if self.storage_type == 'mongodb':
            asset = self._load_from_mongodb('assets', {'id': asset_id})
        else:
            asset = self._load_from_file(self.assets_dir, asset_id)
        
        if not asset:
            return None
        
        # Obtener contenido binario si se solicita
        if include_content:
            try:
                asset_type = asset.get('type', 'misc')
                filepath = os.path.join(self.assets_dir, asset_type, f"{asset_id}.bin")
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        asset['content'] = f.read()
            except Exception as e:
                logger.error(f"Error al cargar contenido de asset: {str(e)}")
        
        return asset
    
    def get_assets(self, asset_type: str = None) -> List[Dict]:
        """
        Obtiene assets según tipo
        
        Args:
            asset_type: Tipo de asset (opcional)
            
        Returns:
            List: Lista de assets del tipo especificado
        """
        assets = []
        
        if self.storage_type == 'mongodb':
            query = {'type': asset_type} if asset_type else {}
            assets = self._list_from_mongodb('assets', query)
        else:
            asset_ids = self._list_files(self.assets_dir)
            for asset_id in asset_ids:
                asset = self._load_from_file(self.assets_dir, asset_id)
                if asset and (not asset_type or asset.get('type') == asset_type):
                    assets.append(asset)
        
        return assets
    
    # === MÉTRICAS ===
    
    def save_metrics(self, channel_id: str, metrics_data: Dict) -> bool:
        """
        Guarda métricas de un canal
        
        Args:
            channel_id: ID del canal
            metrics_data: Datos de métricas
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        # Añadir timestamp si no existe
        if 'timestamp' not in metrics_data:
            metrics_data['timestamp'] = datetime.datetime.now().isoformat()
        
        # Añadir ID de canal
        metrics_data['channel_id'] = channel_id
        
        # Generar ID único para las métricas
        metrics_id = f"metrics_{channel_id}_{int(time.time())}"
        metrics_data['id'] = metrics_id
        
        if self.storage_type == 'mongodb':
            return self._save_to_mongodb('metrics', metrics_data)
        else:
            return self._save_to_file(self.metrics_dir, metrics_id, metrics_data)
    
    def get_channel_metrics(self, channel_id: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Obtiene métricas de un canal en un rango de fechas
        
        Args:
            channel_id: ID del canal
            start_date: Fecha de inicio (ISO format, opcional)
            end_date: Fecha de fin (ISO format, opcional)
            
        Returns:
            List: Lista de métricas del canal
        """
        metrics = []
        
        if self.storage_type == 'mongodb':
            query = {'channel_id': channel_id}
            if start_date:
                query['timestamp'] = {'$gte': start_date}
            if end_date:
                if 'timestamp' not in query:
                    query['timestamp'] = {}
                query['timestamp']['$lte'] = end_date
            
            metrics = self._list_from_mongodb('metrics', query)
        else:
            # Listar todos los archivos de métricas
            metrics_ids = self._list_files(self.metrics_dir)
            for metrics_id in metrics_ids:
                metrics_data = self._load_from_file(self.metrics_dir, metrics_id)
                if metrics_data and metrics_data.get('channel_id') == channel_id:
                    # Filtrar por fecha si se especifica
                    timestamp = metrics_data.get('timestamp')
                    if timestamp:
                        if start_date and timestamp < start_date:
                            continue
                        if end_date and timestamp > end_date:
                            continue
                    metrics.append(metrics_data)
        
        # Ordenar por timestamp
        metrics.sort(key=lambda x: x.get('timestamp', ''))
        return metrics
    
    # === VOCES ===
    
    def save_voice_sample(self, voice_data: Dict, audio_content: bytes = None) -> bool:
        """
        Guarda una muestra de voz
        
        Args:
            voice_data: Metadatos de la voz
            audio_content: Contenido de audio (opcional)
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        voice_id = voice_data.get('id')
        if not voice_id:
            # Generar ID si no existe
            voice_id = f"voice_{int(time.time())}_{voice_data.get('character', 'unknown')}"
            voice_data['id'] = voice_id
        
        # Añadir timestamp de creación/actualización
        if 'created_at' not in voice_data:
            voice_data['created_at'] = datetime.datetime.now().isoformat()
        voice_data['updated_at'] = datetime.datetime.now().isoformat()
        
        # Guardar metadatos
        if self.storage_type == 'mongodb':
            metadata_saved = self._save_to_mongodb('voices', voice_data, {'id': voice_id})
        else:
            metadata_saved = self._save_to_file(self.voices_dir, voice_id, voice_data)
        
        # Guardar contenido de audio si existe
        if audio_content and metadata_saved:
            try:
                # Crear subdirectorio para el personaje
                character = voice_data.get('character', 'misc')
                character_dir = os.path.join(self.voices_dir, character)
                os.makedirs(character_dir, exist_ok=True)
                
                # Guardar contenido
                filepath = os.path.join(character_dir, f"{voice_id}.wav")
                with open(filepath, 'wb') as f:
                    f.write(audio_content)
                return True
            except Exception as e:
                logger.error(f"Error al guardar contenido de voz: {str(e)}")
                return False
        
        return metadata_saved
    
    def get_voice_samples(self, character: str = None) -> List[Dict]:
        """
        Obtiene muestras de voz según personaje
        
        Args:
            character: Nombre del personaje (opcional)
            
        Returns:
            List: Lista de muestras de voz
        """
        voices = []
        
        if self.storage_type == 'mongodb':
            query = {'character': character} if character else {}
            voices = self._list_from_mongodb('voices', query)
        else:
            voice_ids = self._list_files(self.voices_dir)
            for voice_id in voice_ids:
                voice = self._load_from_file(self.voices_dir, voice_id)
                if voice and (not character or voice.get('character') == character):
                    voices.append(voice)
        
        return voices
    
    # === UTILIDADES ===
    
    def backup(self, backup_dir: str = None) -> bool:
        """
        Crea una copia de seguridad de la base de conocimiento
        
        Args:
            backup_dir: Directorio para la copia de seguridad (opcional)
            
        Returns:
            bool: True si se creó correctamente, False en caso contrario
        """
        if not backup_dir:
            backup_dir = os.path.join('backups', f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copiar directorios de datos
            import shutil
            for dir_name in ['channels', 'ctas', 'assets', 'metrics', 'voices']:
                src_dir = os.path.join(self.data_dir, dir_name)
                dst_dir = os.path.join(backup_dir, dir_name)
                if os.path.exists(src_dir):
                    shutil.copytree(src_dir, dst_dir)
            
            logger.info(f"Copia de seguridad creada en {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"Error al crear copia de seguridad: {str(e)}")
            return False
    
    def restore(self, backup_dir: str) -> bool:
        """
        Restaura una copia de seguridad
        
        Args:
            backup_dir: Directorio de la copia de seguridad
            
        Returns:
            bool: True si se restauró correctamente, False en caso contrario
        """
        if not os.path.exists(backup_dir):
            logger.error(f"Directorio de copia de seguridad no encontrado: {backup_dir}")
            return False
        
        try:
            # Copiar directorios de datos
            import shutil
            for dir_name in ['channels', 'ctas', 'assets', 'metrics', 'voices']:
                src_dir = os.path.join(backup_dir, dir_name)
                dst_dir = os.path.join(self.data_dir, dir_name)
                if os.path.exists(src_dir):
                    # Eliminar directorio destino si existe
                    if os.path.exists(dst_dir):
                        shutil.rmtree(dst_dir)
                    # Copiar directorio
                    shutil.copytree(src_dir, dst_dir)
            
            logger.info(f"Copia de seguridad restaurada desde {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"Error al restaurar copia de seguridad: {str(e)}")
            return False

# Punto de entrada para pruebas
if __name__ == "__main__":
    # Crear directorios necesarios
    os.makedirs('logs', exist_ok=True)
    
    # Inicializar base de conocimiento
    kb = KnowledgeBase()
    
    # Prueba: guardar y recuperar un canal
    channel_data = {
        'id': 'test_channel',
        'name': 'Canal de Prueba',
        'niche': 'technology',
        'platforms': ['youtube', 'tiktok'],
        'character': 'tech_expert',
        'created_at': datetime.datetime.now().isoformat(),
        'status': 'active',
        'stats': {
            'videos': 0,
            'subscribers': 0,
            'views': 0,
            'revenue': 0
        }
    }
    
    print(f"Guardando canal de prueba: {kb.save_channel(channel_data)}")
    
    # Recuperar canal
    retrieved_channel = kb.get_channel('test_channel')
    print(f"Canal recuperado: {retrieved_channel['name'] if retrieved_channel else 'No encontrado'}")
    
    # Listar todos los canales
    all_channels = kb.get_channels()
    print(f"Total de canales: {len(all_channels)}")
    
    # Prueba: guardar y recuperar un CTA
    cta_data = {
        'id': 'test_cta',
        'type': 'educational',
        'text': '¡Sigue para más consejos de tecnología!',
        'timing': 6,  # segundos
        'style': 'overlay',
        'reputation_score': 85,
        'performance': []
    }
    
    print(f"Guardando CTA de prueba: {kb.save_cta(cta_data)}")
    
    # Actualizar rendimiento del CTA
    performance_metrics = {
        'views': 1000,
        'clicks': 150,
        'conversion_rate': 0.15,
        'reputation_score': 87
    }
    
    print(f"Actualizando rendimiento del CTA: {kb.update_cta_performance('test_cta', performance_metrics)}")
    
    # Recuperar CTA actualizado
    updated_cta = kb.get_cta('test_cta')
    print(f"CTA actualizado: {updated_cta}")