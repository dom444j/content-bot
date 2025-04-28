"""
Asset Cache - Sistema de caché inteligente para reutilización de assets

Este módulo implementa un sistema de caché que permite almacenar y reutilizar
assets generados (imágenes, audio, video, etc.) para reducir costos de generación
y optimizar el rendimiento del sistema.

Características principales:
- Almacenamiento eficiente de assets por tipo, tema y plataforma
- Políticas de expiración configurables
- Búsqueda semántica para encontrar assets similares
- Estadísticas de ahorro y uso
- Integración con S3/almacenamiento local
"""

import os
import json
import time
import hashlib
import shutil
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from PIL import Image
import io
import pickle

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/asset_cache.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("asset_cache")

class AssetCache:
    """
    Sistema de caché inteligente para assets multimedia.
    
    Permite almacenar y recuperar assets generados (imágenes, audio, videos, etc.)
    para reducir costos de generación y mejorar el rendimiento del sistema.
    """
    
    def __init__(self, 
                config_path: str = "config/cache_config.json",
                use_s3: bool = False,
                local_cache_dir: str = "cache",
                s3_bucket: str = None,
                embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Inicializa el sistema de caché.
        
        Args:
            config_path: Ruta al archivo de configuración
            use_s3: Si es True, usa S3 como almacenamiento, si no, usa almacenamiento local
            local_cache_dir: Directorio local para caché
            s3_bucket: Nombre del bucket S3 (si use_s3 es True)
            embedding_model: Modelo para generar embeddings para búsqueda semántica
        """
        self.config_path = config_path
        self.use_s3 = use_s3
        self.local_cache_dir = local_cache_dir
        self.s3_bucket = s3_bucket
        self.embedding_model = embedding_model
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Inicializar directorios de caché local
        if not self.use_s3:
            self._init_local_cache()
        
        # Inicializar cliente S3 si es necesario
        self.s3_client = None
        if self.use_s3:
            self._init_s3_client()
        
        # Cargar índice de caché
        self.cache_index = self._load_cache_index()
        
        # Inicializar modelo de embeddings para búsqueda semántica
        self.sentence_transformer = None
        self._init_embedding_model()
        
        # Estadísticas de uso
        self.stats = {
            "hits": 0,
            "misses": 0,
            "savings": {
                "estimated_cost": 0.0,
                "time": 0.0
            },
            "storage": {
                "total_size": 0,
                "by_type": {}
            },
            "last_cleanup": datetime.now().isoformat()
        }
        
        # Cargar estadísticas previas si existen
        self._load_stats()
        
        logger.info(f"AssetCache inicializado. Modo: {'S3' if self.use_s3 else 'Local'}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración del caché desde el archivo de configuración."""
        default_config = {
            "expiration": {
                "image": 30,  # días
                "audio": 60,
                "video": 15,
                "text": 90,
                "thumbnail": 30
            },
            "max_size": {
                "image": 1024 * 1024 * 500,  # 500 MB
                "audio": 1024 * 1024 * 200,  # 200 MB
                "video": 1024 * 1024 * 1000,  # 1 GB
                "text": 1024 * 1024 * 50,    # 50 MB
                "thumbnail": 1024 * 1024 * 100  # 100 MB
            },
            "similarity_threshold": 0.85,
            "cleanup_frequency": 7  # días
        }
        
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Combinar con configuración predeterminada
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if subkey not in config[key]:
                                    config[key][subkey] = subvalue
                    return config
            else:
                # Crear directorio si no existe
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                # Guardar configuración predeterminada
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return default_config
    
    def _init_local_cache(self) -> None:
        """Inicializa los directorios de caché local."""
        try:
            # Crear directorio principal de caché
            os.makedirs(self.local_cache_dir, exist_ok=True)
            
            # Crear subdirectorios por tipo de asset
            asset_types = ["image", "audio", "video", "text", "thumbnail"]
            for asset_type in asset_types:
                os.makedirs(os.path.join(self.local_cache_dir, asset_type), exist_ok=True)
            
            # Crear directorio para metadatos
            os.makedirs(os.path.join(self.local_cache_dir, "metadata"), exist_ok=True)
            
            logger.info(f"Directorios de caché local inicializados en {self.local_cache_dir}")
        except Exception as e:
            logger.error(f"Error al inicializar directorios de caché local: {str(e)}")
    
    def _init_s3_client(self) -> None:
        """Inicializa el cliente de S3."""
        try:
            self.s3_client = boto3.client('s3')
            
            # Verificar si el bucket existe
            try:
                self.s3_client.head_bucket(Bucket=self.s3_bucket)
                logger.info(f"Bucket S3 {self.s3_bucket} encontrado")
            except ClientError:
                # Crear bucket si no existe
                self.s3_client.create_bucket(Bucket=self.s3_bucket)
                logger.info(f"Bucket S3 {self.s3_bucket} creado")
            
            # Crear estructura de directorios en S3
            asset_types = ["image", "audio", "video", "text", "thumbnail", "metadata"]
            for asset_type in asset_types:
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=f"{asset_type}/"
                )
            
            logger.info(f"Cliente S3 inicializado para bucket {self.s3_bucket}")
        except Exception as e:
            logger.error(f"Error al inicializar cliente S3: {str(e)}")
            self.use_s3 = False
            self._init_local_cache()
    
    def _init_embedding_model(self) -> None:
        """Inicializa el modelo de embeddings para búsqueda semántica."""
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer(self.embedding_model)
            logger.info(f"Modelo de embeddings {self.embedding_model} inicializado")
        except ImportError:
            logger.warning("sentence-transformers no está instalado. La búsqueda semántica no estará disponible.")
            self.sentence_transformer = None
        except Exception as e:
            logger.error(f"Error al inicializar modelo de embeddings: {str(e)}")
            self.sentence_transformer = None
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Carga el índice de caché desde el almacenamiento."""
        default_index = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "assets": {
                "image": {},
                "audio": {},
                "video": {},
                "text": {},
                "thumbnail": {}
            },
            "semantic_index": {
                "image": {},
                "audio": {},
                "video": {},
                "text": {},
                "thumbnail": {}
            }
        }
        
        try:
            if self.use_s3:
                try:
                    response = self.s3_client.get_object(
                        Bucket=self.s3_bucket,
                        Key="metadata/cache_index.json"
                    )
                    index = json.loads(response['Body'].read().decode('utf-8'))
                    return index
                except ClientError:
                    # Crear índice si no existe
                    self.s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key="metadata/cache_index.json",
                        Body=json.dumps(default_index, indent=4)
                    )
                    return default_index
            else:
                index_path = os.path.join(self.local_cache_dir, "metadata", "cache_index.json")
                if os.path.exists(index_path):
                    with open(index_path, 'r') as f:
                        return json.load(f)
                else:
                    # Crear índice si no existe
                    with open(index_path, 'w') as f:
                        json.dump(default_index, f, indent=4)
                    return default_index
        except Exception as e:
            logger.error(f"Error al cargar índice de caché: {str(e)}")
            return default_index
    
    def _save_cache_index(self) -> None:
        """Guarda el índice de caché en el almacenamiento."""
        try:
            # Actualizar timestamp
            self.cache_index["last_updated"] = datetime.now().isoformat()
            
            if self.use_s3:
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key="metadata/cache_index.json",
                    Body=json.dumps(self.cache_index, indent=4)
                )
            else:
                index_path = os.path.join(self.local_cache_dir, "metadata", "cache_index.json")
                with open(index_path, 'w') as f:
                    json.dump(self.cache_index, f, indent=4)
            
            logger.debug("Índice de caché guardado")
        except Exception as e:
            logger.error(f"Error al guardar índice de caché: {str(e)}")
    
    def _load_stats(self) -> None:
        """Carga las estadísticas de uso del caché."""
        try:
            if self.use_s3:
                try:
                    response = self.s3_client.get_object(
                        Bucket=self.s3_bucket,
                        Key="metadata/cache_stats.json"
                    )
                    self.stats = json.loads(response['Body'].read().decode('utf-8'))
                except ClientError:
                    # Crear estadísticas si no existen
                    self.s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key="metadata/cache_stats.json",
                        Body=json.dumps(self.stats, indent=4)
                    )
            else:
                stats_path = os.path.join(self.local_cache_dir, "metadata", "cache_stats.json")
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                        self.stats = json.load(f)
                else:
                    # Crear estadísticas si no existen
                    with open(stats_path, 'w') as f:
                        json.dump(self.stats, f, indent=4)
            
            logger.debug("Estadísticas de caché cargadas")
        except Exception as e:
            logger.error(f"Error al cargar estadísticas de caché: {str(e)}")
    
    def _save_stats(self) -> None:
        """Guarda las estadísticas de uso del caché."""
        try:
            if self.use_s3:
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key="metadata/cache_stats.json",
                    Body=json.dumps(self.stats, indent=4)
                )
            else:
                stats_path = os.path.join(self.local_cache_dir, "metadata", "cache_stats.json")
                with open(stats_path, 'w') as f:
                    json.dump(self.stats, f, indent=4)
            
            logger.debug("Estadísticas de caché guardadas")
        except Exception as e:
            logger.error(f"Error al guardar estadísticas de caché: {str(e)}")
    
    def _generate_asset_key(self, asset_type: str, metadata: Dict[str, Any]) -> str:
        """
        Genera una clave única para un asset basada en su tipo y metadatos.
        
        Args:
            asset_type: Tipo de asset (image, audio, video, text, thumbnail)
            metadata: Metadatos del asset
        
        Returns:
            Clave única para el asset
        """
        # Crear una representación estable de los metadatos
        metadata_str = json.dumps(metadata, sort_keys=True)
        
        # Generar hash
        hash_obj = hashlib.md5(metadata_str.encode())
        hash_str = hash_obj.hexdigest()
        
        return f"{asset_type}_{hash_str}"
    
    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Genera un embedding para búsqueda semántica.
        
        Args:
            text: Texto para generar embedding
        
        Returns:
            Vector de embedding o None si no se puede generar
        """
        if self.sentence_transformer is None:
            return None
        
        try:
            return self.sentence_transformer.encode(text)
        except Exception as e:
            logger.error(f"Error al generar embedding: {str(e)}")
            return None
    
    def _get_asset_path(self, asset_key: str, asset_type: str) -> str:
        """
        Obtiene la ruta de un asset en el almacenamiento local.
        
        Args:
            asset_key: Clave del asset
            asset_type: Tipo de asset
        
        Returns:
            Ruta al asset
        """
        return os.path.join(self.local_cache_dir, asset_type, asset_key)
    
    def _get_asset_s3_key(self, asset_key: str, asset_type: str) -> str:
        """
        Obtiene la clave S3 de un asset.
        
        Args:
            asset_key: Clave del asset
            asset_type: Tipo de asset
        
        Returns:
            Clave S3 del asset
        """
        return f"{asset_type}/{asset_key}"
    
    def _is_asset_expired(self, asset_info: Dict[str, Any]) -> bool:
        """
        Verifica si un asset ha expirado según la política de expiración.
        
        Args:
            asset_info: Información del asset
        
        Returns:
            True si el asset ha expirado, False en caso contrario
        """
        asset_type = asset_info["type"]
        created_at = datetime.fromisoformat(asset_info["created_at"])
        expiration_days = self.config["expiration"].get(asset_type, 30)
        
        return (datetime.now() - created_at) > timedelta(days=expiration_days)
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calcula la similitud coseno entre dos embeddings.
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
        
        Returns:
            Similitud coseno (0-1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalizar embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2
        
        # Calcular similitud coseno
        return float(np.dot(embedding1_normalized, embedding2_normalized))
    
    def _find_similar_asset(self, 
                           asset_type: str, 
                           metadata: Dict[str, Any], 
                           embedding: Optional[np.ndarray]) -> Optional[str]:
        """
        Busca un asset similar en el caché.
        
        Args:
            asset_type: Tipo de asset
            metadata: Metadatos del asset
            embedding: Embedding del asset
        
        Returns:
            Clave del asset similar o None si no se encuentra
        """
        if embedding is None or asset_type not in self.cache_index["semantic_index"]:
            return None
        
        similarity_threshold = self.config["similarity_threshold"]
        best_match = None
        best_similarity = 0.0
        
        # Buscar en el índice semántico
        for asset_key, asset_embedding in self.cache_index["semantic_index"][asset_type].items():
            # Verificar si el asset existe y no ha expirado
            if asset_key not in self.cache_index["assets"][asset_type]:
                continue
            
            asset_info = self.cache_index["assets"][asset_type][asset_key]
            if self._is_asset_expired(asset_info):
                continue
            
            # Cargar embedding
            embedding_data = np.array(asset_embedding)
            
            # Calcular similitud
            similarity = self._calculate_similarity(embedding, embedding_data)
            
            # Actualizar mejor coincidencia
            if similarity > similarity_threshold and similarity > best_similarity:
                best_match = asset_key
                best_similarity = similarity
        
        return best_match
    
    def _update_storage_stats(self) -> None:
        """Actualiza las estadísticas de almacenamiento."""
        try:
            total_size = 0
            by_type = {}
            
            for asset_type in self.cache_index["assets"]:
                type_size = 0
                
                for asset_key, asset_info in self.cache_index["assets"][asset_type].items():
                    if "size" in asset_info:
                        type_size += asset_info["size"]
                
                by_type[asset_type] = type_size
                total_size += type_size
            
            self.stats["storage"]["total_size"] = total_size
            self.stats["storage"]["by_type"] = by_type
            
            self._save_stats()
            
            logger.debug(f"Estadísticas de almacenamiento actualizadas. Total: {total_size / (1024*1024):.2f} MB")
        except Exception as e:
            logger.error(f"Error al actualizar estadísticas de almacenamiento: {str(e)}")
    
    def _cleanup_expired_assets(self, force: bool = False) -> None:
        """
        Limpia los assets expirados del caché.
        
        Args:
            force: Si es True, fuerza la limpieza aunque no haya pasado el tiempo configurado
        """
        try:
            # Verificar si es necesario hacer limpieza
            last_cleanup = datetime.fromisoformat(self.stats["last_cleanup"])
            cleanup_frequency = self.config["cleanup_frequency"]
            
            if not force and (datetime.now() - last_cleanup) < timedelta(days=cleanup_frequency):
                return
            
            logger.info("Iniciando limpieza de assets expirados")
            
            assets_removed = 0
            space_freed = 0
            
            for asset_type in self.cache_index["assets"]:
                # Crear una lista de assets a eliminar para evitar modificar el diccionario durante la iteración
                assets_to_remove = []
                
                for asset_key, asset_info in self.cache_index["assets"][asset_type].items():
                    if self._is_asset_expired(asset_info):
                        assets_to_remove.append((asset_key, asset_info))
                
                # Eliminar assets expirados
                for asset_key, asset_info in assets_to_remove:
                    # Eliminar archivo
                    if self.use_s3:
                        s3_key = self._get_asset_s3_key(asset_key, asset_type)
                        try:
                            self.s3_client.delete_object(
                                Bucket=self.s3_bucket,
                                Key=s3_key
                            )
                        except ClientError as e:
                            logger.error(f"Error al eliminar asset de S3: {str(e)}")
                    else:
                        asset_path = self._get_asset_path(asset_key, asset_type)
                        if os.path.exists(asset_path):
                            os.remove(asset_path)
                    
                    # Actualizar estadísticas
                    if "size" in asset_info:
                        space_freed += asset_info["size"]
                    
                    # Eliminar del índice
                    del self.cache_index["assets"][asset_type][asset_key]
                    
                    # Eliminar del índice semántico
                    if asset_key in self.cache_index["semantic_index"][asset_type]:
                        del self.cache_index["semantic_index"][asset_type][asset_key]
                    
                    assets_removed += 1
            
            # Actualizar estadísticas
            self.stats["last_cleanup"] = datetime.now().isoformat()
            self._update_storage_stats()
            self._save_cache_index()
            
            logger.info(f"Limpieza completada. Assets eliminados: {assets_removed}, Espacio liberado: {space_freed / (1024*1024):.2f} MB")
        except Exception as e:
            logger.error(f"Error durante la limpieza de assets: {str(e)}")
    
    def _check_storage_limits(self, asset_type: str) -> bool:
        """
        Verifica si se han alcanzado los límites de almacenamiento para un tipo de asset.
        
        Args:
            asset_type: Tipo de asset
        
        Returns:
            True si hay espacio disponible, False si se ha alcanzado el límite
        """
        if asset_type not in self.config["max_size"]:
            return True
        
        max_size = self.config["max_size"][asset_type]
        current_size = self.stats["storage"]["by_type"].get(asset_type, 0)
        
        return current_size < max_size
    
    def _make_space(self, asset_type: str, required_size: int) -> bool:
        """
        Libera espacio eliminando los assets más antiguos de un tipo.
        
        Args:
            asset_type: Tipo de asset
            required_size: Tamaño requerido en bytes
        
        Returns:
            True si se pudo liberar suficiente espacio, False en caso contrario
        """
        try:
            if asset_type not in self.cache_index["assets"]:
                return False
            
            # Ordenar assets por fecha de creación (más antiguos primero)
            assets = [(key, info) for key, info in self.cache_index["assets"][asset_type].items()]
            assets.sort(key=lambda x: x[1]["created_at"])
            
            space_freed = 0
            assets_removed = 0
            
            for asset_key, asset_info in assets:
                # Eliminar archivo
                if self.use_s3:
                    s3_key = self._get_asset_s3_key(asset_key, asset_type)
                    try:
                        self.s3_client.delete_object(
                            Bucket=self.s3_bucket,
                            Key=s3_key
                        )
                    except ClientError as e:
                        logger.error(f"Error al eliminar asset de S3: {str(e)}")
                        continue
                else:
                    asset_path = self._get_asset_path(asset_key, asset_type)
                    if os.path.exists(asset_path):
                        os.remove(asset_path)
                
                # Actualizar estadísticas
                if "size" in asset_info:
                    space_freed += asset_info["size"]
                
                # Eliminar del índice
                del self.cache_index["assets"][asset_type][asset_key]
                
                # Eliminar del índice semántico
                if asset_key in self.cache_index["semantic_index"][asset_type]:
                    del self.cache_index["semantic_index"][asset_type][asset_key]
                
                assets_removed += 1
                
                # Verificar si se ha liberado suficiente espacio
                if space_freed >= required_size:
                    break
            
            # Actualizar estadísticas
            self._update_storage_stats()
            self._save_cache_index()
            
            logger.info(f"Espacio liberado: {space_freed / (1024*1024):.2f} MB, Assets eliminados: {assets_removed}")
            
            return space_freed >= required_size
        except Exception as e:
            logger.error(f"Error al liberar espacio: {str(e)}")
            return False
    
    def store(self, 
             asset_type: str, 
             asset_data: Union[bytes, str, io.BytesIO, np.ndarray], 
             metadata: Dict[str, Any],
             description: str = "",
             force_update: bool = False) -> Dict[str, Any]:
        """
        Almacena un asset en el caché.
        
        Args:
            asset_type: Tipo de asset (image, audio, video, text, thumbnail)
            asset_data: Datos del asset
            metadata: Metadatos del asset
            description: Descripción textual para búsqueda semántica
            force_update: Si es True, actualiza el asset aunque ya exista
        
        Returns:
            Información del asset almacenado
        """
        try:
            # Verificar tipo de asset válido
            valid_types = ["image", "audio", "video", "text", "thumbnail"]
            if asset_type not in valid_types:
                logger.error(f"Tipo de asset no válido: {asset_type}")
                return {
                    "status": "error",
                    "message": f"Tipo de asset no válido: {asset_type}. Debe ser uno de {valid_types}"
                }
            
            # Generar clave única
            asset_key = self._generate_asset_key(asset_type, metadata)
            
            # Verificar si el asset ya existe
            if not force_update and asset_key in self.cache_index["assets"].get(asset_type, {}):
                asset_info = self.cache_index["assets"][asset_type][asset_key]
                
                # Verificar si ha expirado
                if not self._is_asset_expired(asset_info):
                    # Actualizar estadísticas
                    self.stats["hits"] += 1
                    self._save_stats()
                    
                    logger.info(f"Asset encontrado en caché: {asset_key}")
                    
                    return {
                        "status": "success",
                        "message": "Asset encontrado en caché",
                        "asset_key": asset_key,
                        "asset_info": asset_info,
                        "from_cache": True
                    }
            
            # Buscar asset similar si hay descripción
            if description and not force_update:
                embedding = self._generate_embedding(description)
                similar_key = self._find_similar_asset(asset_type, metadata, embedding)
                
                if similar_key:
                    asset_info = self.cache_index["assets"][asset_type][similar_key]
                    
                    # Actualizar estadísticas
                    self.stats["hits"] += 1
                    self._save_stats()
                    
                    logger.info(f"Asset similar encontrado en caché: {similar_key}")
                    
                    return {
                        "status": "success",
                        "message": "Asset similar encontrado en caché",
                        "asset_key": similar_key,
                        "asset_info": asset_info,
                        "from_cache": True,
                        "is_similar": True
                    }
            
            # Preparar datos para almacenamiento
            if isinstance(asset_data, str):
                data = asset_data.encode('utf-8')
                size = len(data)
            elif isinstance(asset_data, np.ndarray):
                # Convertir array a bytes
                data = pickle.dumps(asset_data)
                size = len(data)
            elif isinstance(asset_data, io.BytesIO):
                data = asset_data.getvalue()
                size = len(data)
            else:
                data = asset_data
                size = len(data)
            
            # Verificar límites de almacenamiento
            if not self._check_storage_limits(asset_type):
                # Intentar liberar espacio
                if not self._make_space(asset_type, size):
                    logger.warning(f"No se pudo liberar suficiente espacio para almacenar asset de tipo {asset_type}")
                    return {
                        "status": "error",
                        "message": f"Límite de almacenamiento alcanzado para {asset_type}"
                    }
            
                        # Almacenar asset
            if self.use_s3:
                s3_key = self._get_asset_s3_key(asset_key, asset_type)
                try:
                    self.s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key=s3_key,
                        Body=data
                    )
                except ClientError as e:
                    logger.error(f"Error al almacenar asset en S3: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Error al almacenar asset en S3: {str(e)}"
                    }
            else:
                asset_path = self._get_asset_path(asset_key, asset_type)
                try:
                    with open(asset_path, 'wb') as f:
                        f.write(data)
                except Exception as e:
                    logger.error(f"Error al almacenar asset localmente: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Error al almacenar asset localmente: {str(e)}"
                    }
            
            # Crear información del asset
            asset_info = {
                "id": asset_key,
                "type": asset_type,
                "size": size,
                "metadata": metadata,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0
            }
            
            # Guardar en índice
            if asset_type not in self.cache_index["assets"]:
                self.cache_index["assets"][asset_type] = {}
            
            self.cache_index["assets"][asset_type][asset_key] = asset_info
            
            # Guardar embedding si está disponible
            if description and self.sentence_transformer is not None:
                embedding = self._generate_embedding(description)
                if embedding is not None:
                    if asset_type not in self.cache_index["semantic_index"]:
                        self.cache_index["semantic_index"][asset_type] = {}
                    
                    self.cache_index["semantic_index"][asset_type][asset_key] = embedding.tolist()
            
            # Actualizar estadísticas
            self.stats["misses"] += 1
            self._update_storage_stats()
            self._save_cache_index()
            
            logger.info(f"Asset almacenado: {asset_key} ({size / 1024:.2f} KB)")
            
            return {
                "status": "success",
                "message": "Asset almacenado correctamente",
                "asset_key": asset_key,
                "asset_info": asset_info,
                "from_cache": False
            }
            
        except Exception as e:
            logger.error(f"Error al almacenar asset: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al almacenar asset: {str(e)}"
            }
    
    def retrieve(self, 
                asset_key: str = None, 
                asset_type: str = None, 
                metadata: Dict[str, Any] = None,
                description: str = None) -> Dict[str, Any]:
        """
        Recupera un asset del caché.
        
        Args:
            asset_key: Clave del asset a recuperar
            asset_type: Tipo de asset (requerido si no se proporciona asset_key)
            metadata: Metadatos para buscar el asset (requerido si no se proporciona asset_key)
            description: Descripción para búsqueda semántica
            
        Returns:
            Datos del asset y su información
        """
        try:
            # Verificar parámetros
            if asset_key is None:
                if asset_type is None or metadata is None:
                    return {
                        "status": "error",
                        "message": "Se debe proporcionar asset_key o (asset_type y metadata)"
                    }
                
                # Generar clave a partir de metadatos
                asset_key = self._generate_asset_key(asset_type, metadata)
            else:
                # Extraer tipo de asset de la clave
                parts = asset_key.split("_", 1)
                if len(parts) != 2 or parts[0] not in ["image", "audio", "video", "text", "thumbnail"]:
                    return {
                        "status": "error",
                        "message": f"Clave de asset no válida: {asset_key}"
                    }
                
                asset_type = parts[0]
            
            # Buscar asset en el índice
            if asset_type not in self.cache_index["assets"] or asset_key not in self.cache_index["assets"][asset_type]:
                # Si no se encuentra, intentar búsqueda semántica
                if description and self.sentence_transformer is not None:
                    embedding = self._generate_embedding(description)
                    similar_key = self._find_similar_asset(asset_type, metadata or {}, embedding)
                    
                    if similar_key:
                        asset_key = similar_key
                    else:
                        return {
                            "status": "error",
                            "message": "Asset no encontrado en caché"
                        }
                else:
                    return {
                        "status": "error",
                        "message": "Asset no encontrado en caché"
                    }
            
            # Obtener información del asset
            asset_info = self.cache_index["assets"][asset_type][asset_key]
            
            # Verificar si ha expirado
            if self._is_asset_expired(asset_info):
                # Eliminar asset expirado
                if self.use_s3:
                    s3_key = self._get_asset_s3_key(asset_key, asset_type)
                    try:
                        self.s3_client.delete_object(
                            Bucket=self.s3_bucket,
                            Key=s3_key
                        )
                    except ClientError:
                        pass
                else:
                    asset_path = self._get_asset_path(asset_key, asset_type)
                    if os.path.exists(asset_path):
                        os.remove(asset_path)
                
                # Eliminar del índice
                del self.cache_index["assets"][asset_type][asset_key]
                
                # Eliminar del índice semántico
                if asset_key in self.cache_index["semantic_index"].get(asset_type, {}):
                    del self.cache_index["semantic_index"][asset_type][asset_key]
                
                self._save_cache_index()
                
                return {
                    "status": "error",
                    "message": "Asset expirado"
                }
            
            # Recuperar datos del asset
            if self.use_s3:
                s3_key = self._get_asset_s3_key(asset_key, asset_type)
                try:
                    response = self.s3_client.get_object(
                        Bucket=self.s3_bucket,
                        Key=s3_key
                    )
                    data = response['Body'].read()
                except ClientError as e:
                    logger.error(f"Error al recuperar asset de S3: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Error al recuperar asset de S3: {str(e)}"
                    }
            else:
                asset_path = self._get_asset_path(asset_key, asset_type)
                if not os.path.exists(asset_path):
                    # Eliminar del índice si el archivo no existe
                    del self.cache_index["assets"][asset_type][asset_key]
                    self._save_cache_index()
                    
                    return {
                        "status": "error",
                        "message": "Archivo de asset no encontrado"
                    }
                
                try:
                    with open(asset_path, 'rb') as f:
                        data = f.read()
                except Exception as e:
                    logger.error(f"Error al leer asset: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Error al leer asset: {str(e)}"
                    }
            
            # Actualizar información de acceso
            asset_info["last_accessed"] = datetime.now().isoformat()
            asset_info["access_count"] += 1
            
            # Actualizar estadísticas
            self.stats["hits"] += 1
            
            # Guardar cambios
            self._save_cache_index()
            self._save_stats()
            
            logger.info(f"Asset recuperado: {asset_key}")
            
            return {
                "status": "success",
                "message": "Asset recuperado correctamente",
                "asset_key": asset_key,
                "asset_info": asset_info,
                "data": data
            }
            
        except Exception as e:
            logger.error(f"Error al recuperar asset: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al recuperar asset: {str(e)}"
            }
    
    def delete(self, asset_key: str = None, asset_type: str = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Elimina un asset del caché.
        
        Args:
            asset_key: Clave del asset a eliminar
            asset_type: Tipo de asset (requerido si no se proporciona asset_key)
            metadata: Metadatos para identificar el asset (requerido si no se proporciona asset_key)
            
        Returns:
            Resultado de la operación
        """
        try:
            # Verificar parámetros
            if asset_key is None:
                if asset_type is None or metadata is None:
                    return {
                        "status": "error",
                        "message": "Se debe proporcionar asset_key o (asset_type y metadata)"
                    }
                
                # Generar clave a partir de metadatos
                asset_key = self._generate_asset_key(asset_type, metadata)
            else:
                # Extraer tipo de asset de la clave
                parts = asset_key.split("_", 1)
                if len(parts) != 2 or parts[0] not in ["image", "audio", "video", "text", "thumbnail"]:
                    return {
                        "status": "error",
                        "message": f"Clave de asset no válida: {asset_key}"
                    }
                
                asset_type = parts[0]
            
            # Verificar si el asset existe
            if asset_type not in self.cache_index["assets"] or asset_key not in self.cache_index["assets"][asset_type]:
                return {
                    "status": "error",
                    "message": "Asset no encontrado"
                }
            
            # Obtener información del asset
            asset_info = self.cache_index["assets"][asset_type][asset_key]
            
            # Eliminar archivo
            if self.use_s3:
                s3_key = self._get_asset_s3_key(asset_key, asset_type)
                try:
                    self.s3_client.delete_object(
                        Bucket=self.s3_bucket,
                        Key=s3_key
                    )
                except ClientError as e:
                    logger.error(f"Error al eliminar asset de S3: {str(e)}")
            else:
                asset_path = self._get_asset_path(asset_key, asset_type)
                if os.path.exists(asset_path):
                    os.remove(asset_path)
            
            # Eliminar del índice
            del self.cache_index["assets"][asset_type][asset_key]
            
            # Eliminar del índice semántico
            if asset_key in self.cache_index["semantic_index"].get(asset_type, {}):
                del self.cache_index["semantic_index"][asset_type][asset_key]
            
            # Actualizar estadísticas
            self._update_storage_stats()
            self._save_cache_index()
            
            logger.info(f"Asset eliminado: {asset_key}")
            
            return {
                "status": "success",
                "message": "Asset eliminado correctamente",
                "asset_key": asset_key
            }
            
        except Exception as e:
            logger.error(f"Error al eliminar asset: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al eliminar asset: {str(e)}"
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de uso del caché.
        
        Returns:
            Estadísticas de uso
        """
        try:
            # Actualizar estadísticas de almacenamiento
            self._update_storage_stats()
            
            # Calcular tasa de aciertos
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            # Calcular estadísticas por tipo
            assets_by_type = {}
            for asset_type in self.cache_index["assets"]:
                assets_by_type[asset_type] = len(self.cache_index["assets"][asset_type])
            
            return {
                "status": "success",
                "stats": {
                    "hits": self.stats["hits"],
                    "misses": self.stats["misses"],
                    "hit_rate": hit_rate,
                    "total_requests": total_requests,
                    "savings": self.stats["savings"],
                    "storage": {
                        "total_size_bytes": self.stats["storage"]["total_size"],
                        "total_size_mb": self.stats["storage"]["total_size"] / (1024 * 1024),
                        "by_type": {
                            asset_type: {
                                "size_bytes": size,
                                "size_mb": size / (1024 * 1024)
                            }
                            for asset_type, size in self.stats["storage"]["by_type"].items()
                        }
                    },
                    "assets_count": {
                        "total": sum(len(assets) for assets in self.cache_index["assets"].values()),
                        "by_type": assets_by_type
                    },
                    "last_cleanup": self.stats["last_cleanup"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al obtener estadísticas: {str(e)}"
            }
    
    def clear_cache(self, asset_type: str = None) -> Dict[str, Any]:
        """
        Limpia el caché, eliminando todos los assets o solo los de un tipo específico.
        
        Args:
            asset_type: Tipo de asset a limpiar (si es None, limpia todo el caché)
            
        Returns:
            Resultado de la operación
        """
        try:
            assets_removed = 0
            space_freed = 0
            
            if asset_type:
                # Verificar tipo de asset válido
                valid_types = ["image", "audio", "video", "text", "thumbnail"]
                if asset_type not in valid_types:
                    return {
                        "status": "error",
                        "message": f"Tipo de asset no válido: {asset_type}"
                    }
                
                # Limpiar solo un tipo de asset
                if asset_type in self.cache_index["assets"]:
                    for asset_key, asset_info in self.cache_index["assets"][asset_type].items():
                        # Eliminar archivo
                        if self.use_s3:
                            s3_key = self._get_asset_s3_key(asset_key, asset_type)
                            try:
                                self.s3_client.delete_object(
                                    Bucket=self.s3_bucket,
                                    Key=s3_key
                                )
                            except ClientError:
                                pass
                        else:
                            asset_path = self._get_asset_path(asset_key, asset_type)
                            if os.path.exists(asset_path):
                                os.remove(asset_path)
                        
                        # Actualizar estadísticas
                        if "size" in asset_info:
                            space_freed += asset_info["size"]
                        
                        assets_removed += 1
                    
                    # Limpiar índices
                    self.cache_index["assets"][asset_type] = {}
                    if asset_type in self.cache_index["semantic_index"]:
                        self.cache_index["semantic_index"][asset_type] = {}
            else:
                # Limpiar todo el caché
                for asset_type in self.cache_index["assets"]:
                    for asset_key, asset_info in self.cache_index["assets"][asset_type].items():
                        # Eliminar archivo
                        if self.use_s3:
                            s3_key = self._get_asset_s3_key(asset_key, asset_type)
                            try:
                                self.s3_client.delete_object(
                                    Bucket=self.s3_bucket,
                                    Key=s3_key
                                )
                            except ClientError:
                                pass
                        else:
                            asset_path = self._get_asset_path(asset_key, asset_type)
                            if os.path.exists(asset_path):
                                os.remove(asset_path)
                        
                        # Actualizar estadísticas
                        if "size" in asset_info:
                            space_freed += asset_info["size"]
                        
                        assets_removed += 1
                
                # Reiniciar índices
                for asset_type in self.cache_index["assets"]:
                    self.cache_index["assets"][asset_type] = {}
                
                for asset_type in self.cache_index["semantic_index"]:
                    self.cache_index["semantic_index"][asset_type] = {}
            
            # Actualizar estadísticas
            self._update_storage_stats()
            self._save_cache_index()
            
            logger.info(f"Caché limpiado. Assets eliminados: {assets_removed}, Espacio liberado: {space_freed / (1024*1024):.2f} MB")
            
            return {
                "status": "success",
                "message": f"Caché limpiado correctamente",
                "assets_removed": assets_removed,
                "space_freed_bytes": space_freed,
                "space_freed_mb": space_freed / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error al limpiar caché: {str(e)}")
            return {
                "status": "error",
                "message": f"Error al limpiar caché: {str(e)}"
            }
    
    def update_cost_savings(self, asset_type: str, cost_per_generation: float, time_saved: float) -> None:
        """
        Actualiza las estadísticas de ahorro de costos.
        
        Args:
            asset_type: Tipo de asset
            cost_per_generation: Costo estimado por generación del asset
            time_saved: Tiempo ahorrado en segundos
        """
        try:
            self.stats["savings"]["estimated_cost"] += cost_per_generation
            self.stats["savings"]["time"] += time_saved
            
            self._save_stats()
            
            logger.debug(f"Estadísticas de ahorro actualizadas. Total: ${self.stats['savings']['estimated_cost']:.2f}")
        except Exception as e:
            logger.error(f"Error al actualizar estadísticas de ahorro: {str(e)}")


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar caché
    cache = AssetCache(local_cache_dir="cache/assets")
    
    # Ejemplo: almacenar una imagen
    image_data = b"Datos de imagen simulados"
    image_metadata = {
        "width": 800,
        "height": 600,
        "format": "png",
        "prompt": "Un paisaje de montañas con un lago",
        "model": "stable-diffusion-v1.5"
    }
    
    result = cache.store(
        asset_type="image",
        asset_data=image_data,
        metadata=image_metadata,
        description="Paisaje de montañas con lago al atardecer, cielo naranja, reflejo en el agua"
    )
    
    print(f"Almacenamiento de imagen: {result['status']}")
    
    if result['status'] == 'success':
        # Recuperar la imagen
        asset_key = result['asset_key']
        retrieve_result = cache.retrieve(asset_key=asset_key)
        
        print(f"Recuperación de imagen: {retrieve_result['status']}")
        
        # Buscar por descripción similar
        similar_result = cache.retrieve(
            asset_type="image",
            description="Montañas con un lago y reflejo"
        )
        
        print(f"Búsqueda semántica: {similar_result['status']}")
        
        # Obtener estadísticas
        stats = cache.get_stats()
        print(f"Estadísticas: {stats['status']}")
        if stats['status'] == 'success':
            print(f"Tasa de aciertos: {stats['stats']['hit_rate']:.2f}")
            print(f"Total de assets: {stats['stats']['assets_count']['total']}")