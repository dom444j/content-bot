"""
Módulo de persistencia para el sistema de planificación.

Este módulo proporciona funcionalidades para almacenar y recuperar
el estado del planificador, incluyendo tareas pendientes, historial
de ejecución y configuración del sistema.
"""

import os
import json
import logging
import pickle
import threading
import time
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime, timedelta
import sqlite3
import shutil

from ..core.task_model import Task, TaskLifecycle, TaskPriority

# Configurar logging
logger = logging.getLogger('Scheduler.Utils.Persistence')

class PersistenceManager:
    """
    Gestor de persistencia para el sistema de planificación.
    
    Esta clase se encarga de almacenar y recuperar el estado del planificador,
    incluyendo tareas pendientes, historial de ejecución y configuración.
    Soporta múltiples backends de almacenamiento (archivo, SQLite, MongoDB).
    """
    
    # Tipos de almacenamiento soportados
    STORAGE_FILE = 'file'
    STORAGE_SQLITE = 'sqlite'
    STORAGE_MONGODB = 'mongodb'
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa el gestor de persistencia.
        
        Args:
            config: Configuración opcional para el gestor
        """
        self.config = config or {}
        
        # Tipo de almacenamiento (por defecto: archivo)
        self.storage_type = self.config.get('storage_type', self.STORAGE_FILE)
        
        # Directorio base para almacenamiento
        self.base_dir = self.config.get('base_dir', os.path.join('data', 'scheduler'))
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Configuración específica para cada tipo de almacenamiento
        self.file_config = self.config.get('file_config', {
            'tasks_file': os.path.join(self.base_dir, 'tasks.json'),
            'history_file': os.path.join(self.base_dir, 'task_history.json'),
            'config_file': os.path.join(self.base_dir, 'scheduler_config.json'),
            'backup_dir': os.path.join(self.base_dir, 'backups'),
            'max_backups': 5,
            'backup_interval': 3600  # segundos (1 hora)
        })
        
        self.sqlite_config = self.config.get('sqlite_config', {
            'db_file': os.path.join(self.base_dir, 'scheduler.db'),
            'backup_dir': os.path.join(self.base_dir, 'backups'),
            'max_backups': 5,
            'backup_interval': 3600  # segundos (1 hora)
        })
        
        self.mongodb_config = self.config.get('mongodb_config', {
            'connection_string': 'mongodb://localhost:27017/',
            'database': 'scheduler',
            'tasks_collection': 'tasks',
            'history_collection': 'task_history',
            'config_collection': 'scheduler_config'
        })
        
        # Intervalo de auto-guardado (segundos)
        self.autosave_interval = self.config.get('autosave_interval', 300)  # 5 minutos
        
        # Inicializar almacenamiento
        self._initialize_storage()
        
        # Configurar auto-guardado si está habilitado
        self._stop_autosave = threading.Event()
        if self.config.get('autosave_enabled', True):
            self._autosave_thread = threading.Thread(
                target=self._autosave_loop,
                daemon=True,
                name="PersistenceAutosaveThread"
            )
            self._autosave_thread.start()
            logger.debug("Hilo de auto-guardado iniciado")
        
        logger.info(f"PersistenceManager inicializado con almacenamiento tipo '{self.storage_type}'")
    
    def _initialize_storage(self) -> None:
        """
        Inicializa el almacenamiento según el tipo configurado.
        """
        if self.storage_type == self.STORAGE_FILE:
            # Crear directorios necesarios
            os.makedirs(os.path.dirname(self.file_config['tasks_file']), exist_ok=True)
            os.makedirs(os.path.dirname(self.file_config['history_file']), exist_ok=True)
            os.makedirs(os.path.dirname(self.file_config['config_file']), exist_ok=True)
            os.makedirs(self.file_config['backup_dir'], exist_ok=True)
            
            # Verificar si existen los archivos, si no, crearlos
            for file_path in [self.file_config['tasks_file'], 
                             self.file_config['history_file'], 
                             self.file_config['config_file']]:
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        json.dump({}, f)
            
            logger.debug("Almacenamiento de archivos inicializado")
            
        elif self.storage_type == self.STORAGE_SQLITE:
            # Crear directorio para la base de datos
            os.makedirs(os.path.dirname(self.sqlite_config['db_file']), exist_ok=True)
            os.makedirs(self.sqlite_config['backup_dir'], exist_ok=True)
            
            # Inicializar base de datos SQLite
            conn = sqlite3.connect(self.sqlite_config['db_file'])
            cursor = conn.cursor()
            
            # Crear tablas si no existen
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    task_data TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_history (
                    id TEXT PRIMARY KEY,
                    task_data TEXT,
                    created_at TEXT,
                    completed_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scheduler_config (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.debug("Almacenamiento SQLite inicializado")
            
        elif self.storage_type == self.STORAGE_MONGODB:
            # MongoDB se inicializa bajo demanda
            # Verificar si podemos conectar
            try:
                import pymongo
                client = pymongo.MongoClient(self.mongodb_config['connection_string'])
                db = client[self.mongodb_config['database']]
                # Crear índices si no existen
                db[self.mongodb_config['tasks_collection']].create_index('id', unique=True)
                db[self.mongodb_config['history_collection']].create_index('id', unique=True)
                db[self.mongodb_config['config_collection']].create_index('key', unique=True)
                logger.debug("Almacenamiento MongoDB inicializado")
            except ImportError:
                logger.error("No se pudo importar pymongo. Instale con: pip install pymongo")
                raise
            except Exception as e:
                logger.error(f"Error al inicializar MongoDB: {str(e)}")
                raise
        else:
            logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
            raise ValueError(f"Tipo de almacenamiento no soportado: {self.storage_type}")
    
    def save_tasks(self, tasks: List[Task]) -> bool:
        """
        Guarda una lista de tareas en el almacenamiento.
        
        Args:
            tasks: Lista de tareas a guardar
            
        Returns:
            True si se guardaron correctamente, False en caso contrario
        """
        try:
            if self.storage_type == self.STORAGE_FILE:
                return self._save_tasks_to_file(tasks)
            elif self.storage_type == self.STORAGE_SQLITE:
                return self._save_tasks_to_sqlite(tasks)
            elif self.storage_type == self.STORAGE_MONGODB:
                return self._save_tasks_to_mongodb(tasks)
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return False
        except Exception as e:
            logger.error(f"Error al guardar tareas: {str(e)}")
            return False
    
    def load_tasks(self) -> List[Task]:
        """
        Carga tareas desde el almacenamiento.
        
        Returns:
            Lista de tareas cargadas
        """
        try:
            if self.storage_type == self.STORAGE_FILE:
                return self._load_tasks_from_file()
            elif self.storage_type == self.STORAGE_SQLITE:
                return self._load_tasks_from_sqlite()
            elif self.storage_type == self.STORAGE_MONGODB:
                return self._load_tasks_from_mongodb()
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return []
        except Exception as e:
            logger.error(f"Error al cargar tareas: {str(e)}")
            return []
    
    def save_task_history(self, tasks: List[Task]) -> bool:
        """
        Guarda el historial de tareas en el almacenamiento.
        
        Args:
            tasks: Lista de tareas completadas para el historial
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            if self.storage_type == self.STORAGE_FILE:
                return self._save_history_to_file(tasks)
            elif self.storage_type == self.STORAGE_SQLITE:
                return self._save_history_to_sqlite(tasks)
            elif self.storage_type == self.STORAGE_MONGODB:
                return self._save_history_to_mongodb(tasks)
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return False
        except Exception as e:
            logger.error(f"Error al guardar historial de tareas: {str(e)}")
            return False
    
    def load_task_history(self, limit: int = 1000, 
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> List[Task]:
        """
        Carga el historial de tareas desde el almacenamiento.
        
        Args:
            limit: Número máximo de tareas a cargar
            start_date: Fecha de inicio para filtrar (opcional)
            end_date: Fecha de fin para filtrar (opcional)
            
        Returns:
            Lista de tareas del historial
        """
        try:
            if self.storage_type == self.STORAGE_FILE:
                return self._load_history_from_file(limit, start_date, end_date)
            elif self.storage_type == self.STORAGE_SQLITE:
                return self._load_history_from_sqlite(limit, start_date, end_date)
            elif self.storage_type == self.STORAGE_MONGODB:
                return self._load_history_from_mongodb(limit, start_date, end_date)
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return []
        except Exception as e:
            logger.error(f"Error al cargar historial de tareas: {str(e)}")
            return []
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Guarda la configuración del planificador.
        
        Args:
            config: Diccionario con la configuración
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        try:
            if self.storage_type == self.STORAGE_FILE:
                return self._save_config_to_file(config)
            elif self.storage_type == self.STORAGE_SQLITE:
                return self._save_config_to_sqlite(config)
            elif self.storage_type == self.STORAGE_MONGODB:
                return self._save_config_to_mongodb(config)
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return False
        except Exception as e:
            logger.error(f"Error al guardar configuración: {str(e)}")
            return False
    
    def load_config(self) -> Dict[str, Any]:
        """
        Carga la configuración del planificador.
        
        Returns:
            Diccionario con la configuración
        """
        try:
            if self.storage_type == self.STORAGE_FILE:
                return self._load_config_from_file()
            elif self.storage_type == self.STORAGE_SQLITE:
                return self._load_config_from_sqlite()
            elif self.storage_type == self.STORAGE_MONGODB:
                return self._load_config_from_mongodb()
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            return {}
    
    def create_backup(self) -> str:
        """
        Crea una copia de seguridad del almacenamiento actual.
        
        Returns:
            Ruta de la copia de seguridad creada o cadena vacía si falló
        """
        try:
            if self.storage_type == self.STORAGE_FILE:
                return self._create_file_backup()
            elif self.storage_type == self.STORAGE_SQLITE:
                return self._create_sqlite_backup()
            elif self.storage_type == self.STORAGE_MONGODB:
                return self._create_mongodb_backup()
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return ""
        except Exception as e:
            logger.error(f"Error al crear copia de seguridad: {str(e)}")
            return ""
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Restaura una copia de seguridad.
        
        Args:
            backup_path: Ruta de la copia de seguridad a restaurar
            
        Returns:
            True si se restauró correctamente, False en caso contrario
        """
        try:
            if self.storage_type == self.STORAGE_FILE:
                return self._restore_file_backup(backup_path)
            elif self.storage_type == self.STORAGE_SQLITE:
                return self._restore_sqlite_backup(backup_path)
            elif self.storage_type == self.STORAGE_MONGODB:
                return self._restore_mongodb_backup(backup_path)
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return False
        except Exception as e:
            logger.error(f"Error al restaurar copia de seguridad: {str(e)}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        Lista las copias de seguridad disponibles.
        
        Returns:
            Lista de diccionarios con información de las copias de seguridad
        """
        try:
            if self.storage_type == self.STORAGE_FILE:
                return self._list_file_backups()
            elif self.storage_type == self.STORAGE_SQLITE:
                return self._list_sqlite_backups()
            elif self.storage_type == self.STORAGE_MONGODB:
                return self._list_mongodb_backups()
            else:
                logger.error(f"Tipo de almacenamiento no soportado: {self.storage_type}")
                return []
        except Exception as e:
            logger.error(f"Error al listar copias de seguridad: {str(e)}")
            return []
    
    def cleanup_old_backups(self, max_backups: Optional[int] = None) -> int:
        """
        Elimina copias de seguridad antiguas.
        
        Args:
            max_backups: Número máximo de copias a mantener (usa config por defecto)
            
        Returns:
            Número de copias eliminadas
        """
        try:
            if max_backups is None:
                if self.storage_type == self.STORAGE_FILE:
                    max_backups = self.file_config['max_backups']
                elif self.storage_type == self.STORAGE_SQLITE:
                    max_backups = self.sqlite_config['max_backups']
                else:
                    max_backups = 5
            
            backups = self.list_backups()
            if len(backups) <= max_backups:
                return 0
            
            # Ordenar por fecha (más antiguo primero)
            backups.sort(key=lambda x: x.get('created_at', ''))
            
            # Eliminar las más antiguas
            to_delete = backups[:-max_backups]
            deleted = 0
            
            for backup in to_delete:
                path = backup.get('path', '')
                if path and os.path.exists(path):
                    os.remove(path)
                    deleted += 1
                    logger.debug(f"Copia de seguridad eliminada: {path}")
            
            return deleted
        except Exception as e:
            logger.error(f"Error al limpiar copias de seguridad antiguas: {str(e)}")
            return 0
    
    def stop(self) -> None:
        """
        Detiene el gestor de persistencia y guarda el estado actual.
        """
        if hasattr(self, '_autosave_thread') and self._autosave_thread.is_alive():
            self._stop_autosave.set()
            self._autosave_thread.join(timeout=5.0)
            logger.debug("Hilo de auto-guardado detenido")
        
        logger.info("PersistenceManager detenido")
    
    # Métodos privados para implementaciones específicas
    
    def _save_tasks_to_file(self, tasks: List[Task]) -> bool:
        """Guarda tareas en archivo JSON"""
        tasks_dict = {task.id: task.to_dict() for task in tasks}
        
        with open(self.file_config['tasks_file'], 'w') as f:
            json.dump(tasks_dict, f, indent=2)
        
        logger.debug(f"Guardadas {len(tasks)} tareas en archivo")
        return True
    
    def _load_tasks_from_file(self) -> List[Task]:
        """Carga tareas desde archivo JSON"""
        if not os.path.exists(self.file_config['tasks_file']):
            return []
        
        with open(self.file_config['tasks_file'], 'r') as f:
            tasks_dict = json.load(f)
        
        tasks = []
        for task_id, task_data in tasks_dict.items():
            try:
                task = Task.from_dict(task_data)
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error al cargar tarea {task_id}: {str(e)}")
        
        logger.debug(f"Cargadas {len(tasks)} tareas desde archivo")
        return tasks
    
    def _save_history_to_file(self, tasks: List[Task]) -> bool:
        """Guarda historial de tareas en archivo JSON"""
        # Cargar historial existente
        history = {}
        if os.path.exists(self.file_config['history_file']):
            with open(self.file_config['history_file'], 'r') as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = {}
        
        # Añadir nuevas tareas al historial
        for task in tasks:
            history[task.id] = task.to_dict()
        
        # Guardar historial actualizado
        with open(self.file_config['history_file'], 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.debug(f"Guardadas {len(tasks)} tareas en historial")
        return True
    
    def _load_history_from_file(self, limit: int, 
                               start_date: Optional[datetime],
                               end_date: Optional[datetime]) -> List[Task]:
        """Carga historial de tareas desde archivo JSON"""
        if not os.path.exists(self.file_config['history_file']):
            return []
        
        with open(self.file_config['history_file'], 'r') as f:
            history_dict = json.load(f)
        
        # Convertir a lista de tareas
        tasks = []
        for task_id, task_data in history_dict.items():
            try:
                # Filtrar por fecha si es necesario
                if start_date or end_date:
                    task_date = datetime.fromisoformat(task_data.get('completed_at', ''))
                    
                    if start_date and task_date < start_date:
                        continue
                    
                    if end_date and task_date > end_date:
                        continue
                
                task = Task.from_dict(task_data)
                tasks.append(task)
                
                # Limitar número de tareas
                if len(tasks) >= limit:
                    break
            except Exception as e:
                logger.error(f"Error al cargar tarea del historial {task_id}: {str(e)}")
        
        logger.debug(f"Cargadas {len(tasks)} tareas del historial")
        return tasks
    
    def _save_config_to_file(self, config: Dict[str, Any]) -> bool:
        """Guarda configuración en archivo JSON"""
        with open(self.file_config['config_file'], 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.debug("Configuración guardada en archivo")
        return True
    
    def _load_config_from_file(self) -> Dict[str, Any]:
        """Carga configuración desde archivo JSON"""
        if not os.path.exists(self.file_config['config_file']):
            return {}
        
        with open(self.file_config['config_file'], 'r') as f:
            config = json.load(f)
        
        logger.debug("Configuración cargada desde archivo")
        return config
    
    def _create_file_backup(self) -> str:
        """Crea una copia de seguridad de los archivos"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(self.file_config['backup_dir'], f'backup_{timestamp}')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copiar archivos
        for key, file_path in [
            ('tasks', self.file_config['tasks_file']),
            ('history', self.file_config['history_file']),
            ('config', self.file_config['config_file'])
        ]:
            if os.path.exists(file_path):
                backup_file = os.path.join(backup_dir, os.path.basename(file_path))
                shutil.copy2(file_path, backup_file)
        
        # Crear archivo de metadatos
        metadata = {
            'created_at': datetime.now().isoformat(),
            'storage_type': self.storage_type,
            'files': [os.path.basename(f) for f in [
                self.file_config['tasks_file'],
                self.file_config['history_file'],
                self.file_config['config_file']
            ]]
        }
        
        with open(os.path.join(backup_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Copia de seguridad creada en {backup_dir}")
        return backup_dir
    
    def _restore_file_backup(self, backup_path: str) -> bool:
        """Restaura una copia de seguridad de archivos"""
        if not os.path.exists(backup_path) or not os.path.isdir(backup_path):
            logger.error(f"Ruta de copia de seguridad no válida: {backup_path}")
            return False
        
        # Verificar metadatos
        metadata_file = os.path.join(backup_path, 'metadata.json')
        if not os.path.exists(metadata_file):
            logger.error(f"Archivo de metadatos no encontrado en {backup_path}")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('storage_type') != self.storage_type:
            logger.error(f"Tipo de almacenamiento incompatible: {metadata.get('storage_type')}")
            return False
        
        # Restaurar archivos
        for key, file_path in [
            ('tasks', self.file_config['tasks_file']),
            ('history', self.file_config['history_file']),
            ('config', self.file_config['config_file'])
        ]:
            backup_file = os.path.join(backup_path, os.path.basename(file_path))
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, file_path)
        
        logger.info(f"Copia de seguridad restaurada desde {backup_path}")
        return True
    
    def _list_file_backups(self) -> List[Dict[str, Any]]:
        """Lista las copias de seguridad de archivos disponibles"""
        backups = []
        
        if not os.path.exists(self.file_config['backup_dir']):
            return backups
        
        for item in os.listdir(self.file_config['backup_dir']):
            backup_dir = os.path.join(self.file_config['backup_dir'], item)
            if not os.path.isdir(backup_dir):
                continue
            
            metadata_file = os.path.join(backup_dir, 'metadata.json')
            if not os.path.exists(metadata_file):
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                backups.append({
                    'id': item,
                    'path': backup_dir,
                    'created_at': metadata.get('created_at', ''),
                    'storage_type': metadata.get('storage_type', ''),
                    'files': metadata.get('files', [])
                })
            except Exception as e:
                logger.error(f"Error al leer metadatos de copia de seguridad {item}: {str(e)}")
        
        # Ordenar por fecha (más reciente primero)
        backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return backups
    
    # Implementaciones para SQLite
    
    def _save_tasks_to_sqlite(self, tasks: List[Task]) -> bool:
        """Guarda tareas en base de datos SQLite"""
        conn = sqlite3.connect(self.sqlite_config['db_file'])
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        try:
            # Usar transacción para garantizar atomicidad
            cursor.execute('BEGIN TRANSACTION')
            
            for task in tasks:
                task_data = json.dumps(task.to_dict())
                
                # Insertar o actualizar tarea
                cursor.execute('''
                    INSERT OR REPLACE INTO tasks (id, task_data, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                ''', (task.id, task_data, task.created_at.isoformat(), now))
            
            cursor.execute('COMMIT')
            conn.close()
            
            logger.debug(f"Guardadas {len(tasks)} tareas en SQLite")
            return True
            
        except Exception as e:
            cursor.execute('ROLLBACK')
            conn.close()
            logger.error(f"Error al guardar tareas en SQLite: {str(e)}")
            return False
    
    def _load_tasks_from_sqlite(self) -> List[Task]:
        """Carga tareas desde base de datos SQLite"""
        conn = sqlite3.connect(self.sqlite_config['db_file'])
        cursor = conn.cursor()
        
        cursor.execute('SELECT task_data FROM tasks')
        rows = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in rows:
            try:
                task_data = json.loads(row[0])
                task = Task.from_dict(task_data)
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error al cargar tarea desde SQLite: {str(e)}")
        
        logger.debug(f"Cargadas {len(tasks)} tareas desde SQLite")
        return tasks
    
    def _save_history_to_sqlite(self, tasks: List[Task]) -> bool:
        """Guarda historial de tareas en base de datos SQLite"""
        conn = sqlite3.connect(self.sqlite_config['db_file'])
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        try:
            # Usar transacción para garantizar atomicidad
            cursor.execute('BEGIN TRANSACTION')
            
            for task in tasks:
                task_data = json.dumps(task.to_dict())
                completed_at = task.completed_at.isoformat() if task.completed_at else now
                
                # Insertar o actualizar tarea en historial
                cursor.execute('''
                    INSERT OR REPLACE INTO task_history (id, task_data, created_at, completed_at)
                    VALUES (?, ?, ?, ?)
                ''', (task.id, task_data, task.created_at.isoformat(), completed_at))
            
            cursor.execute('COMMIT')
            conn.close()
            
            logger.debug(f"Guardadas {len(tasks)} tareas en historial SQLite")
            return True
            
        except Exception as e:
            cursor.execute('ROLLBACK')
            conn.close()
            logger.error(f"Error al guardar historial en SQLite: {str(e)}")
            return False
    
        def _load_history_from_sqlite(self, limit: int, 
                                 start_date: Optional[datetime],
                                 end_date: Optional[datetime]) -> List[Task]:
        """Carga historial de tareas desde base de datos SQLite"""
        conn = sqlite3.connect(self.sqlite_config['db_file'])
        cursor = conn.cursor()
        
        # Construir consulta SQL con filtros opcionales
        query = 'SELECT task_data FROM task_history'
        params = []
        
        if start_date or end_date:
            query += ' WHERE '
            conditions = []
            
            if start_date:
                conditions.append('completed_at >= ?')
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append('completed_at <= ?')
                params.append(end_date.isoformat())
            
            query += ' AND '.join(conditions)
        
        # Añadir límite
        query += ' ORDER BY completed_at DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        tasks = []
        for row in rows:
            try:
                task_data = json.loads(row[0])
                task = Task.from_dict(task_data)
                tasks.append(task)
            except Exception as e:
                logger.error(f"Error al cargar tarea del historial desde SQLite: {str(e)}")
        
        logger.debug(f"Cargadas {len(tasks)} tareas del historial desde SQLite")
        return tasks
    
    def _save_config_to_sqlite(self, config: Dict[str, Any]) -> bool:
        """Guarda configuración en base de datos SQLite"""
        conn = sqlite3.connect(self.sqlite_config['db_file'])
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        try:
            # Usar transacción para garantizar atomicidad
            cursor.execute('BEGIN TRANSACTION')
            
            # Eliminar configuración anterior
            cursor.execute('DELETE FROM scheduler_config')
            
            # Insertar nueva configuración
            for key, value in config.items():
                # Convertir valor a JSON si es necesario
                if not isinstance(value, str):
                    value = json.dumps(value)
                
                cursor.execute('''
                    INSERT INTO scheduler_config (key, value, updated_at)
                    VALUES (?, ?, ?)
                ''', (key, value, now))
            
            cursor.execute('COMMIT')
            conn.close()
            
            logger.debug("Configuración guardada en SQLite")
            return True
            
        except Exception as e:
            cursor.execute('ROLLBACK')
            conn.close()
            logger.error(f"Error al guardar configuración en SQLite: {str(e)}")
            return False
    
    def _load_config_from_sqlite(self) -> Dict[str, Any]:
        """Carga configuración desde base de datos SQLite"""
        conn = sqlite3.connect(self.sqlite_config['db_file'])
        cursor = conn.cursor()
        
        cursor.execute('SELECT key, value FROM scheduler_config')
        rows = cursor.fetchall()
        conn.close()
        
        config = {}
        for key, value in rows:
            try:
                # Intentar cargar como JSON
                try:
                    config[key] = json.loads(value)
                except json.JSONDecodeError:
                    # Si no es JSON, usar valor como string
                    config[key] = value
            except Exception as e:
                logger.error(f"Error al cargar configuración desde SQLite para clave {key}: {str(e)}")
        
        logger.debug(f"Configuración cargada desde SQLite: {len(config)} claves")
        return config
    
    def _create_sqlite_backup(self) -> str:
        """Crea una copia de seguridad de la base de datos SQLite"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = os.path.join(self.sqlite_config['backup_dir'], f'backup_{timestamp}')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Ruta del archivo de backup
        backup_file = os.path.join(backup_dir, os.path.basename(self.sqlite_config['db_file']))
        
        # Crear copia de seguridad
        conn = sqlite3.connect(self.sqlite_config['db_file'])
        backup_conn = sqlite3.connect(backup_file)
        
        conn.backup(backup_conn)
        
        conn.close()
        backup_conn.close()
        
        # Crear archivo de metadatos
        metadata = {
            'created_at': datetime.now().isoformat(),
            'storage_type': self.storage_type,
            'db_file': os.path.basename(self.sqlite_config['db_file'])
        }
        
        with open(os.path.join(backup_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Copia de seguridad SQLite creada en {backup_dir}")
        return backup_dir
    
    def _restore_sqlite_backup(self, backup_path: str) -> bool:
        """Restaura una copia de seguridad de la base de datos SQLite"""
        if not os.path.exists(backup_path) or not os.path.isdir(backup_path):
            logger.error(f"Ruta de copia de seguridad no válida: {backup_path}")
            return False
        
        # Verificar metadatos
        metadata_file = os.path.join(backup_path, 'metadata.json')
        if not os.path.exists(metadata_file):
            logger.error(f"Archivo de metadatos no encontrado en {backup_path}")
            return False
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('storage_type') != self.storage_type:
            logger.error(f"Tipo de almacenamiento incompatible: {metadata.get('storage_type')}")
            return False
        
        # Ruta del archivo de backup
        backup_file = os.path.join(backup_path, metadata.get('db_file', ''))
        if not os.path.exists(backup_file):
            logger.error(f"Archivo de base de datos no encontrado en {backup_path}")
            return False
        
        # Cerrar todas las conexiones existentes
        # (Esto es una simplificación, en un sistema real necesitaríamos un mecanismo más robusto)
        
        # Restaurar base de datos
        try:
            # Crear copia de seguridad temporal antes de restaurar
            temp_backup = self._create_sqlite_backup()
            
            # Restaurar desde backup
            backup_conn = sqlite3.connect(backup_file)
            conn = sqlite3.connect(self.sqlite_config['db_file'])
            
            backup_conn.backup(conn)
            
            backup_conn.close()
            conn.close()
            
            logger.info(f"Copia de seguridad SQLite restaurada desde {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error al restaurar copia de seguridad SQLite: {str(e)}")
            return False
    
    def _list_sqlite_backups(self) -> List[Dict[str, Any]]:
        """Lista las copias de seguridad de SQLite disponibles"""
        backups = []
        
        if not os.path.exists(self.sqlite_config['backup_dir']):
            return backups
        
        for item in os.listdir(self.sqlite_config['backup_dir']):
            backup_dir = os.path.join(self.sqlite_config['backup_dir'], item)
            if not os.path.isdir(backup_dir):
                continue
            
            metadata_file = os.path.join(backup_dir, 'metadata.json')
            if not os.path.exists(metadata_file):
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                db_file = os.path.join(backup_dir, metadata.get('db_file', ''))
                
                backups.append({
                    'id': item,
                    'path': backup_dir,
                    'db_file': db_file,
                    'created_at': metadata.get('created_at', ''),
                    'storage_type': metadata.get('storage_type', '')
                })
            except Exception as e:
                logger.error(f"Error al leer metadatos de copia de seguridad {item}: {str(e)}")
        
        # Ordenar por fecha (más reciente primero)
        backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return backups
    
    # Implementaciones para MongoDB
    
    def _save_tasks_to_mongodb(self, tasks: List[Task]) -> bool:
        """Guarda tareas en MongoDB"""
        try:
            import pymongo
            client = pymongo.MongoClient(self.mongodb_config['connection_string'])
            db = client[self.mongodb_config['database']]
            collection = db[self.mongodb_config['tasks_collection']]
            
            # Convertir tareas a diccionarios
            task_dicts = []
            for task in tasks:
                task_dict = task.to_dict()
                task_dict['_id'] = task.id  # Usar ID de tarea como _id en MongoDB
                task_dicts.append(task_dict)
            
            # Usar operación bulk para eficiencia
            if task_dicts:
                # Eliminar tareas existentes con los mismos IDs
                ids = [task.id for task in tasks]
                collection.delete_many({'_id': {'$in': ids}})
                
                # Insertar nuevas tareas
                collection.insert_many(task_dicts)
            
            logger.debug(f"Guardadas {len(tasks)} tareas en MongoDB")
            return True
            
        except ImportError:
            logger.error("No se pudo importar pymongo. Instale con: pip install pymongo")
            return False
        except Exception as e:
            logger.error(f"Error al guardar tareas en MongoDB: {str(e)}")
            return False
    
    def _load_tasks_from_mongodb(self) -> List[Task]:
        """Carga tareas desde MongoDB"""
        try:
            import pymongo
            client = pymongo.MongoClient(self.mongodb_config['connection_string'])
            db = client[self.mongodb_config['database']]
            collection = db[self.mongodb_config['tasks_collection']]
            
            # Obtener todas las tareas
            task_dicts = list(collection.find({}))
            
            tasks = []
            for task_dict in task_dicts:
                try:
                    # Eliminar _id de MongoDB para evitar conflictos
                    if '_id' in task_dict:
                        del task_dict['_id']
                    
                    task = Task.from_dict(task_dict)
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Error al cargar tarea desde MongoDB: {str(e)}")
            
            logger.debug(f"Cargadas {len(tasks)} tareas desde MongoDB")
            return tasks
            
        except ImportError:
            logger.error("No se pudo importar pymongo. Instale con: pip install pymongo")
            return []
        except Exception as e:
            logger.error(f"Error al cargar tareas desde MongoDB: {str(e)}")
            return []
    
    def _save_history_to_mongodb(self, tasks: List[Task]) -> bool:
        """Guarda historial de tareas en MongoDB"""
        try:
            import pymongo
            client = pymongo.MongoClient(self.mongodb_config['connection_string'])
            db = client[self.mongodb_config['database']]
            collection = db[self.mongodb_config['history_collection']]
            
            # Convertir tareas a diccionarios
            task_dicts = []
            for task in tasks:
                task_dict = task.to_dict()
                task_dict['_id'] = task.id  # Usar ID de tarea como _id en MongoDB
                task_dicts.append(task_dict)
            
            # Usar operación bulk para eficiencia
            if task_dicts:
                # Usar upsert para insertar o actualizar
                operations = [
                    pymongo.UpdateOne(
                        {'_id': task_dict['_id']},
                        {'$set': task_dict},
                        upsert=True
                    ) for task_dict in task_dicts
                ]
                
                collection.bulk_write(operations)
            
            logger.debug(f"Guardadas {len(tasks)} tareas en historial MongoDB")
            return True
            
        except ImportError:
            logger.error("No se pudo importar pymongo. Instale con: pip install pymongo")
            return False
        except Exception as e:
            logger.error(f"Error al guardar historial en MongoDB: {str(e)}")
            return False
    
    def _load_history_from_mongodb(self, limit: int, 
                                  start_date: Optional[datetime],
                                  end_date: Optional[datetime]) -> List[Task]:
        """Carga historial de tareas desde MongoDB"""
        try:
            import pymongo
            client = pymongo.MongoClient(self.mongodb_config['connection_string'])
            db = client[self.mongodb_config['database']]
            collection = db[self.mongodb_config['history_collection']]
            
            # Construir filtro
            filter_query = {}
            
            if start_date or end_date:
                filter_query['completed_at'] = {}
                
                if start_date:
                    filter_query['completed_at']['$gte'] = start_date.isoformat()
                
                if end_date:
                    filter_query['completed_at']['$lte'] = end_date.isoformat()
            
            # Obtener tareas con límite y ordenadas por fecha
            task_dicts = list(collection.find(
                filter_query,
                sort=[('completed_at', pymongo.DESCENDING)],
                limit=limit
            ))
            
            tasks = []
            for task_dict in task_dicts:
                try:
                    # Eliminar _id de MongoDB para evitar conflictos
                    if '_id' in task_dict:
                        del task_dict['_id']
                    
                    task = Task.from_dict(task_dict)
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Error al cargar tarea del historial desde MongoDB: {str(e)}")
            
            logger.debug(f"Cargadas {len(tasks)} tareas del historial desde MongoDB")
            return tasks
            
        except ImportError:
            logger.error("No se pudo importar pymongo. Instale con: pip install pymongo")
            return []
        except Exception as e:
            logger.error(f"Error al cargar historial desde MongoDB: {str(e)}")
            return []
    
    def _save_config_to_mongodb(self, config: Dict[str, Any]) -> bool:
        """Guarda configuración en MongoDB"""
        try:
            import pymongo
            client = pymongo.MongoClient(self.mongodb_config['connection_string'])
            db = client[self.mongodb_config['database']]
            collection = db[self.mongodb_config['config_collection']]
            
            # Eliminar configuración anterior
            collection.delete_many({})
            
            # Insertar nueva configuración
            config_items = []
            for key, value in config.items():
                config_items.append({
                    '_id': key,
                    'value': value,
                    'updated_at': datetime.now().isoformat()
                })
            
            if config_items:
                collection.insert_many(config_items)
            
            logger.debug("Configuración guardada en MongoDB")
            return True
            
        except ImportError:
            logger.error("No se pudo importar pymongo. Instale con: pip install pymongo")
            return False
        except Exception as e:
            logger.error(f"Error al guardar configuración en MongoDB: {str(e)}")
            return False
    
    def _load_config_from_mongodb(self) -> Dict[str, Any]:
        """Carga configuración desde MongoDB"""
        try:
            import pymongo
            client = pymongo.MongoClient(self.mongodb_config['connection_string'])
            db = client[self.mongodb_config['database']]
            collection = db[self.mongodb_config['config_collection']]
            
            # Obtener toda la configuración
            config_items = list(collection.find({}))
            
            config = {}
            for item in config_items:
                key = item.get('_id')
                value = item.get('value')
                
                if key:
                    config[key] = value
            
            logger.debug(f"Configuración cargada desde MongoDB: {len(config)} claves")
            return config
            
        except ImportError:
            logger.error("No se pudo importar pymongo. Instale con: pip install pymongo")
            return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración desde MongoDB: {str(e)}")
            return {}
    
    def _create_mongodb_backup(self) -> str:
        """Crea una copia de seguridad de MongoDB"""
        try:
            import pymongo
            from pymongo.errors import OperationFailure
            import subprocess
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = os.path.join(self.sqlite_config['backup_dir'], f'backup_{timestamp}')
            os.makedirs(backup_dir, exist_ok=True)
            
            # Extraer información de conexión
            connection_parts = self.mongodb_config['connection_string'].split('/')
            host_port = connection_parts[2].split(':')
            host = host_port[0]
            port = host_port[1] if len(host_port) > 1 else '27017'
            
            # Nombre de la base de datos
            db_name = self.mongodb_config['database']
            
            # Ruta para el archivo de backup
            backup_file = os.path.join(backup_dir, f"{db_name}.dump")
            
            # Ejecutar mongodump
            try:
                # Intentar usar mongodump si está disponible
                cmd = f"mongodump --host {host} --port {port} --db {db_name} --out {backup_dir}"
                subprocess.run(cmd, shell=True, check=True)
                success = True
            except (subprocess.SubprocessError, FileNotFoundError):
                # Si mongodump falla, usar PyMongo para exportar colecciones
                client = pymongo.MongoClient(self.mongodb_config['connection_string'])
                db = client[db_name]
                
                # Exportar cada colección como JSON
                for collection_name in [
                    self.mongodb_config['tasks_collection'],
                    self.mongodb_config['history_collection'],
                    self.mongodb_config['config_collection']
                ]:
                    collection = db[collection_name]
                    docs = list(collection.find({}))
                    
                    # Convertir ObjectId a string para serialización JSON
                    for doc in docs:
                        if '_id' in doc and not isinstance(doc['_id'], str):
                            doc['_id'] = str(doc['_id'])
                    
                    # Guardar como JSON
                    collection_file = os.path.join(backup_dir, f"{collection_name}.json")
                    with open(collection_file, 'w') as f:
                        json.dump(docs, f, indent=2)
                
                success = True
            
            # Crear archivo de metadatos
            metadata = {
                'created_at': datetime.now().isoformat(),
                'storage_type': self.storage_type,
                'database': db_name,
                'collections': [
                    self.mongodb_config['tasks_collection'],
                    self.mongodb_config['history_collection'],
                    self.mongodb_config['config_collection']
                ]
            }
            
            with open(os.path.join(backup_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if success:
                logger.info(f"Copia de seguridad MongoDB creada en {backup_dir}")
                return backup_dir
            else:
                logger.error("No se pudo crear copia de seguridad de MongoDB")
                return ""
                
        except ImportError:
            logger.error("No se pudo importar pymongo. Instale con: pip install pymongo")
            return ""
        except Exception as e:
            logger.error(f"Error al crear copia de seguridad MongoDB: {str(e)}")
            return ""
    
    def _restore_mongodb_backup(self, backup_path: str) -> bool:
        """Restaura una copia de seguridad de MongoDB"""
        try:
            import pymongo
            from pymongo.errors import OperationFailure
            import subprocess
            
            if not os.path.exists(backup_path) or not os.path.isdir(backup_path):
                logger.error(f"Ruta de copia de seguridad no válida: {backup_path}")
                return False
            
            # Verificar metadatos
            metadata_file = os.path.join(backup_path, 'metadata.json')
            if not os.path.exists(metadata_file):
                logger.error(f"Archivo de metadatos no encontrado en {backup_path}")
                return False
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if metadata.get('storage_type') != self.storage_type:
                logger.error(f"Tipo de almacenamiento incompatible: {metadata.get('storage_type')}")
                return False
            
            # Extraer información de conexión
            connection_parts = self.mongodb_config['connection_string'].split('/')
            host_port = connection_parts[2].split(':')
            host = host_port[0]
            port = host_port[1] if len(host_port) > 1 else '27017'
            
            # Nombre de la base de datos
            db_name = self.mongodb_config['database']
            
            # Intentar restaurar con mongorestore
            try:
                # Verificar si existe directorio de dump
                dump_dir = os.path.join(backup_path, db_name)
                if os.path.exists(dump_dir) and os.path.isdir(dump_dir):
                    cmd = f"mongorestore --host {host} --port {port} --db {db_name} {dump_dir}"
                    subprocess.run(cmd, shell=True, check=True)
                    success = True
                else:
                    # Restaurar desde archivos JSON
                    client = pymongo.MongoClient(self.mongodb_config['connection_string'])
                    db = client[db_name]
                    
                    # Restaurar cada colección desde JSON
                    for collection_name in metadata.get('collections', []):
                        collection_file = os.path.join(backup_path, f"{collection_name}.json")
                        
                        if os.path.exists(collection_file):
                            with open(collection_file, 'r') as f:
                                docs = json.load(f)
                            
                            # Limpiar colección existente
                            db[collection_name].delete_many({})
                            
                            # Insertar documentos
                            if docs:
                                db[collection_name].insert_many(docs)
                    
                    success = True
            except (subprocess.SubprocessError, FileNotFoundError, Exception) as e:
                logger.error(f"Error al restaurar MongoDB: {str(e)}")
                success = False
            
            if success:
                logger.info(f"Copia de seguridad MongoDB restaurada desde {backup_path}")
                return True
            else:
                logger.error("No se pudo restaurar copia de seguridad de MongoDB")
                return False
                
        except ImportError:
            logger.error("No se pudo importar pymongo. Instale con: pip install pymongo")
            return False
        except Exception as e:
            logger.error(f"Error al restaurar copia de seguridad MongoDB: {str(e)}")
            return False
    
    def _list_mongodb_backups(self) -> List[Dict[str, Any]]:
        """Lista las copias de seguridad de MongoDB disponibles"""
        backups = []
        
        if not os.path.exists(self.sqlite_config['backup_dir']):
            return backups
        
        for item in os.listdir(self.sqlite_config['backup_dir']):
            backup_dir = os.path.join(self.sqlite_config['backup_dir'], item)
            if not os.path.isdir(backup_dir):
                continue
            
            metadata_file = os.path.join(backup_dir, 'metadata.json')
            if not os.path.exists(metadata_file):
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if metadata.get('storage_type') == self.STORAGE_MONGODB:
                    backups.append({
                        'id': item,
                        'path': backup_dir,
                        'created_at': metadata.get('created_at', ''),
                        'storage_type': metadata.get('storage_type', ''),
                        'database': metadata.get('database', ''),
                        'collections': metadata.get('collections', [])
                    })
            except Exception as e:
                logger.error(f"Error al leer metadatos de copia de seguridad {item}: {str(e)}")
        
        # Ordenar por fecha (más reciente primero)
        backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return backups
    
    def _autosave_loop(self) -> None:
        """Bucle de auto-guardado que se ejecuta en un hilo separado"""
        logger.debug("Iniciando bucle de auto-guardado")
        
        while not self._stop_autosave.is_set():
            try:
                # Esperar el intervalo de auto-guardado o hasta que se solicite detener
                if self._stop_autosave.wait(timeout=self.autosave_interval):
                    break
                
                # Crear copia de seguridad
                self.create_backup()
                
                # Limpiar copias de seguridad antiguas
                self.cleanup_old_backups()
                
                logger.debug("Auto-guardado completado")
                
            except Exception as e:
                logger.error(f"Error en bucle de auto-guardado: {str(e)}")
                # Esperar un poco antes de reintentar en caso de error
                time.sleep(10)
        
        logger.debug("Bucle de auto-guardado detenido")