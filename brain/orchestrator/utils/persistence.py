"""
Sistema de persistencia para el Orchestrator
"""
import os
import json
import sqlite3
import pickle
import datetime
import logging
import threading
import time
from typing import Dict, Any, Optional, List

# Configuración
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

class PersistenceManager:
    """Gestor de persistencia para el Orchestrator"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.db_path = self.config.get("db_path", os.path.join(DATA_DIR, "orchestrator.db"))
        self.json_dir = self.config.get("json_dir", os.path.join(DATA_DIR, "json"))
        self.snapshot_dir = self.config.get("snapshot_dir", os.path.join(DATA_DIR, "snapshots"))
        
        # Crear directorios
        os.makedirs(self.json_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Inicializar base de datos
        self._init_database()
        
        # Configurar snapshot automático
        self.snapshot_interval = self.config.get("snapshot_interval", 3600)  # 1 hora por defecto
        self.snapshot_active = True
        self.snapshot_thread = threading.Thread(target=self._auto_snapshot, daemon=True)
        self.snapshot_thread.start()
    
    def _init_database(self):
        """Inicializa la base de datos SQLite"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tabla de canales
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS channels (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                ''')
                
                # Tabla de tareas
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    channel_id TEXT,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (channel_id) REFERENCES channels (id)
                )
                ''')
                
                # Tabla de actividades
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS activities (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    action TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metadata TEXT
                )
                ''')
                
                # Tabla de métricas
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    channel_id TEXT,
                    platform TEXT,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (channel_id) REFERENCES channels (id)
                )
                ''')
                
                # Tabla de configuración
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                ''')
                
                # Índices para mejorar rendimiento
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_channel_id ON tasks (channel_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks (type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_activities_timestamp ON activities (timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_activities_category ON activities (category)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_channel_id ON metrics (channel_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_platform ON metrics (platform)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics (timestamp)')
                
                conn.commit()
                logger.info("Base de datos inicializada correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar base de datos: {str(e)}")
            raise
    
    def save_channel(self, channel_id: str, channel_data: Dict[str, Any], status: str) -> bool:
        """Guarda un canal en la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.datetime.now().isoformat()
                
                cursor.execute(
                    "INSERT OR REPLACE INTO channels (id, data, status, updated_at) VALUES (?, ?, ?, ?)",
                    (channel_id, json.dumps(channel_data), status, now)
                )
                
                conn.commit()
                
                # También guardar como JSON para respaldo
                json_path = os.path.join(self.json_dir, f"channel_{channel_id}.json")
                with open(json_path, "w") as f:
                    json.dump({
                        "id": channel_id,
                        "data": channel_data,
                        "status": status,
                        "updated_at": now
                    }, f, indent=2)
                
                return True
        except Exception as e:
            logger.error(f"Error al guardar canal {channel_id}: {str(e)}")
            return False
    
    def load_channel(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Carga un canal desde la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT data, status FROM channels WHERE id = ?", (channel_id,))
                result = cursor.fetchone()
                
                if result:
                    data, status = result
                    channel_data = json.loads(data)
                    channel_data["status"] = status
                    return channel_data
                
                return None
        except Exception as e:
            logger.error(f"Error al cargar canal {channel_id}: {str(e)}")
            
            # Intentar cargar desde JSON como fallback
            try:
                json_path = os.path.join(self.json_dir, f"channel_{channel_id}.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        channel_json = json.load(f)
                        channel_data = channel_json["data"]
                        channel_data["status"] = channel_json["status"]
                        return channel_data
            except Exception as json_error:
                logger.error(f"Error al cargar canal {channel_id} desde JSON: {str(json_error)}")
            
            return None
    
    def load_all_channels(self) -> List[Dict[str, Any]]:
        """Carga todos los canales desde la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT id, data, status FROM channels")
                results = cursor.fetchall()
                
                channels = []
                for channel_id, data, status in results:
                    channel_data = json.loads(data)
                    channel_data["id"] = channel_id
                    channel_data["status"] = status
                    channels.append(channel_data)
                
                return channels
        except Exception as e:
            logger.error(f"Error al cargar todos los canales: {str(e)}")
            
            # Intentar cargar desde JSON como fallback
            try:
                channels = []
                for filename in os.listdir(self.json_dir):
                    if filename.startswith("channel_") and filename.endswith(".json"):
                        with open(os.path.join(self.json_dir, filename), "r") as f:
                            channel_json = json.load(f)
                            channel_data = channel_json["data"]
                            channel_data["id"] = channel_json["id"]
                            channel_data["status"] = channel_json["status"]
                            channels.append(channel_data)
                return channels
            except Exception as json_error:
                logger.error(f"Error al cargar canales desde JSON: {str(json_error)}")
            
            return []
    
    def save_task(self, task_id: str, task_type: str, channel_id: Optional[str], 
                 status: str, priority: int, task_data: Dict[str, Any]) -> bool:
        """Guarda una tarea en la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.datetime.now().isoformat()
                
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO tasks 
                    (id, type, channel_id, status, priority, data, created_at, updated_at) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id, 
                        task_type, 
                        channel_id, 
                        status, 
                        priority, 
                        json.dumps(task_data), 
                        task_data.get("created_at", now), 
                        now
                    )
                )
                
                conn.commit()
                
                # También guardar como JSON para respaldo
                json_path = os.path.join(self.json_dir, f"task_{task_id}.json")
                with open(json_path, "w") as f:
                    json.dump({
                        "id": task_id,
                        "type": task_type,
                        "channel_id": channel_id,
                        "status": status,
                        "priority": priority,
                        "data": task_data,
                        "created_at": task_data.get("created_at", now),
                        "updated_at": now
                    }, f, indent=2)
                
                return True
        except Exception as e:
            logger.error(f"Error al guardar tarea {task_id}: {str(e)}")
            return False
    
    def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Carga una tarea desde la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    SELECT type, channel_id, status, priority, data, created_at, updated_at 
                    FROM tasks WHERE id = ?
                    """, 
                    (task_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    task_type, channel_id, status, priority, data, created_at, updated_at = result
                    task_data = json.loads(data)
                    return {
                        "id": task_id,
                        "type": task_type,
                        "channel_id": channel_id,
                        "status": status,
                        "priority": priority,
                        "data": task_data,
                        "created_at": created_at,
                        "updated_at": updated_at
                    }
                
                return None
        except Exception as e:
            logger.error(f"Error al cargar tarea {task_id}: {str(e)}")
            
            # Intentar cargar desde JSON como fallback
            try:
                json_path = os.path.join(self.json_dir, f"task_{task_id}.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        return json.load(f)
            except Exception as json_error:
                logger.error(f"Error al cargar tarea {task_id} desde JSON: {str(json_error)}")
            
            return None
    
    def load_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Carga tareas por estado desde la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    SELECT id, type, channel_id, status, priority, data, created_at, updated_at 
                    FROM tasks WHERE status = ?
                    """, 
                    (status,)
                )
                results = cursor.fetchall()
                
                tasks = []
                for task_id, task_type, channel_id, status, priority, data, created_at, updated_at in results:
                    task_data = json.loads(data)
                    tasks.append({
                        "id": task_id,
                        "type": task_type,
                        "channel_id": channel_id,
                        "status": status,
                        "priority": priority,
                        "data": task_data,
                        "created_at": created_at,
                        "updated_at": updated_at
                    })
                
                return tasks
        except Exception as e:
            logger.error(f"Error al cargar tareas con estado {status}: {str(e)}")
            return []
    
    def load_tasks_by_channel(self, channel_id: str) -> List[Dict[str, Any]]:
        """Carga tareas por canal desde la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    SELECT id, type, status, priority, data, created_at, updated_at 
                    FROM tasks WHERE channel_id = ?
                    """, 
                    (channel_id,)
                )
                results = cursor.fetchall()
                
                tasks = []
                for task_id, task_type, status, priority, data, created_at, updated_at in results:
                    task_data = json.loads(data)
                    tasks.append({
                        "id": task_id,
                        "type": task_type,
                        "channel_id": channel_id,
                        "status": status,
                        "priority": priority,
                        "data": task_data,
                        "created_at": created_at,
                        "updated_at": updated_at
                    })
                
                return tasks
        except Exception as e:
            logger.error(f"Error al cargar tareas para canal {channel_id}: {str(e)}")
            return []
    
    def delete_task(self, task_id: str) -> bool:
        """Elimina una tarea de la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                conn.commit()
                
                # Eliminar también el archivo JSON
                json_path = os.path.join(self.json_dir, f"task_{task_id}.json")
                if os.path.exists(json_path):
                    os.remove(json_path)
                
                return True
        except Exception as e:
            logger.error(f"Error al eliminar tarea {task_id}: {str(e)}")
            return False
    
    def save_activity(self, activity_id: str, category: str, action: str, 
                     description: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Guarda una actividad en la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.datetime.now().isoformat()
                
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO activities 
                    (id, timestamp, category, action, description, metadata) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        activity_id, 
                        now, 
                        category, 
                        action, 
                        description, 
                        json.dumps(metadata) if metadata else None
                    )
                )
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error al guardar actividad {activity_id}: {str(e)}")
            return False
    
    def load_activities(self, category: Optional[str] = None, 
                       limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Carga actividades desde la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if category:
                    cursor.execute(
                        """
                        SELECT id, timestamp, category, action, description, metadata 
                        FROM activities 
                        WHERE category = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ? OFFSET ?
                        """, 
                        (category, limit, offset)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, timestamp, category, action, description, metadata 
                        FROM activities 
                        ORDER BY timestamp DESC 
                        LIMIT ? OFFSET ?
                        """, 
                        (limit, offset)
                    )
                
                results = cursor.fetchall()
                
                activities = []
                for activity_id, timestamp, category, action, description, metadata in results:
                    activities.append({
                        "id": activity_id,
                        "timestamp": timestamp,
                        "category": category,
                        "action": action,
                        "description": description,
                        "metadata": json.loads(metadata) if metadata else None
                    })
                
                return activities
        except Exception as e:
            logger.error(f"Error al cargar actividades: {str(e)}")
            return []
    
    def save_metric(self, metric_id: str, metric_type: str, value: float, 
                   channel_id: Optional[str] = None, platform: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Guarda una métrica en la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.datetime.now().isoformat()
                
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO metrics 
                    (id, timestamp, channel_id, platform, metric_type, value, metadata) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metric_id, 
                        now, 
                        channel_id, 
                        platform, 
                        metric_type, 
                        value, 
                        json.dumps(metadata) if metadata else None
                    )
                )
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error al guardar métrica {metric_id}: {str(e)}")
            return False
    
    def load_metrics(self, metric_type: Optional[str] = None, 
                    channel_id: Optional[str] = None, 
                    platform: Optional[str] = None,
                    start_time: Optional[str] = None,
                    end_time: Optional[str] = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Carga métricas desde la base de datos con filtros"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT id, timestamp, channel_id, platform, metric_type, value, metadata FROM metrics WHERE 1=1"
                params = []
                
                if metric_type:
                    query += " AND metric_type = ?"
                    params.append(metric_type)
                
                if channel_id:
                    query += " AND channel_id = ?"
                    params.append(channel_id)
                
                if platform:
                    query += " AND platform = ?"
                    params.append(platform)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                metrics = []
                for metric_id, timestamp, ch_id, plat, m_type, value, metadata in results:
                    metrics.append({
                        "id": metric_id,
                        "timestamp": timestamp,
                        "channel_id": ch_id,
                        "platform": plat,
                        "metric_type": m_type,
                        "value": value,
                        "metadata": json.loads(metadata) if metadata else None
                    })
                
                return metrics
        except Exception as e:
            logger.error(f"Error al cargar métricas: {str(e)}")
            return []
    
    def save_config(self, key: str, value: Any) -> bool:
        """Guarda un valor de configuración en la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.datetime.now().isoformat()
                
                # Convertir valor a JSON si es necesario
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value)
                else:
                    value_str = str(value)
                
                cursor.execute(
                    "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, value_str, now)
                )
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error al guardar configuración {key}: {str(e)}")
            return False
    
    def load_config(self, key: str, default: Any = None) -> Any:
        """Carga un valor de configuración desde la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
                result = cursor.fetchone()
                
                if result:
                    value_str = result[0]
                    
                    # Intentar convertir a JSON si es posible
                    try:
                        return json.loads(value_str)
                    except json.JSONDecodeError:
                        return value_str
                
                return default
        except Exception as e:
            logger.error(f"Error al cargar configuración {key}: {str(e)}")
            return default
    
    def load_all_config(self) -> Dict[str, Any]:
        """Carga toda la configuración desde la base de datos"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT key, value FROM config")
                results = cursor.fetchall()
                
                config = {}
                for key, value_str in results:
                    # Intentar convertir a JSON si es posible
                    try:
                        config[key] = json.loads(value_str)
                    except json.JSONDecodeError:
                        config[key] = value_str
                
                return config
        except Exception as e:
            logger.error(f"Error al cargar toda la configuración: {str(e)}")
            return {}
    
    def create_snapshot(self) -> str:
        """Crea un snapshot de la base de datos"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            snapshot_file = os.path.join(self.snapshot_dir, f"orchestrator_db_{timestamp}.db")
            
            # Cerrar todas las conexiones antes de copiar
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA wal_checkpoint(FULL)")
            
            # Copiar archivo de base de datos
            shutil.copy2(self.db_path, snapshot_file)
            
            # Crear snapshot de archivos JSON
            json_snapshot_dir = os.path.join(self.snapshot_dir, f"json_{timestamp}")
            os.makedirs(json_snapshot_dir, exist_ok=True)
            
            for filename in os.listdir(self.json_dir):
                if filename.endswith(".json"):
                    shutil.copy2(
                        os.path.join(self.json_dir, filename),
                        os.path.join(json_snapshot_dir, filename)
                    )
            
            logger.info(f"Snapshot creado: {snapshot_file}")
            return snapshot_file
        except Exception as e:
            logger.error(f"Error al crear snapshot: {str(e)}")
            return ""
    
    def restore_snapshot(self, snapshot_path: str) -> bool:
        """Restaura un snapshot de la base de datos"""
        try:
            if not os.path.exists(snapshot_path):
                logger.error(f"Snapshot no encontrado: {snapshot_path}")
                return False
            
            # Cerrar todas las conexiones antes de restaurar
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA wal_checkpoint(FULL)")
            
            # Hacer backup del actual antes de restaurar
            backup_path = f"{self.db_path}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            shutil.copy2(self.db_path, backup_path)
            
            # Restaurar base de datos
            shutil.copy2(snapshot_path, self.db_path)
            
            # Restaurar archivos JSON si existe el directorio
            snapshot_basename = os.path.basename(snapshot_path)
            timestamp = snapshot_basename.replace("orchestrator_db_", "").replace(".db", "")
            json_snapshot_dir = os.path.join(self.snapshot_dir, f"json_{timestamp}")
            
            if os.path.exists(json_snapshot_dir):
                # Backup de JSONs actuales
                json_backup_dir = os.path.join(self.snapshot_dir, f"json_backup_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
                os.makedirs(json_backup_dir, exist_ok=True)
                
                for filename in os.listdir(self.json_dir):
                    if filename.endswith(".json"):
                        shutil.copy2(
                            os.path.join(self.json_dir, filename),
                            os.path.join(json_backup_dir, filename)
                        )
                
                # Restaurar JSONs
                for filename in os.listdir(json_snapshot_dir):
                    if filename.endswith(".json"):
                        shutil.copy2(
                            os.path.join(json_snapshot_dir, filename),
                            os.path.join(self.json_dir, filename)
                        )
            
            logger.info(f"Snapshot restaurado: {snapshot_path}")
            return True
        except Exception as e:
            logger.error(f"Error al restaurar snapshot: {str(e)}")
            return False
    
    def _auto_snapshot(self):
        """Realiza snapshots automáticos periódicamente"""
        while self.snapshot_active:
            try:
                self.create_snapshot()
                logger.info(f"Snapshot automático creado")
            except Exception as e:
                logger.error(f"Error en snapshot automático: {str(e)}")
            
            # Esperar el intervalo configurado
            time.sleep(self.snapshot_interval)
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Lista todos los snapshots disponibles"""
        try:
            snapshots = []
            for filename in os.listdir(self.snapshot_dir):
                if filename.startswith("orchestrator_db_") and filename.endswith(".db"):
                    file_path = os.path.join(self.snapshot_dir, filename)
                    timestamp = filename.replace("orchestrator_db_", "").replace(".db", "")
                    
                    # Convertir timestamp a datetime
                    dt = datetime.datetime.strptime(timestamp, "%Y%m%d%H%M%S")
                    
                    snapshots.append({
                        "path": file_path,
                        "timestamp": dt.isoformat(),
                        "size": os.path.getsize(file_path)
                    })
            
            # Ordenar por timestamp (más reciente primero)
            return sorted(snapshots, key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            logger.error(f"Error al listar snapshots: {str(e)}")
            return []
    
    def cleanup_old_snapshots(self, max_age_days: int = 30, min_keep: int = 5) -> int:
        """Limpia snapshots antiguos"""
        try:
            snapshots = self.list_snapshots()
            
            # Siempre mantener al menos min_keep snapshots
            if len(snapshots) <= min_keep:
                return 0
            
            now = datetime.datetime.now()
            max_age = datetime.timedelta(days=max_age_days)
            deleted_count = 0
            
            # Ordenar por timestamp (más antiguo primero)
            snapshots_sorted = sorted(snapshots, key=lambda x: x["timestamp"])
            
            # Eliminar snapshots antiguos, pero mantener al menos min_keep
            for i, snapshot in enumerate(snapshots_sorted):
                if i >= len(snapshots_sorted) - min_keep:
                    break
                
                snapshot_time = datetime.datetime.fromisoformat(snapshot["timestamp"])
                age = now - snapshot_time
                
                if age > max_age:
                    # Eliminar snapshot
                    os.remove(snapshot["path"])
                    
                    # Eliminar directorio JSON asociado si existe
                    timestamp = os.path.basename(snapshot["path"]).replace("orchestrator_db_", "").replace(".db", "")
                    json_dir = os.path.join(self.snapshot_dir, f"json_{timestamp}")
                    if os.path.exists(json_dir) and os.path.isdir(json_dir):
                        shutil.rmtree(json_dir)
                    
                    deleted_count += 1
            
            return deleted_count
        except Exception as e:
            logger.error(f"Error al limpiar snapshots antiguos: {str(e)}")
            return 0
    
    def stop(self):
        """Detiene el hilo de snapshots automáticos"""
        self.snapshot_active = False
        if self.snapshot_thread.is_alive():
            self.snapshot_thread.join(timeout=5.0)
        
        # Crear un último snapshot antes de cerrar
        try:
            self.create_snapshot()
        except Exception as e:
            logger.error(f"Error al crear snapshot final: {str(e)}")

# Instancia global
persistence_manager = PersistenceManager()