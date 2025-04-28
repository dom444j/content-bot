"""
Notifier - Sistema de notificaciones y alertas

Este módulo gestiona las notificaciones del sistema:
- Alertas de shadowbans y problemas de distribución
- Notificaciones de saturación de nichos
- Alertas de rendimiento (caídas de CTR, oportunidades)
- Notificaciones de tareas completadas
- Soporte para niveles de alerta (info, warning, critical)
- Escalado de alertas, supresión de duplicados y reintentos
- Personalización por usuario, agrupación inteligente y webhooks personalizados
"""

import os
import sys
import json
import logging
import datetime
import smtplib
import requests
import time
import random
import threading
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
from urllib.parse import urlparse

# Añadir directorio raíz al path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar componentes necesarios
from data.knowledge_base import KnowledgeBase

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'notifier.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Notifier')

class Notifier:
    """
    Sistema de notificaciones que gestiona alertas y mensajes
    para diferentes eventos del sistema de monetización.
    Implementa Singleton, niveles de alerta, personalización por usuario,
    agrupación inteligente y webhooks personalizados.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Notifier, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el sistema de notificaciones si aún no está inicializado"""
        if self._initialized:
            return
            
        logger.info("Inicializando Notifier...")
        
        # Cargar base de conocimiento
        self.kb = KnowledgeBase()
        
        # Cargar configuración de estrategia
        self.strategy_file = os.path.join('config', 'strategy.json')
        self.strategy = self._load_strategy()
        
        # Cargar configuración de plataformas
        self.platforms_file = os.path.join('config', 'platforms.json')
        self.platforms = self._load_platforms()
        
        # Cargar configuración de usuarios
        self.user_configs_file = os.path.join('config', 'user_configs.json')
        self.user_configs = self._load_user_configs()
        
        # Configuración de alertas
        self.alerts_config = self.strategy.get('analytics', {}).get('alerts', {})
        
        # Configuración de niveles de alerta
        self.alert_levels_config = {
            'info': {
                'priority': 3,
                'channels': ['console', 'log'],
                'emoji': 'ℹ️',
                'color': 5592575,  # Azul (Discord)
                'prefix': '[INFO]',
                'suppression_window': 3600,  # 1 hora
                'escalation': None
            },
            'warning': {
                'priority': 2,
                'channels': ['console', 'log', 'email', 'telegram'],
                'emoji': '⚠️',
                'color': 16776960,  # Amarillo (Discord)
                'prefix': '[WARNING]',
                'suppression_window': 1800,  # 30 minutos
                'escalation': {
                    'level': 'critical',
                    'delay': 3600  # Escalar a critical tras 1 hora
                }
            },
            'critical': {
                'priority': 1,
                'channels': ['console', 'log', 'email', 'telegram', 'discord', 'custom_webhook'],
                'emoji': '🚨',
                'color': 16711680,  # Rojo (Discord)
                'prefix': '[CRITICAL]',
                'suppression_window': 900,  # 15 minutos
                'escalation': None
            }
        }
        
        # Configuración de agrupación inteligente
        self.grouping_config = {
            'time_window': self.alerts_config.get('grouping', {}).get('time_window', 300),  # 5 minutos
            'max_group_size': self.alerts_config.get('grouping', {}).get('max_group_size', 10),
            'type_similarity_threshold': 0.8
        }
        
        # Historial de notificaciones
        self.notification_history = []
        
        # Cache para supresión de notificaciones
        self.suppression_cache = defaultdict(list)
        
        # Métricas de notificaciones
        self.notification_metrics = {
            'total_sent': 0,
            'success_by_channel': defaultdict(int),
            'failures_by_channel': defaultdict(int),
            'escalated_notifications': 0,
            'suppressed_notifications': 0,
            'grouped_notifications': 0,
            'average_delivery_time': 0,
            'user_metrics': defaultdict(lambda: {
                'total_sent': 0,
                'success_by_channel': defaultdict(int),
                'failures_by_channel': defaultdict(int)
            })
        }
        
        # Configuración de reintentos
        self.max_retries = 3
        self.base_retry_delay = 5
        
        # Configuración de canales de notificación
        self.notification_channels = {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your_email@gmail.com',
                'password': 'your_app_password',
                'recipients': ['your_email@gmail.com']
            },
            'telegram': {
                'enabled': False,
                'bot_token': 'YOUR_TELEGRAM_BOT_TOKEN',
                'chat_id': 'YOUR_TELEGRAM_CHAT_ID'
            },
            'discord': {
                'enabled': False,
                'webhook_url': 'YOUR_DISCORD_WEBHOOK_URL'
            },
            'custom_webhook': {
                'enabled': False,
                'webhooks': [],  # Lista de {url, format, headers}
                'timeout': 10
            },
            'console': {
                'enabled': True
            },
            'log': {
                'enabled': True
            }
        }
        
        # Validar configuración
        self._validate_channels_config()
        self._validate_user_configs()
        
        # Iniciar hilo para escalado de alertas
        threading.Thread(target=self._alert_escalation_loop, daemon=True).start()
        
        self._initialized = True
        logger.info("Notifier inicializado correctamente")
    
    def _load_strategy(self) -> Dict:
        """Carga la configuración de estrategia desde el archivo JSON"""
        try:
            if os.path.exists(self.strategy_file):
                with open(self.strategy_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de estrategia no encontrado: {self.strategy_file}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar estrategia: {str(e)}")
            return {}
    
    def _load_platforms(self) -> Dict:
        """Carga la configuración de plataformas desde el archivo JSON"""
        try:
            if os.path.exists(self.platforms_file):
                with open(self.platforms_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de plataformas no encontrado: {self.platforms_file}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar plataformas: {str(e)}")
            return {}
    
    def _load_user_configs(self) -> Dict:
        """Carga la configuración de usuarios desde el archivo JSON"""
        try:
            if os.path.exists(self.user_configs_file):
                with open(self.user_configs_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración de usuarios no encontrado: {self.user_configs_file}")
                return {}
        except Exception as e:
            logger.error(f"Error al cargar configuración de usuarios: {str(e)}")
            return {}
    
    def _validate_channels_config(self):
        """Valida la configuración de los canales de notificación"""
        for channel, config in self.notification_channels.items():
            if config.get('enabled', False):
                if channel == 'email':
                    required = ['smtp_server', 'smtp_port', 'username', 'password', 'recipients']
                    if not all(key in config for key in required) or not config['recipients']:
                        config['enabled'] = False
                        logger.warning(f"Deshabilitado canal email: configuración incompleta")
                elif channel == 'telegram':
                    required = ['bot_token', 'chat_id']
                    if not all(key in config for key in required):
                        config['enabled'] = False
                        logger.warning(f"Deshabilitado canal telegram: configuración incompleta")
                elif channel == 'discord':
                    if 'webhook_url' not in config or not config['webhook_url']:
                        config['enabled'] = False
                        logger.warning(f"Deshabilitado canal discord: configuración incompleta")
                elif channel == 'custom_webhook':
                    if not config.get('webhooks') or not all(
                        'url' in wh and wh['url'] for wh in config['webhooks']
                    ):
                        config['enabled'] = False
                        logger.warning(f"Deshabilitado canal custom_webhook: configuración incompleta")
    
    def _validate_user_configs(self):
        """Valida la configuración de usuarios"""
        for user_id, config in list(self.user_configs.items()):
            if not config.get('channels') or not config.get('alert_preferences'):
                logger.warning(f"Configuración inválida para usuario {user_id}, eliminando")
                del self.user_configs[user_id]
                continue
            
            # Validar canales
            for channel in config['channels']:
                if channel not in self.notification_channels:
                    logger.warning(f"Canal {channel} no soportado para usuario {user_id}")
                    config['channels'].remove(channel)
            
            # Validar niveles de alerta
            for level in config['alert_preferences'].get('levels', {}):
                if level not in self.alert_levels_config:
                    logger.warning(f"Nivel {level} no soportado para usuario {user_id}")
                    del config['alert_preferences']['levels'][level]
    
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calcula el retraso para reintentos con backoff exponencial"""
        return self.base_retry_delay * (2 ** retry_count) + random.uniform(0, 0.1)
    
    def _get_user_alert_level(self, user_id: str, level: str) -> str:
        """Obtiene el nivel de alerta personalizado para un usuario"""
        if user_id not in self.user_configs:
            return level
        
        user_level_map = self.user_configs[user_id].get('alert_preferences', {}).get('levels', {})
        return user_level_map.get(level, level)
    
    def _get_user_channels(self, user_id: str, level: str) -> List[str]:
        """Obtiene los canales preferidos para un usuario y nivel"""
        if user_id not in self.user_configs:
            return self.alert_levels_config[level]['channels']
        
        user_channels = self.user_configs[user_id].get('channels', [])
        level_channels = self.alert_levels_config[level]['channels']
        return [ch for ch in user_channels if ch in level_channels and self.notification_channels[ch].get('enabled', False)]
    
    def _apply_user_template(self, user_id: str, message: str, subject: str, data: Dict) -> tuple:
        """Aplica una plantilla personalizada para el mensaje de un usuario"""
        if user_id not in self.user_configs:
            return subject, message
        
        template = self.user_configs[user_id].get('message_template', {
            'subject': '{subject}',
            'message': '{message}\n\nDetalles:\n{details}'
        })
        
        details = '\n'.join(f"- {k}: {v}" for k, v in data.items()) if data else 'Ninguno'
        new_subject = template['subject'].format(subject=subject, **data)
        new_message = template['message'].format(message=message, details=details, **data)
        
        return new_subject, new_message
    
    def send_notification(self, notification_type: str, subject: str, message: str, 
                         data: Dict = None, level: str = 'info', 
                         channels: List[str] = None, user_id: str = None) -> str:
        """
        Envía una notificación a través de los canales configurados
        """
        if level not in self.alert_levels_config:
            logger.warning(f"Nivel de alerta no válido: {level}, usando 'info'")
            level = 'info'
        
        # Aplicar personalización por usuario
        level = self._get_user_alert_level(user_id, level) if user_id else level
        subject, message = self._apply_user_template(user_id, message, subject, data or {})
        
        # Generar ID de notificación
        notification_id = f"{notification_type}_{int(datetime.datetime.now().timestamp())}"
        
        # Verificar supresión
        if self._is_notification_suppressed(notification_type, level):
            self.notification_metrics['suppressed_notifications'] += 1
            logger.info(f"Notificación {notification_id} suprimida por política de supresión")
            return notification_id
        
        # Crear objeto de notificación
        notification = {
            'id': notification_id,
            'type': notification_type,
            'subject': subject,
            'message': message,
            'data': data or {},
            'level': level,
            'timestamp': datetime.datetime.now().isoformat(),
            'channels_sent': [],
            'delivery_time': 0,
            'escalation_status': 'none',
            'user_id': user_id
        }
        
        # Determinar canales a utilizar
        level_config = self.alert_levels_config[level]
        if channels is None:
            channels = self._get_user_channels(user_id, level) if user_id else level_config['channels']
        
        # Registrar en cache de supresión
        self.suppression_cache[notification_type].append({
            'timestamp': datetime.datetime.now().timestamp(),
            'notification_id': notification_id
        })
        
        # Enviar a cada canal
        start_time = time.time()
        for channel in channels:
            if channel not in self.notification_channels or not self.notification_channels[channel].get('enabled', False):
                logger.warning(f"Canal de notificación no disponible o deshabilitado: {channel}")
                continue
            
            try:
                self._send_to_channel(channel, notification, user_id)
                notification['channels_sent'].append(channel)
                self.notification_metrics['success_by_channel'][channel] += 1
                if user_id:
                    self.notification_metrics['user_metrics'][user_id]['success_by_channel'][channel] += 1
                    self.notification_metrics['user_metrics'][user_id]['total_sent'] += 1
            except Exception as e:
                logger.error(f"Error al enviar notificación a {channel}: {str(e)}")
                self.notification_metrics['failures_by_channel'][channel] += 1
                if user_id:
                    self.notification_metrics['user_metrics'][user_id]['failures_by_channel'][channel] += 1
                threading.Thread(target=self._retry_channel, 
                               args=(channel, notification, 0, user_id), 
                               daemon=True).start()
        
        # Calcular tiempo de entrega
        notification['delivery_time'] = time.time() - start_time
        self._update_delivery_metrics(notification['delivery_time'])
        
        # Registrar en historial
        self.notification_history.append(notification)
        self.notification_metrics['total_sent'] += 1
        if user_id:
            self.notification_metrics['user_metrics'][user_id]['total_sent'] += 1
        
        # Programar escalado si aplica
        if level_config['escalation']:
            threading.Thread(target=self._schedule_escalation, 
                           args=(notification_id, level_config['escalation']), 
                           daemon=True).start()
        
        logger.info(f"Notificación {notification_id} enviada a {len(notification['channels_sent'])} canales")
        return notification_id
    
    def _retry_channel(self, channel: str, notification: Dict, retry_count: int, user_id: str = None):
        """Reintenta enviar una notificación a un canal fallido"""
        if retry_count >= self.max_retries:
            logger.error(f"Reintentos agotados para canal {channel}, notificación {notification['id']}")
            return
        
        try:
            delay = self._calculate_retry_delay(retry_count)
            time.sleep(delay)
            self._send_to_channel(channel, notification, user_id)
            notification['channels_sent'].append(f"{channel}_retry_{retry_count + 1}")
            self.notification_metrics['success_by_channel'][channel] += 1
            if user_id:
                self.notification_metrics['user_metrics'][user_id]['success_by_channel'][channel] += 1
            logger.info(f"Reintento exitoso para canal {channel}, notificación {notification['id']}")
        except Exception as e:
            logger.warning(f"Reintento {retry_count + 1}/{self.max_retries} fallido para canal {channel}: {str(e)}")
            threading.Thread(target=self._retry_channel, 
                           args=(channel, notification, retry_count + 1, user_id), 
                           daemon=True).start()
    
    def _update_delivery_metrics(self, delivery_time: float):
        """Actualiza las métricas de tiempo de entrega"""
        total_notifications = self.notification_metrics['total_sent']
        if total_notifications == 0:
            return
        
        current_avg = self.notification_metrics['average_delivery_time']
        new_avg = ((current_avg * (total_notifications - 1)) + delivery_time) / total_notifications
        self.notification_metrics['average_delivery_time'] = new_avg
    
    def _is_notification_suppressed(self, notification_type: str, level: str) -> bool:
        """Verifica si una notificación debe ser suprimida"""
        if notification_type not in self.suppression_cache:
            return False
        
        suppression_window = self.alert_levels_config[level]['suppression_window']
        current_time = datetime.datetime.now().timestamp()
        
        self.suppression_cache[notification_type] = [
            entry for entry in self.suppression_cache[notification_type]
            if current_time - entry['timestamp'] <= suppression_window
        ]
        
        return len(self.suppression_cache[notification_type]) > 0
    
    def _schedule_escalation(self, notification_id: str, escalation_config: Dict):
        """Programa el escalado de una notificación"""
        time.sleep(escalation_config['delay'])
        
        for notification in self.notification_history:
            if notification['id'] == notification_id and notification['escalation_status'] == 'none':
                self._escalate_alert(notification, escalation_config['level'])
                break
    
    def _escalate_alert(self, notification: Dict, new_level: str):
        """Escala una notificación a un nivel superior"""
        notification['escalation_status'] = 'escalated'
        self.notification_metrics['escalated_notifications'] += 1
        
        new_subject = f"[ESCALATED] {notification['subject']}"
        new_message = (
            f"⚠️ ALERTA ESCALADA desde {notification['level'].upper()} a {new_level.upper()}\n\n"
            f"Original: {notification['message']}\n\n"
            f"Razón: No resuelta tras {self.alert_levels_config[notification['level']]['escalation']['delay']} segundos"
        )
        
        self.send_notification(
            notification_type=f"{notification['type']}_escalated",
            subject=new_subject,
            message=new_message,
            data={
                **notification['data'],
                'original_level': notification['level'],
                'escalation_time': datetime.datetime.now().isoformat()
            },
            level=new_level,
            user_id=notification['user_id']
        )
        logger.info(f"Notificación {notification['id']} escalada a {new_level}")
    
    def _alert_escalation_loop(self):
        """Bucle para manejar escalado periódico"""
        while True:
            try:
                current_time = datetime.datetime.now().timestamp()
                for notification in self.notification_history:
                    if notification['escalation_status'] != 'none':
                        continue
                    
                    level_config = self.alert_levels_config.get(notification['level'], {})
                    escalation = level_config.get('escalation')
                    if not escalation:
                        continue
                    
                    notification_time = datetime.datetime.fromisoformat(notification['timestamp']).timestamp()
                    if current_time - notification_time >= escalation['delay']:
                        self._escalate_alert(notification, escalation['level'])
                
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error en bucle de escalado: {str(e)}")
                time.sleep(60)
    
    def _send_to_channel(self, channel: str, notification: Dict, user_id: str = None):
        """Envía una notificación a un canal específico"""
        level_config = self.alert_levels_config[notification['level']]
        if channel == 'email':
            self._send_email(notification, level_config, user_id)
        elif channel == 'telegram':
            self._send_telegram(notification, level_config, user_id)
        elif channel == 'discord':
            self._send_discord(notification, level_config)
        elif channel == 'custom_webhook':
            self._send_custom_webhook(notification, level_config, user_id)
        elif channel == 'console':
            self._send_console(notification, level_config)
        elif channel == 'log':
            self._send_log(notification, level_config)
        else:
            logger.warning(f"Canal de notificación desconocido: {channel}")
    
    def _send_email(self, notification: Dict, level_config: Dict, user_id: str = None):
        """Envía notificación por email"""
        config = self.notification_channels['email']
        recipients = config['recipients']
        
        if user_id and user_id in self.user_configs:
            user_recipients = self.user_configs[user_id].get('email_recipients', [])
            if user_recipients:
                recipients = user_recipients
        
        msg = MIMEMultipart()
        msg['From'] = config['username']
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"{level_config['prefix']} {notification['subject']}"
        
        body = f"{level_config['emoji']} {notification['message']}\n\n"
        
        if notification['data']:
            body += "Detalles adicionales:\n"
            for key, value in notification['data'].items():
                body += f"- {key}: {value}\n"
        
        body += f"\nFecha: {notification['timestamp']}"
        body += f"\nID: {notification['id']}"
        if notification['escalation_status'] == 'escalated':
            body += f"\nEstado: Escalada desde {notification['data'].get('original_level', 'desconocido')}"
        
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            logger.info(f"Email enviado a {len(recipients)} destinatarios")
        except Exception as e:
            logger.error(f"Error al enviar email: {str(e)}")
            raise
    
    def _send_telegram(self, notification: Dict, level_config: Dict, user_id: str = None):
        """Envía notificación por Telegram"""
        config = self.notification_channels['telegram']
        chat_id = config['chat_id']
        
        if user_id and user_id in self.user_configs:
            user_chat_id = self.user_configs[user_id].get('telegram_chat_id')
            if user_chat_id:
                chat_id = user_chat_id
        
        message = f"{level_config['emoji']} *{notification['subject']}*\n\n{notification['message']}\n"
        
        if notification['data']:
            message += "\n*Detalles adicionales:*\n"
            for key, value in notification['data'].items():
                message += f"- {key}: {value}\n"
        
        if notification['escalation_status'] == 'escalated':
            message += f"\n*Escalada desde*: {notification['data'].get('original_level', 'desconocido')}"
        
        url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            logger.error(f"Error al enviar mensaje a Telegram: {response.text}")
            raise Exception(f"Error de Telegram: {response.status_code}")
        
        logger.info("Mensaje enviado a Telegram correctamente")
    
    def _send_discord(self, notification: Dict, level_config: Dict):
        """Envía notificación por Discord"""
        config = self.notification_channels['discord']
        
        embed = {
            'title': f"{level_config['emoji']} {notification['subject']}",
            'description': notification['message'],
            'color': level_config['color'],
            'fields': [],
            'timestamp': notification['timestamp']
        }
        
        if notification['data']:
            for key, value in notification['data'].items():
                embed['fields'].append({
                    'name': key,
                    'value': str(value),
                    'inline': True
                })
        
        if notification['escalation_status'] == 'escalated':
            embed['fields'].append({
                'name': 'Estado',
                'value': f"Escalada desde {notification['data'].get('original_level', 'desconocido')}",
                'inline': True
            })
        
        payload = {
            'embeds': [embed],
            'username': 'Content Bot Notifier'
        }
        
        response = requests.post(config['webhook_url'], json=payload)
        
        if response.status_code != 204:
            logger.error(f"Error al enviar mensaje a Discord: {response.text}")
            raise Exception(f"Error de Discord: {response.status_code}")
        
        logger.info("Mensaje enviado a Discord correctamente")
    
    def _send_custom_webhook(self, notification: Dict, level_config: Dict, user_id: str = None):
        """Envía notificación a webhooks personalizados"""
        config = self.notification_channels['custom_webhook']
        webhooks = config['webhooks']
        
        if user_id and user_id in self.user_configs:
            user_webhooks = self.user_configs[user_id].get('custom_webhooks', [])
            if user_webhooks:
                webhooks = user_webhooks
        
        for webhook in webhooks:
            url = webhook['url']
            format_type = webhook.get('format', 'json')
            headers = webhook.get('headers', {})
            timeout = config.get('timeout', 10)
            
            try:
                # Validar URL
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    logger.warning(f"URL de webhook inválida: {url}")
                    continue
                
                # Preparar payload
                payload = {
                    'notification_id': notification['id'],
                    'type': notification['type'],
                    'subject': notification['subject'],
                    'message': notification['message'],
                    'level': notification['level'],
                    'timestamp': notification['timestamp'],
                    'data': notification['data'],
                    'escalation_status': notification['escalation_status']
                }
                
                if format_type == 'xml':
                    root = ET.Element('notification')
                    for key, value in payload.items():
                        child = ET.SubElement(root, key)
                        child.text = str(value)
                    payload_data = ET.tostring(root, encoding='unicode')
                    headers['Content-Type'] = 'application/xml'
                else:
                    payload_data = json.dumps(payload)
                    headers['Content-Type'] = 'application/json'
                
                # Enviar solicitud
                response = requests.post(url, data=payload_data, headers=headers, timeout=timeout)
                
                if response.status_code not in (200, 201, 204):
                    logger.error(f"Error al enviar webhook a {url}: {response.text}")
                    raise Exception(f"Error de webhook: {response.status_code}")
                
                logger.info(f"Webhook enviado correctamente a {url}")
            
            except Exception as e:
                logger.error(f"Error al enviar webhook a {url}: {str(e)}")
                raise
    
    def _send_console(self, notification: Dict, level_config: Dict):
        """Muestra notificación en consola"""
        print(f"\n{level_config['prefix']} {notification['subject']}")
        print("-" * 50)
        print(f"{level_config['emoji']} {notification['message']}")
        
        if notification['data']:
            print("\nDetalles adicionales:")
            for key, value in notification['data'].items():
                print(f"- {key}: {value}")
        
        print(f"\nFecha: {notification['timestamp']}")
        if notification['escalation_status'] == 'escalated':
            print(f"Estado: Escalada desde {notification['data'].get('original_level', 'desconocido')}")
        if notification['user_id']:
            print(f"Usuario: {notification['user_id']}")
        print("-" * 50)
        
        logger.info("Notificación mostrada en consola")
    
    def _send_log(self, notification: Dict, level_config: Dict):
        """Registra notificación en el log"""
        log_message = f"{level_config['prefix']} {notification['subject']}: {notification['message']}"
        if notification['user_id']:
            log_message += f" (Usuario: {notification['user_id']})"
        
        if notification['level'] == 'critical':
            logger.critical(log_message)
        elif notification['level'] == 'warning':
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _group_notifications_intelligently(self, notifications: List[Dict]) -> List[List[Dict]]:
        """Agrupa notificaciones de manera inteligente"""
        if not notifications:
            return []
        
        grouped = []
        current_group = []
        last_time = None
        time_window = self.grouping_config['time_window']
        max_group_size = self.grouping_config['max_group_size']
        
        # Ordenar por prioridad y timestamp
        sorted_notifications = sorted(
            notifications,
            key=lambda n: (
                self.alert_levels_config[n['level']]['priority'],
                datetime.datetime.fromisoformat(n['timestamp']).timestamp()
            )
        )
        
        for notif in sorted_notifications:
            notif_time = datetime.datetime.fromisoformat(notif['timestamp']).timestamp()
            
            # Verificar si entra en la ventana temporal
            if last_time is None or (notif_time - last_time) <= time_window:
                if len(current_group) < max_group_size:
                    current_group.append(notif)
                else:
                    grouped.append(current_group)
                    current_group = [notif]
            else:
                grouped.append(current_group)
                current_group = [notif]
            
            last_time = notif_time
        
        if current_group:
            grouped.append(current_group)
        
        # Fusionar grupos por tipo si son similares
        final_groups = []
        for group in grouped:
            type_counts = defaultdict(int)
            for notif in group:
                type_counts[notif['type']] += 1
            
            # Si un tipo domina el grupo, mantenerlo unido
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
            if dominant_type and type_counts[dominant_type] / len(group) >= self.grouping_config['type_similarity_threshold']:
                final_groups.append(group)
            else:
                # Dividir por tipo
                type_groups = defaultdict(list)
                for notif in group:
                    type_groups[notif['type']].append(notif)
                final_groups.extend(type_groups.values())
        
        return [g for g in final_groups if g]
    
    def notify_batch(self, notifications: List[Dict]) -> List[str]:
        """
        Envía un lote de notificaciones agrupadas
        """
        notification_ids = []
        
        # Agrupar notificaciones inteligentemente
        grouped_notifications = self._group_notifications_intelligently(notifications)
        
        for group in grouped_notifications:
            if not group:
                continue
            
            grouped_message = "📢 Notificaciones Agrupadas\n\n"
            grouped_data = {}
            highest_level = 'info'
            user_id = group[0].get('user_id')
            
            # Verificar que todas las notificaciones sean del mismo usuario
            if not all(notif.get('user_id') == user_id for notif in group):
                logger.warning("Intento de agrupar notificaciones de diferentes usuarios, enviando individualmente")
                for notif in group:
                    notification_id = self.send_notification(
                        notification_type=notif['type'],
                        subject=notif['subject'],
                        message=notif['message'],
                        data=notif['data'],
                        level=notif['level'],
                        user_id=notif['user_id']
                    )
                    notification_ids.append(notification_id)
                continue
            
            for idx, notif in enumerate(group):
                notification_type = notif.get('type', 'batch')
                subject = notif.get('subject', 'Notificación agrupada')
                message = notif.get('message', '')
                data = notif.get('data', {})
                level = notif.get('level', 'info')
                
                if self.alert_levels_config[level]['priority'] < self.alert_levels_config[highest_level]['priority']:
                    highest_level = level
                
                grouped_message += f"[{idx + 1}] {self.alert_levels_config[level]['prefix']} {subject}\n"
                grouped_message += f"{message}\n\n"
                grouped_data[f"notification_{idx + 1}"] = {
                    'type': notification_type,
                    'subject': subject,
                    'message': message,
                    'data': data,
                    'level': level
                }
            
            notification_id = self.send_notification(
                notification_type="batch_notification",
                subject="Notificaciones Agrupadas",
                message=grouped_message,
                data=grouped_data,
                level=highest_level,
                user_id=user_id
            )
            
            notification_ids.append(notification_id)
            self.notification_metrics['grouped_notifications'] += len(group)
        
        return notification_ids
    
    def notify_shadowban(self, platform: str, channel_id: str, metrics: Dict, user_id: str = None):
        """
        Envía notificación de posible shadowban
        """
        threshold = self.alerts_config.get('risk', {}).get('shadowban_probability', 0.7)
        if user_id and user_id in self.user_configs:
            threshold = self.user_configs[user_id].get('alert_preferences', {}).get('shadowban_threshold', threshold)
        
        views_drop = metrics.get('views_drop', 0)
        engagement_drop = metrics.get('engagement_drop', 0)
        distribution_change = metrics.get('distribution_change', 0)
        
        shadowban_probability = (views_drop + engagement_drop + distribution_change) / 3
        
        if shadowban_probability >= threshold:
            subject = f"Posible shadowban detectado en {platform}"
            message = (
                f"Se ha detectado una posible restricción de distribución (shadowban) "
                f"en el canal {channel_id} de {platform}.\n\n"
                f"Probabilidad estimada: {shadowban_probability:.2%}"
            )
            
            self.send_notification(
                notification_type="shadowban",
                subject=subject,
                message=message,
                data={
                    'platform': platform,
                    'channel_id': channel_id,
                    'views_drop': f"{views_drop:.2%}",
                    'engagement_drop': f"{engagement_drop:.2%}",
                    'distribution_change': f"{distribution_change:.2%}",
                    'probability': f"{shadowban_probability:.2%}"
                },
                level='warning',
                user_id=user_id
            )
    
    def notify_niche_saturation(self, niche: str, metrics: Dict, user_id: str = None):
        """
        Envía notificación de saturación de nicho
        """
        saturation_config = self.strategy.get('optimization_strategies', {}).get('niche_saturation', {})
        
        view_decline = metrics.get('view_decline_rate', 0)
        engagement_decline = metrics.get('engagement_decline_rate', 0)
        competition_increase = metrics.get('competition_increase_rate', 0)
        
        thresholds = saturation_config.get('metrics', {
            'view_decline_rate': 0.2,
            'engagement_decline_rate': 0.15,
            'competition_increase_rate': 0.3
        })
        if user_id and user_id in self.user_configs:
            user_thresholds = self.user_configs[user_id].get('alert_preferences', {}).get('niche_saturation', {})
            thresholds.update(user_thresholds)
        
        threshold_exceeded = (
            view_decline > thresholds.get('view_decline_rate', 0.2) or
            engagement_decline > thresholds.get('engagement_decline_rate', 0.15) or
            competition_increase > thresholds.get('competition_increase_rate', 0.3)
        )
        
        if threshold_exceeded:
            subject = f"Saturación detectada en nicho: {niche}"
            message = (
                f"Se ha detectado una posible saturación en el nicho {niche}.\n\n"
                f"Las métricas indican un aumento de competencia y/o disminución de rendimiento."
            )
            
            self.send_notification(
                notification_type="niche_saturation",
                subject=subject,
                message=message,
                data={
                    'niche': niche,
                    'view_decline': f"{view_decline:.2%}",
                    'engagement_decline': f"{engagement_decline:.2%}",
                    'competition_increase': f"{competition_increase:.2%}"
                },
                level='info',
                user_id=user_id
            )
            
            if saturation_config.get('actions', {}).get('suggest_pivot', False):
                self._suggest_niche_pivot(niche, user_id)
    
    def _suggest_niche_pivot(self, current_niche: str, user_id: str = None):
        """Sugiere nichos alternativos basados en tendencias actuales"""
        alternative_niches = {
            'finance': ['technology', 'health'],
            'health': ['finance', 'humor'],
            'gaming': ['technology', 'humor'],
            'technology': ['finance', 'gaming'],
            'humor': ['gaming', 'health']
        }
        
        suggestions = alternative_niches.get(current_niche, [])
        
        if suggestions:
            subject = f"Sugerencia de pivote para nicho {current_niche}"
            message = (
                f"Basado en la saturación detectada en el nicho {current_niche}, "
                f"se recomienda considerar un pivote hacia los siguientes nichos:\n\n"
                f"- {suggestions[0]}\n"
                f"- {suggestions[1]}\n\n"
                f"Estos nichos muestran mejor potencial de crecimiento y menor saturación."
            )
            
            self.send_notification(
                notification_type="niche_pivot",
                subject=subject,
                message=message,
                data={
                    'current_niche': current_niche,
                    'suggested_niches': suggestions
                },
                level='info',
                user_id=user_id
            )
    
    def notify_performance_drop(self, channel_id: str, platform: str, metrics: Dict, user_id: str = None):
        """
        Envía notificación de caída de rendimiento
        """
        thresholds = self.alerts_config.get('performance_drop', {
            'views': 0.3,
            'engagement': 0.25,
            'conversion': 0.2
        })
        if user_id and user_id in self.user_configs:
            user_thresholds = self.user_configs[user_id].get('alert_preferences', {}).get('performance_drop', {})
            thresholds.update(user_thresholds)
        
        alerts = []
        if metrics.get('views_drop', 0) > thresholds.get('views', 0.3):
            alerts.append(f"Caída de vistas: {metrics['views_drop']:.2%}")
        
        if metrics.get('engagement_drop', 0) > thresholds.get('engagement', 0.25):
            alerts.append(f"Caída de engagement: {metrics['engagement_drop']:.2%}")
        
        if metrics.get('conversion_drop', 0) > thresholds.get('conversion', 0.2):
            alerts.append(f"Caída de conversión: {metrics['conversion_drop']:.2%}")
        
        if alerts:
            subject = f"Caída de rendimiento en {platform}"
            message = (
                f"Se ha detectado una caída significativa en el rendimiento "
                f"del canal {channel_id} en {platform}:\n\n"
                f"- {alerts[0]}\n"
            )
            
            if len(alerts) > 1:
                for alert in alerts[1:]:
                    message += f"- {alert}\n"
            
            max_drop = max(
                metrics.get('views_drop', 0),
                metrics.get('engagement_drop', 0),
                metrics.get('conversion_drop', 0)
            )
            
            level = 'warning' if max_drop > 0.5 else 'info'
            
            self.send_notification(
                notification_type="performance_drop",
                subject=subject,
                message=message,
                data={
                    'channel_id': channel_id,
                    'platform': platform,
                    **{k: f"{v:.2%}" for k, v in metrics.items() if k.endswith('_drop')}
                },
                level=level,
                user_id=user_id
            )
    
    def notify_opportunity(self, opportunity_type: str, data: Dict, user_id: str = None):
        """
        Envía notificación de oportunidad detectada
        """
        opportunity_messages = {
            'trending_topic': {
                'subject': 'Tema tendencia detectado',
                'message': 'Se ha detectado un tema en tendencia que podría ser relevante para tu contenido.',
                'level': 'info'
            },
            'audience_growth': {
                'subject': 'Oportunidad de crecimiento de audiencia',
                'message': 'Se ha detectado un segmento de audiencia con alto potencial de crecimiento.',
                'level': 'info'
            },
            'monetization': {
                'subject': 'Oportunidad de monetización',
                'message': 'Se ha detectado una nueva oportunidad para monetizar tu contenido.',
                'level': 'warning'
            },
            'collaboration': {
                'subject': 'Oportunidad de colaboración',
                'message': 'Se ha identificado un creador potencial para colaboración.',
                'level': 'info'
            },
            'platform_feature': {
                'subject': 'Nueva característica de plataforma',
                'message': 'Se ha lanzado una nueva característica que podría beneficiar a tu canal.',
                'level': 'info'
            },
            'content_gap': {
                'subject': 'Brecha de contenido identificada',
                'message': 'Se ha identificado una brecha de contenido que podrías aprovechar.',
                'level': 'info'
            }
        }
        
        config = opportunity_messages.get(opportunity_type, {
            'subject': f'Nueva oportunidad: {opportunity_type}',
            'message': 'Se ha detectado una nueva oportunidad para tu contenido.',
            'level': 'info'
        })
        
        subject = config['subject']
        message = config['message'] + '\n\n'
        
        if opportunity_type == 'trending_topic':
            message += (
                f"Tema: {data.get('topic', 'No especificado')}\n"
                f"Volumen de búsqueda: {data.get('search_volume', 'No disponible')}\n"
                f"Crecimiento: {data.get('growth_rate', '0')}%\n\n"
                f"Este tema está ganando popularidad y se alinea con tu contenido. "
                f"Considera crear contenido relacionado pronto para aprovechar esta tendencia."
            )
        
        elif opportunity_type == 'audience_growth':
            message += (
                f"Segmento: {data.get('segment', 'No especificado')}\n"
                f"Tamaño estimado: {data.get('estimated_size', 'No disponible')}\n"
                f"Tasa de crecimiento: {data.get('growth_rate', '0')}%\n\n"
                f"Este segmento de audiencia está creciendo rápidamente y muestra interés "
                f"en temas relacionados con tu contenido. Considera adaptar tu estrategia "
                f"para atraer a este público."
            )
        
        elif opportunity_type == 'monetization':
            message += (
                f"Tipo: {data.get('type', 'No especificado')}\n"
                f"Potencial estimado: {data.get('estimated_revenue', 'No disponible')}\n\n"
                f"Esta oportunidad de monetización podría aumentar tus ingresos. "
                f"Revisa los detalles y considera implementarla en tu estrategia."
            )
        
        elif opportunity_type == 'collaboration':
            message += (
                f"Creador: {data.get('creator_name', 'No especificado')}\n"
                f"Plataforma: {data.get('platform', 'No disponible')}\n"
                f"Audiencia: {data.get('audience_size', 'No disponible')}\n\n"
                f"Este creador tiene una audiencia similar y complementaria a la tuya. "
                f"Una colaboración podría beneficiar a ambos canales."
            )
        
        elif opportunity_type == 'platform_feature':
            message += (
                f"Plataforma: {data.get('platform', 'No especificado')}\n"
                f"Característica: {data.get('feature_name', 'No disponible')}\n\n"
                f"Esta nueva característica podría ayudarte a mejorar tu alcance o monetización. "
                f"Considera cómo integrarla en tu estrategia de contenido."
            )
        
        elif opportunity_type == 'content_gap':
            message += (
                f"Tema: {data.get('topic', 'No especificado')}\n"
                f"Demanda estimada: {data.get('demand_level', 'No disponible')}\n"
                f"Competencia: {data.get('competition_level', 'No disponible')}\n\n"
                f"Existe una demanda significativa para este tema con poca oferta de contenido. "
                f"Considera crear contenido para cubrir esta brecha."
            )
        
        self.send_notification(
            notification_type=f"opportunity_{opportunity_type}",
            subject=subject,
            message=message,
            data=data,
            level=config['level'],
            user_id=user_id
        )
    
    def notify_task_completion(self, task_type: str, task_data: Dict, success: bool = True, user_id: str = None):
        """
        Envía notificación de finalización de tarea
        """
        task_messages = {
            'content_creation': 'Creación de contenido',
            'content_optimization': 'Optimización de contenido',
            'analytics_report': 'Informe de análisis',
            'audience_research': 'Investigación de audiencia',
            'competitor_analysis': 'Análisis de competencia',
            'trend_research': 'Investigación de tendencias',
            'monetization_setup': 'Configuración de monetización',
            'engagement_campaign': 'Campaña de engagement',
            'data_backup': 'Respaldo de datos'
        }
        
        task_name = task_messages.get(task_type, f'Tarea: {task_type}')
        
        if success:
            subject = f"✅ {task_name} completada"
            message = f"La tarea de {task_name.lower()} se ha completado con éxito."
            level = 'info'
        else:
            subject = f"❌ {task_name} fallida"
            message = f"La tarea de {task_name.lower()} ha fallado."
            level = 'warning'
        
        if 'name' in task_data:
            message += f"\nNombre: {task_data['name']}"
        
        if 'duration' in task_data:
            message += f"\nDuración: {task_data['duration']} segundos"
        
        if not success and 'error' in task_data:
            message += f"\n\nError: {task_data['error']}"
        
        self.send_notification(
            notification_type=f"task_{task_type}",
            subject=subject,
            message=message,
            data=task_data,
            level=level,
            user_id=user_id
        )
    
    def notify_content_performance(self, content_id: str, platform: str, metrics: Dict, user_id: str = None):
        """
        Envía notificación sobre el rendimiento de un contenido específico
        """
        performance_threshold = self.alerts_config.get('content_performance', {}).get('threshold', 0.2)
        if user_id and user_id in self.user_configs:
            performance_threshold = self.user_configs[user_id].get('alert_preferences', {}).get('performance_threshold', performance_threshold)
        
        views_performance = metrics.get('views_vs_expected', 0)
        engagement_performance = metrics.get('engagement_vs_expected', 0)
        conversion_performance = metrics.get('conversion_vs_expected', 0)
        
        avg_performance = (views_performance + engagement_performance + conversion_performance) / 3
        
        if avg_performance > performance_threshold:
            performance_type = "positive"
            subject = f"📈 Contenido con rendimiento excepcional en {platform}"
            message = f"El contenido está superando las expectativas en {platform}."
            level = "info"
        elif avg_performance < -performance_threshold:
            performance_type = "negative"
            subject = f"📉 Contenido con bajo rendimiento en {platform}"
            message = f"El contenido está por debajo de las expectativas en {platform}."
            level = "warning"
        else:
            return
        
        message += "\n\nMétricas de rendimiento:"
        
        if 'views' in metrics:
            message += f"\n- Vistas: {metrics['views']}"
            if 'views_vs_expected' in metrics:
                deviation = metrics['views_vs_expected'] * 100
                direction = "por encima" if deviation > 0 else "por debajo"
                message += f" ({abs(deviation):.1f}% {direction} de lo esperado)"
        
        if 'engagement_rate' in metrics:
            message += f"\n- Tasa de engagement: {metrics['engagement_rate']:.2%}"
            if 'engagement_vs_expected' in metrics:
                deviation = metrics['engagement_vs_expected'] * 100
                direction = "por encima" if deviation > 0 else "por debajo"
                message += f" ({abs(deviation):.1f}% {direction} de lo esperado)"
        
        if 'conversion_rate' in metrics:
            message += f"\n- Tasa de conversión: {metrics['conversion_rate']:.2%}"
            if 'conversion_vs_expected' in metrics:
                deviation = metrics['conversion_vs_expected'] * 100
                direction = "por encima" if deviation > 0 else "por debajo"
                message += f" ({abs(deviation):.1f}% {direction} de lo esperado)"
        
        if performance_type == "positive":
            message += "\n\nRecomendaciones:"
            message += "\n- Considera crear más contenido similar"
            message += "\n- Analiza qué factores contribuyeron al éxito"
            message += "\n- Promociona este contenido en otras plataformas"
        else:
            message += "\n\nRecomendaciones:"
            message += "\n- Revisa el título, miniaturas y optimización"
            message += "\n- Considera ajustar la estrategia de distribución"
            message += "\n- Evalúa si el contenido se alinea con los intereses de tu audiencia"
        
        self.send_notification(
            notification_type=f"content_performance_{performance_type}",
            subject=subject,
            message=message,
            data={
                'content_id': content_id,
                'platform': platform,
                **metrics
            },
            level=level,
            user_id=user_id
        )
    
    def notify_system_status(self, status: str, details: Dict = None, user_id: str = None):
        """
        Envía notificación sobre el estado del sistema
        """
        details = details or {}
        
        status_config = {
            'ok': {
                'subject': '✅ Sistema funcionando correctamente',
                'message': 'Todos los componentes del sistema están funcionando correctamente.',
                'level': 'info'
            },
            'warning': {
                'subject': '⚠️ Advertencia del sistema',
                'message': 'Se han detectado problemas menores en el sistema.',
                'level': 'warning'
            },
            'error': {
                'subject': '❌ Error en el sistema',
                'message': 'Se han detectado errores en el sistema que requieren atención.',
                'level': 'warning'
            },
            'critical': {
                'subject': '🚨 Error crítico en el sistema',
                'message': 'Se han detectado errores críticos que afectan el funcionamiento del sistema.',
                'level': 'critical'
            }
        }
        
        config = status_config.get(status, status_config['warning'])
        
        message = config['message'] + '\n\n'
        
        if 'affected_components' in details:
            message += "Componentes afectados:\n"
            for component in details['affected_components']:
                message += f"- {component}\n"
            message += "\n"
        
        if 'errors' in details:
            message += "Errores detectados:\n"
            for error in details['errors']:
                message += f"- {error}\n"
            message += "\n"
        
        if 'resource_usage' in details:
            message += "Uso de recursos:\n"
            for resource, usage in details['resource_usage'].items():
                message += f"- {resource}: {usage}\n"
            message += "\n"
        
        if 'recommended_actions' in details:
            message += "Acciones recomendadas:\n"
            for action in details['recommended_actions']:
                message += f"- {action}\n"
        
        self.send_notification(
            notification_type=f"system_status_{status}",
            subject=config['subject'],
            message=message,
            data=details,
            level=config['level'],
            user_id=user_id
        )
    
    def get_notification_history(self, limit: int = 50, notification_type: str = None, 
                               level: str = None, start_date: str = None, 
                               end_date: str = None, user_id: str = None) -> List[Dict]:
        """
        Obtiene historial de notificaciones con filtros opcionales
        """
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.datetime.fromisoformat(start_date)
            except ValueError:
                logger.warning(f"Formato de fecha de inicio no válido: {start_date}")
        
        if end_date:
            try:
                end_datetime = datetime.datetime.fromisoformat(end_date)
            except ValueError:
                logger.warning(f"Formato de fecha de fin no válido: {end_date}")
        
        filtered_notifications = []
        
        for notification in reversed(self.notification_history):
            if notification_type and not notification['type'].startswith(notification_type):
                continue
            
            if level and notification['level'] != level:
                continue
            
            if user_id and notification.get('user_id') != user_id:
                continue
            
            if start_datetime:
                notification_datetime = datetime.datetime.fromisoformat(notification['timestamp'])
                if notification_datetime < start_datetime:
                    continue
            
            if end_datetime:
                notification_datetime = datetime.datetime.fromisoformat(notification['timestamp'])
                if notification_datetime > end_datetime:
                    continue
            
            filtered_notifications.append(notification)
            
            if len(filtered_notifications) >= limit:
                break
        
        return filtered_notifications
    
    def get_notification_metrics(self, user_id: str = None) -> Dict:
        """
        Obtiene métricas de notificaciones
        """
        if user_id:
            user_metrics = self.notification_metrics['user_metrics'].get(user_id, {
                'total_sent': 0,
                'success_by_channel': {},
                'failures_by_channel': {}
            })
            return {
                'total_sent': user_metrics['total_sent'],
                'success_by_channel': dict(user_metrics['success_by_channel']),
                'failures_by_channel': dict(user_metrics['failures_by_channel']),
                'escalated_notifications': self.notification_metrics['escalated_notifications'],
                'suppressed_notifications': self.notification_metrics['suppressed_notifications'],
                'grouped_notifications': self.notification_metrics['grouped_notifications'],
                'average_delivery_time': f"{self.notification_metrics['average_delivery_time']:.3f} seconds"
            }
        
        return {
            'total_sent': self.notification_metrics['total_sent'],
            'success_by_channel': dict(self.notification_metrics['success_by_channel']),
            'failures_by_channel': dict(self.notification_metrics['failures_by_channel']),
            'escalated_notifications': self.notification_metrics['escalated_notifications'],
            'suppressed_notifications': self.notification_metrics['suppressed_notifications'],
            'grouped_notifications': self.notification_metrics['grouped_notifications'],
            'average_delivery_time': f"{self.notification_metrics['average_delivery_time']:.3f} seconds",
            'user_metrics': {k: {
                'total_sent': v['total_sent'],
                'success_by_channel': dict(v['success_by_channel']),
                'failures_by_channel': dict(v['failures_by_channel'])
            } for k, v in self.notification_metrics['user_metrics'].items()}
        }
    
    def update_notification_channels(self, channel_updates: Dict) -> bool:
        """
        Actualiza la configuración de canales de notificación
        """
        try:
            for channel, config in channel_updates.items():
                if channel in self.notification_channels:
                    self.notification_channels[channel].update(config)
                    logger.info(f"Canal de notificación actualizado: {channel}")
                else:
                    self.notification_channels[channel] = config
                    logger.info(f"Nuevo canal de notificación añadido: {channel}")
            
            config_file = os.path.join('config', 'notification_channels.json')
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.notification_channels, f, indent=4)
            
            self._validate_channels_config()
            logger.info("Configuración de canales de notificación guardada")
            return True
        
        except Exception as e:
            logger.error(f"Error al actualizar canales de notificación: {str(e)}")
            return False
    
    def update_user_configs(self, user_configs: Dict) -> bool:
        """
        Actualiza la configuración de usuarios
        """
        try:
            for user_id, config in user_configs.items():
                self.user_configs[user_id] = config
                logger.info(f"Configuración actualizada para usuario: {user_id}")
            
            config_file = os.path.join('config', 'user_configs.json')
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_configs, f, indent=4)
            
            self._validate_user_configs()
            logger.info("Configuración de usuarios guardada")
            return True
        
        except Exception as e:
            logger.error(f"Error al actualizar configuración de usuarios: {str(e)}")
            return False
    
    def cancel_notification(self, notification_id: str) -> bool:
        """
        Cancela una notificación programada (ej. escalado)
        """
        for notification in self.notification_history:
            if notification['id'] == notification_id and notification['escalation_status'] == 'none':
                notification['escalation_status'] = 'cancelled'
                logger.info(f"Notificación {notification_id} cancelada")
                return True
        
        logger.warning(f"Notificación {notification_id} no encontrada o no cancelable")
        return False

# Ejemplo de uso
if __name__ == "__main__":
    os.makedirs('logs', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    notifier = Notifier()
    
    # Ejemplo de configuración de usuario
    user_configs = {
        'user1': {
            'channels': ['email', 'telegram'],
            'email_recipients': ['user1@example.com'],
            'telegram_chat_id': 'USER1_CHAT_ID',
            'custom_webhooks': [
                {'url': 'https://example.com/webhook', 'format': 'json', 'headers': {'Authorization': 'Bearer token'}},
                {'url': 'https://example.com/webhook2', 'format': 'xml'}
            ],
            'alert_preferences': {
                'levels': {'warning': 'critical'},
                'shadowban_threshold': 0.8,
                'performance_threshold': 0.25,
                'niche_saturation': {
                    'view_decline_rate': 0.25,
                    'engagement_decline_rate': 0.2
                }
            },
            'message_template': {
                'subject': 'Alerta: {subject}',
                'message': '{message}\n\nDetalles adicionales:\n{details}'
            }
        }
    }
    
    notifier.update_user_configs(user_configs)
    
    # Ejemplo de notificación para usuario
    notifier.send_notification(
        notification_type="test",
        subject="Prueba de notificación",
        message="Este es un mensaje de prueba del sistema de notificaciones.",
        data={"test_key": "test_value"},
        level="info",
        user_id="user1"
    )
    
    # Ejemplo de notificación agrupada
    notifications = [
        {
            'type': 'test1',
            'subject': 'Prueba 1',
            'message': 'Mensaje de prueba 1',
            'data': {'key': 'value1'},
            'level': 'info',
            'user_id': 'user1',
            'timestamp': datetime.datetime.now().isoformat()
        },
        {
            'type': 'test1',
            'subject': 'Prueba 2',
            'message': 'Mensaje de prueba 2',
            'data': {'key': 'value2'},
            'level': 'warning',
            'user_id': 'user1',
            'timestamp': datetime.datetime.now().isoformat()
        }
    ]
    
    notifier.notify_batch(notifications)
    
    print("Notificaciones enviadas correctamente")