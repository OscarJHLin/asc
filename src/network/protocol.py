"""
通信协议模块

定义节点间通信的消息格式和协议规范
"""

import json
import time
import uuid
from enum import Enum
from typing import Any, Dict, Tuple


class MessageType(Enum):
    """消息类型枚举"""
    HEARTBEAT = "heartbeat"
    DISCOVER = "discover"
    DISCOVER_RESPONSE = "discover_response"
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = "task_result"
    TASK_CANCEL = "task_cancel"
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    ERROR = "error"


class TaskType(Enum):
    """任务类型枚举"""
    INFERENCE = "inference"
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"
    HEALTH_CHECK = "health_check"


class Message:
    """
    消息基类
    
    所有节点间通信消息的基类
    """
    
    def __init__(self,
                 msg_type: str,
                 sender_id: str,
                 sender_ip: str,
                 sender_port: int = 52415,
                 payload: Dict[str, Any] = None,
                 msg_id: str = None):
        self.msg_type = msg_type
        self.sender_id = sender_id
        self.sender_ip = sender_ip
        self.sender_port = sender_port
        self.timestamp = time.time()
        self.payload = payload or {}
        self.msg_id = msg_id or str(uuid.uuid4())[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'msg_type': self.msg_type,
            'sender_id': self.sender_id,
            'sender_ip': self.sender_ip,
            'sender_port': self.sender_port,
            'timestamp': self.timestamp,
            'payload': self.payload,
            'msg_id': self.msg_id,
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        msg = cls(
            msg_type=data.get('msg_type', 'unknown'),
            sender_id=data.get('sender_id', 'unknown'),
            sender_ip=data.get('sender_ip', '127.0.0.1'),
            sender_port=data.get('sender_port', 52415),
            payload=data.get('payload', {}),
            msg_id=data.get('msg_id')
        )
        msg.timestamp = data.get('timestamp', time.time())
        return msg
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """从JSON字符串创建消息"""
        return cls.from_dict(json.loads(json_str))


class TaskMessage(Message):
    """
    任务消息
    
    用于任务分配和结果返回
    """
    
    def __init__(self,
                 msg_type: str,
                 sender_id: str,
                 sender_ip: str,
                 task_id: str = None,
                 task_type: str = None,
                 payload: Dict[str, Any] = None,
                 **kwargs):
        super().__init__(
            msg_type=msg_type,
            sender_id=sender_id,
            sender_ip=sender_ip,
            payload=payload,
            **kwargs
        )
        self.task_id = task_id or str(uuid.uuid4())[:12]
        self.task_type = task_type or TaskType.INFERENCE.value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = super().to_dict()
        data['task_id'] = self.task_id
        data['task_type'] = self.task_type
        return data


class ProtocolValidator:
    """
    协议验证器
    
    验证消息格式和内容的合法性
    """
    
    REQUIRED_FIELDS = ['msg_type', 'sender_id', 'sender_ip', 'timestamp']
    
    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证消息
        
        Returns:
            (是否有效, 错误信息)
        """
        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        msg_type = data.get('msg_type')
        if msg_type not in [t.value for t in MessageType]:
            return False, f"Unknown message type: {msg_type}"
        
        timestamp = data.get('timestamp')
        if not isinstance(timestamp, (int, float)):
            return False, "Invalid timestamp"
        
        if time.time() - timestamp > 300:
            return False, "Message expired"
        
        return True, ""
    
    @classmethod
    def validate_task(cls, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证任务消息
        
        Returns:
            (是否有效, 错误信息)
        """
        valid, error = cls.validate(data)
        if not valid:
            return valid, error
        
        if 'task_id' not in data:
            return False, "Missing task_id"
        
        if 'task_type' not in data:
            return False, "Missing task_type"
        
        task_type = data.get('task_type')
        if task_type not in [t.value for t in TaskType]:
            return False, f"Unknown task type: {task_type}"
        
        return True, ""
