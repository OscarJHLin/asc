"""
推理引擎模块

封装 llama.cpp 的推理功能，支持本地和分布式推理
"""

import os
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any
from ..core.config import Config


class InferenceEngine:
    """
    推理引擎类
    
    管理模型加载和推理任务执行
    """
    
    def __init__(self, config: Config = None):
        """
        初始化推理引擎
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.backend = self.config.get('inference', 'backend', 'llama.cpp')
        
        # 查找 llama.cpp 可执行文件
        self.llama_cli = self._find_llama_cli()
        self.llama_server = self._find_llama_server()
        
        self.current_model: Optional[str] = None
        self.model_info: Optional[Dict] = None
        
        self._server_process: Optional[subprocess.Popen] = None
    
    def _find_llama_cli(self) -> Optional[str]:
        """查找 llama-cli 可执行文件"""
        script_dir = Path(__file__).parent.parent.parent.resolve()
        # 项目内路径
        candidates = [
            script_dir / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe",
            script_dir / "llama.cpp" / "build" / "bin" / "llama-cli",
            script_dir / "llama.cpp" / "build-linux" / "bin" / "llama-cli",
        ]
        
        # 父目录路径（开发环境）
        parent_dir = script_dir.parent
        candidates.extend([
            parent_dir / "exo" / "llama.cpp" / "build" / "bin" / "Release" / "llama-cli.exe",
            parent_dir / "exo" / "llama.cpp" / "build" / "bin" / "llama-cli",
            parent_dir / "exo" / "llama.cpp" / "build-linux" / "bin" / "llama-cli",
        ])
        
        for path in candidates:
            if path.exists():
                return str(path.resolve())
        return None
    
    def _find_llama_server(self) -> Optional[str]:
        """查找 llama-server 可执行文件"""
        script_dir = Path(__file__).parent.parent.parent.resolve()
        # 项目内路径
        candidates = [
            script_dir / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            script_dir / "llama.cpp" / "build" / "bin" / "llama-server",
            script_dir / "llama.cpp" / "build-linux" / "bin" / "llama-server",
        ]
        
        # 父目录路径（开发环境）
        parent_dir = script_dir.parent
        candidates.extend([
            parent_dir / "exo" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
            parent_dir / "exo" / "llama.cpp" / "build" / "bin" / "llama-server",
            parent_dir / "exo" / "llama.cpp" / "build-linux" / "bin" / "llama-server",
        ])
        
        for path in candidates:
            if path.exists():
                return str(path.resolve())
        return None
    
    def load_model(self, model_path: str) -> bool:
        """
        加载模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        if not os.path.exists(model_path):
            print(f"[Engine] 模型文件不存在: {model_path}")
            return False
        
        self.current_model = model_path
        
        # 获取模型信息
        self.model_info = self._get_model_info(model_path)
        
        print(f"[Engine] 模型已加载: {Path(model_path).name}")
        if self.model_info:
            print(f"[Engine] 模型信息: {self.model_info}")
        
        return True
    
    def _get_model_info(self, model_path: str) -> Optional[Dict]:
        """获取模型信息"""
        try:
            # 使用 llama-gguf 工具获取信息
            script_dir = Path(__file__).parent.parent.parent.resolve()
            gguf_tool = script_dir / "llama.cpp" / "build" / "bin" / "Release" / "llama-gguf.exe"
            
            if not gguf_tool.exists():
                return None
            
            result = subprocess.run(
                [str(gguf_tool), "dump", model_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # 解析输出
                info = {}
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info[key.strip()] = value.strip()
                return info
            
        except Exception as e:
            print(f"[Engine] 获取模型信息失败: {e}")
        
        return None
    
    def infer(self, 
              prompt: str,
              max_tokens: int = None,
              temperature: float = None,
              top_p: float = None,
              stream: bool = False) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            prompt: 提示词
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            stream: 是否流式输出
            
        Returns:
            推理结果字典
        """
        if not self.current_model:
            return {'error': '未加载模型'}
        
        if not self.llama_cli:
            return {'error': '未找到 llama-cli'}
        
        # 使用默认配置
        max_tokens = max_tokens or self.config.get('inference', 'max_tokens', 128)
        temperature = temperature or self.config.get('inference', 'temperature', 0.7)
        top_p = top_p or self.config.get('inference', 'top_p', 0.9)
        
        # 构建命令
        cmd = [
            self.llama_cli,
            '-m', self.current_model,
            '-p', prompt,
            '-n', str(max_tokens),
            '--temp', str(temperature),
            '--top-p', str(top_p),
            '--no-display-prompt',
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                output = result.stdout.strip()
                
                # 估算token数
                tokens_generated = len(output.split())
                
                return {
                    'success': True,
                    'output': output,
                    'tokens_generated': tokens_generated,
                    'elapsed_time': elapsed,
                    'tokens_per_sec': tokens_generated / elapsed if elapsed > 0 else 0,
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr,
                    'elapsed_time': elapsed,
                }
        
        except subprocess.TimeoutExpired:
            return {'error': '推理超时'}
        except Exception as e:
            return {'error': str(e)}
    
    def infer_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        流式推理
        
        Args:
            prompt: 提示词
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        # 流式输出需要启动 llama-server
        # 这里简化处理，直接返回完整结果
        result = self.infer(prompt, **kwargs)
        
        if result.get('success'):
            yield result['output']
        else:
            yield f"Error: {result.get('error', 'Unknown error')}"
    
    def start_server(self, port: int = 8080) -> bool:
        """
        启动推理服务器
        
        Args:
            port: 服务器端口
            
        Returns:
            是否启动成功
        """
        if not self.current_model:
            print("[Engine] 请先加载模型")
            return False
        
        if not self.llama_server:
            print("[Engine] 未找到 llama-server")
            return False
        
        if self._server_process:
            print("[Engine] 服务器已在运行")
            return True
        
        cmd = [
            self.llama_server,
            '-m', self.current_model,
            '--port', str(port),
        ]
        
        try:
            self._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待服务器启动
            time.sleep(2)
            
            if self._server_process.poll() is None:
                print(f"[Engine] 推理服务器已启动 (端口: {port})")
                return True
            else:
                print("[Engine] 服务器启动失败")
                return False
        
        except Exception as e:
            print(f"[Engine] 启动服务器失败: {e}")
            return False
    
    def stop_server(self) -> None:
        """停止推理服务器"""
        if self._server_process:
            self._server_process.terminate()
            self._server_process.wait(timeout=5)
            self._server_process = None
            print("[Engine] 推理服务器已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取引擎状态
        
        Returns:
            状态信息字典
        """
        return {
            'backend': self.backend,
            'model_loaded': self.current_model is not None,
            'model_path': self.current_model,
            'model_info': self.model_info,
            'llama_cli_available': self.llama_cli is not None,
            'llama_server_available': self.llama_server is not None,
            'server_running': self._server_process is not None,
        }
    
    def __del__(self):
        """析构时停止服务器"""
        self.stop_server()
