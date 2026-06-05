"""
推理引擎模块

封装 llama.cpp 的推理功能，支持本地和分布式推理
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from ..core.config import Config
from ..core.logging import get_logger

logger = get_logger('engine')


class InferenceEngine:
    """
    推理引擎类
    
    管理模型加载和推理任务执行，通过 llama.cpp 命令行接口调用
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.backend = self.config.get('inference', 'backend', 'llama.cpp')
        
        # 查找可执行文件
        self.llama_cli = self._find_executable('llama-cli')
        self.llama_server = self._find_executable('llama-server')
        
        self.current_model: Optional[str] = None
        self.model_info: Optional[Dict] = None
        self._server_process: Optional[subprocess.Popen] = None
        
        if self.llama_cli:
            logger.info(f"找到推理引擎: {self.llama_cli}")
        else:
            logger.warning("未找到 llama-cli，推理功能不可用")
    
    def _find_executable(self, name: str) -> Optional[str]:
        """
        查找可执行文件
        
        搜索顺序：
        1. 项目内的 llama.cpp 目录
        2. 环境变量 PATH
        3. 常见安装路径
        """
        project_root = Path(__file__).parent.parent.parent.resolve()
        
        # 项目内路径
        candidates = [
            project_root / "llama.cpp" / "build" / "bin" / "Release" / f"{name}.exe",
            project_root / "llama.cpp" / "build" / "bin" / name,
            project_root / "llama.cpp" / "build-linux" / "bin" / name,
        ]
        
        # 用户自定义路径（通过环境变量）
        custom_path = os.getenv('ASC_LLAMA_PATH')
        if custom_path:
            candidates.insert(0, Path(custom_path) / f"{name}.exe")
            candidates.insert(1, Path(custom_path) / name)
        
        for path in candidates:
            if path.exists():
                return str(path.resolve())
        
        # 检查系统 PATH
        import shutil
        found = shutil.which(name)
        if found:
            return found
        
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
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        self.current_model = os.path.abspath(model_path)
        self.model_info = self._get_model_info(model_path)
        
        logger.info(f"模型已加载: {Path(model_path).name}")
        return True
    
    def _get_model_info(self, model_path: str) -> Optional[Dict]:
        """获取模型基本信息"""
        try:
            file_size = os.path.getsize(model_path)
            return {
                'path': model_path,
                'size_mb': file_size // (1024 * 1024),
                'format': 'GGUF',
            }
        except Exception as e:
            logger.warning(f"获取模型信息失败: {e}")
            return None
    
    def infer(self, 
              prompt: str,
              max_tokens: int = None,
              temperature: float = None,
              top_p: float = None) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            prompt: 提示词
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            
        Returns:
            推理结果字典
        """
        if not self.current_model:
            return {'success': False, 'error': '未加载模型'}
        
        if not self.llama_cli:
            return {'success': False, 'error': '未找到 llama-cli，请编译 llama.cpp 或设置 ASC_LLAMA_PATH'}
        
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
            '--single-turn',
        ]
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                output = result.stdout.strip()
                
                # 过滤 llama-cli 的 banner 输出
                # banner 以 "> prompt" 开头标记实际推理结果
                lines = output.split('\n')
                filtered_lines = []
                in_response = False
                for line in lines:
                    if line.startswith('> ') and not in_response:
                        in_response = True
                        continue  # 跳过 "> prompt" 行
                    if in_response:
                        filtered_lines.append(line)
                
                if filtered_lines:
                    output = '\n'.join(filtered_lines).strip()
                
                # 从输出中解析性能数据
                tokens_per_sec = 0.0
                tokens_generated = 0
                prompt_tps = 0.0
                
                for line in result.stderr.split('\n') if result.stderr else []:
                    if 'prompt eval' in line.lower() and 't/s' in line:
                        try:
                            prompt_tps = float(line.split('=')[-1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass
                    elif 'eval time' in line.lower() and 't/s' in line:
                        try:
                            tokens_per_sec = float(line.split('=')[-1].strip().split()[0])
                            tokens_generated = int(line.split('/')[0].split()[-1])
                        except (ValueError, IndexError):
                            pass
                
                if tokens_generated == 0:
                    tokens_generated = len(output.split())
                    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
                
                return {
                    'success': True,
                    'output': output,
                    'tokens_generated': tokens_generated,
                    'elapsed_time': elapsed,
                    'tokens_per_sec': tokens_per_sec,
                    'prompt_tps': prompt_tps,
                }
            else:
                error_msg = result.stderr.strip() if result.stderr else 'Unknown error'
                return {
                    'success': False,
                    'error': error_msg,
                    'elapsed_time': elapsed,
                }
        
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': '推理超时（300秒）'}
        except FileNotFoundError:
            return {'success': False, 'error': f'无法执行: {self.llama_cli}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def infer_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        流式推理（简化实现）
        
        Args:
            prompt: 提示词
            
        Yields:
            生成的文本片段
        """
        result = self.infer(prompt, **kwargs)
        if result.get('success'):
            yield result['output']
        else:
            yield f"Error: {result.get('error', 'Unknown error')}"
    
    def start_server(self, port: int = None) -> bool:
        """
        启动推理服务器（llama-server）
        
        Args:
            port: 服务器端口
            
        Returns:
            是否启动成功
        """
        if not self.current_model:
            logger.error("请先加载模型")
            return False
        
        if not self.llama_server:
            logger.error("未找到 llama-server")
            return False
        
        if self._server_process:
            logger.info("服务器已在运行")
            return True
        
        port = port or self.config.get('inference', 'server_port', 8081)
        
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
            time.sleep(2)
            
            if self._server_process.poll() is None:
                logger.info(f"推理服务器已启动 (端口: {port})")
                return True
            else:
                logger.error("服务器启动失败")
                return False
        
        except Exception as e:
            logger.error(f"启动服务器失败: {e}")
            return False
    
    def stop_server(self) -> None:
        """停止推理服务器"""
        if self._server_process:
            self._server_process.terminate()
            self._server_process.wait(timeout=5)
            self._server_process = None
            logger.info("推理服务器已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
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
        self.stop_server()
