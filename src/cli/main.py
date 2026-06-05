"""
命令行接口模块

提供简洁的命令行操作接口
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.core.cluster import Cluster
from src.core.config import Config
from src.core.logging import get_logger
from src.core.node import Node
from src.inference.engine import InferenceEngine
from src.network.discovery import NodeDiscovery
from src.ui.server import WebUIServer

logger = get_logger('cli')


class ASCCLI:
    """
    ASC 命令行接口
    
    提供一键启动、节点管理、推理任务等功能
    """
    
    def __init__(self):
        self.config = Config()
        self.cluster = Cluster(self.config)
        self.engine = InferenceEngine(self.config)
        self.discovery = None
        self.ui = None
    
    def run(self):
        """运行 CLI"""
        parser = argparse.ArgumentParser(
            description='ASC (All System Cluster) - 分布式AI推理框架',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  %(prog)s                          # 启动节点并自动发现
  %(prog)s start --mode master      # 启动主节点
  %(prog)s start --mode worker --master-ip 192.168.1.100
  %(prog)s infer --model model.gguf --prompt "Hello"
  %(prog)s status                   # 查看节点状态
  %(prog)s discover                 # 发现网络节点
            """
        )
        
        parser.add_argument('--version', action='version', version='ASC 1.0.0')
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # start 命令
        start_parser = subparsers.add_parser('start', help='启动节点')
        start_parser.add_argument('--mode', choices=['auto', 'master', 'worker'], 
                               default='auto', help='节点模式')
        start_parser.add_argument('--master-ip', help='主节点IP (worker模式)')
        start_parser.add_argument('--port', type=int, help='节点端口')
        start_parser.add_argument('--rpc-port', type=int, help='RPC端口')
        start_parser.add_argument('--config', help='配置文件路径')
        start_parser.add_argument('--no-ui', action='store_true', help='禁用Web UI')
        start_parser.add_argument('--no-discover', action='store_true', help='禁用自动发现')
        
        # infer 命令
        infer_parser = subparsers.add_parser('infer', help='运行推理')
        infer_parser.add_argument('--model', required=True, help='模型路径')
        infer_parser.add_argument('--prompt', required=True, help='提示词')
        infer_parser.add_argument('--max-tokens', type=int, default=128, help='最大token数')
        infer_parser.add_argument('--temperature', type=float, default=0.7, help='温度')
        
        # status 命令
        subparsers.add_parser('status', help='查看状态')
        
        # discover 命令
        subparsers.add_parser('discover', help='发现节点')
        
        args = parser.parse_args()
        
        if hasattr(args, 'config') and args.config:
            self.config = Config(args.config)
        
        if hasattr(args, 'port') and args.port:
            self.config.set('node', 'port', args.port)
        if hasattr(args, 'rpc_port') and args.rpc_port:
            self.config.set('node', 'rpc_port', args.rpc_port)
        
        if args.command == 'start' or args.command is None:
            self._cmd_start(args)
        elif args.command == 'infer':
            self._cmd_infer(args)
        elif args.command == 'status':
            self._cmd_status()
        elif args.command == 'discover':
            self._cmd_discover()
        else:
            parser.print_help()
    
    def _cmd_start(self, args):
        """启动节点"""
        print("=" * 50)
        print("  ASC (All System Cluster)")
        print("  分布式AI推理框架 v1.0.0")
        print("=" * 50)
        
        self.cluster.start()
        
        no_discover = getattr(args, 'no_discover', False)
        if not no_discover:
            self.discovery = NodeDiscovery(
                self.config,
                on_node_found=self._on_node_found
            )
            self.discovery.start()
        
        no_ui = getattr(args, 'no_ui', False)
        if not no_ui:
            self.ui = WebUIServer(self.cluster, self.config, self.engine)
            self.ui.start()
        
        mode = getattr(args, 'mode', 'auto')
        master_ip = getattr(args, 'master_ip', None)
        
        if mode == 'worker' and master_ip:
            logger.info(f"工作节点模式，连接主节点: {master_ip}")
        else:
            logger.info(f"节点已启动: {self.cluster.local_node}")
        
        print("\n按 Ctrl+C 停止服务")
        
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在停止服务...")
            self._shutdown()
    
    def _cmd_infer(self, args):
        """运行推理"""
        print("=" * 50)
        print("  ASC - 推理任务")
        print("=" * 50)
        
        if not self.engine.load_model(args.model):
            print("[ERROR] 模型加载失败")
            sys.exit(1)
        
        logger.info(f"提示词: {args.prompt}")
        logger.info("正在推理...")
        
        result = self.engine.infer(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        if result.get('success'):
            print("\n推理结果:")
            output = result['output']
            try:
                print(output)
            except UnicodeEncodeError:
                encoding = sys.stdout.encoding or 'utf-8'
                print(output.encode(encoding, errors='replace').decode(encoding))
            
            print("\n统计:")
            print(f"  生成token数: {result.get('tokens_generated', 0)}")
            print(f"  耗时: {result.get('elapsed_time', 0):.2f}s")
            print(f"  速度: {result.get('tokens_per_sec', 0):.2f} tokens/s")
            if result.get('prompt_tps', 0) > 0:
                print(f"  Prompt速度: {result.get('prompt_tps', 0):.2f} tokens/s")
        else:
            print(f"[ERROR] 推理失败: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    def _cmd_status(self):
        """查看状态"""
        print("=" * 50)
        print("  ASC - 节点状态")
        print("=" * 50)
        
        info = self.cluster.get_cluster_info()
        
        print(f"节点总数: {info['node_count']}")
        print(f"在线节点: {info['online_count']}")
        print(f"总CPU核心: {info['total_cpu_cores']}")
        print(f"总内存: {info['total_memory_mb']} MB")
        
        engine_status = self.engine.get_status()
        print("\n推理引擎:")
        print(f"  后端: {engine_status['backend']}")
        print(f"  llama-cli: {'可用' if engine_status['llama_cli_available'] else '不可用'}")
        print(f"  llama-server: {'可用' if engine_status['llama_server_available'] else '不可用'}")
        print(f"  模型已加载: {'是' if engine_status['model_loaded'] else '否'}")
        
        print("\n节点列表:")
        print("-" * 50)
        for node_info in info['nodes']:
            print(f"  {node_info['name']} ({node_info['ip']})")
            print(f"    状态: {node_info['status']} | 平台: {node_info['platform']}")
            print(f"    CPU: {node_info['resources']['cpu_percent']:.1f}% | 内存: {node_info['resources']['memory_percent']:.1f}%")
            print()
    
    def _cmd_discover(self):
        """发现节点"""
        print("=" * 50)
        print("  ASC - 节点发现")
        print("=" * 50)
        
        discovery = NodeDiscovery(self.config)
        nodes = discovery.discover_nodes()
        
        print(f"发现 {len(nodes)} 个节点:")
        for node in nodes:
            print(f"  - {node.name} @ {node.ip}:{node.port}")
    
    def _on_node_found(self, node: Node):
        """发现新节点回调"""
        logger.info(f"发现新节点: {node}")
        self.cluster.add_node(node)
    
    def _shutdown(self):
        """关闭服务"""
        if self.ui:
            self.ui.stop()
        if self.discovery:
            self.discovery.stop()
        self.cluster.stop()
        logger.info("服务已停止")


def main():
    cli = ASCCLI()
    cli.run()


if __name__ == '__main__':
    main()
