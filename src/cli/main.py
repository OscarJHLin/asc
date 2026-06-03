"""
命令行接口模块

提供简洁的命令行操作接口
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.core.cluster import Cluster
from src.core.node import Node, NodeStatus
from src.network.discovery import NodeDiscovery
from src.inference.engine import InferenceEngine
from src.ui.server import WebUIServer


class ASCCLI:
    """
    ASC 命令行接口类
    
    提供一键启动、节点管理、推理任务等功能
    """
    
    def __init__(self):
        self.config = Config()
        self.cluster = Cluster(self.config)
        self.discovery = None
        self.engine = InferenceEngine(self.config)
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
  %(prog)s start --mode worker --master-ip 192.168.1.100  # 启动工作节点
  %(prog)s infer --model model.gguf --prompt "Hello"  # 运行推理
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
        infer_parser.add_argument('--cluster', action='store_true', help='使用集群推理')
        
        # status 命令
        status_parser = subparsers.add_parser('status', help='查看状态')
        
        # discover 命令
        discover_parser = subparsers.add_parser('discover', help='发现节点')
        
        args = parser.parse_args()
        
        # 加载配置
        if args.config:
            self.config = Config(args.config)
        
        # 应用命令行参数
        if hasattr(args, 'port') and args.port:
            self.config.set('node', 'port', args.port)
        if hasattr(args, 'rpc_port') and args.rpc_port:
            self.config.set('node', 'rpc_port', args.rpc_port)
        
        # 执行命令
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
        print("=" * 60)
        print("ASC (All System Cluster) - 分布式AI推理框架")
        print("=" * 60)
        
        # 启动集群管理
        self.cluster.start()
        
        # 启动节点发现
        no_discover = getattr(args, 'no_discover', False)
        if not no_discover:
            self.discovery = NodeDiscovery(
                self.config,
                on_node_found=self._on_node_found
            )
            self.discovery.start()
        
        # 启动 Web UI
        no_ui = getattr(args, 'no_ui', False)
        if not no_ui:
            self.ui = WebUIServer(self.cluster, self.config)
            self.ui.start()
        
        # 根据模式处理
        mode = getattr(args, 'mode', 'auto')
        master_ip = getattr(args, 'master_ip', None)
        
        if mode == 'worker' and master_ip:
            print(f"[CLI] 工作节点模式，连接主节点: {master_ip}")
        else:
            print(f"[CLI] 节点已启动")
            print(f"[CLI] 节点信息: {self.cluster.local_node}")
        
        print("\n按 Ctrl+C 停止服务")
        
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[CLI] 正在停止服务...")
            self._shutdown()
    
    def _cmd_infer(self, args):
        """运行推理"""
        print("=" * 60)
        print("ASC - 推理任务")
        print("=" * 60)
        
        # 加载模型
        if not self.engine.load_model(args.model):
            print("[CLI] 模型加载失败")
            return
        
        print(f"[CLI] 提示词: {args.prompt}")
        print(f"[CLI] 正在推理...")
        
        # 执行推理
        result = self.engine.infer(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        if result.get('success'):
            print(f"\n[CLI] 推理结果:")
            # 处理 Windows 终端编码问题
            output = result['output']
            try:
                print(output)
            except UnicodeEncodeError:
                # 过滤掉无法编码的字符
                import sys
                encoding = sys.stdout.encoding or 'utf-8'
                safe_output = output.encode(encoding, errors='replace').decode(encoding)
                print(safe_output)
            print(f"\n[CLI] 统计:")
            print(f"  生成token数: {result.get('tokens_generated', 0)}")
            print(f"  耗时: {result.get('elapsed_time', 0):.2f}s")
            print(f"  速度: {result.get('tokens_per_sec', 0):.2f} tokens/s")
        else:
            print(f"[CLI] 推理失败: {result.get('error', 'Unknown error')}")
    
    def _cmd_status(self):
        """查看状态"""
        print("=" * 60)
        print("ASC - 节点状态")
        print("=" * 60)
        
        info = self.cluster.get_cluster_info()
        
        print(f"节点总数: {info['node_count']}")
        print(f"在线节点: {info['online_count']}")
        print(f"总CPU核心: {info['total_cpu_cores']}")
        print(f"总内存: {info['total_memory_mb']} MB")
        
        print("\n节点列表:")
        print("-" * 60)
        for node_info in info['nodes']:
            print(f"  {node_info['name']} ({node_info['ip']})")
            print(f"    状态: {node_info['status']}")
            print(f"    平台: {node_info['platform']}")
            print(f"    CPU: {node_info['resources']['cpu_percent']:.1f}%")
            print(f"    内存: {node_info['resources']['memory_percent']:.1f}%")
            print()
    
    def _cmd_discover(self):
        """发现节点"""
        print("=" * 60)
        print("ASC - 节点发现")
        print("=" * 60)
        
        discovery = NodeDiscovery(self.config)
        nodes = discovery.discover_nodes()
        
        print(f"发现 {len(nodes)} 个节点:")
        for node in nodes:
            print(f"  - {node.name} @ {node.ip}:{node.port}")
    
    def _on_node_found(self, node: Node):
        """发现新节点回调"""
        print(f"[CLI] 发现新节点: {node}")
        self.cluster.add_node(node)
    
    def _shutdown(self):
        """关闭服务"""
        if self.ui:
            self.ui.stop()
        if self.discovery:
            self.discovery.stop()
        self.cluster.stop()
        print("[CLI] 服务已停止")


def main():
    """主入口"""
    cli = ASCCLI()
    cli.run()


if __name__ == '__main__':
    main()
