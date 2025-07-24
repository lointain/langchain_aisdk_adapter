#!/usr/bin/env python3
"""
后端启动脚本

这个脚本用于启动 FastAPI 后端服务。
它会自动检查依赖、设置环境变量，并启动服务器。
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """检查 Python 版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要 Python 3.8 或更高版本")
        print(f"当前版本: {sys.version}")
        sys.exit(1)
    print(f"✅ Python 版本: {sys.version.split()[0]}")

def check_dependencies():
    """检查依赖是否安装"""
    backend_dir = Path(__file__).parent / "backend"
    requirements_file = backend_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ 错误: requirements.txt 文件不存在")
        sys.exit(1)
    
    print("📦 检查依赖...")
    
    # 读取依赖列表
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    missing_deps = []
    for req in requirements:
        package_name = req.split('==')[0].split('>=')[0].split('<=')[0]
        try:
            __import__(package_name.replace('-', '_'))
        except ImportError:
            missing_deps.append(req)
    
    if missing_deps:
        print("❌ 缺少以下依赖:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\n请运行以下命令安装依赖:")
        print(f"   cd {backend_dir}")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ 所有依赖已安装")

def setup_environment():
    """设置环境变量"""
    print("🔧 设置环境变量...")
    
    # 设置默认环境变量（如果未设置）
    env_vars = {
        'OPENAI_API_KEY': 'your-openai-api-key-here',
        'LANGCHAIN_TRACING_V2': 'false',
        'LANGCHAIN_API_KEY': '',
        'PYTHONPATH': str(Path(__file__).parent / "backend")
    }
    
    for key, default_value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = default_value
            if key == 'OPENAI_API_KEY' and default_value == 'your-openai-api-key-here':
                print(f"⚠️  警告: 请设置 {key} 环境变量")
            else:
                print(f"✅ 设置 {key}")
        else:
            print(f"✅ {key} 已设置")

def start_server():
    """启动 FastAPI 服务器"""
    backend_dir = Path(__file__).parent / "backend"
    
    print("🚀 启动 FastAPI 服务器...")
    print(f"📁 工作目录: {backend_dir}")
    print("🌐 服务地址: http://localhost:8000")
    print("📖 API 文档: http://localhost:8000/docs")
    print("\n按 Ctrl+C 停止服务器\n")
    
    try:
        # 切换到后端目录
        os.chdir(backend_dir)
        
        # 启动 uvicorn 服务器
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动服务器失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        sys.exit(1)

def main():
    """主函数"""
    print("🔥 LangChain AI SDK Adapter - 后端启动器")
    print("=" * 50)
    
    try:
        check_python_version()
        check_dependencies()
        setup_environment()
        start_server()
    except KeyboardInterrupt:
        print("\n👋 启动已取消")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()