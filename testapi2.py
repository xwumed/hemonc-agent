"""
快速API测试脚本 - 验证LLM API连接是否正常

使用方法：
  python testapi2.py                    # 使用默认模型 (local_llama)
  python testapi2.py --config openai_gpt4o   # 使用 OpenAI GPT-4o
  python testapi2.py --config openai_gpt5_1  # 使用 OpenAI GPT-5.1
  python testapi2.py --config local_gpt      # 使用本地 GPT-OSS
  python testapi2.py --list             # 列出所有可用配置
"""

import sys
import time
import tomllib
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def load_all_configs():
    """加载所有模型配置"""
    config_path = Path("config.toml")
    with open(config_path, "rb") as f:
        full_config = tomllib.load(f)
    
    # 过滤出模型配置（排除 embedding, reranker, paths 等）
    model_configs = {
        name: cfg for name, cfg in full_config.items()
        if name not in ['embedding', 'reranker', 'paths'] and isinstance(cfg, dict) and 'model_name' in cfg
    }
    return model_configs

def get_client_for_config(config_name='local_llama'):
    """根据配置名称获取客户端和模型信息"""
    configs = load_all_configs()
    
    if config_name not in configs:
        print(f"❌ 配置 '{config_name}' 不存在")
        print(f"可用配置: {', '.join(configs.keys())}")
        sys.exit(1)
    
    config = configs[config_name]
    env_prefix = config.get('env_prefix', 'LOCAL')
    
    # 从环境变量获取 API 配置
    api_key = os.getenv(f"{env_prefix}_API_KEY", "EMPTY")
    api_base = os.getenv(f"{env_prefix}_API_BASE")
    
    if not api_base:
        print(f"❌ 未设置 {env_prefix}_API_BASE 环境变量")
        print(f"请在 .env 文件中设置相应的环境变量")
        sys.exit(1)
    
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=config.get('timeout', 300)
    )
    
    return client, config['model_name'], api_base

def list_available_configs():
    """列出所有可用的配置"""
    configs = load_all_configs()
    
    print("\n📋 可用的模型配置:")
    print("=" * 70)
    for name, cfg in configs.items():
        model_name = cfg.get('model_name', 'N/A')
        env_prefix = cfg.get('env_prefix', 'N/A')
        print(f"  {name:20s} -> {model_name:40s} [{env_prefix}]")
    print("=" * 70)
    print("\n使用方法: python testapi2.py --config <配置名>\n")

def quick_test(config_name='local_llama'):
    """快速测试API连接"""
    try:
        # 获取客户端和配置信息
        client, model_name, api_base = get_client_for_config(config_name)
        
        print(f"\n🔍 测试API连接...")
        print(f"   配置: {config_name}")
        print(f"   模型: {model_name}")
        print(f"   API: {api_base}\n")
        
        # 发送测试请求
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say 'API is working' in one sentence."}
            ],
            max_tokens=50
        )
        elapsed = time.perf_counter() - start_time
        
        # 显示结果
        content = response.choices[0].message.content
        print(f"✅ API连接正常!")
        print(f"   耗时: {elapsed:.2f}秒")
        print(f"   回复: {content}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API连接失败: {e}\n")
        print(f"💡 请检查:")
        print(f"   1. API服务是否启动")
        print(f"   2. config.toml 中的 [{config_name}] 配置是否正确")
        print(f"   3. .env 文件中的相应环境变量是否设置")
        print(f"      (需要设置 {get_env_prefix(config_name)}_API_BASE 和 {get_env_prefix(config_name)}_API_KEY)\n")
        return False

def get_env_prefix(config_name):
    """获取配置的环境变量前缀"""
    try:
        configs = load_all_configs()
        return configs.get(config_name, {}).get('env_prefix', 'LOCAL')
    except:
        return 'LOCAL'

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_available_configs()
            sys.exit(0)
        elif sys.argv[1] == "--config" and len(sys.argv) > 2:
            config_name = sys.argv[2]
            quick_test(config_name)
        elif sys.argv[1] == "--help":
            print(__doc__)
            sys.exit(0)
        else:
            print("用法错误！")
            print(__doc__)
            sys.exit(1)
    else:
        # 默认使用 local_llama
        quick_test('local_llama')
