"""
统一的配置管理器，避免在多个文件中重复配置加载代码
使用单例模式确保全局唯一实例
"""
import os
import tomllib
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI


class ConfigManager:
    """
    配置管理器单例类
    负责加载配置、初始化客户端、提供统一的配置访问接口和路径管理
    """
    _instance: Optional['ConfigManager'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # 加载环境变量
        load_dotenv()
        
        # 确定配置文件路径
        self.config_path = Path(__file__).parent / "config.toml"
        if not self.config_path.exists():
            # 如果不在当前目录，尝试当前工作目录
            self.config_path = Path("config.toml")
        
        # 加载完整配置
        with open(self.config_path, "rb") as f:
            self._full_config = tomllib.load(f)
        
        # 加载各section配置
        self._base_config = self._load_section("local_gpt")
        self._embedding_config = self._load_section("embedding")
        self._reranker_config = self._load_section("reranker")
        
        # 加载路径配置
        self._paths_config = self._full_config.get("paths", {})
        
        # 初始化客户端（延迟初始化，避免导入时的问题）
        self._embed_client: Optional[OpenAI] = None
        self._external_client: Optional[OpenAI] = None
        self._async_external_client: Optional[AsyncOpenAI] = None
        
        # 常量
        self.EMBED_MODEL = self._embedding_config["model_name"]
        self.RERANKER_MODEL = "Qwen3-Reranker-8B"
        self.RERANKER_URL = self._reranker_config.get("api_base")
        self.MODEL_NAME = self._base_config["model_name"]
        self.REASONING_EFFORT = self._base_config.get("reasoning_effort")
        self.REASONING_VERBOSITY = self._base_config.get("reasoning_verbosity")
        
        # 当前目录（config_manager.py 所在目录）
        self.base_dir = Path(__file__).parent
        
        # 存储路径常量（从配置读取，如果没有则使用默认值，直接使用当前目录）
        self.ESMO_STORAGE = self.base_dir / self._paths_config.get("esmo_storage", "ESMO")
        self.NCCN_STORAGE = self.base_dir / self._paths_config.get("nccn_storage", "NCCN")
        self.HEMA_STORAGE = self.base_dir / self._paths_config.get("hema_storage", "HEMA")
        self.ESMO_DB_STORAGE = self.base_dir / self._paths_config.get("esmo_db_storage", "ESMO_chroma_db_qwen")
        self.NCCN_DB_STORAGE = self.base_dir / self._paths_config.get("nccn_db_storage", "NCCN_chroma_db_qwen")
        self.HEMA_DB_STORAGE = self.base_dir / self._paths_config.get("hema_db_storage", "HEMA_chroma_db_qwen")
        self.PATHO_DB_STORAGE = self.base_dir / self._paths_config.get("patho_db_storage", "patho_chroma_db_qwen")
        self.WHO_DB_STORAGE = self.base_dir / self._paths_config.get("who_db_storage", "who_chroma_db_qwen")
        self.UICC_DB_STORAGE = self.base_dir / self._paths_config.get("uicc_db_storage", "uicc_chroma_db_qwen0717")
        self.UICC_DIR = self.base_dir / self._paths_config.get("uicc_dir", "UICC/TRANS0717")
        self.WHO_DIR = self.base_dir / self._paths_config.get("who_dir", "WHOchap")
        self.PATHO_DIR = self.base_dir / self._paths_config.get("patho_dir", "patho")
        self.XML2TXT_ESMO_DIR = self.base_dir / self._paths_config.get("xml2txt_esmo_dir", "xml2txt_outputv2esmo")
        self.XML2TXT_HEMAGUIDE_DIR = self.base_dir / self._paths_config.get("xml2txt_hemaguide_dir", "xml2txt_hemaguidev2")
        self.FIGURE_TXT_DIR = self.base_dir / self._paths_config.get("figure_txt_dir", "figure_txtv2")
        self.PUBMED_ISSN_FILE = self.base_dir / self._paths_config.get("pubmed_issn", "pubmed/issn_new.txt")
        
        self._initialized = True

    def _load_section(self, section_name: str) -> Dict[str, Any]:
        """加载配置文件的某个section，并注入环境变量"""
        if section_name not in self._full_config:
            raise ValueError(f"Section '{section_name}' not found in config.toml")
        
        config = self._full_config[section_name].copy()
        prefix = config.get("env_prefix", "LOCAL")
        
        # 从环境变量注入 API 配置
        config["api_base"] = os.getenv(f"{prefix}_API_BASE")
        config["api_key"] = os.getenv(f"{prefix}_API_KEY", "EMPTY")
        
        return config

    @property
    def embed_client(self) -> OpenAI:
        """获取 embedding 客户端（懒加载）"""
        if self._embed_client is None:
            self._embed_client = OpenAI(
                api_key=self._embedding_config["api_key"],
                base_url=self._embedding_config.get("api_base"),
            )
        return self._embed_client

    @property
    def external_client(self) -> OpenAI:
        """获取外部 LLM 客户端（同步，懒加载）"""
        if self._external_client is None:
            self._external_client = OpenAI(
                api_key=self._base_config["api_key"],
                base_url=self._base_config.get("api_base"),
                timeout=self._base_config.get("timeout", 300),
            )
        return self._external_client

    @property
    def async_external_client(self) -> AsyncOpenAI:
        """获取外部 LLM 客户端（异步，懒加载）"""
        if self._async_external_client is None:
            self._async_external_client = AsyncOpenAI(
                api_key=self._base_config["api_key"],
                base_url=self._base_config.get("api_base"),
                timeout=self._base_config.get("timeout", 300),
            )
        return self._async_external_client

    def get_config(self, section_name: str) -> Dict[str, Any]:
        """获取指定section的配置"""
        return self._load_section(section_name)

    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._full_config.copy()

    def get_path(self, path_key: str) -> Path:
        """获取配置的路径，如果不存在则返回 None"""
        path_str = self._paths_config.get(path_key)
        if path_str:
            # 如果是绝对路径，直接返回；否则相对于 base_dir
            path = Path(path_str)
            if path.is_absolute():
                return path
            return self.base_dir / path_str
        return None

    # 便捷属性访问（注意：必须定义在 ConfigManager 类内）
    @property
    def base_config(self) -> Dict[str, Any]:
        """获取基础配置"""
        return self._base_config.copy()

    @property
    def embedding_config(self) -> Dict[str, Any]:
        """获取 embedding 配置"""
        return self._embedding_config.copy()

    @property
    def reranker_config(self) -> Dict[str, Any]:
        """获取 reranker 配置"""
        return self._reranker_config.copy()


def get_env_var(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    统一读取环境变量，支持必填校验。
    required=True 时，当值缺失或为空会抛出 EnvironmentError。
    """
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


# 创建全局单例实例
config_manager = ConfigManager()

# 导出常用对象，保持向后兼容
embed_client = config_manager.embed_client
external_client = config_manager.external_client
async_external_client = config_manager.async_external_client
EMBED_MODEL = config_manager.EMBED_MODEL
RERANKER_MODEL = config_manager.RERANKER_MODEL
RERANKER_URL = config_manager.RERANKER_URL
MODEL_NAME = config_manager.MODEL_NAME
REASONING_EFFORT = config_manager.REASONING_EFFORT
REASONING_VERBOSITY = config_manager.REASONING_VERBOSITY

# 存储路径常量
ESMO_STORAGE = config_manager.ESMO_STORAGE
NCCN_STORAGE = config_manager.NCCN_STORAGE
HEMA_STORAGE = config_manager.HEMA_STORAGE
ESMO_DB_STORAGE = config_manager.ESMO_DB_STORAGE
NCCN_DB_STORAGE = config_manager.NCCN_DB_STORAGE
HEMA_DB_STORAGE = config_manager.HEMA_DB_STORAGE
UICC_DIR = config_manager.UICC_DIR
WHO_DIR = config_manager.WHO_DIR
PATHO_DIR = config_manager.PATHO_DIR
XML2TXT_ESMO_DIR = config_manager.XML2TXT_ESMO_DIR
XML2TXT_HEMAGUIDE_DIR = config_manager.XML2TXT_HEMAGUIDE_DIR
FIGURE_TXT_DIR = config_manager.FIGURE_TXT_DIR
PATHO_DB_STORAGE = config_manager.PATHO_DB_STORAGE
WHO_DB_STORAGE = config_manager.WHO_DB_STORAGE
UICC_DB_STORAGE = config_manager.UICC_DB_STORAGE
PUBMED_ISSN_FILE = config_manager.PUBMED_ISSN_FILE

