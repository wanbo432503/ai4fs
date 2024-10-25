import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    # OpenAI API配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

    # 自定义大模型服务配置
    CUSTOM_MODEL_API_KEY = os.getenv("CUSTOM_MODEL_API_KEY")
    CUSTOM_MODEL_API_BASE = os.getenv("CUSTOM_MODEL_API_BASE")
    CUSTOM_MODEL_NAME = os.getenv("CUSTOM_MODEL_NAME")

    # 模型选择
    USE_CUSTOM_MODEL = os.getenv("USE_CUSTOM_MODEL", "false").lower() == "true"

    # 其他可选配置
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))

    # 文件上传路径
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")

    # 向量存储路径
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./chroma_db")

config = Config()
