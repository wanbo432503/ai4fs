from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import config

def init_embeddings():
    """获取嵌入模型实例"""
    if config.USE_CUSTOM_EMBEDDINGS:
        try:
            return OllamaEmbeddings(
                base_url=config.EMBEDDING_MODEL_API_BASE,
                model=config.EMBEDDING_MODEL
            )
        except Exception as e:
            print(f"Failed to initialize Ollama embeddings: {str(e)}")
            # 如果 Ollama 初始化失败，回退到 OpenAI
            return init_openai_embeddings()
    else:
        return init_openai_embeddings()

def init_openai_embeddings():
    """初始化 OpenAI 嵌入模型"""
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_base=config.EMBEDDING_MODEL_API_BASE,
        openai_api_key=config.EMBEDDING_MODEL_API_KEY
    )

def init_vector_store(embeddings):
    """初始化向量存储"""
    persist_dir = config.VECTOR_STORE_PATH
    
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name="chat_history",
        collection_metadata={"hnsw:space": "cosine"}
    )

def init_llm():
    """初始化语言模型"""
    if config.USE_CUSTOM_MODEL:
        return ChatOpenAI(
            model_name=config.CUSTOM_MODEL_NAME,
            openai_api_base=config.CUSTOM_MODEL_API_BASE,
            openai_api_key=config.CUSTOM_MODEL_API_KEY,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
    else:
        return ChatOpenAI(
            model_name=config.OPENAI_MODEL_NAME,
            openai_api_base=config.OPENAI_API_BASE,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
