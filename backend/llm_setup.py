from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from config import config

def init_embeddings():
    """获取嵌入模型实例"""
    if config.USE_CUSTOM_EMBEDDINGS:
        return OllamaEmbeddings(
            base_url=config.EMBEDDING_MODEL_API_BASE,
            model=config.EMBEDDING_MODEL
        )
    else:
        return OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            base_url=config.EMBEDDING_MODEL_API_BASE,
            api_key=config.EMBEDDING_MODEL_API_KEY
        )

def init_vector_store(embeddings):
    persist_dir = config.VECTOR_STORE_PATH
    
    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name="chat_history",
        collection_metadata={"hnsw:space": "cosine"}
    )

def init_llm():
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
