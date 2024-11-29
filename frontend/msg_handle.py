import chainlit as cl
from typing import Optional, Tuple
from backend.qa_chain import create_qa_chain, create_chat_chain
from backend.document_loader import process_uploaded_file
from config import config
from backend.chat_history import ChatHistoryManager
from backend.llm_setup import init_embeddings, init_vector_store, init_llm
import re
import requests
from bs4 import BeautifulSoup
import tempfile
import os

# 全局变量
class GlobalComponents:
    embeddings = None
    vector_store = None
    llm = None
    chat_history = None

    @classmethod
    def init(cls):
        """初始化所有全局组件"""
        cls.embeddings = init_embeddings()
        cls.vector_store = init_vector_store(cls.embeddings)
        cls.llm = init_llm()
        cls.chat_history = ChatHistoryManager(cls.vector_store)
        return cls.llm, cls.chat_history

class MessageProcessor:
    @staticmethod
    async def process_message(message: cl.Message, conversation_id: str) -> str:
        """处理用户消息的主入口"""
        if message.elements:
            return await FileHandler.handle_file_message(message, conversation_id)
        
        url = URLHandler.extract_url(message.content)
        if url:
            return await URLHandler.handle_url_message(message, conversation_id, url)
            
        return await MessageProcessor.handle_chat_message(message, conversation_id)

    @staticmethod
    async def handle_chat_message(message: cl.Message, conversation_id: str) -> str:
        """处理普通对话消息"""
        chain = create_chat_chain(GlobalComponents.llm)
        chat_history_text = GlobalComponents.chat_history.get_recent_messages(conversation_id)
        
        text_docs = GlobalComponents.vector_store.similarity_search(
            message.content,
            filter={"conversation_id": conversation_id},
            k=5
        )
        knowledge_text = "\n".join([doc.page_content for doc in text_docs]) if text_docs else ""
        
        inputs = {
            "inputs": {
                "question": message.content,
                "chat_history": chat_history_text,
                "knowledge_text": knowledge_text,
            }
        }
        return await StreamHandler.stream_response(chain, inputs)

class FileHandler:
    @staticmethod
    async def handle_file_message(message: cl.Message, conversation_id: str) -> str:
        """处理文件上传消息"""
        result_text = ""
        for element in message.elements:
            if isinstance(element, cl.File):
                success, msg, file_text = await process_uploaded_file(
                    element, 
                    GlobalComponents.vector_store,
                    config,
                    conversation_id
                )
                await cl.Message(content=msg).send()
                if success:
                    result_text = file_text
                    break
        
        if not result_text:
            return "文件处理失败"
            
        chain = create_qa_chain(GlobalComponents.llm)
        inputs = {
            "inputs": {
                "question": message.content,
                "context": result_text[:28672]
            }
        } 
        return await StreamHandler.stream_response(chain, inputs)

class URLHandler:
    @staticmethod
    def extract_url(text: str) -> Optional[str]:
        """从文本中提取第一个URL"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return urls[0] if urls else None

    @staticmethod
    async def handle_url_message(message: cl.Message, conversation_id: str, url: str) -> str:
        """处理URL消息"""
        status_msg = cl.Message(content=f"正在处理URL: {url}")
        await status_msg.send()
        
        try:
            if url.lower().endswith('.pdf'):
                url_content = await URLHandler._handle_pdf_url(url, conversation_id)
            else:
                url_content = await URLHandler._fetch_url_content(url)
                
            if isinstance(url_content, str) and url_content.startswith("获取URL内容时出错"):
                status_msg.content = url_content
                await status_msg.update()
                return await MessageProcessor.handle_chat_message(message, conversation_id)
                
            chain = create_qa_chain(GlobalComponents.llm)
            inputs = {
                "inputs": {
                    "question": message.content,
                    "context": url_content[:28672]
                }
            }
            return await StreamHandler.stream_response(chain, inputs)
            
        except Exception as e:
            status_msg.content = f"处理URL时出错: {str(e)}"
            await status_msg.update()
            return await MessageProcessor.handle_chat_message(message, conversation_id)

    @staticmethod
    async def _handle_pdf_url(url: str, conversation_id: str) -> str:
        """处理PDF URL"""
        response = requests.get(url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name
            
        try:
            success, _, content = await process_uploaded_file(
                cl.File(name=os.path.basename(url), path=temp_path),
                GlobalComponents.vector_store,
                config,
                conversation_id
            )
            return content if success else "PDF处理失败"
        finally:
            os.unlink(temp_path)

    @staticmethod
    async def _fetch_url_content(url: str) -> str:
        """获取URL内容"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return ' '.join(chunk for chunk in chunks if chunk)
            
        except Exception as e:
            return f"获取URL内容时出错: {str(e)}"

class StreamHandler:
    @staticmethod
    async def stream_response(chain, inputs: dict) -> str:
        """统一处理流式响应"""
        msg = cl.Message(content="")
        await msg.send()
        
        full_response = ""
        async for chunk in chain(**inputs):
            await msg.stream_token(chunk)
            full_response += chunk
            
        msg.content = full_response
        await msg.update()
        
        return full_response

# 导出初始化函数
def init_everything():
    """初始化所有组件"""
    return GlobalComponents.init()
