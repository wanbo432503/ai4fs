from typing import List, Dict
from datetime import datetime
import json
from langchain_core.documents import Document

class ChatHistoryManager:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
    def save_message(self, conversation_id: str, role: str, content: str):
        """保存聊天消息到向量存储"""
        metadata = {
            "conversation_id": str(conversation_id),  # 确保是字符串
            "role": str(role),  # 确保是字符串
            "timestamp": datetime.now().isoformat(),  # ISO格式的时间戳字符串
            "type": "chat_message"
        }
        
        # 确保所有元数据值都是基本类型
        for key, value in metadata.items():
            if value is None:
                metadata[key] = ""  # 将 None 转换为空字符串
        
        doc = Document(
            page_content=str(content),  # 确保内容是字符串
            metadata=metadata
        )
        
        self.vector_store.add_documents([doc])
        
    def get_conversation_history(self, conversation_id: str) -> List[Dict]:
        """获取特定会话的历史记录"""
        where_clause = {
            "$and": [
                {"conversation_id": {"$eq": conversation_id}},
                {"type": {"$eq": "chat_message"}}
            ]
        }
        
        results = self.vector_store.get(
            where=where_clause,
            include=["metadatas", "documents"]
        )
        
        if not results or not results['ids']:
            return []
            
        messages = []
        for doc, metadata in zip(results['documents'], results['metadatas']):
            messages.append({
                "role": metadata["role"],
                "content": doc,
                "timestamp": metadata["timestamp"]
            })
        
        # 确保按时间戳排序
        return sorted(messages, key=lambda x: x["timestamp"])

    def get_recent_messages(self, conversation_id: str, limit: int = 5) -> str:
        """获取最近的几条对话记录并格式化"""
        try:
            messages = self.get_conversation_history(conversation_id)
            
            # 获取最近的消息
            recent_messages = messages[-limit:] if len(messages) > limit else messages
            
            # 格式化对话历史
            formatted_history = []
            for msg in recent_messages:
                role = "用户" if msg["role"] == "user" else "助手"
                formatted_history.append(f"{role}: {msg['content']}")
            
            return "\n".join(formatted_history)
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return ""
        
    def generate_conv_summary(self, conversation_id: str) -> str:
        """生成生成标题所需的对话内容"""
        conv_messages = self.get_conversation_history(conversation_id)
        conv = ""
        for message in conv_messages:
            if message["role"] == "user":
                conv += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                conv += f"Assistant: {message['content']}\n"

        return conv