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
            "conversation_id": conversation_id,
            "role": role,
            "timestamp": datetime.now().isoformat(),
            "type": "chat_message"
        }
        
        doc = Document(
            page_content=content,
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
            
        return sorted(messages, key=lambda x: x["timestamp"])

    def get_recent_messages(self, conversation_id: str, limit: int = 5) -> str:
        """获取最近的几条对话记录并格式化"""
        messages = self.get_conversation_history(conversation_id)
        
        # 获取最近的消息
        recent_messages = messages[-limit:] if len(messages) > limit else messages
        
        # 格式化对话历史
        formatted_history = []
        for msg in recent_messages:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)
