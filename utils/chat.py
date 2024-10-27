from typing import List
import chainlit as cl
from config import config
import uuid

async def send_welcome_message():
    """发送欢迎消息和新建会话按钮"""
    await cl.Message(content="欢迎来到AI知识库，请上传文件提供知识").send()

async def reset_chat():
    """重置会话设置"""
    # 生成新的会话ID
    conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conversation_id)
    
    await cl.Message(content="正在开启新的会话...").send()
    await cl.ChatSettings(defaults={"model": config.CUSTOM_MODEL_NAME}).send()
