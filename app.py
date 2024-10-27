import os
import chainlit as cl
from dotenv import load_dotenv
import shutil
import mimetypes
from utils.document_loader import load_document, process_uploaded_file
from utils.llm_setup import init_embeddings, init_vector_store, init_llm
from utils.qa_chain import create_qa_chain, create_chat_chain
from config import config
from utils.chat import send_welcome_message, reset_chat
import uuid
from utils.chat_history import ChatHistoryManager

# 加载环境变量
load_dotenv()

# 初始化组件
embeddings = init_embeddings()
vector_store = init_vector_store(embeddings)
llm = init_llm()

# 初始化聊天历史管理器
chat_history = ChatHistoryManager(vector_store)

@cl.on_chat_start
async def start():
    # 生成新的会话ID
    conversation_id = str(uuid.uuid4())
    cl.user_session.set("conversation_id", conversation_id)
    
    # 发送欢迎消息
    await send_welcome_message()
    
    # 加载历史消息
    messages = chat_history.get_conversation_history(conversation_id)
    for msg in messages:
        await cl.Message(
            content=msg["content"],
            author=msg["role"]
        ).send()

@cl.action_callback("new_chat")
async def on_new_chat(action):
    """处理新建聊天的回调"""
    await reset_chat()
    await start()

@cl.on_message
async def main(message: cl.Message):
    conversation_id = cl.user_session.get("conversation_id")
    
    # 确保向量存储已初始化
    if not vector_store:
        await cl.Message(content="数据库尚未初始化，系统启动不正常...").send()
        return

        # 检查消息是否包含附件
    if message.elements:  # 如果有上传文件，上传文档，并向量化存储起来
        for element in message.elements:
            if isinstance(element, cl.File):
                print(f"Processing uploaded file: {element.name}")
                success, msg = await process_uploaded_file(element, vector_store, config)
                await cl.Message(content=msg).send()
                if not success:
                    continue

    # 创建消息元素
    msg = cl.Message(content="")
    await msg.send()  # 先发送空消息
    
    full_response = ""
    try:
        # 保存用户消息
        chat_history.save_message(
            conversation_id=conversation_id,
            role="user",
            content=message.content
        )
        
        if message.elements:  # 如果有上传文件，使用 QA 链
            # 创建检索链
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            chain = create_qa_chain(llm, retriever)
            async for chunk in chain(message.content):
                # print(f"Chunk: {chunk}")
                await msg.stream_token(chunk)
                full_response += chunk
        else:  # 如果是普通对话，使用聊天链
            chain = create_chat_chain(llm)
            chat_history_text = chat_history.get_recent_messages(conversation_id)
            async for chunk in chain.astream({
                "question": message.content,
                "chat_history": chat_history_text
            }):
                # print(f"Chunk: {chunk}")
                await msg.stream_token(chunk)
                full_response += chunk
            
        # 更新最终消息
        msg.content = full_response
        await msg.update()
        
        # 保存AI回复
        chat_history.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response
        )
            
    except Exception as e:
        error_msg = f"处理您的问题时出错：{str(e)}"
        msg.content = error_msg
        await msg.update()
        print(f"Error: {error_msg}")  # 添加错误日志

# 这里继续保留原来的 @cl.on_message 处理函数...
