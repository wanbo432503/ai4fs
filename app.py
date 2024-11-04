import chainlit as cl
from chainlit.types import ThreadDict
from dotenv import load_dotenv
from utils.document_loader import load_document, process_uploaded_file
from utils.llm_setup import init_embeddings, init_vector_store, init_llm
from utils.qa_chain import create_qa_chain, create_chat_chain
from config import config
from utils.chat_history import ChatHistoryManager
from typing import Optional
import chainlit.data as cl_data
from utils.data_layer import AI4FSDataLayer

# 加载环境变量
load_dotenv()

# 初始化组件
embeddings = init_embeddings()
vector_store = init_vector_store(embeddings)
llm = init_llm()

# 初始化聊天历史管理器
chat_history = ChatHistoryManager(vector_store)

# 设置自定义数据层
cl_data._data_layer = AI4FSDataLayer()


@cl.on_chat_start
async def start():
    try:       
        # 如果没有会话或会话已过期，创建新会话，采用cl.context.session.thread_id作为thread的key
        await cl.Message(content="正在开启新的会话...").send()
        await cl.ChatSettings(defaults={"model": config.CUSTOM_MODEL_NAME}).send()
        
    except Exception as e:
        print(f"初始化会话时出错: {str(e)}")


@cl.on_message
async def main(message: cl.Message):
    conversation_id = cl.context.session.thread_id
    
    try:
        # 检查向量存储是否初始化
        if not vector_store:
            await cl.Message(content="数据库尚未初始化，系统启动不正常...").send()
            return
            
        # 保存用户消息
        chat_history.save_message(
            conversation_id=conversation_id,
            role="user", 
            content=message.content
        )
        
        # 根据是否有文件上传选择不同的处理流程
        full_response = await handle_message(message, conversation_id)

        # 保存AI回复
        chat_history.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response
        )
            
    except Exception as e:
        error_msg = f"处理您的问题时出错：{str(e)}"
        cl.Message(content=error_msg).send()


async def handle_message(message: cl.Message, conversation_id: str) -> str:
    """处理用户消息,返回AI回复内容"""
    if message.elements:
        return await handle_file_message(message)
    else:
        return await handle_chat_message(message, conversation_id)
        
async def handle_file_message(message: cl.Message) -> str:
    """处理包含文件的消息"""
    # 处理文件上传
    for element in message.elements:
        if isinstance(element, cl.File):
            print(f"Processing uploaded file: {element.name}")
            success, msg = await process_uploaded_file(element, vector_store, config)
            await cl.Message(content=msg).send()
            if not success:
                continue
                
    # 创建QA链处理问题
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    chain = create_qa_chain(llm, retriever)
    
    # 创建消息对象用于流式输出
    msg = cl.Message(content="")
    await msg.send()
    
    # 获取回复
    full_response = ""
    async for chunk in chain(message.content):
        await msg.stream_token(chunk)
        full_response += chunk
        
    # 更新最终消息内容    
    msg.content = full_response
    await msg.update()
    
    return full_response

async def handle_chat_message(message: cl.Message, conversation_id: str) -> str:
    """处理普通对话消息"""
    chain = create_chat_chain(llm)
    chat_history_text = chat_history.get_recent_messages(conversation_id)
    
    # 创建消息对象用于流式输出
    msg = cl.Message(content="")
    await msg.send()
    
    full_response = ""
    async for chunk in chain.astream({
        "question": message.content,
        "chat_history": chat_history_text
    }):
        await msg.stream_token(chunk)
        full_response += chunk
        
    # 更新最终消息内容
    msg.content = full_response 
    await msg.update()
    
    return full_response


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print(f"resume {thread['id']}")


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier=username,
            metadata={"role": "admin", "provider": "credentials"}
        )
    return None
