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
    # 创建消息元素
    msg = cl.Message(content="")
    await msg.send()  # 先发送空消息
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
            print(f"test get message of conversation: {conversation_id}")
            chat_history_text = chat_history.get_recent_messages(conversation_id)
            print(f"Chat history text: {chat_history_text}")
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
