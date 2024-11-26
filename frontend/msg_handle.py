import chainlit as cl
from backend.qa_chain import create_qa_chain, create_chat_chain
from backend.document_loader import process_uploaded_file
from config import config
from backend.chat_history import ChatHistoryManager
from backend.llm_setup import init_embeddings, init_vector_store, init_llm

embeddings = None
vector_store = None
llm = None
chat_history = None

def init_everything():
    # 初始化组件
    global embeddings, vector_store, llm, chat_history
    embeddings = init_embeddings()
    vector_store = init_vector_store(embeddings)
    llm = init_llm()
    # 初始化聊天历史管理器
    chat_history = ChatHistoryManager(vector_store)
    return llm, chat_history

async def handle_message(message: cl.Message, conversation_id: str) -> str:
    """处理用户消息,返回AI回复内容"""
    if message.elements:
        return await handle_file_message(message, conversation_id)
    else:
        return await handle_chat_message(message, conversation_id)
        
async def handle_file_message(message: cl.Message, conversation_id: str) -> str:
    """处理包含文件的消息"""
    # 处理文件上传
    for element in message.elements:
        if isinstance(element, cl.File):
            print(f"Processing uploaded file: {element.name}")
            success, msg, result_text = await process_uploaded_file(element, vector_store, config, conversation_id)
            await cl.Message(content=msg).send()
            if not success:
                continue
                
    chain = create_qa_chain(llm)
    
    # 创建消息对象用于流式输出
    msg = cl.Message(content="")
    await msg.send()
    
    # 获取回复
    full_response = ""
    async for chunk in chain(message.content, result_text):
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
    text_docs = vector_store.similarity_search(message.content,
                                               filter={"conversation_id": conversation_id},
                                               k=5)
    if text_docs:
        knowledge_text = "\n".join([doc.page_content for doc in text_docs])
    else:
        knowledge_text = ""
    
    # 创建消息对象
    msg = cl.Message(content="")
    await msg.send()
    
    # 初始化响应变量
    full_response = ""
    
    # 获取回复并流式输出
    async for chunk in chain({
        "question": message.content,
        "chat_history": chat_history_text,
        "knowledge_text": knowledge_text,
    }):
        await msg.stream_token(chunk)
        full_response += chunk
    
    # 更新消息内容
    msg.content = full_response
    await msg.update()
    
    return full_response
