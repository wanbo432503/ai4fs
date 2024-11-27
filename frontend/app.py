import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chainlit as cl
from chainlit.types import ThreadDict
from dotenv import load_dotenv
from backend.qa_chain import create_conv_summary_chain
from config import config
from typing import Optional
import chainlit.data as cl_data
from frontend.data_layer import AI4FSDataLayer
from frontend.msg_handle import handle_message, init_everything


# 加载环境变量
load_dotenv()
llm, chat_history = init_everything()

# 设置自定义数据层
cl_data._data_layer = AI4FSDataLayer()

first_msg = True
title_generated = False

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
        # 保存用户消息
        chat_history.save_message(
            conversation_id=conversation_id,
            role="user", 
            content=message.content
        )
        
        # 据是否有文件上传选择不同的处理流程
        full_response = await handle_message(message, conversation_id)

        # 保存AI回复
        chat_history.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=full_response
        )
        
        global title_generated
        # 如果还没有生成标题，则生成标题
        if not title_generated:
            message_history = chat_history.get_conversation_history(conversation_id)
            if len([msg for msg in message_history if msg["role"] == "user"]) >= 3:
                conversations = chat_history.generate_conv_summary(conversation_id)
                conv_summary_chain = create_conv_summary_chain(llm)
                if conv_summary_chain is not None:
                    title = conv_summary_chain.invoke({"chat_history": conversations})
                    await cl_data._data_layer.update_thread(message.thread_id, name=title)
                    title_generated = True
                else:
                    print("conv_summary_chain is None")
            
    except Exception as e:
        print(f"on message error: {str(e)}")
        error_msg = f"On Message Error: {str(e)}"
        await cl.Message(content=error_msg).send()


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
