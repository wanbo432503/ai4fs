from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults, TavilySearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.utils.function_calling import convert_to_openai_function
import json
from config import config
from openai import AsyncOpenAI
import asyncio
from datetime import datetime

def generate_func_tools(tools):
    tool_list = [convert_to_openai_function(t) for t in tools]
    func_tools = []
    for tool in tool_list:
        # OpenAI 需要指定 function 类型，参考https://platform.openai.com/docs/guides/function-calling
        func_tools.append({
            "type": "function",
            "function": tool
        })
    return func_tools

def create_qa_chain(llm):
    template = """基于以下已知信息，简洁和专业地回答用户的问题。
    如果无法从中得到答案，请说 "抱歉，我无法从文档中找到相关信息。"
    
    已知信息：
    {context}
    
    问题：
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    
    async def qa_chain(inputs: dict):
        async for chunk in chain.astream({
            "context": inputs.get("context", ""),
            "question": inputs["question"]
        }):
            yield chunk
            
    return qa_chain

def create_chat_chain(llm):
    """创建支持工具调用的聊天链"""
    template = """请以专业、友好的语气回答用户的问题。如果需要搜索相关信息，请使用搜索工具。当前时间为：{current_time}。
    
    最近的对话历史：
    {chat_history}
    
    上传的相关文件内容：
    {knowledge_text}
    
    当前问题：{question}
    """
    
    # 初始化搜索工具列表并确保工具名称匹配
    tools = []
    try:
        tools.append(DuckDuckGoSearchResults(
            name="duckduckgo_results_json",
            max_results=2
        ))
        
        if config.TAVILY_API_KEY:
            tools.append(TavilySearchResults(
                name="tavily_search_results_json",
                api_key=config.TAVILY_API_KEY
            ))
            
        # 添加工具验证
        if not tools:
            raise ValueError("没有可用的搜索工具")
            
        # 创建工具映射前先验证工具是否正确初始化
        tool_map = {tool.name: tool for tool in tools if tool and hasattr(tool, 'invoke')}
        if not tool_map:
            raise ValueError("工具映射创建失败")
            
        generated_tools = generate_func_tools(tools)
        
    except Exception as e:
        print(f"工具初始化错误: {str(e)}")
        # 提供一个基础的回退方案
        return create_basic_chat_chain(llm)
    
    if config.USE_CUSTOM_MODEL:
        # 使用异步 OpenAI 客户端
        client = AsyncOpenAI(
            api_key=config.CUSTOM_MODEL_API_KEY,
            base_url=config.CUSTOM_MODEL_API_BASE,
        )
        model_name = config.CUSTOM_MODEL_NAME
    else:
        client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
        )
        model_name = config.OPENAI_MODEL_NAME
    
    async def chat_chain_with_tools(inputs: dict):
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            messages = [{
                "role": "user", 
                "content": template.format(
                    current_time=current_time,
                    chat_history=inputs.get("chat_history", ""),
                    knowledge_text=inputs.get("knowledge_text", ""),
                    question=inputs["question"]
                )
            }]
            
            complete_tool_calls = {}
            failed_tools = set()
            
            # 改为非流式调用
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=generated_tools,
                temperature=0.7,
                stream=False  # 设置为非流式
            )
            
            # 处理工具调用
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        tool_name = tool_call.function.name
                        
                        if tool_name in failed_tools:
                            continue
                            
                        tool = tool_map.get(tool_name)
                        if tool is None:
                            print(f"Tool not found: {tool_name}")
                            continue
                            
                        tool_response = tool.invoke(function_args)
                        
                        # 添加工具调用到消息历史
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": tool_call.id,
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments
                                },
                                "type": "function"
                            }]
                        })
                        messages.append({
                            "role": "tool",
                            "content": str(tool_response),
                            "tool_call_id": tool_call.id
                        })
                        
                    except Exception as e:
                        print(f"Tool invocation error: {str(e)}")
                        failed_tools.add(tool_name)
                        
                        if "rate" in str(e).lower():
                            alternate_tool_name = next(
                                (name for name in tool_map.keys() 
                                 if name != tool_name and name not in failed_tools),
                                None
                            )
                            
                            if alternate_tool_name:
                                try:
                                    alternate_tool = tool_map[alternate_tool_name]
                                    tool_response = alternate_tool.invoke(function_args)
                                    
                                    messages.append({
                                        "role": "assistant", 
                                        "content": None,
                                        "tool_calls": [{
                                            "id": tool_call.id,
                                            "function": {
                                                "name": alternate_tool_name,
                                                "arguments": tool_call.function.arguments
                                            },
                                            "type": "function"
                                        }]
                                    })
                                    messages.append({
                                        "role": "tool",
                                        "content": str(tool_response),
                                        "tool_call_id": tool_call.id
                                    })
                                except Exception as e2:
                                    print(f"Alternate tool failed: {str(e2)}")
                                    failed_tools.add(alternate_tool_name)
                                    yield "抱歉，搜索服务暂时不可用，请稍后再试。"
                            else:
                                yield "抱歉，所有搜索服务都暂时不可用，请稍后再试。"
                
                # 获取最终响应
                final_response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                
                try:
                    async for chunk in final_response:
                        if chunk.choices[0].delta.content is not None:
                            yield chunk.choices[0].delta.content
                except Exception as e:
                    print(f"流式响应处理错误: {str(e)}")
                    yield "处理响应时发生错误，请稍后重试。"
            else:
                # 直接响应处理
                try:
                    for char in response.choices[0].message.content:
                        await asyncio.sleep(0.02)
                        yield char
                except Exception as e:
                    print(f"字符流处理错误: {str(e)}")
                    yield "处理响应时发生错误，请稍后重试。"
                
        except Exception as e:
            print(f"聊天链错误: {str(e)}")
            yield f"处理您的问题时出错：{str(e)}"
    
    return chat_chain_with_tools

def create_conv_summary_chain(llm):
    """创建生成对话摘要的链"""
    template = """你扮演一个对话总结器，为以下用户与AI助手之间的对话创建简洁标题。
    根据对话内容，生成一个标题，要求不超过6个字，且要能精准捕捉对话的核心主题。
    
    Conversation：
    {chat_history}
    
    输出标题：
    """
    try:
        prompt = ChatPromptTemplate.from_template(template=template)
        parser = StrOutputParser()
        chain = prompt | llm | parser
    except Exception as e:
        print(f"Error in create_conv_summary_chain: {str(e)}")

    return chain


def create_basic_chat_chain(llm):
    """创建基础对话链（不包含工具调用）"""
    template = """请以专业、友好的语气回答用户的问题。当前时间为：{current_time}。
    
    最近的对话历史：
    {chat_history}
    
    当前问题：{question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    
    async def basic_chat_chain(inputs: dict):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            async for chunk in chain.astream({
                "current_time": current_time,
                "chat_history": inputs.get("chat_history", ""),
                "question": inputs["question"]
            }):
                yield chunk
        except Exception as e:
            yield f"处理您的问题时出错：{str(e)}"
    
    return basic_chat_chain
    
