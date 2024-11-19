from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults, TavilySearchResults
from langchain_core.utils.function_calling import convert_to_openai_function
import json
from config import config
from openai import AsyncOpenAI
import asyncio
from datetime import datetime

def generate_tool_for_moonshot(tools):
    tool_list = [convert_to_openai_function(t) for t in tools]
    moonshot_tools = []
    for tool in tool_list:
        # moonshot 需要指定 function 类型
        moonshot_tools.append({
            "type": "function",
            "function": tool
        })
    return moonshot_tools

def create_qa_chain(llm, retriever):
    template = """基于以下已知信息，简洁和专业地回答用户的问题。
    如果无法从中得到答案，请说 "抱歉，我无法从文档中找到相关信息。"
    
    已知信息：
    {context}
    
    问题：
    {question}
    
    请按以下格式输出:
        1. 先给出完整的回答
        2. 然后列出"参考来源："
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    
    async def qa_chain(question: str):
        # 1. 检索相关文档 - 只检索文档类型的内容
        docs = retriever.invoke(
            question,
            where={"type": "document"}  # 添加类型过滤
        )
        print(f"Docs: {docs}")
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # 2. 使用 chain.astream 进行流式输出
        async for chunk in chain.astream({
            "context": context,
            "question": question
        }):
            yield chunk
            
    return qa_chain

def create_chat_chain(llm):
    """创建支持工具调用的聊天链"""
    template = """请以专业、友好的语气回答用户的问题。如果需要搜索相关信息，请使用搜索工具。当前时间为：{current_time}。
    
    最近的对话历史：
    {chat_history}
    
    当前问题：{question}
    """
    
    # 初始化工具列表并确保工具名称匹配
    tools = []
    tools.append(DuckDuckGoSearchResults(
        name="duckduckgo_results_json",
        max_results=2
    ))
    
    # 如果配置了Tavily API密钥,则添加Tavily搜索工具
    if config.TAVILY_API_KEY:
        tools.append(TavilySearchResults(
            name="tavily_search_results_json",
            api_key=config.TAVILY_API_KEY
        ))
    
    # 创建工具名称到工具实例的映射
    tool_map = {tool.name: tool for tool in tools}
    generated_tools = generate_tool_for_moonshot(tools)
    
    # 使用异步 OpenAI 客户端
    client = AsyncOpenAI(
        api_key=config.CUSTOM_MODEL_API_KEY,
        base_url=config.CUSTOM_MODEL_API_BASE,
    )
    
    async def chat_chain_with_tools(inputs: dict):
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            messages = [{
                "role": "user", 
                "content": template.format(
                    current_time=current_time,
                    chat_history=inputs.get("chat_history", ""),
                    question=inputs["question"]
                )
            }]
            
            complete_tool_calls = {}
            
            response = await client.chat.completions.create(
                model="moonshot-v1-32k",
                messages=messages,
                tools=generated_tools,
                temperature=0.7,
                stream=True,
            )
            
            async for chunk in response:
                if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        # 初始化或更新工具调用信息
                        if tool_call.index not in complete_tool_calls:
                            complete_tool_calls[tool_call.index] = {
                                "id": tool_call.id,
                                "name": tool_call.function.name if hasattr(tool_call.function, 'name') else None,
                                "arguments": ""
                            }
                        
                        # 累积参数字符串
                        if hasattr(tool_call.function, 'arguments'):
                            complete_tool_calls[tool_call.index]["arguments"] += tool_call.function.arguments
                            
                        # 如果工具调用信息完整，执行工具调用
                        if complete_tool_calls[tool_call.index]["name"] and complete_tool_calls[tool_call.index]["arguments"]:
                            try:
                                tool_call_info = complete_tool_calls[tool_call.index]
                                
                                # 确保参数字符串是完整的JSON
                                if not tool_call_info["arguments"].strip().endswith("}"):
                                    continue
                                    
                                function_args = json.loads(tool_call_info["arguments"])
                                
                                # 使用工具映射获取工具实例
                                tool = tool_map.get(tool_call_info["name"])
                                if tool is None:
                                    print(f"Tool not found: {tool_call_info['name']}")
                                    continue
                                    
                                try:
                                    tool_response = tool.invoke(function_args)
                                except Exception as e:
                                    print(f"Tool invocation error: {str(e)}")
                                    continue
                                
                                # 将工具响应添加到消息历史
                                messages.append({
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [{
                                        "id": tool_call_info["id"],
                                        "function": {
                                            "name": tool_call_info["name"],
                                            "arguments": tool_call_info["arguments"]
                                        },
                                        "type": "function"
                                    }]
                                })
                                messages.append({
                                    "role": "tool",
                                    "content": str(tool_response),
                                    "tool_call_id": tool_call_info["id"]
                                })
                                
                                # 获取最终响应
                                final_response = await client.chat.completions.create(
                                    model="moonshot-v1-32k",
                                    messages=messages,
                                    temperature=0.7,
                                    stream=True,
                                )

                                async for final_chunk in final_response:
                                    if final_chunk.choices[0].delta.content:
                                        yield final_chunk.choices[0].delta.content
                            except json.JSONDecodeError as e:
                                print(f"JSON parsing error: {str(e)}, tool_call_info: {tool_call_info}")
                                continue
                else:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"Error in chat chain: {str(e)}")
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
    
    prompt = ChatPromptTemplate.from_template(template=template)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    
    return chain
    
async def simulate_stream(text: str):
    for char in text:
        yield char
        await asyncio.sleep(0.02)  # 每个字符间隔20ms
    
