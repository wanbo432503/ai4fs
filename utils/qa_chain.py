from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_core.utils.function_calling import convert_to_openai_function
import json
from config import config
from openai import AsyncOpenAI

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
    template = """请以专业、友好的语气回答用户的问题。如果需要搜索相关信息，请使用搜索工具。
    
    最近的对话历史：
    {chat_history}
    
    当前问题：{question}
    """
    
    tools = [DuckDuckGoSearchResults(max_results=5)]
    generated_tools = generate_tool_for_moonshot(tools)
    
    # 使用 AsyncOpenAI 替代 OpenAI
    client = AsyncOpenAI(
        api_key=config.CUSTOM_MODEL_API_KEY,
        base_url=config.CUSTOM_MODEL_API_BASE,
    )
    
    async def chat_chain_with_tools(inputs: dict):
        try:
            messages = [{
                "role": "user",
                "content": template.format(
                    chat_history=inputs.get("chat_history", ""),
                    question=inputs["question"]
                )
            }]
            
            # 第一次调用，获取可能的工具调用
            response = await client.chat.completions.create(
                model="moonshot-v1-32k",
                messages=messages,
                tools=generated_tools,
                temperature=0.7
            )
            
            if response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # 执行工具调用
                    tool = next((t for t in tools if t.name == function_name), None)
                    if tool:
                        tool_response = tool.invoke(function_args)
                        
                        # 将工具响应添加到消息历史
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call.model_dump()]
                        })
                        messages.append({
                            "role": "tool",
                            "content": str(tool_response),
                            "tool_call_id": tool_call.id
                        })
                        
                        # 获取最终响应
                        final_response = await client.chat.completions.create(
                            model="moonshot-v1-32k",
                            messages=messages,
                            temperature=0.7
                        )
                        return final_response.choices[0].message.content
                    
            else:
                # 如果没有工具调用，直接返回响应内容
                return response.choices[0].message.content
                    
        except Exception as e:
            print(f"Error in chat chain: {str(e)}")
            return f"处理您的问题时出错：{str(e)}"
    
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
    
