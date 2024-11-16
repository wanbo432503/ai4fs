from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
    """创建支持流式输出的聊天链"""
    template = """请以专业、友好的语气回答用户的问题。
    
    最近的对话历史：
    {chat_history}
    
    当前问题：{question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()
    chain = prompt | llm | parser
    
    return chain

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
    
