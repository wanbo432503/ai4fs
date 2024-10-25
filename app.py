import os
import chainlit as cl
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import config
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from typing import List
import shutil
import mimetypes  # 添加到文件顶部的导入部分

# 加载环境变量
load_dotenv()

# 初始化向量存储
embeddings = OllamaEmbeddings(
    model="bge-m3",  # 使用BGE模型
    base_url="http://localhost:11434"  # Ollama默认URL
)

vector_store = Chroma(
    embedding_function=embeddings, 
    persist_directory=config.VECTOR_STORE_PATH
)

# 初始化 OpenAI LLM
llm = ChatOpenAI(
    model_name=config.CUSTOM_MODEL_NAME,
    openai_api_base=config.CUSTOM_MODEL_API_BASE,
    openai_api_key=config.CUSTOM_MODEL_API_KEY,
    temperature=config.TEMPERATURE,
    max_tokens=config.MAX_TOKENS
)

@cl.on_chat_start
async def start():
    global vector_store
    # 初始化向量存储
    vector_store = Chroma(
        collection_name="knowledge_base",
        persist_directory=config.VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    
    await cl.Message(content="欢迎来到AI知识库，请上传文件提供知识").send()


@cl.on_message
async def main(message: cl.Message):
    # 检查消息是否包含附件
    if message.elements:
        for element in message.elements:
            if isinstance(element, cl.File):
                try:
                    # 获取文件的MIME类型
                    mime_type, _ = mimetypes.guess_type(element.name)
                    
                    # 扩展支持的文件类型
                    if mime_type in [
                        'application/pdf',
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'application/msword',
                        'text/csv',
                        'text/plain',
                        'text/markdown'  # 支持 markdown 文件
                    ] or element.name.endswith(('.csv', '.txt', '.md')):  # 使用文件扩展名作为备选判断
                        # 保存和处理文件
                        file_name = element.name
                        save_path = os.path.join(config.UPLOAD_FOLDER, file_name)
                        shutil.copy(element.path, save_path)
                        
                        documents = load_document(element.path)
                        text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
                        texts = text_splitter.split_documents(documents)
                        vector_store.add_documents(texts)
                        
                        await cl.Message(content=f"✅ 文件 {file_name} 已成功处理并添加到知识库").send()
                        continue
                    else:
                        await cl.Message(content=f"❌ 不支持的文件类型：{mime_type}。请上传 PDF 或 Word 文档。").send()
                        continue
                except Exception as e:
                    await cl.Message(content=f"处理文件时出错：{str(e)}").send()
                    continue

    # 确保向量存储已初始化
    if not vector_store:
        await cl.Message(content="数据库尚未初始化，系统启动不正常...").send()
        return
    
    # 创建检索链
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # 返回最相关的3个文档片段
    )
    
    # 创建提示模板
    template = """基于以下已知信息，简洁和专业地回答用户的问题。
    如果无法从中得到答案，请说 "抱歉，我无法从文档中找到相关信息。"
    
    已知信息：
    {context}
    
    问题：{question}"""
    
    # 创建自定义提示
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # 创建对话链
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # 创建消息元素
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # 执行检索问答
        print("开始执行检索问答...")
        response = await chain.ainvoke({
            "query": message.content
        }, callbacks=[cl.AsyncLangchainCallbackHandler()])
        print("检索问答完成")
        
        # 获取答案和来源文档
        print("正在提取答案和来源文档...")
        answer = response["result"]
        source_docs = response["source_documents"]
        print(f"找到 {len(source_docs)} 个相关文档")
        
        # 格式化来源文档信息
        print("正在格式化来源信息...")
        sources = []
        for i, doc in enumerate(source_docs, 1):
            source = f"\n来源 {i}:\n"
            source += f"- 文件: {doc.metadata.get('source', '未知')}\n"
            # source += f"- 内容片段: {doc.page_content[:200]}...\n"
            sources.append(source)
        
        # 组合最终回复
        print("正在生成最终回复...")
        final_response = f"{answer}\n\n参考来源：{''.join(sources)}"
        
        # 使用 msg.content 来更新消息内容
        print(f"正在更新消息...{final_response}")
        msg.content = final_response
        await msg.update()
        print("回复完成")
        
    except Exception as e:
        msg.content = f"处理您的问题时出错：{str(e)}"
        await msg.update()

def load_document(file_path: str):
    """
    根据文件类型加载文档
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的文档列表
    """
    # 获取文件扩展名（转换为小写）
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # 根据文件类型选择相应的加载器
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension in ['.doc', '.docx']:
        loader = Docx2txtLoader(file_path)
    elif file_extension == '.csv':
        loader = CSVLoader(
            file_path,
            encoding='utf-8',
            csv_args={
                'delimiter': ',',
                'quotechar': '"',
                'fieldnames': None  # 如果CSV有标题行，设为None会自动使用第一行作为标题
            }
        )
    elif file_extension in ['.txt', '.md']:
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        # 尝试使用通用加载器
        try:
            loader = UnstructuredFileLoader(file_path)
        except:
            raise ValueError(f"不支持的文件类型: {file_extension}")
    
    return loader.load()

if __name__ == "__main__":
    cl.run()
