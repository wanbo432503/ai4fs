from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import mimetypes
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5) 


def load_document(file_path: str):
    """根据文件类型加载文档"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
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
                'fieldnames': None
            }
        )
    elif file_extension in ['.txt', '.md']:
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        try:
            loader = UnstructuredFileLoader(file_path)
        except:
            raise ValueError(f"不支持的文件类型: {file_extension}")
    
    return loader.load()

async def process_uploaded_file(element, vector_store, config, conversation_id):
    """
    处理上传的文件并添加到向量存储中。

    参数:
        element (cl.File): Chainlit文件对象
        vector_store: 向量存储对象
        config: 配置对象
        conversation_id: 会话ID

    返回:
        tuple: (bool, str, str) - (是否成功, 消息, 文档内容)
    """
    try:
        # 确保上传目录存在
        os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
        
        # 获取文件的MIME类型
        mime_type, _ = mimetypes.guess_type(element.name)
        
        # 检查文件类型是否支持
        supported_mimes = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/csv',
            'text/plain',
            'text/markdown'
        ]
        
        if mime_type in supported_mimes or element.name.endswith(('.csv', '.txt', '.md')):
            # 保存文件
            file_name = element.name
            save_path = os.path.join(config.UPLOAD_FOLDER, file_name)
            
            # 使用 shutil.copy2 来保留文件元数据
            shutil.copy2(element.path, save_path)
            
            # 处理文档
            documents = load_document(save_path)  # 使用保存后的文件路径
            result_text = ""
            for doc in documents:
                result_text += doc.page_content
                doc.metadata.update({   
                    "type": "document",
                    "file_name": file_name,
                    "mime_type": mime_type,
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": conversation_id
                })
            
            # 提交到线程池执行
            future = executor.submit(add_documents_to_vector_store, documents, vector_store)
            
            return True, f"✅ 文件 {file_name} 已成功处理并添加到知识库", result_text
        else:
            return False, f"❌ 不支持的文件类型：{mime_type}。请上传 PDF 或 Word 文档。", ""
            
    except Exception as e:
        return False, f"处理文件时出错：{str(e)}", ""
    
def add_documents_to_vector_store(documents, vector_store):
    """将文档添加到向量存储中"""
    text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    vector_store.add_documents(texts)
    print("VectorDB: File index complete...")