# ai4fs (AI for File System)

ai4fs 是一个创新项目，旨在利用AI技术增强操作系统的文件系统界面，提供全新的交互方式。

## 项目概述

ai4fs 通过整合人工智能技术，为用户提供一个智能化的文件系统交互界面。该项目使用Python和Chainlit构建，支持文档和图片上传，实现了一个直观、高效的聊天式界面。

## 主要特性

- 智能文件系统交互
- 基于聊天的用户界面
- 支持多种文档格式(PDF、Word、CSV、TXT等)
- AI辅助文件管理和分析
- 支持历史对话记录
- 支持多用户管理
- 支持文档向量化存储和检索

## 技术栈

- Python
- Chainlit
- LangChain
- ChromaDB
- Ollama (用于本地embedding)
- OpenAI API / 自定义大模型API

## 安装

1. 克隆仓库：
   ```
   git clone https://github.com/wanbo432503/ai4fs.git
   cd ai4fs
   ```

2. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

3. 配置环境变量：
   在项目根目录下创建 `.env` 文件，设置以下配置：
   ```
   # OpenAI API配置
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_API_BASE=https://api.openai.com/v1
   OPENAI_MODEL_NAME=gpt-3.5-turbo

   # 自定义大模型服务配置(可选)
   CUSTOM_MODEL_API_KEY=your_custom_model_api_key
   CUSTOM_MODEL_API_BASE=https://your-custom-model-api-endpoint.com/v1
   CUSTOM_MODEL_NAME=your_custom_model_name

   # 模型选择
   USE_CUSTOM_MODEL=false
   ```

## Ollama 设置

本项目使用Ollama来运行本地的BGE embedding模型：

1. 安装Ollama:
   访问 https://ollama.ai/download 下载安装包

2. 下载BGE模型:
   ```
   # macOS
   brew install ollama
   # Ubuntu
   apt install ollama
   
   ollama pull bge-m3
   ```

3. 运行Ollama服务:
   ```
   ollama serve
   ```

## 使用方法

1. 启动应用：
   ```
   chainlit run app.py
   ```

2. 访问界面：
   打开浏览器访问 http://localhost:8000

3. 登录系统：
   默认用户名/密码: admin/admin

## 功能说明

- 文件上传：支持PDF、Word、CSV、TXT等格式
- 智能对话：可以询问文档内容相关的问题
- 历史记录：支持查看历史对话记录
- 多用户支持：可配置多个用户访问权限

## 项目结构

```
ai4fs/
├── app.py              # 主应用入口
├── config.py           # 配置文件
├── utils/
│   ├── chat_history.py    # 聊天历史管理
│   ├── data_layer.py      # 数据持久化层
│   ├── document_loader.py # 文档加载器
│   ├── llm_setup.py       # LLM配置
│   └── qa_chain.py        # 问答链
├── data/               # 数据存储目录
└── requirements.txt    # 依赖包列表
```

## 贡献指南

我们欢迎所有形式的贡献，包括但不限于：

- 提交问题和建议
- 改进文档
- 提交代码修复
- 添加新功能

## 开源协议

本项目采用 MIT 协议开源 - 详见 [LICENSE](LICENSE) 文件