# ai4fs (AI for File System)

ai4fs 是一个创新项目，旨在利用AI技术增强操作系统的文件系统界面，提供全新的交互方式。

## 项目概述

ai4fs 通过整合人工智能技术，为用户提供一个智能化的文件系统交互界面。该项目使用Python和Chainlit构建，支持文档和图片上传，实现了一个直观、高效的聊天式界面。

## 主要特性

- 智能文件系统交互
- 基于聊天的用户界面
- 支持文档和图片上传
- AI辅助文件管理和分析

## 技术栈

- Python
- Chainlit
- (其他可能使用的AI库或框架，如OpenAI API、Hugging Face Transformers等)

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

## 使用方法

运行以下命令启动应用：

```
chainlit run app.py
```

启动后,在浏览器中打开显示的本地地址(通常是 http://localhost:8000)即可访问ai4fs界面。

## 功能演示

1. 文件上传:点击界面上的"上传"按钮,选择要上传的文档或图片。
2. 智能交互:在聊天框中输入您的问题或指令,如"帮我整理下载文件夹"或"查找最近修改的PDF文件"。
3. AI分析:系统会分析您的请求并给出相应的建议或执行相关操作。

## 配置

在项目根目录下创建 `.env` 文件,并设置以下环境变量:

```
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# 自定义大模型服务配置
CUSTOM_MODEL_API_KEY=your_custom_model_api_key
CUSTOM_MODEL_API_BASE=https://your-custom-model-api-endpoint.com/v1
CUSTOM_MODEL_NAME=your_custom_model_name

# 模型选择
USE_CUSTOM_MODEL=false

```

请将 `your_openai_api_key` 替换为您的实际OpenAI API密钥。

## Ollama 设置

本项目使用Ollama来运行本地的BGE embedding模型。请按照以下步骤设置Ollama:

1. 安装Ollama:
   访问 https://ollama.ai/download 并下载适用于macOS的Ollama安装包。

2. 下载BGE模型:
   在终端中运行以下命令:
   ```
   # brew install ollama # for macos
   # apt install ollama # for ubuntu
   ollama pull bge-m3
   ```

3. 运行Ollama服务:
   在使用本应用之前,请确保Ollama服务正在运行。在终端中运行:
   ```
   ollama serve
   ```

注意: 请保持Ollama服务运行,否则embedding功能将无法正常工作。
