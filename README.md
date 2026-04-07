# Multi-Agent RAG 智能客服系统

## 项目简介

Multi-Agent RAG 智能客服系统是一个基于 LangGraph 和 RAG 技术的多智能体系统，能够处理旅游路线规划、企业文档智能问答、娱乐互动等多种任务。系统采用模块化架构设计，通过多智能体协同工作，为用户提供精准、高效的智能服务。

## 功能特性

- **旅游路线规划**：根据用户需求生成详细的旅游路线
- **企业文档智能问答**：基于 PDF 文档的智能检索和问答
- **娱乐互动**：生成幽默笑话，提升用户体验
- **多智能体协作**：多个专业智能体协同工作，提供专业服务
- **实时反馈**：采用流式输出，提供实时响应
- **友好的 Web 界面**：使用 Gradio 构建的用户友好界面

## 技术栈

### 后端技术
- **LangGraph**：构建多智能体状态图
- **LangChain**：LLM 应用框架
- **ChromaDB**：向量数据库
- **SentenceTransformer**：文本嵌入模型（BGE-large-zh）
- **PyPDF2 / pdfplumber**：PDF 文档处理

### 前端技术
- **Gradio**：快速构建 Web 界面

### 大语言模型
- **DeepSeek**：用于问题分类、旅行规划、笑话生成
- **通义千问**：用于 RAG 问答生成

### 其他
- **MCP (Model Context Protocol)**：工具调用协议（高德地图 API）
- **Python 3.8+**：开发语言

## 项目结构

```
multi_agent/
├── chromadb/           # 向量数据库存储
├── documents/          # PDF 文档目录
├── .env                # 环境变量配置
├── .gitignore          # Git 忽略文件
├── Director.py         # 多智能体协调器
├── Multi_Server.py     # Web 界面
├── README.md           # 项目说明文档
└── rag_agent.py        # RAG 文档检索
```

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository-url>
cd multi_agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件，添加以下环境变量：

```
# DeepSeek API
DEEPSEEK_API_KEY=your_deepseek_api_key

# 通义千问 API
DASHSCOPE_API_KEY=your_dashscope_api_key

# 可选：模型配置
DASHSCOPE_MODEL=qwen-plus
```

### 4. 准备文档

将需要进行智能问答的 PDF 文档放入 `documents/` 目录。

### 5. 下载嵌入模型

下载 BGE-large-zh 中文嵌入模型，放置在 `models/BAAI/bge-large-zh-v1___5/` 目录。

## 使用方法

### 启动系统

```bash
python Multi_Server.py
```

系统会自动启动 Web 服务器，默认在 `http://127.0.0.1:7860` 访问。

### 功能使用

1. **旅游路线规划**：输入类似 "规划从西村到中山大学的路线" 的问题
2. **企业文档问答**：输入与文档相关的问题，如 "客户经理考核办法有哪些内容"
3. **笑话生成**：输入 "讲一个笑话" 或具体要求，如 "讲一个郭德纲的笑话"
4. **其他问题**：系统会返回相应的提示

## 系统截图

### 主界面

### 旅游路线规划示例

<img width="1793" height="603" alt="{04313D4B-72BC-410F-B081-27E788E327CB}" src="https://github.com/user-attachments/assets/970469b5-42dd-43b1-baf6-b17b3d07e200" />


### 企业文档问答示例

<img width="1884" height="617" alt="{C6DA7701-88E3-4237-8E2C-371E529819EC}" src="https://github.com/user-attachments/assets/bd4106e7-caef-4af9-bfce-df8ad92ec62e" />


### 笑话生成示例

<img width="1886" height="452" alt="{C1239E5F-6BB8-48E2-A3E9-0B28D2094FD5}" src="https://github.com/user-attachments/assets/2209b02c-048f-47c7-80a3-6e52268bd092" />


## 核心模块说明

### 1. rag_agent.py

- **RAGAgent 类**：负责 PDF 文档的加载、处理和向量存储
- **CustomEmbeddings 类**：封装 BGE-large-zh 中文嵌入模型
- **主要方法**：
  - `load_pdf_files()`：加载 PDF 文件，提取文本和表格
  - `process_documents()`：处理文档并创建向量存储
  - `query()`：执行查询，返回答案和相关文档片段
  - `update_vectorstore()`：更新向量存储

### 2. Director.py

- **多智能体协调**：使用 LangGraph 构建状态图
- **节点说明**：
  - `supervisor_node`：监督节点，负责问题分类
  - `travel_node`：旅游规划节点
  - `joke_node`：笑话生成节点
  - `company_node`：公司信息节点
  - `other_node`：其他问题节点
- **路由逻辑**：根据问题类型分发到相应的智能体

### 3. Multi_Server.py

- **Web 界面**：使用 Gradio 构建
- **功能**：
  - 提供用户输入界面
  - 显示回答结果
  - 显示相关文档片段

## 技术亮点

1. **多智能体协同**：使用 LangGraph 实现多个专业智能体协同工作
2. **RAG 技术**：结合向量检索和生成式 AI，提供准确的文档问答
3. **PDF 智能处理**：支持文本和表格的提取与转换
4. **向量数据库优化**：使用 ChromaDB 实现向量存储持久化


## 扩展建议

1. **增加更多智能体**：可以添加天气查询、新闻推荐等更多功能
2. **优化向量检索**：调整 chunk_size 和 overlap 参数，提高检索精度
3. **添加对话历史**：实现多轮对话功能
4. **用户认证**：添加用户登录和权限管理
5. **性能优化**：使用缓存、异步处理等技术提升性能
6. **部署到云服务**：部署到云服务器，提供公共访问


### 常见问题

1. **API 密钥错误**：
   - 检查 `.env` 文件中的 API 密钥是否正确
   - 确保 API 密钥具有足够的权限

2. **向量存储初始化失败**：
   - 检查嵌入模型路径是否正确
   - 确保 ChromaDB 目录具有写入权限

3. **PDF 文档解析失败**：
   - 检查 PDF 文件是否损坏
   - 尝试使用不同的 PDF 阅读器打开验证

4. **Web 界面无法访问**：
   - 检查端口是否被占用
   - 尝试使用不同的浏览器访问


