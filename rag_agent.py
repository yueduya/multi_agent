import os
import requests
import pdfplumber
from PyPDF2 import PdfReader
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

load_dotenv()

class CustomEmbeddings:
    def __init__(self):
        device = torch.device('cpu')
        model_path = r'H:\TraTrae_code\models\BAAI\bge-large-zh-v1___5'
        self.model = SentenceTransformer(model_path, device=device)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

class RAGAgent:
    def __init__(self, pdf_folder, persist_directory="./chromadb", model_name="thenlper/gte-small"):
        self.pdf_folder = pdf_folder
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = CustomEmbeddings()
    
    def load_pdf_files(self):
        """加载文件夹中的所有 PDF 文件，包括文本和表格"""
        all_chunks = []
        doc_id = 0
        
        for file in os.listdir(self.pdf_folder):
            if file.endswith(".pdf"):
                file_path = os.path.join(self.pdf_folder, file)
                doc_id += 1
                
                try:
                    # 首先尝试使用 PyPDF2 提取文本
                    reader = PdfReader(file_path)
                    total_pages = len(reader.pages)
                    
                    print(f"正在处理 PDF 文件: {file} (共 {total_pages} 页)")
                    
                    for page_no, page in enumerate(reader.pages, start=1):
                        # 显示进度
                        if page_no % 10 == 0 or page_no == total_pages:
                            print(f"  进度: {page_no}/{total_pages} 页")
                        
                        # 使用 PyPDF2 提取文本
                        text = page.extract_text() or ""
                        
                        if text:
                            all_chunks.append({
                                "id": f"{doc_id}_text_{page_no}",
                                "text": text,
                                "metadata": {
                                    "doc_id": doc_id,
                                    "doc_name": file,
                                    "page": page_no,
                                    "type": "text"
                                }
                            })
                        
                        # 尝试使用 pdfplumber 提取表格
                        try:
                            with pdfplumber.open(file_path) as pdf:
                                if page_no <= len(pdf.pages):
                                    pdf_page = pdf.pages[page_no - 1]
                                    tables = pdf_page.extract_tables()
                                    
                                    if tables:
                                        for idx, table in enumerate(tables, start=1):
                                            # 将表格转换为 Markdown 格式
                                            table_md = self._table_to_markdown(table)
                                            
                                            if table_md:
                                                all_chunks.append({
                                                    "id": f"{doc_id}_table_{page_no}_{idx}",
                                                    "text": table_md,
                                                    "metadata": {
                                                        "doc_id": doc_id,
                                                        "doc_name": file,
                                                        "page": page_no,
                                                        "type": "table",
                                                        "table_index": idx
                                                    }
                                                })
                                        print(f"  - 第 {page_no} 页: 提取到 {len(tables)} 个表格")
                        except Exception as table_error:
                            # 如果 pdfplumber 失败，继续处理
                            pass
                    
                    print(f"已加载 PDF 文件: {file} (共 {len(all_chunks)} 个文本块)")
                        
                except Exception as e:
                    print(f"处理 PDF 文件 {file} 时出错: {str(e)}")
        
        return all_chunks
    
    def _table_to_markdown(self, table):
        """将表格转换为 Markdown 格式"""
        if not table:
            return ""
        
        markdown = []
        for i, row in enumerate(table):
            # 处理空单元格
            row = [str(cell) if cell is not None else "" for cell in row]
            markdown.append("| " + " | ".join(row) + " |")
            
            # 添加表头分隔线
            if i == 0:
                separator = "|" + "|".join(["---"] * len(row)) + "|"
                markdown.append(separator)
        
        return "\n".join(markdown)
    
    def process_documents(self, docs):
        """处理文档并创建向量存储"""
        # 将字典转换为 Document 对象
        documents = []
        for chunk in docs:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)
        
        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        texts = text_splitter.split_documents(documents)
        
        # 创建持久化的 Chroma 向量存储
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        print(f"向量存储已创建并持久化到 {self.persist_directory}")
    
    def initialize_qa_chain(self):
        """初始化问答链"""
        if not self.vectorstore:
            # 如果向量存储不存在，尝试从持久化目录加载
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                # 检查向量存储是否为空
                if self.vectorstore._collection.count() == 0:
                    print("向量存储为空，需要更新")
                    return False
                print("已从持久化目录加载向量存储")
            except Exception as e:
                print(f"加载向量存储失败: {str(e)}")
                return False
        return True
    
    def query(self, question):
        """执行查询"""
        if not self.initialize_qa_chain():
            return "错误: 无法初始化问答链", []
        
        try:
            # 使用向量检索获取相关文档
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)
            
            # 打印检索到的相似度最高的文本块
            print(f"\n{'='*60}")
            print(f"检索到 {len(docs)} 个相关文本块:")
            print(f"{'='*60}")
            
            for i, doc in enumerate(docs, 1):
                print(f"\n【文本块 {i}】")
                print(f"内容: {doc.page_content}")
                if hasattr(doc, 'metadata') and doc.metadata:
                    print(f"元数据: {doc.metadata}")
            
            print(f"\n{'='*60}\n")
            
            # 构建上下文
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # 构建提示词
            prompt = f"""基于以下上下文信息回答问题。如果上下文中没有相关信息，请说明。

上下文：
{context}

问题：{question}

回答："""
            
            # 调用通义千问 API
            llm = self._get_llm()
            result = llm.invoke(prompt)
            
            # 准备返回的chunk信息
            chunks_info = []
            for i, doc in enumerate(docs, 1):
                chunk_info = {
                    "id": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                }
                chunks_info.append(chunk_info)
            
            return result.content, chunks_info
        except Exception as e:
            return f"查询出错: {str(e)}", []
    
    def update_vectorstore(self):
        """更新向量存储"""
        docs = self.load_pdf_files()
        if docs:
            self.process_documents(docs)
            self.initialize_qa_chain()
            return "向量存储已更新"
        else:
            return "没有找到 PDF 文件或无法提取文本"
    
    def _get_llm(self):
        """创建通义千问 LLM"""
        # 优先使用新的环境变量名称
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        
        model_name = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        
        try:
            llm = ChatTongyi(
                model=model_name,
                api_key=api_key
            )
            return llm
        except Exception as e:
            raise ValueError(f"创建通义千问 LLM 失败: {str(e)}")

if __name__ == "__main__":
    print("=== RAG Agent 启动 ===")
    pdf_folder = "./documents"
    print(f"PDF 文件夹: {pdf_folder}")
    
    print("\n正在创建 RAG Agent...")
    agent = RAGAgent(pdf_folder)
    print("RAG Agent 创建成功")
    
    print("\n正在更新向量存储...")
    print(agent.update_vectorstore())
    
    print("\n开始问答 (输入 'quit' 退出)")
    while True:
        question = input("\n请输入问题: ").strip()
        if question.lower() == 'quit':
            break
        if question:
            answer, chunks = agent.query(question)
            print(f"\n回答: {answer}")
            if chunks:
                print("\n相关文档片段:")
                for i, chunk in enumerate(chunks, 1):
                    print(f"\n片段 {i}:")
                    print(chunk['content'])
                    if chunk['metadata']:
                        print(f"来源: {chunk['metadata'].get('doc_name', '未知')} 第{chunk['metadata'].get('page', '未知')}页")