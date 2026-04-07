import asyncio
from operator import add
from typing import TypedDict, Annotated
import os
import dashscope
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.config import get_stream_writer
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import logging
from openai import OpenAI
from rag_agent import RAGAgent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

nodes = ["superviser", "travel", "company", "joke", "other"]



# api_key = os.getenv("DASHSCOPE_API_KEY")

# llm = ChatTongyi(
#     models="qwen-plus",
#     api_key=api_key
# )




api_key = os.getenv("DEEPSEEK_API_KEY")

# 直接使用 LangChain 的 ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

# 初始化RAGAgent实例
rag_agent = RAGAgent("./documents")
# 只有当向量存储不存在或为空时才更新
if not rag_agent.initialize_qa_chain():
    print("向量存储不存在或为空，正在更新...")
    rag_agent.update_vectorstore()
else:
    print("向量存储已存在，跳过更新")

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    type: str


builder = StateGraph(State)


def supervisor_node(state: State):
    print("supervisor_node")
    writer = get_stream_writer()
    writer({"node", ">>>>>>supervisor_node"})
    # 根据用户的问题，对问题进行分类，分类结果保存到type当中
    prompt = """你是一个专业的客服助手，负责对用户的问题进行分类，并将任务分给其他Agent执行。
    如果用户的问题是和旅游路线规划相关的，那就返回 travel 。
    如果用户的问题是希望讲一个笑话，那就返回 joke 。
    如果用户的问题是了解公司内容或经理考核相关，那就返回 company 。
    如果是其他的问题，返回 other 。
    除了这几个选项外（结果只能是一个单词），不要返回任何其他的内容。
    """

    prompts = [{"role": "system", "content":prompt},
               {"role": "user", "content":state["messages"][0]}]

    # 如果已经有type了，说明已经由其他节点处理完成了，直接返回
    if "type" in state:
        writer({"supervisor_step": f"已经获得{state['type']}智能体处理结果"})
        return {"type": END}
    else:
        response = llm.invoke(prompts)
        typeRes = response.content
        writer({"supervisor_step": f"问题分类结果是{typeRes}"})
        if typeRes in nodes:
            return {"type": typeRes}
        else:
            raise ValueError("type is not in ( travel, company, joke, other)")


# 全局缓存
_mcp_tools = None
_mcp_client = None


# async def get_mcp_tools():
#     """获取 MCP 工具（异步初始化）"""
#     global _mcp_tools, _mcp_client

#     if _mcp_tools is None:
#         _mcp_client = MultiServerMCPClient({
#             "amap-maps": {
#                 "type": "sse",
#                 "url": "https://mcp.api-inference.modelscope.net/e2a9333c4e9642/sse",
#                 "headers": {
#                     "Authorization": "Bearer ms-ab5e800e-65eb-401e-8eb2-dfa097553932"
#                 }
#                 }
#         })
#         _mcp_tools = await _mcp_client.get_tools()

#     return _mcp_tools


# def travel_node(state: State):
#     print("travel_node")
#     writer = get_stream_writer()
#     writer({"node": ">>>>>>travel_node"})

#     user_message = state["messages"][0] if state.get("messages") else ""
#     system_prompt = "你是一个专业旅行规划助手，根据用户问题，生成一个旅游路线规划。请用中文回答，并返回一个不超过150字的规划结果"

#     try:
#         # 获取工具
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         try:
#             tools = loop.run_until_complete(get_mcp_tools())
#         finally:
#             loop.close()

#         # 创建 agent（新方式）
#         agent = create_agent(
#             model=llm,
#             tools=tools
#         )

#         # ✅ 正确的输入格式
#         response = agent.invoke({
#             "messages": [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_message}
#             ]
#         })

#         final_response = response['messages'][-1].content
#         writer({"travel_result": final_response})

#         # ✅ 返回字典
#         return {
#             "messages": [HumanMessage(content=final_response)],
#             "type": "travel"
#         }

#     except Exception as e:
#         error_msg = f"旅行规划失败: {str(e)}"
#         writer({"error": error_msg})
#         return {
#             "messages": [HumanMessage(content=error_msg)],
#             "type": "error"
#         }
# 
# 全局缓存
_mcp_tools = None
_mcp_client = None

async def get_mcp_tools():
    """获取 MCP 工具（异步初始化）"""
    global _mcp_tools, _mcp_client

    if _mcp_tools is None:
        _mcp_client = MultiServerMCPClient({
             "amap-maps": {
                "transport": "sse",  # 使用 transport 而不是 type
                "url": "https://mcp.api-inference.modelscope.net/e2a9333c4e9642/sse",
                "headers": {
                    "Authorization": "Bearer ms-ab5e800e-65eb-401e-8eb2-dfa097553932"
                }
            }
        })
        _mcp_tools = await _mcp_client.get_tools()

    return _mcp_tools

def travel_node(state: State):
    print("travel_node")
    writer = get_stream_writer()
    writer({"node": ">>>>>>travel_node"})


    # 确保user_message是字符串
    if state.get("messages") and state["messages"]:
        if hasattr(state["messages"][0], "content"):
            user_message = state["messages"][0].content
        else:
            user_message = str(state["messages"][0])
    else:
        user_message = ""
    system_prompt = "你是一个专业旅行规划助手，根据用户问题，生成一个旅游路线规划。请用中文回答，并返回一个不超过500字的规划结果"

    try:
        # 直接调用llm生成旅行规划
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        response = llm.invoke(prompt)
        final_response = response.content
        writer({"travel_result": final_response})

        return {
            "messages": [HumanMessage(content=final_response)],
            "type": "travel"
        }

    except Exception as e:
        error_msg = f"旅行规划失败: {str(e)}"
        writer({"error": error_msg})
        return {
            "messages": [HumanMessage(content=error_msg)],
            "type": "error"
        }


def joke_node(state: State):
    print("joke_node")
    writer = get_stream_writer()
    writer({"node": ">>>>>>joke_node"})

    system_prompt = "你是一个笑话大师，根据用户问题，返回一个不超过150个字的笑话。"
    prompt = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": state["messages"][0]}]

    response = llm.invoke(prompt)

    return {"messages":[HumanMessage(content=response.content)], "type": "joke"}


def company_node(state: State):
    print("company_node")
    writer = get_stream_writer()
    writer({"node": ">>>>>>company_node"})

    user_message = state["messages"][0] if state.get("messages") else ""
    
    try:
        # 使用RAGAgent查询
        answer, chunks = rag_agent.query(user_message)
        
        # 构建包含chunk信息的响应
        response_content = f"{answer}\n\n【相关文档片段】\n"
        for i, chunk in enumerate(chunks, 1):
            response_content += f"\n片段 {i}:\n{chunk['content']}\n"
            if chunk['metadata']:
                response_content += f"来源: {chunk['metadata'].get('doc_name', '未知')} 第{chunk['metadata'].get('page', '未知')}页\n"
        
        writer({"company_result": answer})
        writer({"chunks": chunks})
        
        return {
            "messages": [HumanMessage(content=response_content)],
            "type": "company"
        }
    except Exception as e:
        error_msg = f"公司信息查询失败: {str(e)}"
        writer({"error": error_msg})
        return {
            "messages": [HumanMessage(content=error_msg)],
            "type": "error"
        }


def other_node(state: State):
    print("other_node")
    writer = get_stream_writer()
    writer({"node": ">>>>>>other_node"})

    return {"messages": [HumanMessage(content="我暂时无法回答这个问题")], "type": "other"}

# 条件路由
def routing_func(state: State):
    if state["type"] == "travel":
        return "travel_node"
    elif state["type"] == "joke":
        return "joke_node"
    elif state["type"] == "company":
        return "company_node"
    elif state["type"] == END:
        return END
    else:
        return "other_node"







# 构建图
builder = StateGraph(State)

# 添加节点
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("travel_node", travel_node )
builder.add_node("joke_node", joke_node)
builder.add_node("company_node", company_node)
builder.add_node("other_node", other_node)


# 添加边
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", routing_func, ["travel_node", "joke_node", "company_node", "other_node", END])
builder.add_edge("travel_node", "supervisor_node")
builder.add_edge("joke_node", "supervisor_node")
builder.add_edge("company_node", "supervisor_node")
builder.add_edge("other_node", "supervisor_node")



# 构建Graph
checkpoint = InMemorySaver()
graph = builder.compile(checkpointer=checkpoint)


# 执行任务测试代码
if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    for chunk in graph.stream({"messages": ["规划西村到中山大学的路线"]}
            , config
            , stream_mode="values"):
        print(chunk)

    # res = graph.invoke({"messages": ["今天天气怎么样"]}
    #                    , config
    #                    , stream_mode="values")
    # print(res["messages"][-1].content)

