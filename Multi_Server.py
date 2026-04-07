import gradio as gr
import random
from Director import graph


def process_input(text):
    config = {
        "configurable": {
            "thread_id": random.randint(1, 1000)
        }
    }

    result = graph.invoke({"messages": [text]}, config)
    response = result["messages"][-1].content
    
    # 分离答案和RAG结果
    if "【相关文档片段】" in response:
        parts = response.split("【相关文档片段】")
        answer = parts[0].strip()
        rag_result = "【相关文档片段】" + parts[1].strip()
    else:
        answer = response
        rag_result = ""
    
    return answer, rag_result


with gr.Blocks() as demo:
    gr.Markdown("# LangGraph Multi-Agent")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 可以问路线规划，公司信息，讲笑话，快来试试吧。")
            inputs_text = gr.Textbox(label="问题*", placeholder="请输入你的问题", value="讲一个郭德纲的笑话")
            btn_start = gr.Button(value="Start", variant="primary")
        with gr.Column():
            output_text = gr.Textbox(label="回答")
            rag_output = gr.Textbox(label="相关文档片段", lines=10)

    btn_start.click(process_input, inputs=[inputs_text], outputs=[output_text, rag_output])

demo.launch()