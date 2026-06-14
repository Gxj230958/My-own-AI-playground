from openai import OpenAI
import sys
import signal
import json
from functionCallRegistry import function_registry, function_desc

# 初始化OpenAI客户端
client = OpenAI(
    api_key="sk-0b870c21ee08424da20f00e028041b99",  # 请替换为您的API密钥
    base_url="https://api.deepseek.com"
)

# 设置系统消息
system_message = {"role": "system", "content": "你是一个用于对话场景的智能助手，请正确、简洁、比较口语化地回答问题。你能够使用提供的tools（函数）来回答问题，有必要时需要从用户提问中抽取函数所需要的参数"}

# 初始化消息列表，保存对话历史
messages = [system_message]

def handle_interrupt(signal, frame):
    """处理Ctrl+C中断"""
    print("\n程序已终止")
    sys.exit(0)

# 注册中断处理函数
signal.signal(signal.SIGINT, handle_interrupt)

def chat_with_model():
    """实现与模型的连续对话"""
    try:
        while True:
            # 获取用户输入
            user_input = input("Q: ")
            
            # 将用户消息添加到消息列表（保存上下文）
            user_message = {"role": "user", "content": user_input}
            messages.append(user_message)
            
            print("A: ", end="", flush=True)
            
            # 创建聊天请求，包含函数调用功能
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=function_desc,
                stream=True
            )
            
            # 存储完整回答，用于添加到消息历史
            full_response = ""
            has_tool_call = False
            tool_call_message = None
            
            # 处理流式响应
            for chunk in response:
                # 检查是否为函数调用
                if chunk.choices[0].delta.tool_calls and len(chunk.choices[0].delta.tool_calls) > 0:
                    has_tool_call = True
                    # 构建函数调用消息
                    if not tool_call_message:
                        tool_call_message = {"role": "assistant", "tool_calls": [], "content": None}
                    
                    # 更新tools调用数据
                    delta_tool_calls = chunk.choices[0].delta.tool_calls
                    for delta_tool_call in delta_tool_calls:
                        if len(tool_call_message["tool_calls"]) <= delta_tool_call.index:
                            # 添加新的tool_call
                            tool_call_message["tool_calls"].append({
                                "id": delta_tool_call.id if delta_tool_call.id else "",
                                "function": {
                                    "name": delta_tool_call.function.name if delta_tool_call.function and delta_tool_call.function.name else "",
                                    "arguments": delta_tool_call.function.arguments if delta_tool_call.function and delta_tool_call.function.arguments else ""
                                },
                                "type": "function"
                            })
                        else:
                            # 更新现有tool_call
                            if delta_tool_call.id:
                                tool_call_message["tool_calls"][delta_tool_call.index]["id"] = delta_tool_call.id
                            if delta_tool_call.function and delta_tool_call.function.name:
                                tool_call_message["tool_calls"][delta_tool_call.index]["function"]["name"] = delta_tool_call.function.name
                            if delta_tool_call.function and delta_tool_call.function.arguments:
                                tool_call_message["tool_calls"][delta_tool_call.index]["function"]["arguments"] += delta_tool_call.function.arguments
                
                # 如果有常规内容，显示并添加到完整回答
                elif chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            # 如果是函数调用
            if has_tool_call and tool_call_message and tool_call_message["tool_calls"]:
                messages.append(tool_call_message)
                
                # 执行所有函数调用
                for tool_call in tool_call_message["tool_calls"]:
                    function_name = tool_call["function"]["name"]
                    try:
                        function_arguments = json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        function_arguments = {}
                    
                    # 调用函数并获取结果
                    function_result = function_registry.get(function_name)(function_arguments)
                    
                    # 将函数结果添加到消息列表
                    tool_message = {"role": "tool", "tool_call_id": tool_call["id"], "content": function_result}
                    messages.append(tool_message)
                
                # 获取最终回答
                print("", end="", flush=True)  # 确保输出在同一行
                final_response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True
                )
                
                # 输出最终回答
                final_full_response = ""
                for chunk in final_response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        final_full_response += content
                
                # 添加助手最终回答到消息列表
                if final_full_response:
                    messages.append({"role": "assistant", "content": final_full_response})
            else:
                # 对于非函数调用的普通回答，直接添加到消息列表
                if full_response:
                    messages.append({"role": "assistant", "content": full_response})
            
            print("\n")
    
    except Exception as e:
        print(f"\n发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    chat_with_model()
